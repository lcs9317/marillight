#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FastAPI RAG server (512MB RAM safe)
- Characters in CHAR_DIR are chunked & indexed.
- Embeddings stored as memmap(float16) to avoid RAM spikes.
- Queries use REMOTE embeddings by default (Gemini/OpenAI) to prevent OOM.
- Documents can be embedded remotely too (set REMOTE_DOC_EMBEDDINGS=1) or locally via fastembed.

ENV (sane defaults for Koyeb 512MB):
  CHAR_DIR=./characters
  CACHE_DIR=./cache
  USE_EMBEDDINGS=1
  # Remote embedding controls
  REMOTE_EMBEDDINGS=1               # queries: 1=use remote (default), 0=local
  REMOTE_DOC_EMBEDDINGS=0           # docs:    1=use remote (safer), 0=local (default)
  EMBED_PROVIDER=gemini             # gemini | openai
  GEMINI_EMBED_MODEL=text-embedding-004
  OPENAI_EMBED_MODEL=text-embedding-3-small
  GEMINI_API_KEY=...
  OPENAI_API_KEY=...
  # fastembed (used only when local is chosen)
  EMB_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  EMB_BATCH=8
  DOT_BLOCK=4096
  HF_HOME=/opt/hf
  OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  MALLOC_ARENA_MAX=2
"""

import os, re, glob, json, pickle, hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

from fastapi import FastAPI
from pydantic import BaseModel

# =========================
# Config
# =========================
CHAR_DIR         = os.environ.get("CHAR_DIR", "./characters")
CACHE_DIR        = os.environ.get("CACHE_DIR", "./cache")
INDEX_CACHE      = os.path.join(CACHE_DIR, "index.pkl")
EMB_CACHE        = os.path.join(CACHE_DIR, "embeddings.npy")
META_CACHE       = os.path.join(CACHE_DIR, "meta.json")

USE_EMB_ENV      = os.environ.get("USE_EMBEDDINGS", "1")
BACKEND          = "emb" if USE_EMB_ENV == "1" else "tfidf"

# Chunking
CHUNK_SIZE       = int(os.environ.get("CHUNK_SIZE", "900"))
CHUNK_OVERLAP    = int(os.environ.get("CHUNK_OVERLAP", "250"))
TOP_K_DEFAULT    = int(os.environ.get("TOP_K", "4"))
ACTIVE_NAME_HINT = os.environ.get("CURRENT_CHARACTER", "마릴라이트|Marillight")

# Remote embedding controls
REMOTE_EMBEDDINGS      = os.environ.get("REMOTE_EMBEDDINGS", "1") == "1"     # queries
REMOTE_DOC_EMBEDDINGS  = os.environ.get("REMOTE_DOC_EMBEDDINGS", "0") == "1" # docs (default off)
EMBED_PROVIDER         = os.environ.get("EMBED_PROVIDER", "gemini")          # gemini|openai
GEMINI_EMBED_MODEL     = os.environ.get("GEMINI_EMBED_MODEL", "text-embedding-004")
OPENAI_EMBED_MODEL     = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Local fastembed model (when used)
EMB_MODEL        = os.environ.get("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMB_BATCH        = int(os.environ.get("EMB_BATCH", "8"))

# Optional LLM (for answer drafting)
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY")
PROVIDER         = os.environ.get("PROVIDER")  # "openai" | "gemini" | None


# =========================
# Utils
# =========================
def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    if len(text) <= size:
        return [text]
    out: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        out.append(text[start:end])
        if end == len(text): break
        start = max(end - overlap, 0)
    return out

def file_fingerprint(paths: List[str]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        try:
            st = os.stat(p)
            h.update(p.encode())
            h.update(str(st.st_mtime_ns).encode())
            h.update(str(st.st_size).encode())
        except FileNotFoundError:
            pass
    return h.hexdigest()

def choose_active_name(all_names: List[str]) -> str:
    hint = ACTIVE_NAME_HINT.lower()
    if not all_names: return ""
    for n in all_names:
        if any(tok in n.lower() for tok in re.split(r"[|,]", hint)):
            return n
    return all_names[0]


# =========================
# Remote embeddings (Gemini/OpenAI)
# =========================
def embed_remote(texts: List[str], *, provider: str) -> List[List[float]]:
    """Return list of embedding vectors (python lists) from the chosen provider."""
    provider = provider.lower()
    if provider == "openai":
        import openai
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        openai.api_key = key
        model = os.environ.get("OPENAI_EMBED_MODEL", OPENAI_EMBED_MODEL)
        resp = openai.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]
    else:
        import google.generativeai as genai
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=key)
        model = os.environ.get("GEMINI_EMBED_MODEL", GEMINI_EMBED_MODEL)
        out: List[List[float]] = []
        # google-genai embeds one content per call; keep batch loop small for rate limits
        for t in texts:
            r = genai.embed_content(model=model, content=t, task_type="retrieval_document")
            out.append(r["embedding"])
        return out


# =========================
# Data load
# =========================
def load_documents(char_dir: str) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    paths = []
    for ext in ("*.txt", "*.md"):
        paths.extend(glob.glob(os.path.join(char_dir, ext)))
    docs, ids, metas = [], [], []
    for p in sorted(paths):
        name = os.path.splitext(os.path.basename(p))[0]
        body = read_text(p).strip()
        if not body: 
            continue
        chunks = chunk_text(body, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, ch in enumerate(chunks):
            docs.append(ch)
            ids.append(f"{name}-{i:04d}")
            metas.append({"char": name, "chunk": i, "path": p})
    return docs, ids, metas


# =====================
# Retriever
# =====================
class Retriever:
    def __init__(self, backend="emb"):
        self.backend = backend
        self.docs: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self.emb = None            # np.ndarray or np.memmap
        self.model = None          # fastembed.TextEmbedding
        self.vectorizer = None     # TfidfVectorizer
        self.matrix = None         # TF-IDF matrix
        self.doc_provider = None   # "gemini" | "openai" | "fastembed"
        self.doc_model = None      # model name
        self.dim = None            # embedding dimension

    def fit(self, docs: List[str], ids: List[str], metas: List[Dict[str, Any]]):
        self.docs, self.ids, self.metas = docs, ids, metas
        if self.backend == "tfidf":
            self._fit_tfidf()
        else:
            # Set provider/model from META_CACHE if present
            if os.path.isfile(META_CACHE):
                try:
                    meta = json.load(open(META_CACHE, "r", encoding="utf-8"))
                    self.doc_provider = meta.get("provider")
                    self.doc_model = meta.get("model")
                    self.dim = meta.get("dim")
                except Exception:
                    pass
            self._fit_embeddings()

    def _fit_tfidf(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.95)
        self.matrix = self.vectorizer.fit_transform(self.docs)

    def _lazy_load_model(self):
        if self.model is None:
            from fastembed import TextEmbedding
            cache_dir = os.environ.get("HF_HOME", "/tmp/hf")
            name = os.environ.get("EMB_MODEL", EMB_MODEL)
            try:
                self.model = TextEmbedding(model_name=name, cache_dir=cache_dir)
            except ValueError:
                # unsupported model name -> fallback to multilingual MiniLM
                fallback = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                print(f"[warn] Unsupported model '{name}'. Falling back to '{fallback}'.")
                self.model = TextEmbedding(model_name=fallback, cache_dir=cache_dir)

    def _fit_embeddings(self):
        """
        Build doc embeddings into memmap. Uses remote if REMOTE_DOC_EMBEDDINGS=1,
        otherwise fastembed locally (streamed).
        """
        import numpy as np
        os.makedirs(os.path.dirname(EMB_CACHE), exist_ok=True)

        # If cache matches, just mmap
        if os.path.isfile(EMB_CACHE) and os.path.isfile(META_CACHE):
            try:
                meta = json.load(open(META_CACHE, "r", encoding="utf-8"))
                if meta.get("finger") == CURRENT_FINGER:
                    self.emb = np.load(EMB_CACHE, mmap_mode="r")
                    self.doc_provider = meta.get("provider")
                    self.doc_model = meta.get("model")
                    self.dim = int(meta.get("dim"))
                    return
            except Exception:
                pass

        # Build fresh
        if REMOTE_DOC_EMBEDDINGS:
            # --- Remote (Gemini/OpenAI) ---
            provider = EMBED_PROVIDER
            # probe for dim
            probe = embed_remote([self.docs[0]], provider=provider)[0]
            dim = len(probe)
            self.dim = dim
            dtype = np.float16
            from numpy.lib.format import open_memmap
            mmap = open_memmap(EMB_CACHE, mode="w+", dtype=dtype, shape=(len(self.docs), dim))
            B = int(os.environ.get("EMB_BATCH", "8"))
            for s in range(0, len(self.docs), B):
                chunk = self.docs[s:s+B]
                vecs = embed_remote(chunk, provider=provider)
                arr = np.asarray(vecs, dtype=np.float32)
                arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
                mmap[s:s+arr.shape[0], :] = arr.astype(dtype)
            del mmap
            self.emb = np.load(EMB_CACHE, mmap_mode="r")
            meta = {"finger": CURRENT_FINGER, "dim": dim, "dtype": str(self.emb.dtype),
                    "provider": provider, "model": GEMINI_EMBED_MODEL if provider=="gemini" else OPENAI_EMBED_MODEL}
            with open(META_CACHE, "w", encoding="utf-8") as f:
                json.dump(meta, f)
            self.doc_provider = meta["provider"]
            self.doc_model = meta["model"]
        else:
            # --- Local fastembed streamed ---
            from fastembed import TextEmbedding
            self._lazy_load_model()
            probe = list(self.model.embed([self.docs[0]]))[0]
            dim = len(probe)
            self.dim = dim
            dtype = np.float16
            from numpy.lib.format import open_memmap
            mmap = open_memmap(EMB_CACHE, mode="w+", dtype=dtype, shape=(len(self.docs), dim))
            B = int(os.environ.get("EMB_BATCH", "8"))
            for s in range(0, len(self.docs), B):
                chunk = self.docs[s:s+B]
                vecs = list(self.model.embed(chunk))
                arr = np.asarray(vecs, dtype=np.float32)
                arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
                mmap[s:s+arr.shape[0], :] = arr.astype(dtype)
            del mmap
            self.emb = np.load(EMB_CACHE, mmap_mode="r")
            meta = {"finger": CURRENT_FINGER, "dim": dim, "dtype": str(self.emb.dtype),
                    "provider": "fastembed", "model": os.environ.get("EMB_MODEL", EMB_MODEL)}
            with open(META_CACHE, "w", encoding="utf-8") as f:
                json.dump(meta, f)
            self.doc_provider = "fastembed"
            self.doc_model = meta["model"]

    def _q_embed(self, q: str) -> "np.ndarray":
        """Embed a single query to match doc space/provider/dim."""
        import numpy as np
        # Decide provider by doc cache
        provider = self.doc_provider or ("gemini" if REMOTE_EMBEDDINGS else "fastembed")
        if provider in ("gemini", "openai"):
            qv = np.asarray(embed_remote([q], provider=provider)[0], dtype=np.float32)
        else:
            # local fastembed
            self._lazy_load_model()
            qv = np.asarray(list(self.model.embed([q]))[0], dtype=np.float32)
        # normalize
        qv /= (np.linalg.norm(qv) + 1e-12)
        return qv

    def query(self, q: str, top_k: int = 4) -> List[Tuple[float, Dict[str, Any]]]:
        if self.backend == "tfidf":
            import numpy as np
            qv = self.vectorizer.transform([q])
            scores = (self.matrix @ qv.T).toarray().ravel()
            idx = np.argsort(-scores)[:top_k]
            return [(float(scores[i]), {"id": self.ids[i], "meta": self.metas[i], "text": self.docs[i]}) for i in idx]
        else:
            import numpy as np
            qv = self._q_embed(q)
            # block matvec for memmap
            if hasattr(self.emb, "filename"):  # np.memmap proxy
                block = int(os.environ.get("DOT_BLOCK", "4096"))
                sims = np.empty(len(self.ids), dtype=np.float32)
                for s in range(0, len(self.ids), block):
                    A = np.asarray(self.emb[s:s+block], dtype=np.float32)
                    sims[s:s+A.shape[0]] = A @ qv
            else:
                A = np.asarray(self.emb, dtype=np.float32)
                sims = A @ qv
            idx = np.argsort(-sims)[:top_k]
            return [(float(sims[i]), {"id": self.ids[i], "meta": self.metas[i], "text": self.docs[i]}) for i in idx]


# =====================
# Index build/load
# =====================
@dataclass
class IndexPack:
    backend: str
    docs: List[str]
    ids: List[str]
    metas: List[Dict[str, Any]]
    retriever: Retriever
    finger: str
    active: str

def load_paths(char_dir: str) -> List[str]:
    paths = []
    for ext in ("*.txt", "*.md"):
        paths.extend(glob.glob(os.path.join(char_dir, ext)))
    return paths

CURRENT_FINGER = ""  # set during build

def build_index(char_dir: str, backend: str = "emb") -> IndexPack:
    global CURRENT_FINGER
    os.makedirs(CACHE_DIR, exist_ok=True)
    paths = load_paths(char_dir)
    docs, ids, metas = load_documents(char_dir)
    CURRENT_FINGER = file_fingerprint(paths) + f"|{backend}|{CHUNK_SIZE}|{CHUNK_OVERLAP}"

    # index.pkl cache (docs/ids/metas)
    try:
        if os.path.isfile(INDEX_CACHE):
            c = pickle.load(open(INDEX_CACHE, "rb"))
            if c.get("finger") == CURRENT_FINGER:
                docs, ids, metas = c["docs"], c["ids"], c["metas"]
            else:
                raise KeyError
        else:
            raise FileNotFoundError
    except Exception:
        with open(INDEX_CACHE, "wb") as f:
            pickle.dump({"docs": docs, "ids": ids, "metas": metas, "finger": CURRENT_FINGER}, f)

    retr = Retriever(backend=backend)

    # If embeddings cache exists & matches, just mmap and return
    if backend == "emb" and os.path.isfile(EMB_CACHE) and os.path.isfile(META_CACHE):
        try:
            meta = json.load(open(META_CACHE, "r", encoding="utf-8"))
            if meta.get("finger") == CURRENT_FINGER:
                import numpy as np
                retr.docs, retr.ids, retr.metas = docs, ids, metas
                retr.emb = np.load(EMB_CACHE, mmap_mode="r")
                retr.dim = int(meta.get("dim"))
                retr.doc_provider = meta.get("provider")
                retr.doc_model = meta.get("model")
                names = sorted({m["char"] for m in metas})
                active = choose_active_name(names)
                return IndexPack(backend, docs, ids, metas, retr, CURRENT_FINGER, active)
        except Exception:
            pass

    # Build embeddings now
    if backend == "tfidf":
        retr.fit(docs, ids, metas)
    else:
        retr.fit(docs, ids, metas)

    names = sorted({m["char"] for m in metas})
    active = choose_active_name(names)
    return IndexPack(backend, docs, ids, metas, retr, CURRENT_FINGER, active)


# =====================
# Optional LLM for answers
# =====================
class ChatReq(BaseModel):
    query: str
    top_k: Optional[int] = None

def system_prompt(active_name: str) -> str:
    return (
        "당신은 세계관 문서에서 근거를 찾아 대답하는 조수입니다. "
        "출처 문장에 맞춰 간결하고 정확하게 답하세요."
        + (f"\n현재 활성 캐릭터 힌트: {active_name}" if active_name else "")
    )

def generate_with_openai(sys_prompt: str, context: str, user: str) -> str:
    import openai
    openai.api_key = OPENAI_API_KEY
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "system", "content": f"[컨텍스트]\n{context}"},
        {"role": "user", "content": user},
    ]
    resp = openai.chat.completions.create(model=model, messages=messages, temperature=0.5)
    return resp.choices[0].message.content

def generate_with_gemini(sys_prompt: str, context: str, user: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"))
    prompt = f"{sys_prompt}\n\n[컨텍스트]\n{context}\n\n[사용자]\n{user}"
    resp = model.generate_content(prompt)
    return resp.text

def fallback_generate(sys_prompt: str, context: str, user: str) -> str:
    snippet = "\n".join([l for l in context.splitlines() if l.strip()][:10])
    return f"{snippet}\n\n(참고로 위 문맥을 바탕으로 요약했습니다.)"

def generate_answer(context_docs: List[str], user_input: str) -> str:
    context = "\n\n---\n\n".join(context_docs)
    sp = system_prompt(PACK.active if PACK else "")
    if PROVIDER == "openai" and OPENAI_API_KEY:
        try:
            return generate_with_openai(sp, context, user_input)
        except Exception:
            pass
    if PROVIDER == "gemini" and GEMINI_API_KEY:
        try:
            return generate_with_gemini(sp, context, user_input)
        except Exception:
            pass
    return fallback_generate(sp, context, user_input)


# =====================
# FastAPI
# =====================
app = FastAPI(title="RAG Server (memmap + remote embeddings)")

PACK: Optional[IndexPack] = None

@app.on_event("startup")
def _startup():
    global PACK
    PACK = build_index(CHAR_DIR, backend=BACKEND)

@app.get("/health")
def health():
    ok = PACK is not None
    meta = {}
    try:
        m = json.load(open(META_CACHE, "r", encoding="utf-8"))
        meta = {"provider": m.get("provider"), "model": m.get("model"), "dim": m.get("dim")}
    except Exception:
        pass
    return {"ok": ok, "backend": BACKEND, "active": getattr(PACK, "active", None), "emb": meta}

@app.get("/stats")
def stats():
    if PACK is None: return {"ok": False}
    names = sorted({m["char"] for m in PACK.metas})
    return {"ok": True, "backend": PACK.backend, "chars": names, "docs": len(PACK.docs)}

@app.post("/reload")
def reload_index():
    global PACK
    PACK = build_index(CHAR_DIR, backend=BACKEND)
    return {"ok": True, "docs": len(PACK.docs), "chars": sorted({m["char"] for m in PACK.metas})}

class ChatResp(BaseModel):
    ok: bool
    answer: str
    contexts: List[Dict[str, Any]]

@app.post("/chat", response_model=ChatResp)
def chat(req: ChatReq):
    if PACK is None:
        return {"ok": False, "answer": "", "contexts": []}
    k = req.top_k or TOP_K_DEFAULT
    hits = PACK.retriever.query(req.query, top_k=k)
    ctx_docs = [h[1]["text"] for h in hits]
    answer = generate_answer(ctx_docs, req.query)
    results = [
        {"score": float(h[0]), "id": h[1]["id"], "char": h[1]["meta"]["char"], "chunk": h[1]["meta"]["chunk"]}
        for h in hits
    ]
    return {"ok": True, "answer": answer, "contexts": results}

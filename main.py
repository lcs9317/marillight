#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FastAPI RAG server (512MB RAM friendly)
- Character .txt files in CHAR_DIR are chunked and indexed
- Embedding backend uses fastembed with streaming memmap (.npy) to avoid OOM
- TF-IDF backend kept as fallback but default is embeddings
- Cached index and embeddings are reused across restarts

Environment variables (sane defaults for Koyeb Free):
  CHAR_DIR=./characters
  CACHE_DIR=./cache
  USE_EMBEDDINGS=1                 # 1: use embeddings, 0: TF-IDF
  EMB_MODEL=intfloat/multilingual-e5-small
  EMB_BATCH=8                       # smaller -> lower peak RAM
  DOT_BLOCK=4096                    # block size for sim calc on memmap
  HF_HOME=/opt/hf                   # model cache
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
# Config (환경변수로 조절)
# =========================
CHAR_DIR         = os.environ.get("CHAR_DIR", "./characters")
CACHE_DIR        = os.environ.get("CACHE_DIR", "./cache")
INDEX_CACHE      = os.path.join(CACHE_DIR, "index.pkl")
EMB_CACHE        = os.path.join(CACHE_DIR, "embeddings.npy")
META_CACHE       = os.path.join(CACHE_DIR, "meta.json")

# 임베딩 기본 ON (필요시만 TF-IDF로 내릴 수 있게 스위치 유지)
USE_EMB_ENV      = os.environ.get("USE_EMBEDDINGS", "1")
BACKEND          = "emb" if USE_EMB_ENV == "1" else "tfidf"

# 청크/검색 파라미터
CHUNK_SIZE       = int(os.environ.get("CHUNK_SIZE", "900"))
CHUNK_OVERLAP    = int(os.environ.get("CHUNK_OVERLAP", "250"))
TOP_K_DEFAULT    = int(os.environ.get("TOP_K", "4"))
ACTIVE_NAME_HINT = os.environ.get("CURRENT_CHARACTER", "마릴라이트|Marillight")

# fastembed 임베딩 모델 (다국어 권장)
EMB_MODEL        = os.environ.get("EMB_MODEL", "intfloat/multilingual-e5-small")
EMB_BATCH        = int(os.environ.get("EMB_BATCH", "8"))

# LLM 선택(없어도 동작)
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY")
PROVIDER         = os.environ.get("PROVIDER")  # "openai" | "gemini" | None


# =========================
# 유틸리티
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
    # 힌트와 가장 잘 매칭되는 이름을 active로
    hint = ACTIVE_NAME_HINT.lower()
    best = all_names[0] if all_names else ""
    for n in all_names:
        if any(tok in n.lower() for tok in re.split(r"[|,]", hint)):
            return n
    return best

# =========================
# 데이터 적재
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
# 검색기
# =====================
class Retriever:
    def __init__(self, backend="emb"):
        self.backend = backend
        self.docs: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self.emb = None
        self.model = None
        self.vectorizer = None
        self.matrix = None

    def fit(self, docs: List[str], ids: List[str], metas: List[Dict[str, Any]]):
        self.docs, self.ids, self.metas = docs, ids, metas
        if self.backend == "tfidf":
            self._fit_tfidf()
        else:
            self._fit_fastembed()

    def _fit_tfidf(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.95)
        self.matrix = self.vectorizer.fit_transform(self.docs)

    def _lazy_load_model(self):
        if self.model is None:
            from fastembed import TextEmbedding
            cache_dir = os.environ.get("HF_HOME", "/tmp/hf")
            self.model = TextEmbedding(model_name=EMB_MODEL, cache_dir=cache_dir)

    def _fit_fastembed(self):
        import numpy as np
        from fastembed import TextEmbedding
        from numpy.lib.format import open_memmap
        cache_dir = os.environ.get("HF_HOME", "/tmp/hf")
        # 로드
        self.model = TextEmbedding(model_name=EMB_MODEL, cache_dir=cache_dir)
        docs = self.docs
        if not docs:
            self.emb = np.empty((0, 0), dtype=np.float32)
            return
        # 차원 파악용 1회 추론
        probe = list(self.model.embed([docs[0]]))[0]
        dim = len(probe)
        os.makedirs(os.path.dirname(EMB_CACHE), exist_ok=True)
        B = int(os.environ.get("EMB_BATCH", "8"))
        dtype = np.float16  # 절반 용량
        # .npy 형식의 파일 백드 memmap (np.load(mmap_mode='r')로 재사용 가능)
        mmap = open_memmap(EMB_CACHE, mode="w+", dtype=dtype, shape=(len(docs), dim))
        for s in range(0, len(docs), B):
            chunk = docs[s:s+B]
            vecs = list(self.model.embed(chunk))
            import numpy as np
            arr = np.asarray(vecs, dtype=np.float32)
            # 정규화
            arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
            mmap[s:s+arr.shape[0], :] = arr.astype(dtype)
        del mmap  # flush
        # 읽기 전용 메모리맵으로 붙임
        import numpy as np
        self.emb = np.load(EMB_CACHE, mmap_mode="r")

    def query(self, q: str, top_k: int = 4) -> List[Tuple[float, Dict[str, Any]]]:
        if self.backend == "tfidf":
            import numpy as np
            qv = self.vectorizer.transform([q])
            scores = (self.matrix @ qv.T).toarray().ravel()
            idx = np.argsort(-scores)[:top_k]
            return [(float(scores[i]), {"id": self.ids[i], "meta": self.metas[i], "text": self.docs[i]}) for i in idx]
        else:
            import numpy as np
            self._lazy_load_model()
            qv = list(self.model.embed([q]))[0]
            qv = np.asarray(qv, dtype=np.float32)
            qv = qv / (np.linalg.norm(qv) + 1e-12)
            # memmap이면 블록 곱으로 RAM 피크 방지
            if isinstance(self.emb, np.memmap):
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
# 인덱스 빌드/로드
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

def build_index(char_dir: str, backend: str = "emb") -> IndexPack:
    os.makedirs(CACHE_DIR, exist_ok=True)
    paths = []
    for ext in ("*.txt", "*.md"):
        paths.extend(glob.glob(os.path.join(char_dir, ext)))
    docs, ids, metas = load_documents(char_dir)
    finger = file_fingerprint(paths) + f"|{backend}|{CHUNK_SIZE}|{CHUNK_OVERLAP}"

    # index.pkl 캐시
    if os.path.isfile(INDEX_CACHE):
        try:
            with open(INDEX_CACHE, "rb") as f:
                c = pickle.load(f)
            if c.get("finger") == finger:
                docs, ids, metas = c["docs"], c["ids"], c["metas"]
        except Exception:
            pass
    else:
        with open(INDEX_CACHE, "wb") as f:
            pickle.dump({"docs": docs, "ids": ids, "metas": metas, "finger": finger}, f)

    retr = Retriever(backend=backend)

    # ---- 임베딩 캐시가 있으면 mmap 으로 로드(메모리 절약)하고 학습 생략
    if backend == "emb" and os.path.isfile(EMB_CACHE) and os.path.isfile(META_CACHE):
        try:
            meta = json.load(open(META_CACHE, "r", encoding="utf-8"))
            if meta.get("finger") == finger:
                retr.docs, retr.ids, retr.metas = docs, ids, metas
                import numpy as np
                retr.emb = np.load(EMB_CACHE, mmap_mode="r")
                names = sorted({m["char"] for m in metas})
                active = choose_active_name(names)
                return IndexPack(backend, docs, ids, metas, retr, finger, active)
        except Exception:
            pass

    # ---- 최초 1회만 임베딩 생성 후 캐시 저장 (임베딩은 open_memmap으로 스트리밍 생성)
    if backend == "tfidf":
        retr.fit(docs, ids, metas)
    else:
        retr.fit(docs, ids, metas)  # 내부에서 EMB_CACHE로 memmap 생성
        with open(META_CACHE, "w", encoding="utf-8") as f:
            json.dump({"finger": finger, "dim": int(retr.emb.shape[1]), "dtype": str(retr.emb.dtype)}, f)

    names = sorted({m["char"] for m in metas})
    active = choose_active_name(names)
    return IndexPack(backend, docs, ids, metas, retr, finger, active)

# =====================
# LLM (선택 사항)
# =====================
class ChatReq(BaseModel):
    query: str
    top_k: Optional[int] = None

def system_prompt(active_name: str) -> str:
    return (
        "당신은 세계관 문서에서 근거를 찾아 대답하는 조수입니다. "
        "출처 문장에 맞춰 간결하고 정확하게 답하세요. 필요하면 bullet로 정리하세요."
        + (f"\n현재 활성 캐릭터 힌트: {active_name}" if active_name else "")
    )

def generate_with_openai(sys_prompt: str, context: str, user: str) -> str:
    import openai
    openai.api_key = OPENAI_API_KEY
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "system", "content": f"[참고 맥락]\n{context}"},
        {"role": "user", "content": user},
    ]
    resp = openai.chat.completions.create(model=model, messages=messages, temperature=0.6)
    return resp.choices[0].message.content

def generate_with_gemini(sys_prompt: str, context: str, user: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(os.environ.get("GEMINI_MODEL", "gemini-1.5-pro"))
    prompt = f"{sys_prompt}\n\n[참고 맥락]\n{context}\n\n[사용자]\n{user}"
    resp = model.generate_content(prompt)
    return resp.text

def fallback_generate(sys_prompt: str, context: str, user: str) -> str:
    snippet = "\n".join([l for l in context.splitlines() if l.strip()][:10])
    return (
        "관련 정보를 요약해보겠습니다.\n\n"
        f"{snippet}\n\n"
        "위 내용을 바탕으로 답했습니다. 더 궁금한 점 있나요?"
    )

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
app = FastAPI(title="RAG Server (memmap)")

PACK: Optional[IndexPack] = None

@app.on_event("startup")
def _startup():
    global PACK
    PACK = build_index(CHAR_DIR, backend=BACKEND)

@app.get("/health")
def health():
    ok = PACK is not None
    return {"ok": ok, "backend": BACKEND, "active": getattr(PACK, "active", None)}

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

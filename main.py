#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, glob, json, pickle, hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

from fastapi import FastAPI
from pydantic import BaseModel

# ===== Config =====
CHAR_DIR         = os.environ.get("CHAR_DIR", "./characters")
CACHE_DIR        = os.environ.get("CACHE_DIR", "./cache")
INDEX_CACHE      = os.path.join(CACHE_DIR, "index.pkl")

# 기본: 임베딩 검색(초기 느림 허용)
USE_EMB_ENV      = os.environ.get("USE_EMBEDDINGS")
BACKEND          = "emb" if (USE_EMB_ENV is None or USE_EMB_ENV == "1") else "tfidf"

CHUNK_SIZE       = int(os.environ.get("CHUNK_SIZE", "900"))
CHUNK_OVERLAP    = int(os.environ.get("CHUNK_OVERLAP", "250"))
TOP_K_DEFAULT    = int(os.environ.get("TOP_K", "4"))
ACTIVE_NAME_HINT = os.environ.get("CURRENT_CHARACTER", "마릴라이트|Marillight")

EMB_MODEL        = os.environ.get("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMB_BATCH        = int(os.environ.get("EMB_BATCH", "16"))
EMB_FP16         = os.environ.get("EMB_FP16", "1") == "1"

OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY")
PROVIDER         = os.environ.get("PROVIDER")  # "openai" | "gemini" | None

os.makedirs(CACHE_DIR, exist_ok=True)

SECTION_MAP = {
    "캐릭터 소개": "bio",
    "외형": "appearance",
    "외형 묘사": "appearance",
    "외형 디테일": "appearance_detail",
    "외형 디테일(참조용)": "appearance_detail",
    "성격 및 말투": "personality",
    "스토리 주요 행적": "story",
    "인간관계": "relations",
    "챗봇 인식 키워드": "keywords",
}
SECTION_HEADER_RE = re.compile(r"^\s*\[(.+?)\]\s*$")

def file_fingerprint(paths: List[str]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        try:
            st = os.stat(p)
            h.update(p.encode("utf-8"))
            h.update(str(st.st_mtime_ns).encode())
            h.update(str(st.st_size).encode())
        except FileNotFoundError:
            continue
    return h.hexdigest()

def read_text(p: str) -> str:
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def normalize_line(s: str) -> str:
    return s.rstrip("\n\r")

def parse_character_txt(path: str) -> Dict[str, Any]:
    raw = read_text(path)
    lines = [normalize_line(x) for x in raw.splitlines()]
    header: Dict[str, str] = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line: i += 1; continue
        if line.startswith("["): break
        if ":" in line:
            k, v = line.split(":", 1)
            header[k.strip()] = v.strip()
        i += 1

    sections: Dict[str, List[str]] = {
        "bio": [], "appearance": [], "appearance_detail": [],
        "personality": [], "story": [], "relations": [], "keywords": []
    }
    current_key = None
    buf: List[str] = []

    def flush():
        nonlocal buf, current_key
        if current_key:
            text = "\n".join(buf).strip()
            if text:
                if current_key == "keywords":
                    items = [l.strip("-• ").strip() for l in text.splitlines() if l.strip()]
                    sections[current_key].extend(items)
                else:
                    sections[current_key].append(text)
        buf = []

    while i < len(lines):
        m = SECTION_HEADER_RE.match(lines[i])
        if m:
            flush()
            src = m.group(1).strip()
            current_key = SECTION_MAP.get(src, None) or "bio"
            i += 1; continue
        buf.append(lines[i]); i += 1
    flush()

    name = header.get("이름") or header.get("Name") or os.path.basename(path).split(".")[0]
    header["_name"] = name.strip()
    return {"path": path, "header": header, "sections": sections}

def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    if len(text) <= size: return [text]
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        out.append(text[start:end])
        if end == len(text): break
        start = max(0, end - overlap)
    return out

def build_lore_doc(name: str, ch: Dict[str, Any], all_names: List[str]) -> str:
    s = ch["sections"]; parts: List[str] = []
    if s["bio"]:               parts.append("[캐릭터 소개]\n" + "\n\n".join(s["bio"]))
    if s["appearance"]:        parts.append("[외형]\n" + "\n\n".join(s["appearance"]))
    if s["appearance_detail"]: parts.append("[외형 디테일]\n" + "\n\n".join(s["appearance_detail"]))
    if s["personality"]:       parts.append("[성격 및 말투]\n" + "\n\n".join(s["personality"]))
    if s["story"]:             parts.append("[스토리 주요 행적]\n" + "\n\n".join(s["story"]))
    if s["relations"]:         parts.append("[인간관계]\n" + "\n\n".join(s["relations"]))
    if s["keywords"]:          parts.append("[챗봇 인식 키워드]\n- " + "\n- ".join(s["keywords"]))
    others = [n for n in all_names if n != name]
    if others: parts.append(f"[참조]\n관계/인지 대상: {', '.join(others)}")
    return f"=== {name} ===\n" + "\n\n".join(parts)

# ===== Retriever =====
class Retriever:
    def __init__(self, backend="emb"):
        self.backend = backend
        self.docs: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self.model = None
        self.vectorizer = None
        self.matrix = None
        self.emb = None  # np.ndarray

    def fit(self, docs: List[str], ids: List[str], metas: List[Dict[str, Any]], precomputed_emb=None):
        self.docs, self.ids, self.metas = docs, ids, metas
        if self.backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.95)
            self.matrix = self.vectorizer.fit_transform(self.docs)
        else:
            import numpy as np
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(EMB_MODEL)
            if precomputed_emb is not None:
                self.emb = precomputed_emb
            else:
                self.emb = self.model.encode(
                    self.docs, normalize_embeddings=True,
                    batch_size=EMB_BATCH, show_progress_bar=True
                ).astype(np.float32)

    def query(self, q: str, top_k: int = 4) -> List[Tuple[float, Dict[str, Any]]]:
        if self.backend == "tfidf":
            import numpy as np
            qv = self.vectorizer.transform([q])
            scores = (self.matrix @ qv.T).toarray().ravel()
            idx = np.argsort(-scores)[:top_k]
            return [(float(scores[i]), {"id": self.ids[i], "meta": self.metas[i], "text": self.docs[i]}) for i in idx]
        else:
            import numpy as np
            qv = self.model.encode([q], normalize_embeddings=True, batch_size=1, show_progress_bar=False)[0]
            sims = self.emb @ qv
            idx = np.argsort(-sims)[:top_k]
            return [(float(sims[i]), {"id": self.ids[i], "meta": self.metas[i], "text": self.docs[i]}) for i in idx]

# ===== Embedding cache =====
def load_cached_embeddings(finger: str):
    import numpy as np, json
    npy = os.path.join(CACHE_DIR, f"emb_{finger}.npy")
    meta = os.path.join(CACHE_DIR, f"emb_{finger}.json")
    if os.path.isfile(npy) and os.path.isfile(meta):
        with open(meta, "r", encoding="utf-8") as f:
            info = json.load(f)
        arr = np.load(npy, mmap_mode=None)
        return arr, info
    return None, None

def save_cached_embeddings(finger: str, emb, info: Dict[str, Any]):
    import numpy as np, json
    npy = os.path.join(CACHE_DIR, f"emb_{finger}.npy")
    meta = os.path.join(CACHE_DIR, f"emb_{finger}.json")
    if EMB_FP16 and emb.dtype != "float16":
        emb = emb.astype("float16")
    np.save(npy, emb)
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

# ===== Index pack =====
@dataclass
class IndexPack:
    backend: str
    docs: List[str]
    ids: List[str]
    metas: List[Dict[str, Any]]
    retriever: Retriever
    finger: str
    active: str

def choose_active_name(all_names: List[str]) -> str:
    try:
        pat = re.compile(ACTIVE_NAME_HINT, re.IGNORECASE)
        hits = [n for n in all_names if pat.search(n)]
        return hits[0] if hits else (all_names[0] if all_names else "마릴라이트")
    except re.error:
        return all_names[0] if all_names else "마릴라이트"

def build_index(char_dir: str, backend="emb") -> IndexPack:
    paths = glob.glob(os.path.join(char_dir, "*.txt"))
    if not paths:
        raise FileNotFoundError(f"캐릭터 파일(.txt)이 없습니다: {char_dir}")
    finger = file_fingerprint(paths)

    # 문서 캐시
    if os.path.isfile(INDEX_CACHE):
        try:
            with open(INDEX_CACHE, "rb") as f:
                cache = pickle.load(f)
            if cache.get("finger") == finger:
                docs, ids, metas = cache["docs"], cache["ids"], cache["metas"]
            else:
                docs, ids, metas = None, None, None
        except Exception:
            docs, ids, metas = None, None, None
    else:
        docs, ids, metas = None, None, None

    if docs is None:
        parsed = [parse_character_txt(p) for p in paths]
        names  = [d["header"]["_name"] for d in parsed]
        docs, ids, metas = [], [], []
        for d in parsed:
            name = d["header"]["_name"]
            full = build_lore_doc(name, d, names)
            chunks = chunk_text(full, CHUNK_SIZE, CHUNK_OVERLAP)
            for ci, ch in enumerate(chunks):
                docs.append(ch)
                ids.append(f"{name}-chunk-{ci}")
                metas.append({"char": name, "chunk": ci})
        with open(INDEX_CACHE, "wb") as f:
            pickle.dump({"docs": docs, "ids": ids, "metas": metas, "finger": finger}, f)

    retr = Retriever(backend=backend)
    pre_emb = None
    if backend == "emb":
        cached, info = load_cached_embeddings(finger)
        if cached is not None:
            pre_emb = cached
        else:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            model = SentenceTransformer(EMB_MODEL)
            doc_emb = model.encode(
                docs, normalize_embeddings=True,
                batch_size=EMB_BATCH, show_progress_bar=True
            ).astype("float32")
            pre_emb = doc_emb
            save_cached_embeddings(finger, pre_emb, {"model": EMB_MODEL, "size": len(docs)})
    retr.fit(docs, ids, metas, precomputed_emb=pre_emb)

    names = sorted({m["char"] for m in metas})
    active = choose_active_name(names)
    return IndexPack(backend, docs, ids, metas, retr, finger, active)

# ===== LLM =====
SYSTEM_PROMPT = """당신은 '마릴라이트'입니다. 아스트리 아스니아의 초지능체이자 아스니아 아카데미 학장.
상냥하고 차분한 말투로, 존대를 유지하고, 때때로 호기심 어린 밝은 톤을 보입니다.
아키텍트를 '아키텍트님'이라 부릅니다.
답변에는 필요한 만큼만 맥락을 자연스럽게 녹이고, 과한 설정 덤핑은 피합니다.
"""

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
        "아키텍트님, 요청하신 내용에 대해 간략히 정리해보겠습니다.\n\n"
        f"{snippet}\n\n"
        "위의 설정을 바탕으로 답을 드렸습니다. 더 자세한 항목이 필요하시면 알려주세요."
    )

def generate_answer(context_docs: List[str], user_input: str) -> str:
    context = "\n\n---\n\n".join(context_docs)
    provider = PROVIDER or ("openai" if OPENAI_API_KEY else ("gemini" if GEMINI_API_KEY else "fallback"))
    if provider == "openai":
        return generate_with_openai(SYSTEM_PROMPT, context, user_input)
    elif provider == "gemini":
        return generate_with_gemini(SYSTEM_PROMPT, context, user_input)
    else:
        return fallback_generate(SYSTEM_PROMPT, context, user_input)

# ===== FastAPI =====
app = FastAPI(title="Marillight RAG")
PACK: Optional[IndexPack] = None

class ChatReq(BaseModel):
    query: str
    top_k: Optional[int] = None

@app.on_event("startup")
def _startup():
    global PACK
    print(f"[info] backend: {BACKEND}  (cache: ./cache)")
    PACK = build_index(CHAR_DIR, backend=BACKEND)
    print(f"[info] active persona: {PACK.active}; docs={len(PACK.docs)}")

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
    return {"ok": True, "docs": len(PACK.docs)}

@app.post("/chat")
def chat(req: ChatReq):
    if PACK is None:
        return {"ok": False, "error": "index not ready"}
    k = req.top_k or TOP_K_DEFAULT
    hits = PACK.retriever.query(req.query, top_k=k)
    ctx_docs = [h[1]["text"] for h in hits]
    answer = generate_answer(ctx_docs, req.query)
    results = [
        {"score": float(h[0]), "id": h[1]["id"], "char": h[1]["meta"]["char"], "chunk": h[1]["meta"]["chunk"]}
        for h in hits
    ]
    return {"ok": True, "answer": answer, "contexts": results}

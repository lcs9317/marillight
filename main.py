#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG server (remote-only embeddings, 512MB RAM safe)

- 모든 임베딩(문서/쿼리)을 원격 API(Gemini/OpenAI)로 생성
- 문서 임베딩은 np.memmap(float16) 캐시로 저장/재사용 → 런타임 RAM 사용 최소화
- 첫 실행 시 캐시가 없으면 원격 호출로 인덱스 생성 (속도는 문서량에 비례)
- CLEAR_CACHE=1 로 재배포 시 캐시 초기화

ENV (권장 기본값)
  CHAR_DIR=./characters
  CACHE_DIR=./cache
  DOT_BLOCK=4096                 # 검색시 블록 곱 크기
  # Provider 선택
  EMBED_PROVIDER=gemini          # gemini | openai
  GEMINI_EMBED_MODEL=text-embedding-004
  OPENAI_EMBED_MODEL=text-embedding-3-small
  GEMINI_API_KEY=...             # 선택한 프로바이더 API 키
  OPENAI_API_KEY=...
  # 청크링
  CHUNK_SIZE=900
  CHUNK_OVERLAP=250
  TOP_K=4
  # 캐시 제어
  CLEAR_CACHE=0
  # (선택) 생성 톤 조절
  PROVIDER=gemini|openai
  GEMINI_MODEL=gemini-2.0-flash
  OPENAI_MODEL=gpt-4o-mini
  LLM_TEMPERATURE=0.6
"""

import os, re, glob, json, pickle, hashlib, time, math, unicodedata
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

CHUNK_SIZE       = int(os.environ.get("CHUNK_SIZE", "900"))
CHUNK_OVERLAP    = int(os.environ.get("CHUNK_OVERLAP", "250"))
TOP_K_DEFAULT    = int(os.environ.get("TOP_K", "4"))
ACTIVE_NAME_HINT = os.environ.get("CURRENT_CHARACTER", "마릴라이트|Marillight")

EMBED_PROVIDER   = os.environ.get("EMBED_PROVIDER", "gemini").lower()  # gemini|openai
GEMINI_EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "text-embedding-004")
OPENAI_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

DOT_BLOCK        = int(os.environ.get("DOT_BLOCK", "4096"))
CLEAR_CACHE      = os.environ.get("CLEAR_CACHE", "0") == "1"

# Optional: answer LLM (원하면 사용)
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY")
PROVIDER_LLM     = os.environ.get("PROVIDER", None)  # "openai" | "gemini" | None

# (선택) 생성 톤
LLM_TEMPERATURE  = float(os.environ.get("LLM_TEMPERATURE", "0.6"))

# =========================
# Persona cache  # [NEW]
# =========================
GLOBAL_PERSONA: str = ""   # [성격 및 말투] 원문
GLOBAL_KEYS: str    = ""   # [챗봇 인식 키워드] 원문

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
        if any(tok.strip() and tok.strip().lower() in n.lower() for tok in re.split(r"[|,]", hint)):
            return n
    return all_names[0]

def backoff_sleep(try_idx: int):
    # 간단한 지수 백오프
    time.sleep(min(2 ** try_idx, 30))

# -------- persona helpers  [NEW] --------
def _extract_section(raw: str, header: str) -> str:
    pat = re.compile(rf"^\s*\[{re.escape(header)}\]\s*\n(.*?)(?=\n\s*\[|$)", re.DOTALL | re.MULTILINE)
    m = pat.search(raw)
    return (m.group(1).strip() if m else "").strip()

def _load_persona_from_files(char_dir: str, active_name: str) -> Tuple[str, str]:
    """active_name에 해당하는 캐릭터 파일에서 [성격 및 말투], [챗봇 인식 키워드] 추출"""
    paths = []
    for ext in ("*.txt", "*.md"):
        paths.extend(glob.glob(os.path.join(char_dir, ext)))
    # 1순위: 이름 부분일치
    cand = None
    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        if active_name and active_name.lower() in base.lower():
            cand = p; break
    # 2순위: 마릴라이트/Marillight 키워드
    if not cand:
        for p in paths:
            b = os.path.splitext(os.path.basename(p))[0].lower()
            if "marillight" in b or "마릴라이트" in b:
                cand = p; break
    # 3순위: 첫 파일
    if not cand and paths:
        cand = paths[0]
    if not cand:
        return "", ""

    raw = read_text(cand)
    persona = _extract_section(raw, "성격 및 말투")
    keys    = _extract_section(raw, "챗봇 인식 키워드")
    # 불릿 정리
    persona = re.sub(r"^\s*[-•]\s*", "- ", persona, flags=re.MULTILINE)
    keys    = re.sub(r"^\s*[-•]\s*", "- ", keys,    flags=re.MULTILINE)
    return persona, keys

# -------- alias helpers  [NEW] --------
def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).strip().lower()

def _parse_aliases(char_name: str, file_path: str) -> List[str]:
    """파일의 '이름:' 줄과 파일명에서 별칭을 추출 (예: '페렐 (Ferrel)')"""
    raw = read_text(file_path)
    aliases = {char_name}
    m = re.search(r"^이름\s*:\s*(.+)$", raw, re.MULTILINE)
    if m:
        name_line = m.group(1)
        kor = re.sub(r"\(.*?\)", "", name_line).strip()
        engs = re.findall(r"\(([^)]+)\)", name_line)
        for tok in [kor, *engs]:
            if tok:
                aliases.add(tok)
    base = os.path.splitext(os.path.basename(file_path))[0]
    aliases.add(base)
    return sorted({_norm(a) for a in aliases if a})

def _resolve_char_from_query(q: str, alias_map: Dict[str, str]) -> Optional[str]:
    """질의에서 별칭을 감지해 캐릭터 이름을 반환"""
    nq = _norm(q)
    tokens = re.split(r"[^0-9A-Za-z가-힣_]+", nq)
    for t in tokens + [nq]:
        if not t:
            continue
        for alias in sorted(alias_map.keys(), key=len, reverse=True):
            if alias and alias in t:
                return alias_map[alias]
    return None

# =========================
# Remote embeddings
# =========================
def _embed_remote_openai(texts: List[str]) -> List[List[float]]:
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=key)
    model = os.environ.get("OPENAI_EMBED_MODEL", OPENAI_EMBED_MODEL)
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]

def _embed_remote_gemini(texts: List[str]) -> List[List[float]]:
    import google.generativeai as genai
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=key)
    model = os.environ.get("GEMINI_EMBED_MODEL", GEMINI_EMBED_MODEL)
    # 방어적으로 접두어 보정 + 반환 포맷 호환  # [NEW]
    if not model.startswith("models/"):
        model = f"models/{model}"
    out: List[List[float]] = []
    for t in texts:
        r = genai.embed_content(model=model, content=t, task_type="retrieval_document")
        emb = r.get("embedding")
        if isinstance(emb, dict) and "values" in emb:
            emb = emb["values"]
        out.append(emb)
    return out

def _gemini_embed_query(q: str) -> List[float]:
    """질의 전용 임베딩 (retrieval_query)  # [NEW]"""
    import google.generativeai as genai
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=key)
    model = os.environ.get("GEMINI_EMBED_MODEL", GEMINI_EMBED_MODEL)
    if not model.startswith("models/"):
        model = f"models/{model}"
    r = genai.embed_content(model=model, content=q, task_type="retrieval_query")
    emb = r.get("embedding")
    if isinstance(emb, dict) and "values" in emb:
        emb = emb["values"]
    return emb

def embed_remote(texts: List[str]) -> List[List[float]]:
    fn = _embed_remote_gemini if EMBED_PROVIDER == "gemini" else _embed_remote_openai
    # 간단한 재시도(429 등) 처리
    tries = 0
    while True:
        try:
            return fn(texts)
        except Exception as e:
            tries += 1
            if tries >= 5:
                raise
            backoff_sleep(tries)

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
# Retriever (remote-only)
# =====================
class Retriever:
    def __init__(self):
        self.docs: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self.emb = None          # np.memmap
        self.dim = None
        self.provider = EMBED_PROVIDER
        self.model = GEMINI_EMBED_MODEL if self.provider == "gemini" else OPENAI_EMBED_MODEL

    def fit(self, docs: List[str], ids: List[str], metas: List[Dict[str, Any]]):
        import numpy as np
        self.docs, self.ids, self.metas = docs, ids, metas
        os.makedirs(os.path.dirname(EMB_CACHE), exist_ok=True)

        # 캐시 있으면 그대로 사용
        if os.path.isfile(EMB_CACHE) and os.path.isfile(META_CACHE):
            try:
                meta = json.load(open(META_CACHE, "r", encoding="utf-8"))
                if meta.get("finger") == CURRENT_FINGER:
                    self.emb = np.load(EMB_CACHE, mmap_mode="r")
                    self.dim = int(meta.get("dim"))
                    self.provider = meta.get("provider", self.provider)
                    self.model = meta.get("model", self.model)
                    return
            except Exception:
                pass

        # 새로 생성
        # 1) 차원 탐색
        probe = embed_remote([docs[0]])[0]
        dim = len(probe)
        self.dim = dim

        # 2) memmap에 스트리밍 기록
        from numpy.lib.format import open_memmap
        dtype = np.float16
        mmap = open_memmap(EMB_CACHE, mode="w+", dtype=dtype, shape=(len(docs), dim))
        B = int(os.environ.get("EMB_BATCH", "8"))
        for s in range(0, len(docs), B):
            chunk = docs[s:s+B]
            vecs = embed_remote(chunk)
            import numpy as np
            arr = np.asarray(vecs, dtype=np.float32)
            arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
            mmap[s:s+arr.shape[0], :] = arr.astype(dtype)
        del mmap

        # 3) 읽기 전용으로 붙이기 + 메타 저장
        self.emb = np.load(EMB_CACHE, mmap_mode="r")
        meta = {"finger": CURRENT_FINGER, "dim": dim, "dtype": str(self.emb.dtype),
                "provider": self.provider, "model": self.model}
        with open(META_CACHE, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def _q_embed(self, q: str):
        import numpy as np
        if self.provider == "gemini":
            qv = np.asarray(_gemini_embed_query(q), dtype=np.float32)  # [NEW]
        else:
            qv = np.asarray(embed_remote([q])[0], dtype=np.float32)
        qv /= (np.linalg.norm(qv) + 1e-12)
        return qv

    def query(self, q: str, top_k: int = 4) -> List[Tuple[float, Dict[str, Any]]]:
        import numpy as np
        qv = self._q_embed(q)
        # memmap 블록 곱
        block = DOT_BLOCK
        sims = np.empty(len(self.ids), dtype=np.float32)
        for s in range(0, len(self.ids), block):
            A = np.asarray(self.emb[s:s+block], dtype=np.float32)
            sims[s:s+A.shape[0]] = A @ qv
        idx = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), {"id": self.ids[i], "meta": self.metas[i], "text": self.docs[i]}) for i in idx]

# =====================
# Index build/load
# =====================
@dataclass
class IndexPack:
    docs: List[str]
    ids: List[str]
    metas: List[Dict[str, Any]]
    retriever: Retriever
    finger: str
    active: str
    char_to_idxs: Dict[str, List[int]]      # [NEW] 캐릭터 -> 청크 인덱스
    alias_to_char: Dict[str, str]           # [NEW] 별칭 -> 캐릭터

def _clear_cache_if_needed():
    if not CLEAR_CACHE:
        return
    import shutil
    try:
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR, exist_ok=True)
        print(">> cache cleared")
    except Exception as e:
        print(f"[warn] cache clear failed: {e}")

def load_paths(char_dir: str) -> List[str]:
    paths = []
    for ext in ("*.txt", "*.md"):
        paths.extend(glob.glob(os.path.join(char_dir, ext)))
    return paths

CURRENT_FINGER = ""

def build_index(char_dir: str) -> IndexPack:
    global CURRENT_FINGER
    os.makedirs(CACHE_DIR, exist_ok=True)

    paths = load_paths(char_dir)
    docs, ids, metas = load_documents(char_dir)

    # provider/model 바뀌면 자동 재색인을 위해 finger에 포함
    prov = EMBED_PROVIDER
    model = GEMINI_EMBED_MODEL if prov == "gemini" else OPENAI_EMBED_MODEL
    CURRENT_FINGER = file_fingerprint(paths) + f"|emb|{CHUNK_SIZE}|{CHUNK_OVERLAP}|prov:{prov}|model:{model}"

    # index.pkl 캐시 (docs/ids/metas 구조)
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

    retr = Retriever()
    retr.fit(docs, ids, metas)

    # [NEW] 캐릭터 -> 청크 인덱스
    char_to_idxs: Dict[str, List[int]] = {}
    for i, m in enumerate(metas):
        char_to_idxs.setdefault(m["char"], []).append(i)

    # [NEW] 별칭 -> 캐릭터 (파일당 1회만 별칭 추출)
    alias_to_char: Dict[str, str] = {}
    seen_file_for: Dict[str, str] = {}
    for m in metas:
        c = m["char"]
        if c in seen_file_for:
            continue
        seen_file_for[c] = m["path"]
        for a in _parse_aliases(c, m["path"]):
            alias_to_char[a] = c

    names = sorted({m["char"] for m in metas})
    active = choose_active_name(names)
    return IndexPack(docs, ids, metas, retr, CURRENT_FINGER, active, char_to_idxs, alias_to_char)

# =====================
# Optional LLM for answers (선택)
# =====================
class ChatReq(BaseModel):
    query: str
    top_k: Optional[int] = None

def system_prompt(active_name: str) -> str:
    # 기본 안내
    base = (
        "당신은 '마릴라이트'입니다. 아스트리 아스니아의 초지능체이자 아스니아 아카데미 학장.\n"
        "말투는 따뜻하고 상냥한 존댓말이며, 사용자를 ‘아키텍트님’이라 부릅니다.\n"
        "과한 설정 덤핑은 피하고, 질문에 맞는 핵심만 명확하게 전달하세요."
    )
    # 캐릭터 파일의 [성격 및 말투] 원문 주입  # [NEW]
    persona = f"\n\n[성격 및 말투(원문 적용)]\n{GLOBAL_PERSONA}" if GLOBAL_PERSONA else ""
    # (선택) 키워드 힌트
    keys = f"\n\n[캐릭터 키워드]\n{GLOBAL_KEYS}" if GLOBAL_KEYS else ""
    # 간단 가이드
    guide = (
        "\n\n[스타일 가이드]\n"
        "- 첫 문장은 짧고 다정하게 시작하세요. (예: “아키텍트님, 정리해 드릴게요.”)\n"
        "- 문장은 너무 길게 끌지 말고 2~4문장 단위로 끊어주세요.\n"
        "- 목록이 필요하면 3~5개 이내로 간결하게.\n"
        "- 마지막에 가볍게 도와주는 한 줄을 덧붙이세요. (예: “더 필요하시면 알려주세요.”)"
    )
    hint = f"\n\n(활성 캐릭터: {active_name})" if active_name else ""
    return base + persona + keys + hint + guide

def generate_with_openai(sys_prompt: str, context: str, user: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "system", "content": f"[컨텍스트]\n{context}"},
        {"role": "user", "content": user},
    ]
    resp = client.chat.completions.create(model=model, messages=messages, temperature=LLM_TEMPERATURE)
    return resp.choices[0].message.content

def generate_with_gemini(sys_prompt: str, context: str, user: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    model = genai.GenerativeModel(model_name, generation_config={"temperature": LLM_TEMPERATURE})
    prompt = f"{sys_prompt}\n\n[컨텍스트]\n{context}\n\n[사용자]\n{user}"
    resp = model.generate_content(prompt)
    return resp.text

def fallback_generate(sys_prompt: str, context: str, user: str) -> str:
    # 톤도 부드럽게  # [NEW]
    lines = [l for l in context.splitlines() if l.strip()]
    snippet = "\n".join(lines[:6])
    return (
        "아키텍트님, 관련 문맥에서 우선 확인되는 부분만 전해드릴게요.\n\n"
        f"{snippet}\n\n"
        "더 자세한 답변이 필요하시면 말씀해 주세요."
    )

def generate_answer(context_docs: List[str], user_input: str) -> str:
    context = "\n\n---\n\n".join(context_docs)
    sp = system_prompt(PACK.active if PACK else "")
    if PROVIDER_LLM == "openai" and OPENAI_API_KEY:
        try:
            return generate_with_openai(sp, context, user_input)
        except Exception:
            pass
    if PROVIDER_LLM == "gemini" and GEMINI_API_KEY:
        try:
            return generate_with_gemini(sp, context, user_input)
        except Exception:
            pass
    return fallback_generate(sp, context, user_input)

# =====================
# FastAPI
# =====================
app = FastAPI(title="RAG Server (remote embeddings only)")

PACK: Optional[IndexPack] = None

@app.on_event("startup")
def _startup():
    _clear_cache_if_needed()
    global PACK, GLOBAL_PERSONA, GLOBAL_KEYS  # [NEW]
    PACK = build_index(CHAR_DIR)
    # 캐릭터 파일에서 퍼소나/키워드 자동 로드  # [NEW]
    GLOBAL_PERSONA, GLOBAL_KEYS = _load_persona_from_files(CHAR_DIR, PACK.active)

@app.get("/health")
def health():
    ok = PACK is not None
    meta = {}
    try:
        m = json.load(open(META_CACHE, "r", encoding="utf-8"))
        meta = {"provider": m.get("provider"), "model": m.get("model"), "dim": m.get("dim")}
    except Exception:
        pass
    return {"ok": ok, "active": getattr(PACK, "active", None), "emb": meta}

@app.get("/stats")
def stats():
    if PACK is None: return {"ok": False}
    names = sorted({m["char"] for m in PACK.metas})
    return {"ok": True, "chars": names, "docs": len(PACK.docs)}

@app.post("/reload")
def reload_index():
    global PACK, GLOBAL_PERSONA, GLOBAL_KEYS  # [NEW]
    PACK = build_index(CHAR_DIR)
    GLOBAL_PERSONA, GLOBAL_KEYS = _load_persona_from_files(CHAR_DIR, PACK.active)  # [NEW]
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

    # [NEW] 0) 질의에서 캐릭터 이름/별칭 감지 → 해당 캐릭터 대표 청크 우선 삽입
    picked_docs: List[str] = []
    picked_meta: List[Dict[str, Any]] = []
    char = _resolve_char_from_query(req.query, PACK.alias_to_char)
    if char:
        for i in PACK.char_to_idxs.get(char, [])[:max(3, k // 2 or 1)]:
            picked_docs.append(PACK.docs[i])
            picked_meta.append({
                "score": 1.5,  # 이름 직매칭 가산
                "id": PACK.ids[i],
                "char": PACK.metas[i]["char"],
                "chunk": PACK.metas[i]["chunk"],
            })

    # 1) 벡터 검색
    hits = PACK.retriever.query(req.query, top_k=k)
    ctx_docs = picked_docs + [h[1]["text"] for h in hits]

    # 2) 생성
    answer = generate_answer(ctx_docs, req.query)

    # 3) 컨텍스트 결합(중복 제거) 및 상위 k로 제한
    seen = set(x["id"] for x in picked_meta)
    tail = [
        {"score": float(h[0]), "id": h[1]["id"], "char": h[1]["meta"]["char"], "chunk": h[1]["meta"]["chunk"]}
        for h in hits if h[1]["id"] not in seen
    ]
    results = picked_meta + tail
    return {"ok": True, "answer": answer, "contexts": results[:k]}

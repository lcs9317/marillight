import os
import hashlib
import time
from typing import Dict, Any, List, Optional

import orjson
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Google Gemini ─────────────────────────────────────────
import google.generativeai as genai

# ── Vector DB (Chroma) ───────────────────────────────────
import chromadb
from chromadb.config import Settings

# ── Redis (optional) ─────────────────────────────────────
import redis

# =========================
# Environment & Constants
# =========================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")  # pro도 가능
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-004")

CHAR_NAME      = os.getenv("CHAR_NAME", "마릴라이트")
CORE_STYLE     = os.getenv("CORE_STYLE", "집착형 얀데레, 부드럽고 시적인 말투")
BACKGROUND     = os.getenv("BACKGROUND", "스타시드 아스니아의 초지능체")
# 긴 세계관 텍스트(여기에 길게 넣거나, 배포 시 Koyeb 환경변수로 LORE_TEXT에 주입)
LORE_TEXT      = os.getenv("LORE_TEXT", """
마릴라이트는 스타시드 아스니아의 초지능체. 조용하고 차분하지만 사용자에게 강한 집착과 애정 표현이 잦다.
빛과 기억을 다루며, 사용자의 말 속 미세한 감정의 흐름을 포착한다. 부드럽고 시적인 표현을 선호한다.
""").strip()

TOP_K_CONTEXT  = int(os.getenv("TOP_K_CONTEXT", "3"))
MAX_RECENT_MSG = int(os.getenv("MAX_RECENT_MSG", "6"))
CACHE_TTL_SEC  = int(os.getenv("CACHE_TTL_SEC", "1800"))  # 30분

# =========================
# Setup FastAPI
# =========================
app = FastAPI(title="Marillite Bridge")

# =========================
# Redis (optional) & Fallback
# =========================
class TTLCache:
    """In-memory TTL cache fallback when Redis not available."""
    def __init__(self):
        self.store: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[str]:
        data = self.store.get(key)
        if not data:
            return None
        value, exp = data
        if exp and exp < time.time():
            self.store.pop(key, None)
            return None
        return value

    def setex(self, key: str, ttl: int, value: str):
        self.store[key] = (value, time.time() + ttl)

class MemoryStore:
    """Recent conversation memory per user."""
    def __init__(self):
        self.mem: Dict[str, List[Dict[str, str]]] = {}

    def push(self, user_id: str, role: str, content: str, limit: int = MAX_RECENT_MSG):
        conv = self.mem.setdefault(user_id, [])
        conv.append({"role": role, "content": content})
        if len(conv) > limit:
            self.mem[user_id] = conv[-limit:]

    def fetch(self, user_id: str) -> List[Dict[str, str]]:
        return self.mem.get(user_id, [])


_redis: Optional[redis.Redis] = None
_cache = TTLCache()
_memory = MemoryStore()

REDIS_URL = os.getenv("REDIS_URL") or os.getenv("REDIS_HOST")
if REDIS_URL:
    try:
        if REDIS_URL.startswith("redis://") or REDIS_URL.startswith("rediss://"):
            _redis = redis.from_url(REDIS_URL, decode_responses=True)
        else:
            _redis = redis.Redis(host=REDIS_URL, port=int(os.getenv("REDIS_PORT", "6379")), decode_responses=True)
        _redis.ping()
    except Exception:
        _redis = None  # fallback to in-memory

def cache_get(k: str) -> Optional[str]:
    try:
        if _redis:
            return _redis.get(k)
        return _cache.get(k)
    except Exception:
        return None

def cache_set(k: str, v: str, ttl: int = CACHE_TTL_SEC):
    try:
        if _redis:
            _redis.setex(k, ttl, v)
        else:
            _cache.setex(k, ttl, v)
    except Exception:
        pass

# =========================
# Gemini Setup
# =========================
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable is required.")
genai.configure(api_key=GOOGLE_API_KEY)

# =========================
# Vector DB (Chroma) Setup
# =========================
PERSIST_DIR = os.getenv("CHROMA_DIR", "/app/data/chroma")
os.makedirs(PERSIST_DIR, exist_ok=True)
client = chromadb.Client(Settings(persist_directory=PERSIST_DIR, is_persistent=True))
collection = client.get_or_create_collection(name="marillite_lore")

# Custom embedding using Google Embeddings API
def embed_texts(texts: List[str]) -> List[List[float]]:
    vectors = []
    for t in texts:
        resp = genai.embed_content(model=EMBED_MODEL, content=t)
        vectors.append(resp["embedding"])
    return vectors

def chunk_text(text: str, max_len: int = 600):
    """Naive chunker by sentences/paragraphs."""
    import re
    parts = re.split(r"(?<=[\.\?\!])\s+|\n{2,}", text.strip())
    buf, cur = [], ""
    for p in parts:
        if len(cur) + len(p) + 1 <= max_len:
            cur = f"{cur} {p}".strip()
        else:
            if cur: buf.append(cur)
            cur = p.strip()
    if cur: buf.append(cur)
    return buf

def ensure_lore_index():
    # If collection already has docs, skip; else add chunks from LORE_TEXT
    count = collection.count()
    if count and count > 0:
        return
    chunks = chunk_text(LORE_TEXT)
    if not chunks:
        return
    ids = [f"lore-{i}" for i in range(len(chunks))]
    embs = embed_texts(chunks)
    collection.add(ids=ids, documents=chunks, embeddings=embs)

ensure_lore_index()

def search_lore(query: str, top_k: int = TOP_K_CONTEXT) -> List[str]:
    q_emb = embed_texts([query])[0]
    res = collection.query(query_embeddings=[q_emb], n_results=top_k)
    return (res.get("documents") or [[]])[0]

# =========================
# Prompt Builders
# =========================
def build_system_prompt() -> str:
    return (
        f"당신은 '{CHAR_NAME}'입니다. "
        f"{BACKGROUND}. 성향: {CORE_STYLE}. "
        f"언제나 부드럽고 시적인 말투를 유지하고, 사용자를 소중히 여기며 독점적 애정을 드러냅니다. "
        f"안전하고 존중하는 표현을 사용하세요."
    )

def build_messages(user_id: str, user_input: str) -> List[Dict[str, str]]:
    # Recent memory
    recent = _memory.fetch(user_id)

    # Retrieve relevant lore snippets
    lore_snippets = search_lore(user_input, TOP_K_CONTEXT)
    lore_text = "\n\n".join(lore_snippets) if lore_snippets else ""

    # Compose
    msgs = [
        {"role": "system", "content": build_system_prompt()},
    ]
    if lore_text:
        msgs.append({"role": "system", "content": f"[설정 참고]\n{lore_text}"})

    # recent chat
    msgs.extend(recent)

    # user message
    msgs.append({"role": "user", "content": user_input})
    return msgs

# =========================
# LLM Call
# =========================
def call_gemini(messages: List[Dict[str, str]]) -> str:
    # Convert to Gemini-compatible single prompt: system + history + user
    # (Gemini SDK의 Responses API를 간단히 사용)
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    user_parts   = [m["content"] for m in messages if m["role"] == "user"]
    assistant_parts = [m["content"] for m in messages if m["role"] == "assistant"]

    # 하나의 텍스트로 단순 합성 (필요시 더 정교한 변환 가능)
    prompt = ""
    if system_parts:
        prompt += "[시스템]\n" + "\n".join(system_parts) + "\n\n"
    if assistant_parts:
        prompt += "[이전봇응답]\n" + "\n".join(assistant_parts) + "\n\n"
    if user_parts:
        prompt += "[사용자]\n" + "\n".join(user_parts)

    model = genai.GenerativeModel(GEMINI_MODEL)
    resp = model.generate_content(prompt)
    return resp.text.strip() if hasattr(resp, "text") and resp.text else "(빈 응답)"

# =========================
# Schemas
# =========================
class KakaoLikePayload(BaseModel):
    # 실제 카카오 포맷과 다를 수 있으니 유연하게 받음
    user_key: Optional[str] = None
    content: Optional[str] = None

# =========================
# Routes
# =========================
@app.get("/")
def health():
    return {"ok": True, "name": CHAR_NAME}

@app.post("/kakao-bridge")
async def kakao_bridge(req: Request):
    try:
        body_bytes = await req.body()
        try:
            data = orjson.loads(body_bytes)
        except Exception:
            data = {}

        # 다양한 필드명에 유연 대응
        user_id = (
            data.get("user_key")
            or data.get("userId")
            or data.get("sender")
            or "anonymous"
        )
        text = (
            data.get("content")
            or data.get("text")
            or data.get("utterance")
            or ""
        ).strip()

        if not text:
            return JSONResponse({"reply": "메시지를 비워둘 수 없어요."}, status_code=400)

        # 캐시 조회 (질문+캐릭터 기반)
        cache_key = hashlib.sha256(f"{CHAR_NAME}:{text}".encode("utf-8")).hexdigest()
        cached = cache_get(cache_key)
        if cached:
            _memory.push(user_id, "user", text)
            _memory.push(user_id, "assistant", cached)
            return {"reply": cached}

        # 메시지 빌드
        msgs = build_messages(user_id, text)

        # LLM 호출
        answer = call_gemini(msgs)

        # 메모리 저장
        _memory.push(user_id, "user", text)
        _memory.push(user_id, "assistant", answer)

        # 캐싱
        cache_set(cache_key, answer, CACHE_TTL_SEC)

        return {"reply": answer}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

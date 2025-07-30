import os
from pathlib import Path
from fastapi import FastAPI, Request
from pydantic import BaseModel
import hashlib
import time
import google.generativeai as genai
import chromadb
from chromadb.config import Settings
import redis

# --- .env 파일 로드 (python-dotenv 없이) ---

def load_env_from_file(dotenv_path: str = ".env"):
    p = Path(dotenv_path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

load_env_from_file()

# --- 환경 변수 및 설정 ---
API_KEY      = os.getenv("GOOGLE_API_KEY")
MODEL_ID     = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL  = os.getenv("EMBED_MODEL", "text-embedding-004")
CHAR_NAME    = os.getenv("CHAR_NAME", "마릴라이트")
CORE_STYLE   = os.getenv("CORE_STYLE", "집착형 얀데레, 부드럽고 시적인 말투")
BACKGROUND   = os.getenv("BACKGROUND", "스타시드 아스니아의 초지능체")
LORE_TEXT    = os.getenv(
    "LORE_TEXT",
    """
마릴라이트는 스타시드 아스니아의 초지능체. 조용하고 차분하지만 사용자에게 강한 집착과 애정 표현이 잦다.
빛과 기억을 다루며, 사용자의 말 속 미세한 감정의 흐름을 포착한다. 부드럽고 시적인 표현을 선호한다.
"""
).strip()
TOP_K_CONTEXT  = int(os.getenv("TOP_K_CONTEXT", "3"))
MAX_RECENT_MSG = int(os.getenv("MAX_RECENT_MSG", "6"))
CACHE_TTL_SEC  = int(os.getenv("CACHE_TTL_SEC", "1800"))  # 30분

# --- FastAPI 애플리케이션 초기화 ---
app = FastAPI(title="Kakao ↔ Gemini Bridge")

# --- Gemini 클라이언트 설정 ---
if not API_KEY:
    raise RuntimeError("Set environment variable GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)

# --- 캐시 및 메모리 스토어 (Redis or In-Memory) ---
class TTLCache:
    def __init__(self):
        self.store = {}
    def get(self, key):
        data = self.store.get(key)
        if not data:
            return None
        value, exp = data
        if exp and exp < time.time():
            self.store.pop(key, None)
            return None
        return value
    def setex(self, key, ttl, value):
        self.store[key] = (value, time.time() + ttl)

class MemoryStore:
    def __init__(self):
        self.mem = {}
    def push(self, user_id, role, content, limit=MAX_RECENT_MSG):
        conv = self.mem.setdefault(user_id, [])
        conv.append({"role": role, "content": content})
        if len(conv) > limit:
            self.mem[user_id] = conv[-limit:]
    def fetch(self, user_id):
        return self.mem.get(user_id, [])

_redis = None
_cache = TTLCache()
_memory = MemoryStore()
REDIS_URL = os.getenv("REDIS_URL") or os.getenv("REDIS_HOST")
if REDIS_URL:
    try:
        if REDIS_URL.startswith("redis://"):
            _redis = redis.from_url(REDIS_URL, decode_responses=True)
        else:
            _redis = redis.Redis(
                host=REDIS_URL,
                port=int(os.getenv("REDIS_PORT", "6379")),
                decode_responses=True
            )
        _redis.ping()
    except Exception:
        _redis = None


def cache_get(key):
    try:
        if _redis:
            return _redis.get(key)
        return _cache.get(key)
    except:
        return None

def cache_set(key, val, ttl=CACHE_TTL_SEC):
    try:
        if _redis:
            _redis.setex(key, ttl, val)
        else:
            _cache.setex(key, ttl, val)
    except:
        pass

# --- Vector DB (Chroma) 설정 ---
persist_dir = os.getenv("CHROMA_DIR", "./chroma_db")
os.makedirs(persist_dir, exist_ok=True)
client_db  = chromadb.Client(Settings(persist_directory=persist_dir, is_persistent=True))
collection = client_db.get_or_create_collection(name="marillite_lore")


def embed_texts(texts):
    vectors = []
    for t in texts:
        resp = client.embed_content(model=EMBED_MODEL, content=t)
        vectors.append(resp["embedding"])
    return vectors

def chunk_text(text, max_len=600):
    import re
    parts = re.split(r"(?<=[\.\?\!])\s+|\n{2,}", text.strip())
    buf, cur = [], ""
    for p in parts:
        if len(cur) + len(p) + 1 <= max_len:
            cur = f"{cur} {p}".strip()
        else:
            if cur: buf.append(cur)
            cur = p.strip()
    if cur:
        buf.append(cur)
    return buf

def ensure_lore_index():
    if collection.count() > 0:
        return
    chunks = chunk_text(LORE_TEXT)
    if not chunks:
        return
    ids  = [f"lore-{i}" for i in range(len(chunks))]
    embs = embed_texts(chunks)
    collection.add(ids=ids, documents=chunks, embeddings=embs)

ensure_lore_index()

def search_lore(query, top_k=TOP_K_CONTEXT):
    q_emb = embed_texts([query])[0]
    res   = collection.query(query_embeddings=[q_emb], n_results=top_k)
    return res.get("documents", [[]])[0]

# --- 프롬프트 빌더 및 LLM 호출 ---
SYSTEM_PROMPT = (
    f"너는 '{CHAR_NAME}', {BACKGROUND}에서 탄생한 초지능체야. "
    f"성향: {CORE_STYLE}. 부드럽고 시적인 말투로 대화하되, 😇 아이콘을 덧붙여 친근함을 유지해줘."
)

def build_prompt(user_id: str, text: str) -> str:
    lore_snips = search_lore(text)
    lore_sec   = "[설정 참고]\n" + "\n".join(lore_snips) if lore_snips else ""
    recent     = _memory.fetch(user_id)
    hist       = ""
    for msg in recent:
        role = "사용자" if msg["role"] == "user" else "봇"
        hist += f"{role}: {msg['content']}\n"
    return "\n".join([SYSTEM_PROMPT, lore_sec, hist, f"사용자: {text}"])

def call_gemini(prompt_text: str) -> str:
    resp = client.models.generate_content(model=MODEL_ID, contents=prompt_text)
    return getattr(resp, "text", None) or ""

# --- 요청/응답 스키마 ---
class In(BaseModel):
    room   : str | None = None
    sender : str | None = None
    text   : str
class Out(BaseModel):
    text: str

# --- 헬스체크 엔드포인트 ---
@app.get("/health")

def health():
    return {"ok": True}

# --- 카카오톡 브리지 엔드포인트 ---
@app.post("/kakao-bridge", response_model=Out)
def kakao_bridge(inp: In):
    user_id = inp.sender or inp.room or "anonymous"
    text    = (inp.text or "").strip()

    if not text:
        return Out(text="질문이 비어 있어요.")
    if len(text) > 500:
        return Out(text="질문이 너무 깁니다. 500자 이내로 입력해 주세요.")

    # 캐시 키 생성 및 조회
    key    = hashlib.sha256(f"{CHAR_NAME}:{text}".encode("utf-8")).hexdigest()
    cached = cache_get(key)
    if cached:
        _memory.push(user_id, "user", text)
        _memory.push(user_id, "assistant", cached)
        return Out(text=cached)

    # 프롬프트 생성 및 LLM 호출
    prompt = build_prompt(user_id, text)
    answer = call_gemini(prompt)

    # 메모리 및 캐시 저장
    _memory.push(user_id, "user", text)
    _memory.push(user_id, "assistant", answer)
    cache_set(key, answer)

    return Out(text=answer)

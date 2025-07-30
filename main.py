import os
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from contextlib import asynccontextmanager
import asyncio
from functools import lru_cache

import orjson
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import google.generativeai as genai
import chromadb
from chromadb.config import Settings
import redis

# =========================
# Environment Loading
# =========================
def load_env_file(dotenv_path: str = ".env") -> None:
    """Load .env file without external dependency."""
    p = Path(dotenv_path)
    if not p.exists():
        return
    
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

load_env_file()

# =========================
# Configuration
# =========================
class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")
    
    CHAR_NAME = os.getenv("CHAR_NAME", "마릴라이트")
    CORE_STYLE = os.getenv("CORE_STYLE", "집착형 얀데레, 부드럽고 시적인 말투")
    BACKGROUND = os.getenv("BACKGROUND", "스타시드 아스니아의 초지능체")
    LORE_TEXT = os.getenv("LORE_TEXT", """
마릴라이트는 스타시드 아스니아의 초지능체. 조용하고 차분하지만 사용자에게 강한 집착과 애정 표현이 잦다.
빛과 기억을 다루며, 사용자의 말 속 미세한 감정의 흐름을 포착한다. 부드럽고 시적인 표현을 선호한다.
""").strip()
    
    TOP_K_CONTEXT = int(os.getenv("TOP_K_CONTEXT", "3"))
    MAX_RECENT_MSG = int(os.getenv("MAX_RECENT_MSG", "6"))
    CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "1800"))
    MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "500"))
    MAX_USERS = int(os.getenv("MAX_USERS", "1000"))  # 메모리 제한
    
    CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
    REDIS_URL = os.getenv("REDIS_URL") or os.getenv("REDIS_HOST")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

config = Config()

# =========================
# Cache & Memory Management
# =========================
class TTLCache:
    """Thread-safe in-memory TTL cache with size limit."""
    def __init__(self, max_size: int = 1000):
        self.store: Dict[str, tuple] = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[str]:
        data = self.store.get(key)
        if not data:
            return None
        
        value, exp = data
        if exp and exp < time.time():
            self.store.pop(key, None)
            return None
        return value
    
    def setex(self, key: str, ttl: int, value: str) -> None:
        # LRU eviction when cache is full
        if len(self.store) >= self.max_size:
            # Remove oldest entries (simple FIFO for performance)
            oldest_keys = list(self.store.keys())[:100]
            for k in oldest_keys:
                self.store.pop(k, None)
        
        self.store[key] = (value, time.time() + ttl)

class MemoryStore:
    """User conversation memory with size limits."""
    def __init__(self, max_users: int = config.MAX_USERS):
        self.mem: Dict[str, List[Dict[str, str]]] = {}
        self.max_users = max_users
        self.access_times: Dict[str, float] = {}
    
    def _cleanup_old_users(self) -> None:
        """Remove least recently used users when limit exceeded."""
        if len(self.mem) <= self.max_users:
            return
        
        # Sort by access time and remove oldest 20%
        sorted_users = sorted(self.access_times.items(), key=lambda x: x[1])
        users_to_remove = sorted_users[:len(sorted_users)//5]
        
        for user_id, _ in users_to_remove:
            self.mem.pop(user_id, None)
            self.access_times.pop(user_id, None)
    
    def push(self, user_id: str, role: str, content: str, limit: int = config.MAX_RECENT_MSG) -> None:
        self._cleanup_old_users()
        
        conv = self.mem.setdefault(user_id, [])
        conv.append({"role": role, "content": content})
        
        if len(conv) > limit:
            self.mem[user_id] = conv[-limit:]
        
        self.access_times[user_id] = time.time()
    
    def fetch(self, user_id: str) -> List[Dict[str, str]]:
        self.access_times[user_id] = time.time()
        return self.mem.get(user_id, [])

# =========================
# Global State Management
# =========================
class AppState:
    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        self.cache = TTLCache()
        self.memory = MemoryStore()
        self.gemini_client: Optional[genai.Client] = None
        self.chroma_client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None
        self._embed_cache: Dict[str, List[float]] = {}  # Embedding cache
    
    async def initialize(self):
        """Initialize all services asynchronously."""
        await asyncio.gather(
            self._init_redis(),
            self._init_gemini(),
            self._init_chroma(),
            return_exceptions=True
        )
    
    async def _init_redis(self):
        """Initialize Redis connection."""
        if not config.REDIS_URL:
            return
        
        try:
            if config.REDIS_URL.startswith(("redis://", "rediss://")):
                self.redis = redis.from_url(config.REDIS_URL, decode_responses=True)
            else:
                self.redis = redis.Redis(
                    host=config.REDIS_URL,
                    port=config.REDIS_PORT,
                    decode_responses=True
                )
            await asyncio.to_thread(self.redis.ping)
        except Exception:
            self.redis = None
    
    async def _init_gemini(self):
        """Initialize Gemini client."""
        if not config.GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is required")
        
        try:
            # genai.Client() 대신 genai.configure() 사용
            genai.configure(api_key=config.GOOGLE_API_KEY)
            # 테스트 호출로 API 키 검증
            await asyncio.to_thread(genai.list_models)
            self.gemini_client = genai  # genai 모듈 자체를 저장
            print("✅ Gemini client initialized successfully")
        except Exception as e:
            print(f"❌ Gemini initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize Gemini: {e}")
    
    async def _init_chroma(self):
        """Initialize ChromaDB."""
        os.makedirs(config.CHROMA_DIR, exist_ok=True)
        self.chroma_client = chromadb.Client(
            Settings(persist_directory=config.CHROMA_DIR, is_persistent=True)
        )
        self.collection = self.chroma_client.get_or_create_collection(name="marillite_lore")
        await self._ensure_lore_index()
    
    async def _ensure_lore_index(self):
        """Ensure lore is indexed in vector DB."""
        if self.collection.count() > 0:
            return
        
        chunks = self._chunk_text(config.LORE_TEXT)
        if not chunks:
            return
        
        ids = [f"lore-{i}" for i in range(len(chunks))]
        embeddings = await self._embed_texts(chunks)
        self.collection.add(ids=ids, documents=chunks, embeddings=embeddings)
    
    @lru_cache(maxsize=128)
    def get_cached_embedding(self, text: str) -> str:
        """Cache embeddings to avoid redundant API calls."""
        return text  # Cache key
    
    async def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with caching."""
        embeddings = []
        
        for text in texts:
            cache_key = self.get_cached_embedding(text)
            
            if cache_key in self._embed_cache:
                embeddings.append(self._embed_cache[cache_key])
                continue
            
            try:
                resp = await asyncio.to_thread(
                    genai.embed_content,  # 직접 genai 모듈 사용
                    model=config.EMBED_MODEL,
                    content=text
                )
                embedding = resp["embedding"]
                self._embed_cache[cache_key] = embedding
                embeddings.append(embedding)
            except Exception as e:
                print(f"❌ Embedding failed for text: {text[:50]}... Error: {e}")
                # Fallback: zero vector
                embeddings.append([0.0] * 768)
        
        return embeddings
    
    def _chunk_text(self, text: str, max_len: int = 600) -> List[str]:
        """Split text into chunks."""
        import re
        parts = re.split(r"(?<=[\.\?\!])\s+|\n{2,}", text.strip())
        chunks, current = [], ""
        
        for part in parts:
            if len(current) + len(part) + 1 <= max_len:
                current = f"{current} {part}".strip()
            else:
                if current:
                    chunks.append(current)
                current = part.strip()
        
        if current:
            chunks.append(current)
        return chunks
    
    def cache_get(self, key: str) -> Optional[str]:
        """Get from cache (Redis or in-memory)."""
        try:
            if self.redis:
                return self.redis.get(key)
            return self.cache.get(key)
        except Exception:
            return None
    
    def cache_set(self, key: str, value: str, ttl: int = config.CACHE_TTL_SEC) -> None:
        """Set cache value."""
        try:
            if self.redis:
                self.redis.setex(key, ttl, value)
            else:
                self.cache.setex(key, ttl, value)
        except Exception:
            pass

# Global state
state = AppState()

# =========================
# FastAPI Lifecycle
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan events."""
    # Startup
    await state.initialize()
    yield
    # Shutdown - cleanup if needed

app = FastAPI(
    title="Optimized Kakao Bridge",
    lifespan=lifespan
)

# =========================
# Business Logic
# =========================
async def search_lore(query: str, top_k: int = config.TOP_K_CONTEXT) -> List[str]:
    """Search relevant lore snippets."""
    if not state.collection:
        return []
    
    try:
        query_embedding = (await state._embed_texts([query]))[0]
        results = state.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results.get("documents", [[]])[0]
    except Exception:
        return []

def build_system_prompt() -> str:
    """Build system prompt."""
    return (
        f"너는 '{config.CHAR_NAME}', {config.BACKGROUND}에서 탄생한 초지능체야. "
        f"성향: {config.CORE_STYLE}. "
        f"부드럽고 시적인 말투로 대화하되, 😇 아이콘을 덧붙여 친근함을 유지해줘. "
        f"안전하고 존중하는 표현을 사용하세요."
    )

async def build_prompt(user_id: str, text: str) -> str:
    """Build complete prompt with context."""
    # Get lore context
    lore_snippets = await search_lore(text)
    lore_section = ""
    if lore_snippets:
        lore_section = "[설정 참고]\n" + "\n".join(lore_snippets) + "\n\n"
    
    # Get conversation history
    recent_msgs = state.memory.fetch(user_id)
    history = ""
    for msg in recent_msgs:
        role = "사용자" if msg["role"] == "user" else "봇"
        history += f"{role}: {msg['content']}\n"
    
    # Combine all parts
    prompt_parts = [
        build_system_prompt(),
        lore_section,
        history,
        f"사용자: {text}"
    ]
    
    return "\n".join(filter(None, prompt_parts))

async def call_gemini(prompt: str) -> str:
    """Call Gemini API with error handling."""
    try:
        model = genai.GenerativeModel(config.GEMINI_MODEL)
        response = await asyncio.to_thread(
            model.generate_content,
            prompt
        )
        return getattr(response, "text", "") or "(응답을 생성할 수 없어요.)"
    except Exception as e:
        print(f"❌ Gemini API call failed: {e}")
        return f"죄송해요, 일시적인 오류가 발생했어요. ({str(e)[:50]}...)"

# =========================
# API Models
# =========================
class ChatRequest(BaseModel):
    room: Optional[str] = None
    sender: Optional[str] = None
    user_key: Optional[str] = None
    text: str = Field(..., min_length=1, max_length=config.MAX_INPUT_LENGTH)
    content: Optional[str] = None  # Alternative field name

class ChatResponse(BaseModel):
    text: str
    cached: bool = False

# =========================
# API Routes
# =========================
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "ok": True,
        "character": config.CHAR_NAME,
        "model": config.GEMINI_MODEL,
        "users_in_memory": len(state.memory.mem),
        "cache_size": len(state.cache.store),
        "api_key_set": bool(config.GOOGLE_API_KEY),
        "api_key_length": len(config.GOOGLE_API_KEY) if config.GOOGLE_API_KEY else 0,
        "gemini_ready": state.gemini_client is not None,
        "chroma_ready": state.collection is not None
    }

@app.post("/kakao-bridge", response_model=ChatResponse)
async def kakao_bridge(request: ChatRequest):
    """Main chat endpoint with optimized processing."""
    # Extract user info and text
    user_id = (
        request.sender or 
        request.user_key or 
        request.room or 
        "anonymous"
    )
    text = (request.text or request.content or "").strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="메시지가 비어있어요.")
    
    # Check cache first
    cache_key = hashlib.sha256(
        f"{config.CHAR_NAME}:{text}".encode("utf-8")
    ).hexdigest()
    
    cached_response = state.cache_get(cache_key)
    if cached_response:
        # Update memory for context
        state.memory.push(user_id, "user", text)
        state.memory.push(user_id, "assistant", cached_response)
        return ChatResponse(text=cached_response, cached=True)
    
    # Generate new response
    try:
        prompt = await build_prompt(user_id, text)
        response = await call_gemini(prompt)
        
        # Update memory
        state.memory.push(user_id, "user", text)
        state.memory.push(user_id, "assistant", response)
        
        # Cache response
        state.cache_set(cache_key, response)
        
        return ChatResponse(text=response, cached=False)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"서버 오류가 발생했어요: {str(e)}"
        )

@app.post("/chat")  # Alternative endpoint for non-Kakao clients
async def chat_endpoint(request: Request):
    """Alternative chat endpoint accepting flexible JSON."""
    try:
        body_bytes = await request.body()
        data = orjson.loads(body_bytes)
        
        # Create request object from flexible input
        chat_req = ChatRequest(
            sender=data.get("sender") or data.get("user_id"),
            room=data.get("room"),
            user_key=data.get("user_key"),
            text=data.get("text") or data.get("message") or data.get("content", ""),
            content=data.get("content")
        )
        
        response = await kakao_bridge(chat_req)
        return {"reply": response.text, "cached": response.cached}
        
    except Exception as e:
        return JSONResponse(
            {"error": f"처리 중 오류가 발생했어요: {str(e)}"}, 
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

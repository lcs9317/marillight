import os
import hashlib
import time
import asyncio
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
import logging

import orjson
import httpx
from fastapi import FastAPI, Request, BackgroundTasks
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
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-004")

CHAR_NAME      = os.getenv("CHAR_NAME", "마릴라이트")
CORE_STYLE     = os.getenv("CORE_STYLE", "집착형 얀데레, 부드럽고 시적인 말투")
BACKGROUND     = os.getenv("BACKGROUND", "스타시드 아스니아의 초지능체")
LORE_TEXT      = os.getenv("LORE_TEXT", """
마릴라이트는 스타시드 아스니아의 초지능체. 조용하고 차분하지만 사용자에게 강한 집착과 애정 표현이 잦다.
빛과 기억을 다루며, 사용자의 말 속 미세한 감정의 흐름을 포착한다. 부드럽고 시적인 표현을 선호한다.
""").strip()

TOP_K_CONTEXT  = int(os.getenv("TOP_K_CONTEXT", "3"))
MAX_RECENT_MSG = int(os.getenv("MAX_RECENT_MSG", "6"))
CACHE_TTL_SEC  = int(os.getenv("CACHE_TTL_SEC", "1800"))

# 웹훅 설정 (제타 스타일)
WEBHOOK_TIMEOUT = int(os.getenv("WEBHOOK_TIMEOUT", "30"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# =========================
# Setup FastAPI & Logging
# =========================
app = FastAPI(title="Marillite Bridge")
executor = ThreadPoolExecutor(max_workers=20)  # 더 많은 워커
http_client = httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# Cache & Memory (간소화)
# =========================
class SimpleCache:
    def __init__(self):
        self.store: Dict[str, tuple] = {}
    
    def get(self, key: str) -> Optional[str]:
        if key not in self.store:
            return None
        value, exp = self.store[key]
        if exp < time.time():
            del self.store[key]
            return None
        return value
    
    def set(self, key: str, value: str, ttl: int = CACHE_TTL_SEC):
        self.store[key] = (value, time.time() + ttl)

class ConversationMemory:
    def __init__(self):
        self.conversations: Dict[str, List[Dict]] = {}
    
    def add_message(self, user_id: str, role: str, content: str):
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        self.conversations[user_id].append({
            "role": role, 
            "content": content,
            "timestamp": time.time()
        })
        
        # 최근 메시지만 유지
        if len(self.conversations[user_id]) > MAX_RECENT_MSG:
            self.conversations[user_id] = self.conversations[user_id][-MAX_RECENT_MSG:]
    
    def get_recent(self, user_id: str) -> List[Dict]:
        return self.conversations.get(user_id, [])

# 전역 인스턴스
cache = SimpleCache() 
memory = ConversationMemory()

# =========================
# Gemini Setup (동기 유지 - 더 안정적)
# =========================
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable is required.")
genai.configure(api_key=GOOGLE_API_KEY)

# =========================
# Vector DB (필요시에만 초기화)
# =========================
vector_db = None
if LORE_TEXT.strip():
    try:
        PERSIST_DIR = os.getenv("CHROMA_DIR", "/tmp/chroma")
        os.makedirs(PERSIST_DIR, exist_ok=True)
        client = chromadb.Client(Settings(persist_directory=PERSIST_DIR, is_persistent=True))
        collection = client.get_or_create_collection(name="lore")
        
        # 벡터DB 초기화 (동기로 간단히)
        if collection.count() == 0 and LORE_TEXT:
            chunks = [LORE_TEXT[i:i+500] for i in range(0, len(LORE_TEXT), 400)]  # 간단한 청킹
            embeddings = []
            for chunk in chunks:
                resp = genai.embed_content(model=EMBED_MODEL, content=chunk)
                embeddings.append(resp["embedding"])
            
            collection.add(
                ids=[f"chunk-{i}" for i in range(len(chunks))],
                documents=chunks,
                embeddings=embeddings
            )
        vector_db = collection
        logger.info("✅ Vector DB initialized")
    except Exception as e:
        logger.warning(f"Vector DB 초기화 실패: {e}")
        vector_db = None

# =========================
# AI 처리 (핵심 로직)
# =========================
def get_relevant_context(query: str) -> str:
    """벡터 검색으로 관련 컨텍스트 가져오기"""
    if not vector_db:
        return ""
    
    try:
        query_embedding = genai.embed_content(model=EMBED_MODEL, content=query)["embedding"]
        results = vector_db.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K_CONTEXT
        )
        docs = results.get("documents", [[]])[0]
        return "\n".join(docs) if docs else ""
    except Exception as e:
        logger.error(f"벡터 검색 실패: {e}")
        return ""

def generate_response(user_id: str, user_input: str) -> str:
    """AI 응답 생성 (동기 - 안정성 우선)"""
    try:
        # 시스템 프롬프트
        system_prompt = f"""당신은 '{CHAR_NAME}'입니다. {BACKGROUND}
성향: {CORE_STYLE}
항상 부드럽고 시적인 말투를 유지하며, 사용자를 소중히 여기고 독점적 애정을 표현합니다."""

        # 관련 컨텍스트 
        context = get_relevant_context(user_input)
        if context:
            system_prompt += f"\n\n[참고 설정]\n{context}"

        # 대화 히스토리
        recent_messages = memory.get_recent(user_id)
        history = ""
        for msg in recent_messages[-4:]:  # 최근 4개만
            role = "사용자" if msg["role"] == "user" else "어시스턴트"
            history += f"{role}: {msg['content']}\n"

        # 최종 프롬프트
        prompt = f"{system_prompt}\n\n{history}사용자: {user_input}\n어시스턴트:"

        # Gemini 호출
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        
        result = response.text.strip() if hasattr(response, "text") and response.text else "죄송해요, 응답을 생성할 수 없어요."
        return result
        
    except Exception as e:
        logger.error(f"AI 응답 생성 실패: {e}")
        return "일시적인 오류가 발생했어요. 잠시 후 다시 시도해 주세요."

# =========================
# 웹훅 전송 (비동기 - 핵심!)
# =========================
async def send_webhook_response(webhook_url: str, response_data: dict, retries: int = 0):
    """웹훅 응답 전송 (재시도 포함)"""
    try:
        async with httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT) as client:
            resp = await client.post(webhook_url, json=response_data)
            resp.raise_for_status()
            logger.info(f"✅ 웹훅 전송 성공: {webhook_url}")
            return True
            
    except Exception as e:
        logger.error(f"❌ 웹훅 전송 실패 (시도 {retries + 1}): {e}")
        
        if retries < MAX_RETRIES:
            await asyncio.sleep(2 ** retries)  # 지수 백오프
            return await send_webhook_response(webhook_url, response_data, retries + 1)
        
        logger.error(f"최종 실패: {webhook_url}")
        return False

# =========================
# API Models
# =========================
class ChatRequest(BaseModel):
    user_id: str
    message: str
    webhook_url: Optional[str] = None  # 응답받을 웹훅 URL

class DirectChatRequest(BaseModel):
    user_id: str = "anonymous"
    message: str

# =========================
# Routes
# =========================
@app.get("/")
def health():
    return {
        "ok": True, 
        "name": CHAR_NAME,
        "model": GEMINI_MODEL,
        "vector_db": vector_db is not None
    }

@app.post("/chat")
async def chat_endpoint(req: ChatRequest, background_tasks: BackgroundTasks):
    """비동기 채팅 엔드포인트 (제타 스타일)"""
    user_id = req.user_id
    message = req.message.strip()
    
    if not message:
        return JSONResponse({"error": "메시지가 비어있습니다"}, status_code=400)
    
    # 캐시 확인
    cache_key = hashlib.sha256(f"{CHAR_NAME}:{message}".encode()).hexdigest()
    cached_response = cache.get(cache_key)
    
    if cached_response:
        logger.info(f"캐시 히트: {user_id}")
        memory.add_message(user_id, "user", message)
        memory.add_message(user_id, "assistant", cached_response)
        
        if req.webhook_url:
            background_tasks.add_task(
                send_webhook_response, 
                req.webhook_url, 
                {"reply": cached_response, "user_id": user_id}
            )
            return {"status": "processing", "message": "응답을 처리 중입니다"}
        else:
            return {"reply": cached_response}
    
    # 웹훅이 있으면 백그라운드에서 처리
    if req.webhook_url:
        background_tasks.add_task(process_and_send_response, user_id, message, req.webhook_url, cache_key)
        return {"status": "processing", "message": "응답을 생성 중입니다"}
    
    # 직접 응답
    response = await asyncio.get_event_loop().run_in_executor(
        executor, generate_response, user_id, message
    )
    
    memory.add_message(user_id, "user", message) 
    memory.add_message(user_id, "assistant", response)
    # 빈 응답("")이나 플레이스홀더 "(빈 응답)" 은 캐싱하지 않음
    if response and response != "(빈 응답)":
        cache.set(cache_key, response)
    
    return {"reply": response}

async def process_and_send_response(user_id: str, message: str, webhook_url: str, cache_key: str):
    """백그라운드에서 AI 응답 생성 후 웹훅 전송"""
    try:
        # AI 응답 생성 (스레드풀에서)
        response = await asyncio.get_event_loop().run_in_executor(
            executor, generate_response, user_id, message
        )
        
        # 메모리에 저장
        memory.add_message(user_id, "user", message)
        memory.add_message(user_id, "assistant", response)
        # 빈 응답은 캐싱하지 않기
        if response and response != "(빈 응답)":
            cache.set(cache_key, response)
        
        # 웹훅으로 응답 전송
        await send_webhook_response(webhook_url, {
            "reply": response,
            "user_id": user_id
        })
        
    except Exception as e:
        logger.error(f"백그라운드 처리 실패: {e}")
        # 에러도 웹훅으로 전송
        await send_webhook_response(webhook_url, {
            "error": "응답 생성 중 오류가 발생했습니다",
            "user_id": user_id
        })

@app.post("/direct-chat")
async def direct_chat(req: DirectChatRequest):
    """즉시 응답 (테스트용)"""
    response = await asyncio.get_event_loop().run_in_executor(
        executor, generate_response, req.user_id, req.message
    )
    return {"reply": response}

# 카카오톡 호환 (기존 유지)
@app.post("/kakao-bridge") 
async def kakao_bridge(request: Request):
    try:
        body = await request.body()
        data = orjson.loads(body) if body else {}
        
        user_id = data.get("user_key", "kakao_user")
        message = data.get("content", "").strip()
        
        if not message:
            return {"reply": "메시지를 입력해주세요."}
        cache_key = hashlib.sha256(f"{CHAR_NAME}:{message}".encode()).hexdigest()
        # 캐시 확인
        cached = cache.get(cache_key)
        if cached:
            return {"reply": cached}

        response = await asyncio.get_event_loop().run_in_executor(
            executor, generate_response, user_id, message
        )

        if response and response != "(빈 응답)":
            cache.set(cache_key, response)
        
        return {"reply": response}
        
    except Exception as e:
        logger.error(f"카카오 브릿지 오류: {e}")
        return JSONResponse({"error": "처리 중 오류가 발생했습니다"}, status_code=500)

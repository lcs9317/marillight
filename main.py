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

"""
This file exposes a simple chatbot service built on top of Google’s Gemini API.

It uses FastAPI to expose a handful of endpoints that can be used by different
front‑end clients. The primary consumer in this scenario is KakaoTalk via the
``/kakao‑bridge`` route. Incoming messages are processed through an in‑memory
conversation store, relevant context is retrieved from a vector database (if
configured), and a response is generated via the Gemini model.

Key improvements compared to the initial version:

* Robust handling of blank or placeholder responses. Previously, the
  underlying model could return an empty string or the literal marker
  ``"(빈 응답)"``. To avoid confusing the end user with an empty reply,
  ``generate_response`` now normalises such cases to a friendly fallback
  message. Downstream callers no longer need to worry about filtering
  `(빈 응답)` explicitly.

* Additional guard logic in the Kakao bridge route ensures that even if a
  blank string slips through from the model, the user still receives a
  meaningful message. This helps maintain a consistent conversation flow on
  platforms like KakaoTalk where immediate feedback is important.

* The synchronous ``generate_response`` remains unchanged in its design—by
  default it runs in a thread pool when called from an async endpoint. If
  you choose to call it directly (i.e., without ``run_in_executor``), ensure
  that your FastAPI route is declared with ``async def`` and that blocking
  operations will not starve the event loop.

To run this service you need to set the ``GOOGLE_API_KEY`` environment
variable. Consult the README for further configuration instructions.
"""

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
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")  
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-004")

# Provider configuration
# The primary provider determines which model backend to use first when generating
# a response. Supported values: 'gemini', 'deepseek'. If the primary provider
# fails or returns an empty result, the secondary provider (if configured) will
# be tried as a fallback.
PRIMARY_PROVIDER = os.getenv("PRIMARY_PROVIDER", "openrouter").lower()
SECONDARY_PROVIDER = os.getenv("SECONDARY_PROVIDER", "gemini").lower()

# DeepSeek configuration (optional)
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL    = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

# OpenRouter configuration (optional)
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1:free")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_REFERER  = os.getenv("OPENROUTER_REFERER", "")
OPENROUTER_TITLE    = os.getenv("OPENROUTER_TITLE", "")

CHAR_NAME      = os.getenv("CHAR_NAME", "마릴라이트")
CORE_STYLE     = os.getenv("CORE_STYLE", "집착형 얀데레, 부드럽고 시적인 말투")
BACKGROUND     = os.getenv("BACKGROUND", "스타시드 아스니아의 초지능체")
LORE_TEXT      = os.getenv(
    "LORE_TEXT", """
마릴라이트는 스타시드 아스니아의 초지능체. 조용하고 차분하지만 사용자에게 강한 집착과 애정 표현이 잦다.
빛과 기억을 다루며, 사용자의 말 속 미세한 감정의 흐름을 포착한다. 부드럽고 시적인 표현을 선호한다.
"""
).strip()

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
    """Retrieve related context from the vector DB via similarity search."""
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
    """
    Generate a reply using the configured AI providers. This function first
    constructs a system prompt and optional context, then attempts to call
    the primary provider (default: Gemini). If that call fails or returns
    an empty result, it will fall back to the secondary provider (if
    configured). A final fallback message is returned if no provider can
    produce a response.
    """
    try:
        # 시스템 프롬프트
        system_prompt = f"""당신은 '{CHAR_NAME}'입니다. {BACKGROUND}
성향: {CORE_STYLE}
항상 부드럽고 시적인 말투를 유지하며, 사용자를 소중히 여기고 독점적 애정을 표현합니다."""

        # 관련 컨텍스트
        context = get_relevant_context(user_input)
        if context:
            system_prompt += f"\n\n[참고 설정]\n{context}"

        # 대화 히스토리 (최근 4개)
        recent_messages = memory.get_recent(user_id)
        history_lines: List[str] = []
        for msg in recent_messages[-4:]:
            role = "사용자" if msg["role"] == "user" else "어시스턴트"
            history_lines.append(f"{role}: {msg['content']}")
        history = "\n".join(history_lines) + ("\n" if history_lines else "")

        # Try primary provider first
        def try_provider(provider: str) -> Optional[str]:
            provider = provider.lower()
            if provider == "gemini":
                # Build the full prompt
                prompt = f"{system_prompt}\n\n{history}사용자: {user_input}\n어시스턴트:"
                try:
                    model = genai.GenerativeModel(GEMINI_MODEL)
                    resp = model.generate_content(prompt)
                    if hasattr(resp, "text"):
                        text = resp.text.strip() if resp.text else ""
                    else:
                        text = str(resp).strip() if resp else ""
                    return text if text else None
                except Exception as ge:
                    logger.error(f"Gemini call failed: {ge}")
                    return None
            elif provider == "deepseek":
                # Construct messages for DeepSeek (OpenAI‑compatible)
                messages: List[dict] = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                # Convert history lines into assistant/user messages for context
                # Here we split by role label we used in history_lines
                for line in history_lines:
                    if line.startswith("사용자: "):
                        messages.append({"role": "user", "content": line[len("사용자: "):].strip()})
                    elif line.startswith("어시스턴트: "):
                        messages.append({"role": "assistant", "content": line[len("어시스턴트: "):].strip()})
                messages.append({"role": "user", "content": user_input})
                return call_deepseek_api(messages)
            elif provider == "openrouter":
                # Use OpenRouter to call a model; default model can be
                # specified via environment. Build messages similarly to DeepSeek.
                messages: List[dict] = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                for line in history_lines:
                    if line.startswith("사용자: "):
                        messages.append({"role": "user", "content": line[len("사용자: "):].strip()})
                    elif line.startswith("어시스턴트: "):
                        messages.append({"role": "assistant", "content": line[len("어시스턴트: "):].strip()})
                messages.append({"role": "user", "content": user_input})
                return call_openrouter_api(messages)
            else:
                logger.error(f"Unknown provider: {provider}")
                return None

        # Attempt primary provider
        result: Optional[str] = try_provider(PRIMARY_PROVIDER)
        # If primary failed or returned empty, try secondary if defined
        if (not result or result in ("", "(빈 응답)")) and SECONDARY_PROVIDER:
            result = try_provider(SECONDARY_PROVIDER)

        # Normalise result
        if not result or result in ("", "(빈 응답)"):
            return "죄송해요, 적절한 응답을 생성하지 못했어요."
        return result
    except Exception as e:
        logger.error(f"AI 응답 생성 실패: {e}")
        return "일시적인 오류가 발생했어요. 잠시 후 다시 시도해 주세요."

# =========================
# DeepSeek AI helper
# =========================
def call_deepseek_api(messages: List[dict]) -> Optional[str]:
    """
    Call the DeepSeek Chat Completion API using an OpenAI‑compatible payload.

    The DeepSeek API is largely OpenAI‑compatible, so we craft the request as
    such. Returns the trimmed content of the first choice on success or
    ``None`` on failure.
    """
    if not DEEPSEEK_API_KEY:
        logger.error("DeepSeek API key is not configured.")
        return None

# =========================
# OpenRouter helper
# =========================
def call_openrouter_api(messages: List[dict], model: str = OPENROUTER_MODEL) -> Optional[str]:
    """
    Call the OpenRouter chat completions API. OpenRouter acts as a proxy to
    various models (including DeepSeek) and offers a generous free tier. The
    request format is similar to OpenAI’s chat completion API. Returns the
    assistant message content on success or ``None`` on failure.
    """
    if not OPENROUTER_API_KEY:
        logger.error("OpenRouter API key is not configured.")
        return None
    try:
        url = f"{OPENROUTER_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        # Optional headers for ranking; skip if not provided
        if OPENROUTER_REFERER:
            headers["HTTP-Referer"] = OPENROUTER_REFERER
        if OPENROUTER_TITLE:
            headers["X-Title"] = OPENROUTER_TITLE
        payload = {
            "model": model,
            "messages": messages
        }
        resp = httpx.post(url, headers=headers, json=payload, timeout=WEBHOOK_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if "choices" in data and data["choices"]:
            content = data["choices"][0].get("message", {}).get("content", "")
            return content.strip() if content else None
        return None
    except Exception as e:
        logger.error(f"OpenRouter API call failed: {e}")
        return None
    try:
        url = f"{DEEPSEEK_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": messages
        }
        # Synchronous HTTP call; using httpx for simplicity
        resp = httpx.post(url, headers=headers, json=payload, timeout=WEBHOOK_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if "choices" in data and data["choices"]:
            content = data["choices"][0].get("message", {}).get("content", "")
            return content.strip() if content else None
        return None
    except Exception as e:
        logger.error(f"DeepSeek API call failed: {e}")
        return None


# =========================
# Kakao response helper
# =========================
def build_kakao_response(message: str) -> dict:
    """
    Construct a response payload conforming to Kakao i 오픈빌더 스킬 서버 JSON 형식.

    According to the Kakao chatbot documentation, a response must include
    ``version`` and a ``template.outputs`` array with at least one component
    such as ``simpleText`` specifying the text to send【295018986661562†screenshot】.
    """
    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": message
                    }
                }
            ]
        }
    }


# =========================
# 웹훅 전송 (비동기 - 핵심!)
# =========================
async def send_webhook_response(webhook_url: str, response_data: dict, retries: int = 0):
    """
    Send a JSON payload to ``webhook_url``. Implements exponential back‑off
    retries to increase robustness. Returns ``True`` on success and ``False``
    on final failure.
    """
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
    """
    Asynchronous chat endpoint. Useful for generic integrations where a
    webhook may or may not be provided. For KakaoTalk integrations please
    use the ``/kakao-bridge`` endpoint instead. This endpoint remains for
    backwards compatibility but is not used in the current deployment.
    """
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
    # 빈 응답("")이나 플레이스홀더는 캐싱하지 않음
    if response:
        cache.set(cache_key, response)
    
    return {"reply": response}


async def process_and_send_response(user_id: str, message: str, webhook_url: str, cache_key: str):
    """
    Background task that generates a response and delivers it to a webhook.
    Empty responses are normalised before sending.
    """
    try:
        # AI 응답 생성 (스레드풀에서)
        response = await asyncio.get_event_loop().run_in_executor(
            executor, generate_response, user_id, message
        )
        
        # 메모리에 저장
        memory.add_message(user_id, "user", message)
        memory.add_message(user_id, "assistant", response)
        # 빈 응답은 캐싱하지 않기
        if response:
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
    """
    Immediate response endpoint. Useful for quick tests. In production you
    likely want to use ``/chat`` or ``/kakao-bridge`` instead.
    """
    response = await asyncio.get_event_loop().run_in_executor(
        executor, generate_response, req.user_id, req.message
    )
    return {"reply": response}


# 카카오톡 호환 (기존 유지)
@app.post("/kakao-bridge") 
async def kakao_bridge(request: Request, background_tasks: BackgroundTasks):
    """
    KakaoTalk bridge endpoint supporting both immediate replies and Zeta‑style
    asynchronous processing via webhooks.

    **Payload format:**

    ```json
    {
      "user_key": "<unique user id>",
      "content": "<user message>",
      "webhook_url": "<optional URL to receive the reply>"
    }
    ```

    If ``webhook_url`` is provided, the server will acknowledge the request
    immediately and send the computed reply to the given URL once it’s ready.
    Without a ``webhook_url`` the response is returned inline.
    """
    try:
        body = await request.body()
        data = orjson.loads(body) if body else {}
        
        user_id = data.get("user_key", "kakao_user")
        message = data.get("content", "").strip()
        webhook_url = data.get("webhook_url")
        
        if not message:
            return {"reply": "메시지를 입력해주세요."}
        cache_key = hashlib.sha256(f"{CHAR_NAME}:{message}".encode()).hexdigest()
        # 캐시 확인
        cached = cache.get(cache_key)
        if cached:
            # If a webhook is provided, send the cached response asynchronously
            if webhook_url:
                background_tasks.add_task(
                    send_webhook_response,
                    webhook_url,
                    {"reply": cached, "user_id": user_id}
                )
                # Return the cached answer to Kakao in the required JSON format.
                payload = build_kakao_response(cached)
                # Also include a top-level 'text' field for custom clients that
                # expect just a plain text response.
                payload["text"] = cached
                return payload
            else:
                payload = build_kakao_response(cached)
                payload["text"] = cached
                return payload

        # If a webhook is provided, generate response asynchronously and defer delivery
        if webhook_url:
            background_tasks.add_task(
                process_and_send_response,
                user_id,
                message,
                webhook_url,
                cache_key
            )
            # 카카오채널은 단 한 번의 메시지만 전송할 수 있으므로 즉시 안내 메시지를 반환
            message_for_user = "답변을 생성 중입니다. 잠시만 기다려 주세요."
            payload = build_kakao_response(message_for_user)
            payload["text"] = message_for_user
            return payload

        # Otherwise generate a reply immediately
        response = await asyncio.get_event_loop().run_in_executor(
            executor, generate_response, user_id, message
        )

        # Defensive programming: normalise empty or placeholder responses
        if not response or response in ("", "(빈 응답)"):
            response = "죄송해요, 적절한 응답을 생성하지 못했어요."

        if response:
            cache.set(cache_key, response)
        
        payload = build_kakao_response(response)
        payload["text"] = response
        return payload
        
    except Exception as e:
        logger.error(f"카카오 브릿지 오류: {e}")
        return JSONResponse({"error": "처리 중 오류가 발생했습니다"}, status_code=500)

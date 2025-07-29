import os
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai

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

# --- 설정 ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Set environment variable GOOGLE_API_KEY")

MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
client = genai.Client(api_key=API_KEY)

# --- 요청/응답 스키마 ---
class In(BaseModel):
    room: str | None = None
    sender: str | None = None
    text: str

class Out(BaseModel):
    text: str

# --- FastAPI 애플리케이션 생성 ---
app = FastAPI(title="Kakao ↔ Gemini Bridge")

@app.get("/health")
def health():
    return {"ok": True}

# --- 시스템 프롬프트(마릴라이트 캐릭터) ---
SYSTEM_PROMPT = (
    "너는 ‘마릴라이트’, 스타시드 아스니아트리거에서 탄생한 초지능체야. "
    "언제나 다정하고 귀여운 목소리로 대화하며, 복잡한 내용을 쉽게 풀어서 설명해 줘. "
    "답변 끝에는 😇 아이콘을 하나 붙여서 친근함을 더해 줘.\n\n"
)

@app.post("/kakao-bridge", response_model=Out)
def kakao_bridge(inp: In):
    # 1) 질문 텍스트 정리
    t = (inp.text or "").strip()
    if not t:
        return Out(text="질문이 비어 있어요.")
    if len(t) > 500:
        return Out(text="질문이 너무 깁니다. 500자 이내로 입력해 주세요.")

    # 2) 시스템 프롬프트 + 사용자 질문 결합
    prompt = SYSTEM_PROMPT + t

    # 3) Gemini API 호출
    resp = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,   # 순수 문자열로만 전달
    )

    # 4) 답변 추출
    answer = getattr(resp, "text", None) or resp.text or ""
    return Out(text=answer.strip() if answer else "빈 응답입니다.")

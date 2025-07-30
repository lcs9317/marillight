import os
import google.generativeai as genai

# ChromaDB 텔레메트리 비활성화
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# API 키 확인 (환경변수에서만 가져옴)
api_key = os.getenv("GOOGLE_API_KEY")
print(f"API Key set: {bool(api_key)}")
print(f"API Key length: {len(api_key) if api_key else 0}")

if not api_key:
    print("❌ GOOGLE_API_KEY not set!")
    print("Set it with: export GOOGLE_API_KEY=your_api_key_here")
    exit(1)

# Gemini 설정
try:
    genai.configure(api_key=api_key)
    print("✅ Gemini configured")
    
    # 모델 리스트 확인
    models = list(genai.list_models())
    print(f"✅ Available models: {len(models)}")
    
    # 임베딩 테스트 (하드코딩된 모델명 사용)
    embed_model = "models/text-embedding-004"
    test_text = "안녕하세요"
    
    print(f"Testing embedding with model: {embed_model}")
    embed_response = genai.embed_content(
        model=embed_model,
        content=test_text
    )
    print(f"✅ Embedding success! Vector length: {len(embed_response['embedding'])}")
    
    # 간단한 생성 테스트 (하드코딩된 모델명 사용)
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content("안녕하세요!")
    print(f"✅ Test response: {response.text[:100]}...")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

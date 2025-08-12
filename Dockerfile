FROM python:3.11-slim

# 런타임 안정화 및 캐시 경로
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/cache/hf \
    TRANSFORMERS_CACHE=/app/cache/hf \
    SENTENCE_TRANSFORMERS_HOME=/app/cache/hf

WORKDIR /app

# 의존성 먼저 설치 (빌드 캐시 최적화)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 소스 전체 복사 (characters 포함)
COPY . .

# 캐시 디렉토리 만들어두기
RUN mkdir -p /app/cache /app/cache/hf

# === 여기서 "빌드 타임"에 임베딩/인덱스 생성해서 이미지에 굽기 ===
# - main.py의 build_index()를 호출해 ./cache/index.pkl + ./cache/emb_*.npy 생성
# - 모델 파일도 HF 캐시에 받아 이미지에 포함됨(재시작 빨라짐)
ENV CHAR_DIR=/app/character \
    USE_EMBEDDINGS=1 \
    EMB_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    EMB_BATCH=16 \
    EMB_FP16=1
RUN python - <<'PY'
from main import build_index
import os
print("[build] precomputing embeddings/index ...")
pack = build_index(os.environ.get("CHAR_DIR","./character"), backend="emb")
print("[build] done: docs=", len(pack.docs))
PY

# 관례상 8080 노출 (Koyeb는 $PORT를 넘겨줌)
EXPOSE 8080

# FastAPI 실행 (uvicorn 1 worker)
CMD python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1

FROM python:3.11-slim
WORKDIR /app

# ---- System deps (작고 안전하게)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && rm -rf /var/lib/apt/lists/*

# ---- Python deps via requirements.txt
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# ---- App files
COPY main.py /app/main.py

# 캐릭터 리소스 폴더 생성 후 *.txt 복사
RUN mkdir -p /app/character
COPY /character/*.txt /app/character/
# 혹시 섞여 들어온 파일 정리
RUN rm -f /app/character/requirements.txt || true

# ---- Defaults (Koyeb ENV로 덮어써도 됨)
ENV PORT=8000 \
    CHAR_DIR=/app/character \
    CACHE_DIR=/app/cache \
    HF_HOME=/opt/hf \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    FASTEMBED_DISABLE_CUDA=1 \
    EMB_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" \
    USE_EMBEDDINGS=1 \
    CHUNK_SIZE=900 \
    CHUNK_OVERLAP=250 \
    TOP_K=4 \
    CURRENT_CHARACTER="마릴라이트|Marillight"

# 캐시/인덱스 디렉토리
RUN mkdir -p /opt/hf /app/cache

# ---- (1) fastembed 모델 프리캐시
RUN python - <<'PY'
import os
from fastembed import TextEmbedding
m = os.environ.get("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
TextEmbedding(model_name=m, cache_dir=os.environ.get("HF_HOME","/opt/hf"))
print(">> cached fastembed model:", m)
PY

# ---- (2) 인덱스 + 임베딩 사전 생성 (index.pkl / embeddings.npy / meta.json 포함)
RUN python - <<'PY'
import os
os.environ["CHAR_DIR"]  = os.environ.get("CHAR_DIR", "/app/character")
os.environ["CACHE_DIR"] = os.environ.get("CACHE_DIR", "/app/cache")
os.environ["USE_EMBEDDINGS"] = os.environ.get("USE_EMBEDDINGS", "1")
os.environ["EMB_MODEL"] = os.environ.get("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
from main import build_index
pack = build_index(os.environ["CHAR_DIR"], backend="emb" if os.environ["USE_EMBEDDINGS"]=="1" else "tfidf")
print(">> prebaked index docs:", len(pack.docs))
PY

# 비루트 실행 권장
RUN useradd -m appuser && chown -R appuser:appuser /app /opt/hf
USER appuser

# 컨테이너 자체 헬스체크 (Koyeb 측 헬스체크와 병행 가능)
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s CMD curl -fsS http://127.0.0.1:${PORT:-8000}/health || exit 1

EXPOSE 8000
CMD ["bash", "-lc", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT}"]

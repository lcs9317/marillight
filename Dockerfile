FROM python:3.11-slim
WORKDIR /app

# ---- OS deps (onnxruntime가 쓰는 OpenMP 등) ----
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# ---- 런타임 튜닝(메모리 피크 억제) ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    FASTEMBED_DISABLE_CUDA=1 \
    HF_HOME=/opt/hf \
    CHAR_DIR=/app/characters \
    CACHE_DIR=/app/cache \
    USE_EMBEDDINGS=1 \
    EMB_MODEL=intfloat/multilingual-e5-small \
    EMB_BATCH=8 \
    DOT_BLOCK=4096 \
    PORT=8000

# ---- Python deps ----
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# ---- App source ----
# 주의: characters/*.txt 같이 리소스를 포함하려면 레포에 넣어주세요.
# 깔끔하게 하려면 .dockerignore도 추가 권장(.git, node_modules 등 제외)
COPY . /app

# 리소스 폴더 보장
RUN mkdir -p /app/characters /app/cache /opt/hf

# ---- 모델/인덱스 프리베이크(빌드 타임에 수행) ----
# 레포 안에 characters/*.txt 또는 루트의 *.txt가 있다면
# 빌드 시 임베딩을 미리 생성해두어 런타임 스타트업 속도·메모리 모두 안전
RUN python - <<'PY' || true
import os, glob
from main import build_index
os.environ.setdefault("CHAR_DIR","/app/characters")
os.environ.setdefault("CACHE_DIR","/app/cache")
os.environ.setdefault("USE_EMBEDDINGS","1")
# characters가 비어있으면 건너뜀
has_txt = bool(glob.glob("/app/characters/*.txt")) or bool(glob.glob("/app/*.txt"))
if has_txt:
    pack = build_index(os.environ["CHAR_DIR"],
                       backend="emb" if os.environ["USE_EMBEDDINGS"]=="1" else "tfidf")
    print(">> prebaked index docs:", len(pack.docs))
else:
    print(">> no character .txt found; skip pre-bake")
PY

# ---- 비루트 실행 ----
RUN useradd -m appuser && chown -R appuser:appuser /app /opt/hf
USER appuser

# ---- 헬스체크 ----
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

EXPOSE 8000
CMD ["bash","-lc","python -m uvicorn main:app --host 0.0.0.0 --port ${PORT}"]

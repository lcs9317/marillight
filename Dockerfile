# Base
FROM python:3.11-slim
WORKDIR /app

# System deps (slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
    "fastapi==0.115.*" \
    "uvicorn[standard]==0.30.*" \
    "fastembed==0.5.2" \
    "scikit-learn==1.5.*" \
    "numpy==2.0.*" \
    "openai>=1.40.0" \
    "google-generativeai>=0.7.0"

# App files
COPY main.py /app/main.py
RUN mkdir -p /app/character
COPY *.txt /app/character/

# Defaults (Koyeb에서 ENV로 얼마든지 덮어쓸 수 있음)
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

# Dirs
RUN mkdir -p /opt/hf /app/cache

# ---- Build-time model pre-cache (기본 EMB_MODEL을 미리 받아 이미지에 포함)
RUN python - <<'PY'
import os
from fastembed import TextEmbedding
m = os.environ.get("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
TextEmbedding(model_name=m, cache_dir=os.environ.get("HF_HOME","/opt/hf"))
print(">> cached fastembed model:", m)
PY

# Non-root
RUN useradd -m appuser && chown -R appuser:appuser /app /opt/hf
USER appuser

EXPOSE 8000
CMD ["bash", "-lc", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT}"]

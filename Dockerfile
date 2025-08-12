FROM python:3.11-slim
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*

# 런타임 튜닝
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    CHAR_DIR=/app/characters \
    CACHE_DIR=/app/cache \
    DOT_BLOCK=4096 \
    EMBED_PROVIDER=gemini \
    GEMINI_EMBED_MODEL=text-embedding-004 \
    PORT=8000

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
RUN mkdir -p /app/characters /app/cache

# 비루트
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# 헬스체크
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

EXPOSE 8000
CMD ["bash","-lc","python -m uvicorn main:app --host 0.0.0.0 --port ${PORT}"]

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 소스
COPY . .

# 런타임 캐시(이미지에 포함 X)
ENV HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf \
    SENTENCE_TRANSFORMERS_HOME=/tmp/hf

EXPOSE 8080
CMD python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1

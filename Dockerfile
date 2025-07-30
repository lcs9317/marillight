FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .

# 포트를 외부에 노출(문제 해결에는 필수는 아니나 관례상 추가)
EXPOSE 8080

# shell form 사용: /bin/sh -c '...' 로 자동 실행
CMD uvicorn main:app --host 0.0.0.0 --port $PORT

FROM python:3.11-slim

# 선택: 파이썬 런타임 품질
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 소스 전체 복사(templates/static 등 포함)
COPY . .

# 관례상 8080 노출(실제 바인딩은 $PORT 사용)
EXPOSE 8080

# 콘솔 스크립트 이슈를 피하기 위해 module 방식으로 실행
# PORT 미주입 시 로컬에서 8080으로 기본값
CMD python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}

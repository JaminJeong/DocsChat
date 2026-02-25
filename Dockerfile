FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 설치 (lxml, curl 등 의존성)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libxml2-dev \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU 버전 먼저 설치 (CUDA 버전 대비 ~700MB 절약)
# sentence-transformers가 torch를 의존하므로 CPU 버전을 명시적으로 설치
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Python 의존성 설치 (requirements.txt의 나머지 패키지)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 소스코드 복사
COPY . .

# HuggingFace 캐시 디렉토리 생성 (볼륨 마운트 포인트)
RUN mkdir -p /app/.cache/huggingface

# 포트 노출
EXPOSE 8501

# 헬스체크 (Streamlit 서버 응답 확인)
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Streamlit 실행
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]

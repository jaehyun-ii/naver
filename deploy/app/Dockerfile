# llmops_core 코어 앱 이미지 — 게이트웨이(OpenAI 호환) + 콘솔(운영 UI).
# 같은 이미지를 command만 바꿔 gateway/console 두 서비스로 띄운다.
# GPU 의존(serving=vllm/transformers, training=unsloth)은 별도 이미지(deploy/serving/Dockerfile.hf).
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 의존성 캐시 레이어: 메타데이터 먼저 복사
COPY pyproject.toml README.md ./
COPY llmops_core ./llmops_core
COPY config ./config

# 코어 글루 + 게이트웨이(litellm) + 영속 스토어(psycopg2) + 추적(mlflow client).
RUN pip install --no-cache-dir -e ".[gateway,store,tracking]"

EXPOSE 4000 4100

# 기본은 게이트웨이. 콘솔은 compose에서 command 오버라이드.
CMD ["uvicorn", "llmops_core.gateway.app:app", "--host", "0.0.0.0", "--port", "4000"]

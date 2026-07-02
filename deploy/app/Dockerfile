# llmops_core 코어 앱 이미지 — 게이트웨이(OpenAI 호환) + 콘솔(운영 UI).
# 같은 이미지를 command만 바꿔 gateway/console 두 서비스로 띄운다.
# GPU 의존(serving=vllm/transformers, training=unsloth)은 별도 이미지(deploy/serving/Dockerfile.hf).
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 콘솔 real 모드 오케스트레이션용 docker CLI(정적 클라이언트만, 데몬은 호스트 docker.sock 사용).
# x86_64 전제. aarch64면 URL의 x86_64→aarch64로.
RUN curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-27.3.1.tgz \
      | tar xz -C /tmp \
    && mv /tmp/docker/docker /usr/local/bin/docker \
    && rm -rf /tmp/docker

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 의존성 캐시 레이어: 메타데이터 먼저 복사
COPY pyproject.toml README.md ./
COPY llmops_core ./llmops_core
COPY config ./config

# 설치 extra는 빌드 인자로 선택. 기본(gateway 슬림) vs 콘솔 prod(quality/rag/tuning 포함).
#   콘솔 prod: --build-arg EXTRAS="gateway,store,tracking,quality,rag,tuning"
ARG EXTRAS="gateway,store,tracking"
RUN pip install --no-cache-dir -e ".[${EXTRAS}]"

# PII(presidio)용 spaCy 모델 — quality extra 포함 빌드에서만 설치(en 대형 + ko).
RUN if echo "${EXTRAS}" | grep -q quality; then \
      python -m spacy download en_core_web_lg && python -m spacy download ko_core_news_sm ; \
    fi

EXPOSE 4000 4100

# 기본은 게이트웨이. 콘솔은 compose에서 command 오버라이드.
CMD ["uvicorn", "llmops_core.gateway.app:app", "--host", "0.0.0.0", "--port", "4000"]

"""LLMOps 콘솔 — 운영자/사용자 편의 UI의 백엔드 (FastAPI).

OpenAI 호환 게이트웨이(gateway/app.py)와 별개의 제어평면 콘솔이다.
키·예산·모델 관리, 챗 플레이그라운드, 릴리스 승인, 데이터 품질/데이터셋을 한 화면에서 제공.
프론트엔드(React)는 static/에서 서빙한다.

실행:
    uvicorn llmops_core.console.app:app --reload --port 4100
    # 브라우저에서 http://localhost:4100  (master key: 설정값, 기본 sk-master-changeme)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from llmops_core.common.config import get_settings

logger = logging.getLogger(__name__)

from llmops_core.console.routers import (
    audit,
    auth,
    benchmark,
    chat,
    data,
    drift,
    evaluation,
    infra,
    keys,
    models,
    pipeline,
    prompts,
    rag,
    releases,
    safety,
    serving,
    storage,
    tracking,
    traces,
    tuning,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 챗 등 LLM 호출 스팬을 Jaeger로 송출(통합 트레이스 뷰). OTel 미가용 시 graceful.
    # fail-loud: 조용히 삼키지 않고 로깅한다. prod에서는 ERROR(가시화)하되 기동은 계속(크래시 루프 방지).
    try:
        from llmops_core.telemetry import init_telemetry

        init_telemetry()
    except Exception:  # noqa: BLE001
        if get_settings().is_prod:
            logger.error("telemetry 초기화 실패(prod) — 트레이싱 없이 계속", exc_info=True)
        else:
            logger.warning("telemetry 초기화 실패 — 트레이싱 없이 계속", exc_info=True)
    yield


app = FastAPI(title="llmops-core console", version="0.1.0", lifespan=lifespan)

# CORS: 오리진 화이트리스트는 설정(cors_origins)에서. prod에서 "*"는 config 검증기가 차단.
_cors_origins = get_settings().cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(keys.router)
app.include_router(models.router)
app.include_router(chat.router)
app.include_router(releases.router)
app.include_router(data.router)
app.include_router(pipeline.router)
app.include_router(prompts.router)
app.include_router(tuning.router)
app.include_router(infra.router)
app.include_router(evaluation.router)
app.include_router(auth.router)
app.include_router(audit.router)
app.include_router(serving.router)
app.include_router(drift.router)
app.include_router(rag.router)
app.include_router(safety.router)
app.include_router(tracking.router)
app.include_router(storage.router)
app.include_router(traces.router)
app.include_router(benchmark.router)


@app.get("/api/health")
def health() -> dict:
    """LIVENESS — 프로세스 생존만 확인하는 값싼 프로브(의존성 검사 없음)."""
    return {"status": "ok", "service": "console"}


@app.get("/api/ready")
def ready() -> JSONResponse:
    """READINESS — 임계 의존성(Postgres/S3/MLflow/Qdrant 등) 실제 도달 확인.

    모두 정상이면 200, 하나라도 실패면 503 + 의존성별 상태 dict. 짧은 타임아웃으로 병렬 프로브.
    """
    from llmops_core.telemetry.health import check_readiness

    ok, deps = check_readiness()
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ready" if ok else "degraded", "service": "console",
                 "dependencies": deps},
    )


# 프런트엔드(React SPA)는 별도 web(nginx) 컨테이너가 서빙하고 /api를 이 콘솔로 프록시한다.
# 콘솔은 순수 API 서비스다. (deploy/web/, deploy/docker-compose.h100.yaml)

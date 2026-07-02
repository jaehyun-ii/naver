"""LLMOps 콘솔 — 운영자/사용자 편의 UI의 백엔드 (FastAPI).

OpenAI 호환 게이트웨이(gateway/app.py)와 별개의 제어평면 콘솔이다.
키·예산·모델 관리, 챗 플레이그라운드, 릴리스 승인, 데이터 품질/데이터셋을 한 화면에서 제공.
프론트엔드(React)는 static/에서 서빙한다.

실행:
    uvicorn llmops_core.console.app:app --reload --port 4100
    # 브라우저에서 http://localhost:4100  (master key: 설정값, 기본 sk-master-changeme)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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
    try:
        from llmops_core.telemetry import init_telemetry

        init_telemetry()
    except Exception:  # noqa: BLE001
        pass
    yield


app = FastAPI(title="llmops-core console", version="0.1.0", lifespan=lifespan)

# 개발 편의: 프론트 dev 서버(Vite 등)에서의 호출 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    return {"status": "ok", "service": "console"}


_STATIC = Path(__file__).parent / "static"
if _STATIC.exists():
    # Vite 빌드 산출물(React SPA). 해시된 자산은 /assets 하위에 위치한다.
    _ASSETS = _STATIC / "assets"
    if _ASSETS.exists():
        app.mount("/assets", StaticFiles(directory=str(_ASSETS)), name="assets")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_STATIC / "index.html")

    # SPA 폴백(React Router): /api 외의 미매칭 경로는 실제 정적 파일이면 그대로,
    # 아니면 index.html을 돌려줘 클라이언트 라우팅이 딥링크·새로고침에서도 동작하게 한다.
    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str) -> FileResponse:
        if full_path.startswith("api"):
            raise HTTPException(status_code=404, detail="Not Found")
        candidate = (_STATIC / full_path).resolve()
        if candidate.is_file() and str(candidate).startswith(str(_STATIC.resolve())):
            return FileResponse(candidate)
        return FileResponse(_STATIC / "index.html")

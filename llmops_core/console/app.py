"""LLMOps 콘솔 — 운영자/사용자 편의 UI의 백엔드 (FastAPI).

OpenAI 호환 게이트웨이(gateway/app.py)와 별개의 제어평면 콘솔이다.
키·예산·모델 관리, 챗 플레이그라운드, 릴리스 승인, 데이터 품질/데이터셋을 한 화면에서 제공.
프론트엔드(React)는 static/에서 서빙한다.

실행:
    uvicorn llmops_core.console.app:app --reload --port 4100
    # 브라우저에서 http://localhost:4100  (master key: 설정값, 기본 sk-master-changeme)
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from llmops_core.console.routers import (
    audit,
    auth,
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

app = FastAPI(title="llmops-core console", version="0.1.0")

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


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "console"}


_STATIC = Path(__file__).parent / "static"
if _STATIC.exists():

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC / "index.html")

    app.mount("/", StaticFiles(directory=str(_STATIC), html=True), name="static")

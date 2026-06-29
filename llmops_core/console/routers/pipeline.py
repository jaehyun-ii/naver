"""파이프라인(학습·배포 DAG) — 실행·승인 재개·조회.

비GPU 단계는 실제 코어 코드를 실행, GPU/Argo 단계는 모의. 승인 단계는 Release Gateway 실연동.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from llmops_core.common.security import Principal
from llmops_core.console.pipeline_engine import resume_run, start_run
from llmops_core.console.schemas import PipelineRun, RunPipelineBody
from llmops_core.console.security import require_perm
from llmops_core.console.services import services

router = APIRouter(
    prefix="/api/pipeline", tags=["pipeline"],
    dependencies=[Depends(require_perm("read"))],
)


@router.post("/run")
def run(
    body: RunPipelineBody, principal: Principal = Depends(require_perm("pipeline:run")),
) -> PipelineRun:
    svc = services()
    out = start_run(svc, body)
    svc.audit.record(principal.subject, "pipeline.run", target=out.id,
                     detail={"mode": body.mode, "method": body.method})
    return out


@router.post("/runs/{run_id}/resume")
def resume(
    run_id: str, principal: Principal = Depends(require_perm("pipeline:run")),
) -> PipelineRun:
    svc = services()
    try:
        out = resume_run(svc, run_id)
    except KeyError as exc:
        raise HTTPException(404, f"run 없음: {run_id}") from exc
    svc.audit.record(principal.subject, "pipeline.resume", target=run_id,
                     detail={"status": out.status})
    return out


@router.get("/runs")
def list_runs() -> list[PipelineRun]:
    return services().pipelines.list()


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> PipelineRun:
    run = services().pipelines.get(run_id)
    if run is None:
        raise HTTPException(404, f"run 없음: {run_id}")
    return run

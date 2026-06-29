"""입력 데이터 드리프트 탐지 — 기준선 등록 + 현재 텍스트 드리프트 검사 (제어평면).

학습/기준 데이터 분포를 기준선으로 등록하고, 라이브/평가 텍스트의 분포 변화를 PSI/JS로 검사한다.
기준선 등록은 operator 이상, 검사는 viewer 이상. judge·GPU 불필요.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from llmops_core.common.security import Principal
from llmops_core.console.security import require_perm
from llmops_core.console.services import services
from llmops_core.monitoring import compute_drift, summarize

router = APIRouter(prefix="/api/drift", tags=["drift"],
                   dependencies=[Depends(require_perm("read"))])


class BaselineBody(BaseModel):
    name: str
    texts: list[str]


class CheckBody(BaseModel):
    name: str
    texts: list[str]
    thresholds: dict[str, float] | None = None


class AutoRetrainBody(BaseModel):
    baseline_name: str  # 비교 기준선
    current_texts: list[str]  # 운영 데이터 표본(드리프트 검사 대상)
    thresholds: dict[str, float] | None = None
    auto_approve: bool = True  # 게이트 통과분 자동 승인·배포
    # 재학습 데이터/설정(드리프트 감지 시 사용)
    mode: str = "sim"
    method: str = "sft"
    served_name: str = "hcx-seed-tuned"
    train_max_steps: int = 20
    labeled: list[dict] | None = None
    preference: list[dict] | None = None


@router.get("/baselines")
def list_baselines() -> list[dict]:
    return [{"name": b["name"], "n": b["summary"].get("n"),
             "avg_len": round(b["summary"].get("avg_len", 0), 1),
             "created_at": b.get("created_at")}
            for b in services().drift_baselines.list()]


@router.post("/baseline")
def set_baseline(
    body: BaselineBody, principal: Principal = Depends(require_perm("data:write")),
) -> dict:
    if not body.texts:
        raise HTTPException(422, "기준선 텍스트가 비었습니다")
    svc = services()
    summary = summarize(body.texts)
    svc.drift_baselines.save({"name": body.name, "summary": summary, "created_at": time.time()})
    svc.audit.record(principal.subject, "drift.baseline", target=body.name,
                     detail={"n": summary["n"]})
    return {"name": body.name, "n": summary["n"], "avg_len": round(summary["avg_len"], 1)}


@router.post("/check")
def check(body: CheckBody) -> dict:
    """현재 텍스트를 기준선과 비교 → 드리프트 점수·플래그."""
    base = services().drift_baselines.get(body.name)
    if base is None:
        raise HTTPException(404, f"기준선 없음: {body.name}")
    rep = compute_drift(base["summary"], body.texts, thresholds=body.thresholds)
    return {
        "name": body.name, "drift": rep.drift, "scores": rep.scores,
        "thresholds": rep.thresholds, "n_reference": rep.n_reference,
        "n_current": rep.n_current, "details": rep.details,
    }


@router.post("/auto-retrain")
def auto_retrain(
    body: AutoRetrainBody, principal: Principal = Depends(require_perm("pipeline:run")),
) -> dict:
    """폐루프: 드리프트 검사 → 감지 시 재학습 파이프라인 트리거(+자동 승인·배포)."""
    from llmops_core.console.schemas import RunPipelineBody
    from llmops_core.monitoring.retrain import check_and_retrain

    retrain_body = RunPipelineBody(
        name="drift-retrain", mode=body.mode, method=body.method,
        served_name=body.served_name, train_max_steps=body.train_max_steps,
        labeled=body.labeled, preference=body.preference,
    )
    svc = services()
    try:
        result = check_and_retrain(
            svc, baseline_name=body.baseline_name, current_texts=body.current_texts,
            retrain_body=retrain_body, thresholds=body.thresholds,
            auto_approve=body.auto_approve,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    svc.audit.record(principal.subject, "drift.auto_retrain",
                     target=body.baseline_name,
                     detail={"drift": result["drift"], "triggered": result["triggered"]})
    return result

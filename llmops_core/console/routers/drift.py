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

# 통계적으로 의미 있는 드리프트 판정에 필요한 최소 표본 수(표본이 작으면 판정 보류)
DEFAULT_MIN_SAMPLE = 30


def _min_sample(thresholds: dict[str, float] | None) -> int:
    """thresholds 딕셔너리의 'min_sample' 키로 오버라이드 가능(없으면 기본값)."""
    try:
        return int((thresholds or {}).get("min_sample", DEFAULT_MIN_SAMPLE))
    except (TypeError, ValueError):
        return DEFAULT_MIN_SAMPLE


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
    auto_approve: bool = False  # 게이트 통과분 자동 승인·배포(기본 비활성 — 사람 승인)
    # 재학습 데이터/설정(드리프트 감지 시 사용)
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
    min_sample = _min_sample(body.thresholds)
    if len(body.texts) < min_sample:
        # 표본 과소 — 오탐 방지 위해 드리프트 판정을 보류(insufficient_sample)
        return {
            "name": body.name, "drift": False, "verdict": "insufficient_sample",
            "scores": {}, "thresholds": {}, "n_reference": base["summary"].get("n", 0),
            "n_current": len(body.texts), "min_sample": min_sample,
            "details": {"reason": f"표본 {len(body.texts)}건 < 최소 {min_sample}건"},
        }
    rep = compute_drift(base["summary"], body.texts, thresholds=body.thresholds)
    return {
        "name": body.name, "drift": rep.drift, "verdict": "drift" if rep.drift else "no_drift",
        "scores": rep.scores, "thresholds": rep.thresholds, "n_reference": rep.n_reference,
        "n_current": rep.n_current, "min_sample": min_sample, "details": rep.details,
    }


@router.post("/auto-retrain")
def auto_retrain(
    body: AutoRetrainBody, principal: Principal = Depends(require_perm("pipeline:run")),
) -> dict:
    """폐루프: 드리프트 검사 → 감지 시 재학습 파이프라인 트리거(+자동 승인·배포)."""
    from llmops_core.console.schemas import RunPipelineBody
    from llmops_core.monitoring.retrain import check_and_retrain

    retrain_body = RunPipelineBody(
        name="drift-retrain", method=body.method,
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

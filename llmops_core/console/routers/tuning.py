"""모델 학습 파라미터 최적화 — Optuna HPO (콘솔 UI).

학습 컨테이너에서 실제 HPO 실행(각 trial = 짧은 SFT + reference 평가). 분 단위라
백그라운드로 진행하고 결과는 HPORegistry에 저장, 프론트가 폴링한다.
"""

from __future__ import annotations

import secrets
import threading
import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from llmops_core.console.security import require_perm
from llmops_core.console.services import services

router = APIRouter(prefix="/api/tuning", tags=["tuning"], dependencies=[Depends(require_perm("tuning:run"))])

_SAMPLE_LABELED = [
    {"text": "선급증서 유효기간이 지나면?", "response": "즉시 선급기관에 재검사를 신청해 갱신합니다."},
    {"text": "FAT 불합격 시 절차는?", "response": "부적합 항목을 시정조치 요구서로 발행하고 재시험합니다."},
]


class HPOBody(BaseModel):
    method: str = "sft"  # sft | dpo | grpo
    labeled: list[dict] = Field(default_factory=list)  # sft:{text,response} · dpo:{prompt,chosen,rejected} · grpo:{prompt}
    eval: list[dict] = Field(default_factory=list)  # sft:{question,expected} · dpo:{prompt,chosen,rejected} · grpo:{prompt}
    trials: int = 4
    steps: int = 12
    base_model: str | None = None


def _run_real(hpo_id: str, body: HPOBody) -> None:
    rec = services().hpo.get(hpo_id)
    try:
        from llmops_core.console.executor import RealExecutor

        method = body.method or "sft"
        if method == "sft":
            labeled = body.labeled or _SAMPLE_LABELED
            evalset = body.eval or [{"question": x["text"], "expected": x["response"]} for x in labeled]
        else:  # dpo/grpo: 선호/프롬프트 데이터는 그대로, eval 없으면 train으로 대체
            labeled = body.labeled
            evalset = body.eval or body.labeled
        res = RealExecutor().hpo(hpo_id, labeled, evalset, trials=body.trials,
                                 steps=body.steps, base_model=body.base_model, method=method)
        rec.update(status="succeeded", best_params=res.get("best_params"),
                   best_value=res.get("best_value"), trials=res.get("trials", []))
    except Exception as exc:  # noqa: BLE001
        rec.update(status="failed", error=str(exc)[:300])
    finally:
        services().hpo.save(rec)  # 영속(HA)


@router.post("/hpo")
def start_hpo(body: HPOBody) -> dict:
    hpo_id = "hpo-" + secrets.token_urlsafe(5)
    rec = {"id": hpo_id, "method": body.method, "trials_n": body.trials, "status": "running",
           "created_at": time.time(), "trials": [], "best_params": None, "best_value": None}
    services().hpo.add(rec)
    threading.Thread(target=_run_real, args=(hpo_id, body), daemon=True).start()
    return rec


@router.get("/hpo")
def list_hpo() -> list[dict]:
    return services().hpo.list()


@router.get("/hpo/{hpo_id}")
def get_hpo(hpo_id: str) -> dict:
    rec = services().hpo.get(hpo_id)
    if rec is None:
        raise HTTPException(404, f"hpo 없음: {hpo_id}")
    return rec

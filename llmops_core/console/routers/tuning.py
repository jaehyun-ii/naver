"""모델 학습 파라미터 최적화 — Optuna HPO (콘솔 UI).

sim: 즉시 모의 trial(TPE 흉내)로 탐색 곡선을 보여줌. real: 학습 컨테이너에서 실제 HPO 실행
(각 trial = 짧은 SFT + reference 평가). 결과는 HPORegistry에 저장, 프론트가 폴링.
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
    mode: str = "sim"  # sim | real
    labeled: list[dict] = Field(default_factory=list)  # {text,response}
    eval: list[dict] = Field(default_factory=list)  # {question,expected}
    trials: int = 4
    steps: int = 12


def _mock_search(trials: int) -> dict:
    """TPE 흉내 모의 탐색 — 점차 좋아지는 trial 곡선(데모)."""
    import hashlib

    space_r = [8, 16, 32]
    space_a = [16, 32, 64]
    runs = []
    best = None
    for i in range(trials):
        h = int(hashlib.md5(f"trial{i}".encode()).hexdigest()[:6], 16)
        lr = round(5e-5 + (h % 1000) / 1000 * 4.5e-4, 6)
        r = space_r[h % 3]
        a = space_a[(h // 3) % 3]
        score = round(0.15 + 0.1 * (i / max(trials - 1, 1)) + (h % 50) / 1000, 4)
        params = {"learning_rate": lr, "lora_r": r, "lora_alpha": a}
        runs.append({"trial": i, "params": params, "score": score})
        if best is None or score > best["score"]:
            best = {"params": params, "score": score}
    return {"trials": runs, "best_params": best["params"], "best_value": best["score"]}


def _run_real(hpo_id: str, body: HPOBody) -> None:
    rec = services().hpo.get(hpo_id)
    try:
        from llmops_core.console.executor import RealExecutor

        labeled = body.labeled or _SAMPLE_LABELED
        evalset = body.eval or [{"question": x["text"], "expected": x["response"]} for x in labeled]
        res = RealExecutor().hpo(hpo_id, labeled, evalset, trials=body.trials, steps=body.steps)
        rec.update(status="succeeded", best_params=res.get("best_params"),
                   best_value=res.get("best_value"), trials=res.get("trials", []))
    except Exception as exc:  # noqa: BLE001
        rec.update(status="failed", error=str(exc)[:300])
    finally:
        services().hpo.save(rec)  # 영속(HA)


@router.post("/hpo")
def start_hpo(body: HPOBody) -> dict:
    hpo_id = "hpo-" + secrets.token_urlsafe(5)
    rec = {"id": hpo_id, "mode": body.mode, "trials_n": body.trials, "status": "running",
           "created_at": time.time(), "trials": [], "best_params": None, "best_value": None}
    services().hpo.add(rec)
    if body.mode == "real":
        threading.Thread(target=_run_real, args=(hpo_id, body), daemon=True).start()
    else:
        res = _mock_search(body.trials)
        rec.update(status="succeeded", **res)
        services().hpo.save(rec)  # 영속(HA)
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

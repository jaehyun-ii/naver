"""드리프트→재학습→배포 폐루프 — 입력 분포 변화가 감지되면 재학습 파이프라인을 자동 발동.

흐름: 운영 데이터(current_texts)를 기준선과 비교 → 드리프트면 retrain_body로 파이프라인 시작
→ 자동 승인 정책이면 게이트 통과분을 승인·배포(real은 백그라운드 워처가 waiting→승인→배포 재개).
드리프트가 없으면 아무 것도 하지 않는다(불필요한 재학습 방지).
"""

from __future__ import annotations

import threading
import time

from llmops_core.console.pipeline_engine import resume_run, start_run
from llmops_core.console.schemas import RunPipelineBody
from llmops_core.monitoring import compute_drift


def _auto_approve_when_ready(svc, run_id: str, *, approver: str = "auto-retrain",
                             max_wait_s: int = 1800) -> None:
    """real 모드: 백그라운드로 run이 waiting에 도달하면 자동 승인 후 배포 재개."""
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        run = svc.pipelines.get(run_id)
        if run is None:
            return
        if run.status == "waiting" and run.release_id:
            svc.releases.approve(run.release_id, approver)
            resume_run(svc, run_id)
            return
        if run.status in ("failed", "succeeded"):
            return
        time.sleep(2)


def check_and_retrain(
    svc, *, baseline_name: str, current_texts: list[str],
    retrain_body: RunPipelineBody, thresholds: dict | None = None,
    auto_approve: bool = True,
) -> dict:
    """기준선 대비 드리프트 검사 → 감지 시 재학습 파이프라인 트리거(+자동 승인·배포).

    반환: {drift(bool), scores, triggered(bool), run_id?, status?}.
    """
    base = svc.drift_baselines.get(baseline_name)
    if base is None:
        raise KeyError(f"기준선 없음: {baseline_name}")
    rep = compute_drift(base["summary"], current_texts, thresholds=thresholds)
    out = {
        "drift": rep.drift, "scores": rep.scores, "thresholds": rep.thresholds,
        "n_reference": rep.n_reference, "n_current": rep.n_current, "triggered": False,
    }
    if not rep.drift:
        out["reason"] = "드리프트 미감지 — 재학습 생략"
        return out

    # 드리프트 감지 → 재학습 트리거
    run = start_run(svc, retrain_body)
    out.update(triggered=True, run_id=run.id, status=run.status,
               reason="드리프트 감지 — 재학습 트리거")

    if auto_approve:
        if run.mode == "real":
            # real은 백그라운드 진행 → waiting 도달 시 자동 승인하는 워처 가동
            threading.Thread(target=_auto_approve_when_ready, args=(svc, run.id),
                             daemon=True).start()
        elif run.status == "waiting" and run.release_id:
            # sim은 동기 완료 → 즉시 승인·배포
            svc.releases.approve(run.release_id, "auto-retrain")
            run = resume_run(svc, run.id)
            out["status"] = run.status
    return out

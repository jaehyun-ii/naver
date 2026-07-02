"""드리프트→재학습→배포 폐루프 — 입력 분포 변화가 감지되면 재학습 파이프라인을 자동 발동.

흐름: 운영 데이터(current_texts)를 기준선과 비교 → 드리프트면 retrain_body로 파이프라인 시작
→ 자동 승인 정책이면 백그라운드 워처가 waiting→승인→배포 재개.
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
    """백그라운드로 run이 waiting에 도달하면 자동 승인 후 배포 재개."""
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


# 통계적으로 의미 있는 드리프트 판정 최소 표본 + 반복 트리거 방지 쿨다운(초)
DEFAULT_MIN_SAMPLE = 30
DEFAULT_COOLDOWN_S = 600


def check_and_retrain(
    svc, *, baseline_name: str, current_texts: list[str],
    retrain_body: RunPipelineBody, thresholds: dict | None = None,
    auto_approve: bool = False, min_sample: int | None = None,
    cooldown_s: float | None = None,
) -> dict:
    """기준선 대비 드리프트 검사 → 감지 시 재학습 파이프라인 트리거(+선택적 자동 승인·카나리 배포).

    안전장치:
    - 표본 과소(min_sample 미만)면 드리프트 판정 보류(오탐 방지).
    - 쿨다운/중복 방지: 직전 트리거 이후 cooldown_s 내이거나 이전 run이 진행 중이면 생략.
    - auto_approve 기본 False(사람 승인). True여도 배포는 stable 직접 교체가 아닌 카나리로.

    반환: {drift(bool), scores, triggered(bool), run_id?, status?}.
    """
    th = dict(thresholds or {})
    min_sample = int(th.pop("min_sample", min_sample if min_sample is not None
                             else DEFAULT_MIN_SAMPLE))
    cooldown_s = DEFAULT_COOLDOWN_S if cooldown_s is None else cooldown_s

    base = svc.drift_baselines.get(baseline_name)
    if base is None:
        raise KeyError(f"기준선 없음: {baseline_name}")

    # 1) 표본 과소 가드 — 판정 보류
    if len(current_texts) < min_sample:
        return {
            "drift": False, "verdict": "insufficient_sample", "triggered": False,
            "scores": {}, "thresholds": {}, "n_reference": base["summary"].get("n", 0),
            "n_current": len(current_texts), "min_sample": min_sample,
            "reason": f"표본 {len(current_texts)}건 < 최소 {min_sample}건 — 판정 보류",
        }

    rep = compute_drift(base["summary"], current_texts, thresholds=th or None)
    out = {
        "drift": rep.drift, "scores": rep.scores, "thresholds": rep.thresholds,
        "n_reference": rep.n_reference, "n_current": rep.n_current, "triggered": False,
    }
    if not rep.drift:
        out["reason"] = "드리프트 미감지 — 재학습 생략"
        return out

    # 2) 쿨다운/중복 방지 — 이전 재학습이 진행 중이거나 쿨다운 내면 생략
    prev_run_id = base.get("retrain_run_id")
    prev_run = svc.pipelines.get(prev_run_id) if prev_run_id else None
    if prev_run is not None and prev_run.status in ("running", "waiting", "post-running"):
        out.update(reason="이전 재학습 진행 중 — 중복 트리거 방지",
                   active_run_id=prev_run_id)
        return out
    last_at = base.get("last_retrain_at")
    if last_at is not None and (time.time() - last_at) < cooldown_s:
        out.update(reason=f"쿨다운({cooldown_s:.0f}s) 내 — 재학습 생략",
                   active_run_id=prev_run_id)
        return out

    # 3) 드리프트 감지 → 재학습 트리거(백그라운드) — 자동 승인 시 카나리 배포
    run = start_run(svc, retrain_body)
    run.artifacts["deploy_mode"] = "canary"  # 자동 경로는 stable 직접교체 금지 → 카나리
    svc.pipelines.save(run)
    # 쿨다운/중복 방지용 마커 영속
    base["last_retrain_at"] = time.time()
    base["retrain_run_id"] = run.id
    svc.drift_baselines.save(base)

    out.update(triggered=True, run_id=run.id, status=run.status,
               reason="드리프트 감지 — 재학습 트리거(카나리)")

    if auto_approve:
        # waiting 도달 시 자동 승인하는 워처 가동(분 단위 GPU 학습이라 백그라운드)
        threading.Thread(target=_auto_approve_when_ready, args=(svc, run.id),
                         daemon=True).start()
    return out

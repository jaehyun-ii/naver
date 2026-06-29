"""로컬 재학습 스케줄러 — Argo Cron/Events 없이 동작하는 자동화 (동일 파이프라인 트리거).

cron-retrain.yaml(쿠버네티스 CronWorkflow)과 같은 일을 단일 박스에서 수행한다:
주기적으로(또는 1회) 콘솔 파이프라인 API를 호출해 학습→평가→승인→배포를 발동.
정책에 따라 자동 승인(eval 게이트 통과분)까지 진행할 수 있다.

    # 1회 즉시 트리거(테스트/수동)
    python -m llmops_core.pipelines.scheduler --console http://localhost:4101 --once --auto-approve
    # 주기 실행(예: 1시간마다)
    python -m llmops_core.pipelines.scheduler --console http://localhost:4101 --interval 3600
"""

from __future__ import annotations

import argparse
import time

import httpx


def trigger_once(
    console: str, master: str, *, mode: str, method: str, pet: str,
    served_name: str, max_steps: int, auto_approve: bool, timeout: float = 1800.0,
) -> dict:
    """파이프라인 1회 실행(+옵션 자동승인). 최종 run dict 반환."""
    h = {"X-Master-Key": master, "Content-Type": "application/json"}
    with httpx.Client(timeout=timeout) as c:
        body = {"name": "scheduled-retrain", "mode": mode, "method": method, "pet": pet,
                "served_name": served_name, "train_max_steps": max_steps}
        run = c.post(f"{console}/api/pipeline/run", headers=h, json=body).json()
        rid = run["id"]

        # real 모드는 백그라운드 → waiting/실패까지 폴링
        while run.get("status") in ("running", "post-running"):
            time.sleep(3)
            run = c.get(f"{console}/api/pipeline/runs/{rid}", headers=h).json()

        if auto_approve and run.get("status") == "waiting" and run.get("release_id"):
            c.post(f"{console}/api/releases/{run['release_id']}/approve",
                   headers=h, json={"approver": "scheduler"})
            run = c.post(f"{console}/api/pipeline/runs/{rid}/resume", headers=h).json()
            while run.get("status") in ("running", "post-running"):
                time.sleep(3)
                run = c.get(f"{console}/api/pipeline/runs/{rid}", headers=h).json()
        return run


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="로컬 재학습 스케줄러(Argo Cron 대체)")
    p.add_argument("--console", default="http://localhost:4101")
    p.add_argument("--master", default="sk-master-changeme")
    p.add_argument("--mode", default="real", choices=["real", "sim"])
    p.add_argument("--method", default="sft", choices=["sft", "dpo", "grpo"])
    p.add_argument("--pet", default="lora", choices=["lora", "dora"])
    p.add_argument("--served-name", default="hcx-seed-tuned")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--auto-approve", action="store_true", help="eval 게이트 통과분 자동 승인·배포")
    p.add_argument("--interval", type=int, default=0, help="초 단위 주기(0이면 1회). --once와 동일")
    p.add_argument("--once", action="store_true")
    args = p.parse_args(argv)

    def cycle() -> None:
        run = trigger_once(
            args.console, args.master, mode=args.mode, method=args.method, pet=args.pet,
            served_name=args.served_name, max_steps=args.max_steps, auto_approve=args.auto_approve,
        )
        stages = " · ".join(f"{s['name']}:{s['status']}" for s in run.get("stages", []))
        print(f"[scheduler] run={run.get('id')} status={run.get('status')}\n  {stages}", flush=True)

    if args.once or args.interval <= 0:
        cycle()
        return 0
    print(f"[scheduler] 주기 {args.interval}s 재학습 시작 (Ctrl-C 종료)", flush=True)
    while True:
        cycle()
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())

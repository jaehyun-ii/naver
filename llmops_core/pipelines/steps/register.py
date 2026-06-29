"""DAG 단계 4 — MLflow Registry 'Production' 승격 (평가 게이트 통과 전제)."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tenant", default="shared")
    p.add_argument("--model-name", default="hcx-seed-3b")
    p.add_argument("--model-uri", default="models:/sft/latest")
    args = p.parse_args(argv)

    from llmops_core.common.schemas import ModelStage
    from llmops_core.tracking import ExperimentTracker

    tracker = ExperimentTracker()
    ref = tracker.register(args.model_uri, args.model_name, tenant=args.tenant)
    ref = tracker.promote(ref, ModelStage.PRODUCTION)
    print(f"registered {ref.name} v{ref.version} -> {ref.stage.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

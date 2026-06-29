"""DAG 단계 3 — 평가 + 게이트. 미달 시 비0 종료로 Argo DAG를 중단한다.

design 6.3/6.5: 고정 테스트셋으로 ragas 평가 → 임계값 게이트 → MLflow 기록.
데이터 버전별 점수가 자동 비교되도록 data_version을 결과/기록에 새긴다.
"""

from __future__ import annotations

import argparse
import sys

from llmops_core.common.errors import EvalGateFailed
from llmops_core.evaluation import EvalHarness, GatePolicy, load_testset


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--testset", default="evaluation_data/fixed_testset.jsonl")
    p.add_argument("--model-ref", default="hcx-seed-3b")
    p.add_argument("--data-version", default=None)
    p.add_argument("--threshold-faithfulness", type=float, default=0.8)
    p.add_argument("--threshold-context-precision", type=float, default=0.7)
    # 추론 단계(선택): answer 미보유 테스트셋이면 후보 모델로 생성 후 평가
    p.add_argument("--generate", action="store_true")
    p.add_argument("--gateway-url", default="http://localhost:4000")
    p.add_argument("--api-key", default=None)
    args = p.parse_args(argv)

    cases = load_testset(args.testset)
    gate = GatePolicy(
        thresholds={
            "faithfulness": args.threshold_faithfulness,
            "context_precision": args.threshold_context_precision,
        }
    )

    predict = None
    if args.generate:
        from llmops_core.evaluation import PredictConfig

        predict = PredictConfig(
            base_url=args.gateway_url, api_key=args.api_key or "", model=args.model_ref
        )

    harness = EvalHarness(gate=gate)
    result = harness.run(
        "ci-gate",
        cases,
        args.model_ref,
        kind="ragas",
        data_version=args.data_version,
        predict=predict,
    )

    # MLflow 기록 (tracking 서버가 있을 때)
    try:
        from llmops_core.tracking import ExperimentTracker

        tracker = ExperimentTracker()
        with tracker.run({"model_ref": args.model_ref}, data_ver=args.data_version):
            tracker.log_eval(result)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] MLflow 기록 건너뜀: {exc}", file=sys.stderr)

    print(f"metrics={result.metrics} passed={result.passed}")
    if not result.passed:
        # 게이트 미달 → 파이프라인 차단
        try:
            gate.evaluate(result.metrics)
        except EvalGateFailed as exc:
            print(f"[GATE FAILED] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

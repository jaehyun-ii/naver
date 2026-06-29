"""평가 게이트 정책 — CI/파이프라인에서 품질 회귀를 차단하는 임계값 규칙."""

from __future__ import annotations

from dataclasses import dataclass, field

from llmops_core.common.errors import EvalGateFailed


@dataclass
class GatePolicy:
    """메트릭별 최소 임계값. 하나라도 미달이면 게이트 실패.

    예: GatePolicy(thresholds={"faithfulness": 0.8, "context_precision": 0.7})
    """

    thresholds: dict[str, float] = field(default_factory=dict)

    def evaluate(self, metrics: dict[str, float]) -> bool:
        """통과 여부 반환. 미달 시 EvalGateFailed (CI 빌드 실패용)."""
        failures: dict[str, tuple[float, float]] = {}
        for metric, threshold in self.thresholds.items():
            actual = metrics.get(metric)
            if actual is None or actual < threshold:
                failures[metric] = (actual if actual is not None else 0.0, threshold)
        if failures:
            raise EvalGateFailed(failures)
        return True

    def passes(self, metrics: dict[str, float]) -> bool:
        """예외 없이 bool만 필요할 때."""
        try:
            return self.evaluate(metrics)
        except EvalGateFailed:
            return False

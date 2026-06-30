"""평가 하니스 — ragas/deepeval의 metrics 구현체만 임베드 + 자체 게이트.

Confident AI/플랫폼 SaaS는 채용하지 않고 메트릭 함수/클래스만 사용한다.
고정 테스트셋(JSONL, 50~200 케이스)을 로드해 정량 평가 → 게이트 → MLflow 기록.
데이터 버전별 비교(design 6.5)를 위해 data_version을 결과에 새긴다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from llmops_core.common.errors import OptionalDependencyError
from llmops_core.common.schemas import EvalCase, EvalResult
from llmops_core.evaluation.gate import GatePolicy

if TYPE_CHECKING:
    from llmops_core.evaluation.predict import PredictConfig


def load_testset(path: str | Path) -> list[EvalCase]:
    """JSONL 고정 테스트셋 로더. 각 줄 = EvalCase 필드."""
    cases: list[EvalCase] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            cases.append(EvalCase(**json.loads(line)))
    return cases


def run_ragas(cases: list[EvalCase], metrics: list | None = None) -> dict[str, float]:
    """RAG 전용 메트릭(faithfulness, context precision/recall, answer relevancy)."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("ragas", "evaluation") from exc

    metrics = metrics or [faithfulness, answer_relevancy, context_precision, context_recall]
    ds = Dataset.from_list(
        [
            {
                "question": c.question,
                "answer": c.answer or "",
                "contexts": c.contexts,
                "ground_truth": c.expected or "",
            }
            for c in cases
        ]
    )
    result = evaluate(ds, metrics=metrics)
    return {k: float(v) for k, v in result.items()}


def _chars(s: str) -> list[str]:
    """공백 제외 문자 시퀀스 — 한국어 등 CJK에 강건한 토큰 단위."""
    return [c for c in s.lower() if not c.isspace()]


def _char_f1(pred: str, ref: str) -> float:
    """생성답·정답의 문자 다중집합 F1 (judge 불필요, 결정적)."""
    from collections import Counter

    p, r = _chars(pred), _chars(ref)
    if not p or not r:
        return 0.0
    overlap = sum((Counter(p) & Counter(r)).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(r)
    return 2 * prec * rec / (prec + rec)


def run_reference_metrics(cases: list[EvalCase]) -> dict[str, float]:
    """LLM judge 없이 정답(expected) 대비 결정적 메트릭을 계산.

    - reference_f1: 문자 단위 F1 평균(내용 일치도)
    - answer_match: 정답 핵심부 포함 비율(느슨한 정확도)
    로컬·오프라인에서 외부 키 없이 학습 모델 품질을 정량화한다.
    """
    f1s: list[float] = []
    matches: list[float] = []
    for c in cases:
        ans, exp = (c.answer or ""), (c.expected or "")
        f1s.append(_char_f1(ans, exp))
        key = "".join(_chars(exp))[:8]
        matches.append(1.0 if key and key in "".join(_chars(ans)) else 0.0)
    n = max(len(cases), 1)
    return {
        "reference_f1": round(sum(f1s) / n, 4),
        "answer_match": round(sum(matches) / n, 4),
    }


def run_deepeval(
    cases: list[EvalCase],
    metric_factories: list[Callable] | None = None,
) -> dict[str, float]:
    """일반 LLM 품질 메트릭(G-Eval, hallucination, answer relevancy 등) 평균."""
    try:
        from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric
        from deepeval.test_case import LLMTestCase
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("deepeval", "evaluation") from exc

    factories = metric_factories or [AnswerRelevancyMetric, HallucinationMetric]
    scores: dict[str, list[float]] = {}
    for c in cases:
        tc = LLMTestCase(
            input=c.question,
            actual_output=c.answer or "",
            context=c.contexts or None,
            expected_output=c.expected,
        )
        for factory in factories:
            metric = factory()
            metric.measure(tc)
            scores.setdefault(metric.__class__.__name__, []).append(metric.score)
    return {name: sum(v) / len(v) for name, v in scores.items() if v}


class EvalHarness:
    """평가 실행 + 게이트 + 결과 패키징(MLflow 기록은 tracking 모듈에서 연계)."""

    def __init__(self, gate: GatePolicy | None = None) -> None:
        self.gate = gate or GatePolicy()

    def run(
        self,
        suite: str,
        cases: list[EvalCase],
        model_ref: str,
        *,
        kind: str = "ragas",
        data_version: str | None = None,
        predict: "PredictConfig | None" = None,
    ) -> EvalResult:
        # 추론 단계(선택): answer가 비어 있으면 후보 모델로 생성한 뒤 평가
        if predict is not None:
            from llmops_core.evaluation.predict import generate_answers

            cases = generate_answers(cases, predict)

        if kind == "ragas":
            metrics = run_ragas(cases)
        elif kind == "deepeval":
            metrics = run_deepeval(cases)
        elif kind == "reference":
            metrics = run_reference_metrics(cases)
        else:
            raise ValueError(f"알 수 없는 평가 종류: {kind}")

        passed = self.gate.passes(metrics)
        return EvalResult(
            suite=suite,
            model_ref=model_ref,
            data_version=data_version,
            metrics=metrics,
            passed=passed,
            num_cases=len(cases),
        )

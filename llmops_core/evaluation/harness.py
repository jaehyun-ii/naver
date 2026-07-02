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


import re

_WS_RE = re.compile(r"\s+")


def _normalize(s: str) -> str:
    """정규화: 소문자·양끝공백 제거·연속공백 1개로 축약(결정적 채점의 전처리)."""
    return _WS_RE.sub(" ", (s or "").strip().lower())


def _chars(s: str) -> list[str]:
    """공백 제외 문자 시퀀스 — 한국어 등 CJK에 강건한 토큰 단위."""
    return [c for c in s.lower() if not c.isspace()]


def _tokens(s: str) -> list[str]:
    """형태소 대용 토큰 — 정규화 후 공백 분리(공백 없는 CJK는 문자열로 폴백)."""
    norm = _normalize(s)
    if not norm:
        return []
    toks = norm.split(" ")
    # 공백이 없어 토큰이 1개뿐이면(예: CJK 붙임) 문자 단위로 폴백해 F1이 의미를 갖게 한다
    if len(toks) == 1 and len(norm) > 1:
        return [c for c in norm if not c.isspace()]
    return toks


def _multiset_f1(pred_units: list[str], ref_units: list[str]) -> float:
    from collections import Counter

    if not pred_units or not ref_units:
        return 0.0
    overlap = sum((Counter(pred_units) & Counter(ref_units)).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(pred_units), overlap / len(ref_units)
    return 2 * prec * rec / (prec + rec)


def _char_f1(pred: str, ref: str) -> float:
    """생성답·정답의 문자 다중집합 F1 (judge 불필요, 결정적)."""
    return _multiset_f1(_chars(pred), _chars(ref))


def token_f1(pred: str, ref: str) -> float:
    """토큰(공백/형태소) 단위 다중집합 F1 — 내용 일치도."""
    return _multiset_f1(_tokens(pred), _tokens(ref))


def exact_match(pred: str, ref: str) -> float:
    """정규화 후 완전 일치(1.0/0.0) — strip/lowercase/공백축약 기준의 정확도."""
    p, r = _normalize(pred), _normalize(ref)
    return 1.0 if p and r and p == r else 0.0


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L 유사 LCS 기반 F 점수(토큰 LCS 비율)."""
    p, r = _tokens(pred), _tokens(ref)
    if not p or not r:
        return 0.0
    lcs = _lcs_len(p, r)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


def case_metrics(pred: str, ref: str) -> dict[str, float]:
    """단일 케이스의 결정적 메트릭 묶음."""
    return {
        "exact_match": exact_match(pred, ref),
        "token_f1": round(token_f1(pred, ref), 4),
        "char_f1": round(_char_f1(pred, ref), 4),
        "rouge_l": round(rouge_l(pred, ref), 4),
    }


def reference_metrics_detailed(cases: list[EvalCase]) -> dict:
    """정답(expected) 대비 결정적 메트릭을 케이스별 + 집계로 반환(judge 불필요).

    반환: {"aggregate": {...}, "cases": [{"id", exact_match, token_f1, char_f1, rouge_l}]}.
    """
    per_case: list[dict] = []
    acc: dict[str, float] = {"exact_match": 0.0, "token_f1": 0.0, "char_f1": 0.0, "rouge_l": 0.0}
    for c in cases:
        m = case_metrics(c.answer or "", c.expected or "")
        per_case.append({"id": getattr(c, "id", None), **m})
        for k, v in m.items():
            acc[k] += v
    n = max(len(cases), 1)
    aggregate = {k: round(v / n, 4) for k, v in acc.items()}
    # 하위호환 별칭: reference_f1=토큰 F1, answer_match=정규화 완전일치 비율
    aggregate["reference_f1"] = aggregate["token_f1"]
    aggregate["answer_match"] = aggregate["exact_match"]
    return {"aggregate": aggregate, "cases": per_case}


def run_reference_metrics(cases: list[EvalCase]) -> dict[str, float]:
    """LLM judge 없이 정답(expected) 대비 결정적 메트릭 집계를 계산.

    - exact_match/answer_match: 정규화 완전일치 비율(strip·lowercase·공백축약)
    - token_f1/reference_f1: 토큰 단위 F1 평균(내용 일치도)
    - char_f1: 문자 단위 F1 평균(CJK 강건)
    - rouge_l: LCS 기반 F 점수 평균
    """
    return reference_metrics_detailed(cases)["aggregate"]


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

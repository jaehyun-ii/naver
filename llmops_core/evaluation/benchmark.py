"""벤치마크 평가 — 명명·버전 고정된 평가 데이터셋에 모델을 돌려 표준 메트릭 산출.

벤치마크 = 고정 테스트셋(cases). 같은 벤치마크로 여러 모델을 평가해 리더보드로 비교한다.
모델 생성은 generate(question)->answer 콜러블(게이트웨이/ModelClient 경유)로 주입하고,
정답(expected) 대비 결정적 reference 메트릭(judge 불필요)으로 채점한다.
"""

from __future__ import annotations

from typing import Callable

from llmops_core.common.schemas import EvalCase
from llmops_core.evaluation.harness import run_reference_metrics


def run_benchmark(generate: Callable[[str], str], cases: list[dict]) -> dict:
    """cases({question/text, expected/response})를 generate로 추론·채점.

    반환: {metrics: {reference_f1, answer_match}, num_cases, errors}.
    generate 호출 실패는 빈 답으로 처리(graceful)하고 errors로 보고한다.
    """
    scored: list[EvalCase] = []
    errors = 0
    for i, c in enumerate(cases):
        q = c.get("question") or c.get("text") or ""
        exp = c.get("expected") or c.get("response")
        try:
            ans = generate(q) if q else ""
        except Exception:  # noqa: BLE001
            ans, errors = "", errors + 1
        scored.append(EvalCase(id=str(c.get("id") or f"c{i}"), question=q,
                               expected=exp, answer=ans))
    metrics = run_reference_metrics(scored)
    return {"metrics": metrics, "num_cases": len(scored), "errors": errors}

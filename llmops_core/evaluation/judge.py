"""LLM-as-judge — 생성 답변을 judge 모델로 0~1 채점.

judge 모델은 model_list.yaml의 논리명으로 선택(게이트웨이 경유 → 백엔드 자유).
정답이 모호한 생성형 출력에 대한 보조 지표. 결정적 reference 메트릭과 병행한다.
"""

from __future__ import annotations

import re

DEFAULT_JUDGE_PROMPT = (
    "당신은 엄정한 채점자입니다. 질문·정답(참고)·모델답변을 보고 모델답변의 정확성을 "
    "0.0~1.0으로 채점하세요. 0.0=완전히 틀림, 1.0=정답과 동등. 숫자만 한 줄로 답하세요.\n\n"
    "[질문]\n{q}\n[정답]\n{ref}\n[모델답변]\n{ans}\n점수:"
)


def parse_score(text: str) -> float:
    """judge 출력에서 0~1 점수 추출(10점 척도 등 방어)."""
    m = re.search(r"\d+(?:\.\d+)?", text or "")
    if not m:
        return 0.0
    v = float(m.group(0))
    if v > 1.0:
        v = v / 10.0 if v <= 10 else 1.0
    return max(0.0, min(1.0, v))


def run_judge_eval(client, judge_model: str, cases, prompt_template: str | None = None) -> dict:
    """각 케이스(question/expected/answer)를 judge 모델로 채점 → 평균 점수.

    client.complete(model, messages) 인터페이스(common.model_client.ModelClient 호환).
    prompt_template(미지정 시 DEFAULT_JUDGE_PROMPT)은 {q}{ref}{ans} 변수를 받는다.
    모델 호출 실패는 0.0으로 처리(graceful) 후 실패 수를 함께 보고.
    """
    template = prompt_template or DEFAULT_JUDGE_PROMPT
    scores: list[float] = []
    errors = 0
    for c in cases:
        q = getattr(c, "question", "") or ""
        ref = getattr(c, "expected", "") or ""
        ans = getattr(c, "answer", "") or ""
        msgs = [{"role": "user", "content": template.format(q=q, ref=ref, ans=ans)}]
        try:
            txt = client.complete(judge_model, msgs, temperature=0.0, max_tokens=8)
            scores.append(parse_score(txt))
        except Exception:  # noqa: BLE001
            errors += 1
            scores.append(0.0)
    n = max(len(scores), 1)
    return {
        "judge_score": round(sum(scores) / n, 4),
        "judge_n": len(scores),
        "judge_errors": errors,
    }

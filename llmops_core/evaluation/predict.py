"""평가 전 추론 단계 — 후보 모델로 답변 생성(게이트웨이 경유).

평가 하니스는 EvalCase.answer가 채워져 있어야 메트릭을 낼 수 있다. 보통 테스트셋에는
question/expected/contexts만 있으므로, 여기서 후보 모델을 호출해 answer를 생성한다.
호출은 게이트웨이(OpenAI 호환)로 보내 백엔드(vLLM/transformers/CLOVA) 무관하게 동작.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from llmops_core.common.schemas import EvalCase


@dataclass
class PredictConfig:
    base_url: str  # 게이트웨이 주소 (예: http://localhost:4000)
    api_key: str  # 테넌트 가상키
    model: str  # 논리 모델명 (평가 대상)
    temperature: float = 0.0  # 평가 재현성을 위해 기본 0
    max_tokens: int = 512
    timeout: float = 60.0
    # RAG 평가 시 컨텍스트를 프롬프트에 주입(있을 때)
    use_contexts: bool = True


def _build_messages(case: EvalCase, cfg: PredictConfig) -> list[dict]:
    messages: list[dict] = []
    if cfg.use_contexts and case.contexts:
        context = "\n\n".join(case.contexts)
        messages.append(
            {"role": "system", "content": f"다음 컨텍스트를 근거로 답하라:\n{context}"}
        )
    messages.append({"role": "user", "content": case.question})
    return messages


def generate_answers(
    cases: list[EvalCase], cfg: PredictConfig, *, client: httpx.Client | None = None
) -> list[EvalCase]:
    """각 케이스에 대해 모델 답변을 생성해 .answer를 채운 새 리스트 반환.

    이미 answer가 있는 케이스는 건너뛴다(외부 사전생성 결과 존중).
    client를 주입하면 그것을 사용(테스트/커넥션 재사용), 아니면 내부에서 생성·정리한다.
    """
    owns_client = client is None
    client = client or httpx.Client(timeout=cfg.timeout)
    out: list[EvalCase] = []
    try:
        for case in cases:
            if case.answer:
                out.append(case)
                continue
            resp = client.post(
                f"{cfg.base_url.rstrip('/')}/v1/chat/completions",
                headers={"Authorization": f"Bearer {cfg.api_key}"},
                json={
                    "model": cfg.model,
                    "messages": _build_messages(case, cfg),
                    "temperature": cfg.temperature,
                    "max_tokens": cfg.max_tokens,
                },
            )
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"]
            out.append(case.model_copy(update={"answer": answer}))
    finally:
        if owns_client:
            client.close()
    return out

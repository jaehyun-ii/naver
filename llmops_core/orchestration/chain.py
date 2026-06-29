"""오케스트레이션 — langchain_core 프리미티브만 차용(Primitive 모드).

`langchain` 메가패키지는 채용하지 않는다. Runnable/Prompt/Parser 프리미티브로 체인을
명시적으로 직접 구성하고, LLM 호출은 게이트웨이(OpenAI 호환)로 보낸다.
프롬프트는 프롬프트 스토어(라벨)에서 런타임에 가져온다 → 코드 배포 없이 교체.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from llmops_core.common.errors import OptionalDependencyError
from llmops_core.prompts import GitPromptStore


@dataclass
class GatewayLLM:
    """게이트웨이를 향하는 최소 OpenAI 호환 클라이언트 (모델을 직접 부르지 않음)."""

    base_url: str
    api_key: str  # 테넌트 가상키
    model: str
    temperature: float = 0.7

    def invoke(self, messages: list[dict]) -> str:
        resp = httpx.post(
            f"{self.base_url.rstrip('/')}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages, "temperature": self.temperature},
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def build_prompt(template: str):
    """langchain_core.prompts.ChatPromptTemplate 프리미티브로 래핑(선택)."""
    try:
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("langchain-core", "orchestration") from exc
    return ChatPromptTemplate.from_template(template)


class SimpleChain:
    """프롬프트 스토어(라벨) → 렌더 → 게이트웨이 호출, 명시적 체인.

    과도한 추상화 없이 핵심 경로만 직접 구성(디버깅 용이).
    """

    def __init__(self, llm: GatewayLLM, prompt_name: str, store: GitPromptStore | None = None):
        self.llm = llm
        self.prompt_name = prompt_name
        self.store = store or GitPromptStore()

    def run(self, *, label: str = "prod", system: str | None = None, **vars: object) -> str:
        prompt = self.store.by_label(self.prompt_name, label)
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt.render(**vars)})
        return self.llm.invoke(messages)

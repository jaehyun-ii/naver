"""LLM 스팬 시맨틱 규약 — 자체 정의.

모든 LLM 호출 스팬에 동일한 속성 키를 부여해, 백엔드(OTel/Langfuse/ClickHouse)가
무엇이든 추적↔평가↔비용 분석이 일관되게 연결되도록 한다. (관측 백본의 핵심 글루)

OpenInference/OpenLLMetry 관례와 호환되는 `gen_ai.*` / `llm.*` 키를 채택한다.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from llmops_core.common.schemas import TenantContext, Usage


# ── 속성 키 상수 (자체 규약) ──
class Attr:
    SYSTEM = "gen_ai.system"  # 예: "vllm", "clova"
    REQUEST_MODEL = "gen_ai.request.model"  # 논리 모델명
    RESPONSE_MODEL = "gen_ai.response.model"  # 실제 백엔드 모델
    PROMPT_TOKENS = "gen_ai.usage.input_tokens"
    COMPLETION_TOKENS = "gen_ai.usage.output_tokens"
    TOTAL_TOKENS = "gen_ai.usage.total_tokens"
    TEMPERATURE = "gen_ai.request.temperature"
    # 자체 확장: 멀티테넌시/비용 (게이트웨이 정책 평면과 연결)
    TENANT_ID = "llmops.tenant.id"
    KEY_ID = "llmops.key.id"
    COST_USD = "llmops.cost.usd"
    PROMPT_LABEL = "llmops.prompt.label"  # Langfuse/프롬프트 스토어 라벨
    PROMPT_VERSION = "llmops.prompt.version"


def tracer(name: str = "llmops_core") -> trace.Tracer:
    return trace.get_tracer(name)


@contextmanager
def llm_span(
    request_model: str,
    *,
    tenant: TenantContext | None = None,
    system: str = "vllm",
    temperature: float | None = None,
    prompt_label: str | None = None,
    prompt_version: str | None = None,
) -> Iterator[Span]:
    """LLM 호출 1건을 규약 속성과 함께 스팬으로 감싼다.

    사용 예:
        with llm_span("hcx-seed-3b", tenant=ctx) as span:
            resp = await router.acompletion(...)
            record_usage(span, usage, cost_usd=cost)
    """
    with tracer().start_as_current_span(f"chat {request_model}") as span:
        span.set_attribute(Attr.SYSTEM, system)
        span.set_attribute(Attr.REQUEST_MODEL, request_model)
        if temperature is not None:
            span.set_attribute(Attr.TEMPERATURE, temperature)
        if tenant is not None:
            span.set_attribute(Attr.TENANT_ID, tenant.tenant_id)
            span.set_attribute(Attr.KEY_ID, tenant.key_id)
        if prompt_label:
            span.set_attribute(Attr.PROMPT_LABEL, prompt_label)
        if prompt_version:
            span.set_attribute(Attr.PROMPT_VERSION, prompt_version)
        try:
            yield span
        except Exception as exc:  # noqa: BLE001
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            raise


def record_usage(
    span: Span,
    usage: Usage,
    *,
    response_model: str | None = None,
    cost_usd: float | None = None,
) -> None:
    """토큰/비용/응답모델을 스팬에 규약대로 기록."""
    span.set_attribute(Attr.PROMPT_TOKENS, usage.prompt_tokens)
    span.set_attribute(Attr.COMPLETION_TOKENS, usage.completion_tokens)
    span.set_attribute(Attr.TOTAL_TOKENS, usage.total_tokens)
    if response_model:
        span.set_attribute(Attr.RESPONSE_MODEL, response_model)
    if cost_usd is not None:
        span.set_attribute(Attr.COST_USD, cost_usd)

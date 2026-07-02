"""챗 플레이그라운드 — 가상키 인증 → 정책 적용 → 백엔드 호출 → 비용 반영.

백엔드는 litellm Router(게이트웨이와 동일 싱글톤)를 통해 호출하며, 입력 가드레일·출력
PII 마스킹도 게이트웨이와 동일 경로를 재사용한다(정합). 백엔드 장애는 502로 노출한다
(이전의 무조건 에코 폴백은 장애를 은폐하고 가드레일/마스킹을 우회하므로 제거).
오프라인 데모용 에코는 dev(비운영) 환경에서만 opt-in 폴백으로 유지한다.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from llmops_core.common.config import get_settings
from llmops_core.common.errors import AuthError, PolicyViolation
from llmops_core.common.schemas import ChatCompletionRequest, Usage
from llmops_core.console.schemas import ChatBody, ChatReply
from llmops_core.console.services import services
from llmops_core.telemetry import llm_span, record_usage

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _dev_echo_enabled() -> bool:
    """오프라인 데모 에코 폴백 허용 여부 — 비운영(dev)에서만 on(운영은 502로 장애 노출)."""
    return not get_settings().is_prod


def _echo(messages: list, model: str) -> ChatReply:
    """오프라인 데모 백엔드 — 마지막 사용자 메시지를 에코하고 토큰/비용을 추정."""
    last = next((m.content for m in reversed(messages) if m.role == "user"), "")
    content = f"[echo:{model}] {last}"
    prompt_tokens = sum(len(m.content.split()) for m in messages)
    completion_tokens = len(content.split())
    return ChatReply(
        content=content,
        model=model,
        backend="echo",
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        cost_usd=0.0,
    )


@router.post("")
async def chat(body: ChatBody) -> ChatReply:
    svc = services()

    # 1) 가상키 인증
    try:
        ctx = svc.key_store.verify(body.virtual_key)
    except AuthError as exc:
        raise HTTPException(401, str(exc)) from exc

    # 2) 정책 검사 (화이트리스트/RPM/예산)
    try:
        svc.policy.authorize(ctx, body.model)
    except PolicyViolation as exc:
        raise HTTPException(429, str(exc)) from exc

    # 게이트웨이와 동일한 라우터 싱글톤·가드레일 경로를 재사용(요청마다 생성 금지·정합).
    from llmops_core.gateway.app import (
        _apply_output_guardrail,
        get_router,
        guardrails,
    )

    # 3) 입력 가드레일 (프롬프트 인젝션) — 게이트웨이와 동일 적용
    gin = guardrails.check_input([m.model_dump() for m in body.messages])
    if not gin.allowed:
        raise HTTPException(400, f"guardrail: {gin.reason}")

    req = ChatCompletionRequest(
        model=body.model,
        messages=body.messages,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
    )

    # 4) 백엔드 호출 + 트레이싱(OTel 스팬 — Jaeger 통합 뷰)
    with llm_span(body.model, tenant=ctx, temperature=body.temperature) as span:
        try:
            router_ = get_router()
            raw, usage, cache_hit = await router_.acompletion(req)
        except Exception as exc:  # noqa: BLE001 — 백엔드 미가용/litellm 오류
            if not _dev_echo_enabled():
                # 운영: 장애를 은폐하지 않고 502로 노출.
                raise HTTPException(502, f"백엔드 호출 실패: {exc}") from exc
            reply = _echo(body.messages, body.model)  # dev 오프라인 데모 폴백
        else:
            # 5) 출력 가드레일(금칙어 차단·PII 마스킹) — 게이트웨이와 동일 적용
            _apply_output_guardrail(raw)
            cost = 0.0 if cache_hit else router_.estimate_cost(body.model, usage)
            reply = ChatReply(
                content=raw["choices"][0]["message"]["content"],
                model=raw.get("model", body.model),
                backend="router",
                usage=usage.model_dump(),
                cost_usd=cost,
            )
        record_usage(span, Usage(**reply.usage), response_model=reply.model,
                     cost_usd=reply.cost_usd)

    # 6) 비용 반영 (예산 통제)
    svc.policy.record_spend(ctx, reply.cost_usd)
    return reply

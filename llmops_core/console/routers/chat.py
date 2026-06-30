"""챗 플레이그라운드 — 가상키 인증 → 정책 적용 → 백엔드 호출 → 비용 반영.

백엔드 우선순위: litellm Router(설치 시) → 없거나 실패하면 에코(데모용, 오프라인 동작).
정책 평면(화이트리스트/RPM/예산)은 백엔드 가용성과 무관하게 항상 적용된다.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from llmops_core.common.errors import AuthError, PolicyViolation
from llmops_core.common.schemas import ChatCompletionRequest
from llmops_core.console.schemas import ChatBody, ChatReply
from llmops_core.console.services import services

router = APIRouter(prefix="/api/chat", tags=["chat"])


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

    # 3) 백엔드 호출 (litellm Router → 실패 시 에코)
    reply = _echo(body.messages, body.model)
    try:
        from llmops_core.gateway.router import GatewayRouter

        req = ChatCompletionRequest(
            model=body.model,
            messages=body.messages,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )
        router_ = GatewayRouter()
        raw, usage, cache_hit = await router_.acompletion(req)
        cost = 0.0 if cache_hit else router_.estimate_cost(body.model, usage)
        reply = ChatReply(
            content=raw["choices"][0]["message"]["content"],
            model=raw.get("model", body.model),
            backend="router",
            usage=usage.model_dump(),
            cost_usd=cost,
        )
    except Exception:  # noqa: BLE001 — litellm 미설치/백엔드 미가용 시 에코 폴백
        pass

    # 4) 비용 반영 (예산 통제)
    svc.policy.record_spend(ctx, reply.cost_usd)
    return reply

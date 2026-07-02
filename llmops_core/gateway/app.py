"""게이트웨이 FastAPI — OpenAI 호환 단일 진입점.

흐름: 가상키 인증 → 정책 엔진(화이트리스트·RPM·예산) → litellm.Router → 비용 반영
     → 전 과정 OTel 스팬(자체 LLM 규약)으로 트레이싱.
앱은 논리 모델명만 호출하므로 백엔드(vLLM/CLOVA) 교체에 영향받지 않는다.
"""

from __future__ import annotations

import json
import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from llmops_core.common.config import get_settings
from llmops_core.common.errors import AuthError, PolicyViolation
from llmops_core.common.schemas import ChatCompletionRequest, TenantContext, Usage
from llmops_core.common.security import make_audit_log
from llmops_core.common.stores import make_key_store, make_ledger
from llmops_core.gateway.cache import CacheStats
from llmops_core.gateway.guardrails import GuardrailEngine
from llmops_core.gateway.keys import VirtualKeyStore
from llmops_core.gateway.policy import PolicyEngine
from llmops_core.gateway.router import GatewayRouter
from llmops_core.telemetry import init_telemetry, llm_span, record_usage

logger = logging.getLogger(__name__)

# ── 의존 컴포넌트 (store.backend=postgres면 Postgres 영속, 게이트웨이↔콘솔 상태 공유) ──
key_store: VirtualKeyStore = make_key_store()
audit = make_audit_log()  # 인증 실패 등 보안 이벤트 감사 싱크
policy = PolicyEngine(ledger=make_ledger())
guardrails = GuardrailEngine()
cache_stats = CacheStats(enabled=get_settings().cache.enabled)
_router: GatewayRouter | None = None
_rag = None  # 서빙 RAG 파이프라인(필요 시 lazy)


def get_rag():
    global _rag
    if _rag is None:
        from llmops_core.rag import RagPipeline

        _rag = RagPipeline()
    return _rag


def _rag_enabled(req: ChatCompletionRequest) -> bool:
    """요청별 extra.rag override > 설정 rag.serving_enabled."""
    override = req.extra.get("rag")
    if override is not None:
        return bool(override)
    return get_settings().rag.serving_enabled


def get_router() -> GatewayRouter:
    global _router
    if _router is None:
        _router = GatewayRouter()
    return _router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # fail-loud: 텔레메트리 초기화 실패를 조용히 삼키지 않는다. prod는 ERROR로 가시화하되
    # 기동은 계속(트레이싱 부재로 게이트웨이 자체가 죽는 크래시 루프 방지).
    try:
        init_telemetry()
    except Exception:  # noqa: BLE001
        if get_settings().is_prod:
            logger.error("telemetry 초기화 실패(prod) — 트레이싱 없이 계속", exc_info=True)
        else:
            logger.warning("telemetry 초기화 실패 — 트레이싱 없이 계속", exc_info=True)
    yield


app = FastAPI(title="llmops-core gateway", version="0.1.0", lifespan=lifespan)


async def authenticate(authorization: str = Header(...)) -> TenantContext:
    if not authorization.lower().startswith("bearer "):
        _audit_auth_fail("gateway", "malformed_authorization")
        raise HTTPException(401, "Bearer 토큰 필요")
    raw_key = authorization.split(" ", 1)[1].strip()
    try:
        return key_store.verify(raw_key)
    except AuthError as exc:
        _audit_auth_fail("gateway", str(exc))
        raise HTTPException(401, str(exc)) from exc


def _audit_auth_fail(target: str, reason: str) -> None:
    """인증 실패 감사 기록(P8: 실패 로깅 표준화). 싱크 미가용 시 graceful."""
    try:
        audit.record("anonymous", "auth:fail", target=target,
                     result="denied", detail={"reason": reason})
    except Exception:  # noqa: BLE001
        pass


@app.get("/health")
async def health() -> dict[str, str]:
    """LIVENESS — 프로세스 생존만 확인하는 값싼 프로브(의존성 검사 없음)."""
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> JSONResponse:
    """READINESS — 임계 의존성(Postgres/Redis/S3/MLflow/Qdrant 등) 실제 도달 확인.

    모두 정상이면 200, 하나라도 실패면 503 + 의존성별 상태 dict. 짧은 타임아웃으로 병렬 프로브.
    """
    from llmops_core.telemetry.health import check_readiness

    ok, deps = check_readiness()
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ready" if ok else "degraded", "service": "gateway",
                 "dependencies": deps},
    )


# ── 관리자: 가상키 발급 (master_key로 보호) — 멀티테넌시 제어 평면 ──
class IssueKeyRequest(BaseModel):
    tenant_id: str
    allowed_models: list[str] = []
    monthly_budget_usd: float | None = None
    rpm_limit: int | None = None


def _require_master(x_master_key: str | None = Header(default=None)) -> None:
    # 상수시간 비교(타이밍 공격 방지). 헤더 미존재(None)는 무효로 취급.
    if x_master_key is None or not secrets.compare_digest(
        x_master_key, get_settings().gateway.master_key
    ):
        _audit_auth_fail("gateway/admin", "master_key_mismatch")
        raise HTTPException(403, "master key 불일치")


@app.post("/admin/keys")
async def issue_key(
    body: IssueKeyRequest, _: None = Depends(_require_master)
) -> dict:
    raw = key_store.issue(
        body.tenant_id,
        allowed_models=body.allowed_models,
        monthly_budget_usd=body.monthly_budget_usd,
        rpm_limit=body.rpm_limit,
    )
    return {"tenant_id": body.tenant_id, "virtual_key": raw}


@app.get("/admin/cache/stats")
async def get_cache_stats(_: None = Depends(_require_master)) -> dict:
    """캐시 히트율(비용 절감 관측)."""
    return cache_stats.stats()


@app.get("/v1/models")
async def list_models(ctx: TenantContext = Depends(authenticate)) -> dict:
    models = get_router().logical_models
    if ctx.allowed_models:
        models = [m for m in models if m in ctx.allowed_models]
    return {"object": "list", "data": [{"id": m, "object": "model"} for m in models]}


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    ctx: TenantContext = Depends(authenticate),
) -> Response:
    router = get_router()

    # 0) 입력 크기 상한 (자원고갈 방지) — 초과 시 400
    gw = get_settings().gateway
    if len(req.messages) > gw.max_messages:
        raise HTTPException(400, f"메시지 수 초과 (max={gw.max_messages})")
    total_chars = sum(len(m.content or "") for m in req.messages)
    if total_chars > gw.max_input_chars:
        raise HTTPException(400, f"입력 길이 초과 (max={gw.max_input_chars} chars)")

    # 1) 정책 검사 (멀티테넌시 제어 평면)
    try:
        policy.authorize(ctx, req.model)
    except PolicyViolation as exc:
        raise HTTPException(429, str(exc)) from exc

    # 2) 입력 가드레일 (프롬프트 인젝션) — 원본 사용자 입력에 적용
    gin = guardrails.check_input([m.model_dump() for m in req.messages])
    if not gin.allowed:
        raise HTTPException(400, f"guardrail: {gin.reason}")

    # 2.5) 서빙 RAG — 검색·컨텍스트 주입(평가 --rag와 동일 경로). control 필드는 제거.
    rag_on = _rag_enabled(req)
    req.extra.pop("rag", None)
    if rag_on:
        from llmops_core.common.schemas import ChatMessage

        injected, _hits = get_rag().inject_context([m.model_dump() for m in req.messages])
        req.messages = [ChatMessage(**m) for m in injected]

    # 3-STREAM) stream=true면 SSE로 청크 릴레이(버퍼링 금지). 사용량/비용은 종료 시 집계.
    if req.stream:
        return await _stream_completion(router, req, ctx)

    # 3) 라우팅 호출 + 트레이싱 (캐싱은 litellm 네이티브가 Router 내부에서 처리)
    with llm_span(req.model, tenant=ctx, temperature=req.temperature) as span:
        raw, usage, cache_hit = await router.acompletion(req)
        cache_stats.record(cache_hit)
        cost = 0.0 if cache_hit else router.estimate_cost(req.model, usage)
        record_usage(span, usage, response_model=raw.get("model"), cost_usd=cost)

    # 4) 출력 가드레일 (금칙어·PII 마스킹) — 첫 choice 메시지에 적용
    _apply_output_guardrail(raw)

    # 5) 비용 반영 (캐시 적중 시 0 — 예산 통제)
    policy.record_spend(ctx, cost)

    return JSONResponse(raw)


async def _stream_completion(
    router: GatewayRouter, req: ChatCompletionRequest, ctx: TenantContext
) -> StreamingResponse:
    """OpenAI 호환 SSE 스트리밍.

    litellm async 스트림 청크를 `data: {json}\\n\\n` 형식으로 릴레이하고 `data: [DONE]`로
    종료한다. 사용량/비용은 stream_options.include_usage 로 오는 마지막 usage 청크에서
    best-effort 집계해 스팬 기록·예산 반영한다.
    한계: 출력 가드레일(금칙어/PII 마스킹)은 청크 단위로는 적용하지 않는다(스트리밍 특성).
          출력 마스킹이 필수인 테넌트는 비스트리밍 경로를 사용해야 한다.
    """
    stream = await router.astream(req)

    async def event_gen():
        final_usage = Usage()
        model_name = req.model
        with llm_span(req.model, tenant=ctx, temperature=req.temperature) as span:
            try:
                async for chunk in stream:
                    raw = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
                    u = raw.get("usage")
                    if u:
                        final_usage = Usage(
                            prompt_tokens=u.get("prompt_tokens", 0),
                            completion_tokens=u.get("completion_tokens", 0),
                            total_tokens=u.get("total_tokens", 0),
                        )
                    if raw.get("model"):
                        model_name = raw["model"]
                    yield f"data: {json.dumps(raw, default=str)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                cost = router.estimate_cost(req.model, final_usage)
                record_usage(span, final_usage, response_model=model_name, cost_usd=cost)
                policy.record_spend(ctx, cost)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


def _apply_output_guardrail(raw: dict) -> None:
    """응답 choices의 메시지 콘텐츠에 출력 가드레일 적용(차단 시 400, 마스킹 시 치환)."""
    for choice in raw.get("choices", []):
        msg = choice.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        gout = guardrails.check_output(content)
        if not gout.allowed:
            raise HTTPException(400, f"guardrail: {gout.reason}")
        if gout.text is not None and gout.text != content:
            msg["content"] = gout.text


@app.exception_handler(PolicyViolation)
async def _policy_handler(_, exc: PolicyViolation):  # pragma: no cover
    return JSONResponse(status_code=429, content={"error": str(exc)})

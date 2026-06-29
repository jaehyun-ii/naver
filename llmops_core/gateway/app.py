"""게이트웨이 FastAPI — OpenAI 호환 단일 진입점.

흐름: 가상키 인증 → 정책 엔진(화이트리스트·RPM·예산) → litellm.Router → 비용 반영
     → 전 과정 OTel 스팬(자체 LLM 규약)으로 트레이싱.
앱은 논리 모델명만 호출하므로 백엔드(vLLM/CLOVA) 교체에 영향받지 않는다.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from llmops_core.common.config import get_settings
from llmops_core.common.errors import AuthError, PolicyViolation
from llmops_core.common.schemas import ChatCompletionRequest, TenantContext
from llmops_core.common.stores import make_key_store, make_ledger
from llmops_core.gateway.keys import VirtualKeyStore
from llmops_core.gateway.policy import PolicyEngine
from llmops_core.gateway.router import GatewayRouter
from llmops_core.telemetry import init_telemetry, llm_span, record_usage

# ── 의존 컴포넌트 (store.backend=postgres면 Postgres 영속, 게이트웨이↔콘솔 상태 공유) ──
key_store: VirtualKeyStore = make_key_store()
policy = PolicyEngine(ledger=make_ledger())
_router: GatewayRouter | None = None


def get_router() -> GatewayRouter:
    global _router
    if _router is None:
        _router = GatewayRouter()
    return _router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_telemetry()
    yield


app = FastAPI(title="llmops-core gateway", version="0.1.0", lifespan=lifespan)


async def authenticate(authorization: str = Header(...)) -> TenantContext:
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Bearer 토큰 필요")
    raw_key = authorization.split(" ", 1)[1].strip()
    try:
        return key_store.verify(raw_key)
    except AuthError as exc:
        raise HTTPException(401, str(exc)) from exc


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ── 관리자: 가상키 발급 (master_key로 보호) — 멀티테넌시 제어 평면 ──
class IssueKeyRequest(BaseModel):
    tenant_id: str
    allowed_models: list[str] = []
    monthly_budget_usd: float | None = None
    rpm_limit: int | None = None


def _require_master(x_master_key: str = Header(...)) -> None:
    if x_master_key != get_settings().gateway.master_key:
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
) -> JSONResponse:
    router = get_router()

    # 1) 정책 검사 (멀티테넌시 제어 평면)
    try:
        policy.authorize(ctx, req.model)
    except PolicyViolation as exc:
        raise HTTPException(429, str(exc)) from exc

    # 2) 라우팅 호출 + 트레이싱
    with llm_span(req.model, tenant=ctx, temperature=req.temperature) as span:
        raw, usage = await router.acompletion(req)
        cost = router.estimate_cost(req.model, usage)
        record_usage(span, usage, response_model=raw.get("model"), cost_usd=cost)

    # 3) 비용 반영 (예산 통제)
    policy.record_spend(ctx, cost)

    return JSONResponse(raw)


@app.exception_handler(PolicyViolation)
async def _policy_handler(_, exc: PolicyViolation):  # pragma: no cover
    return JSONResponse(status_code=429, content={"error": str(exc)})

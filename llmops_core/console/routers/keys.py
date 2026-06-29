"""가상키 관리 — 발급/조회/폐기 (멀티테넌시 제어평면 UI)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from llmops_core.common.security import Principal
from llmops_core.console.schemas import IssuedKey, IssueKeyBody, KeyView
from llmops_core.console.security import require_perm
from llmops_core.console.services import services

router = APIRouter(prefix="/api/keys", tags=["keys"], dependencies=[Depends(require_perm("read"))])


def _view(ctx) -> KeyView:
    svc = services()
    return KeyView(
        key_id=ctx.key_id,
        tenant_id=ctx.tenant_id,
        allowed_models=ctx.allowed_models,
        monthly_budget_usd=ctx.monthly_budget_usd,
        rpm_limit=ctx.rpm_limit,
        spent_usd=round(svc.policy.ledger.month_to_date(ctx.tenant_id), 6),
    )


@router.get("")
def list_keys() -> list[KeyView]:
    return [_view(c) for c in services().key_store.list_keys()]


@router.post("")
def issue_key(
    body: IssueKeyBody, principal: Principal = Depends(require_perm("keys:write")),
) -> IssuedKey:
    svc = services()
    raw = svc.key_store.issue(
        body.tenant_id,
        allowed_models=body.allowed_models,
        monthly_budget_usd=body.monthly_budget_usd,
        rpm_limit=body.rpm_limit,
    )
    ctx = svc.key_store.verify(raw)
    svc.audit.record(principal.subject, "key.issue", target=ctx.key_id,
                     detail={"tenant": body.tenant_id})
    return IssuedKey(virtual_key=raw, **_view(ctx).model_dump())


@router.delete("/{key_id}")
def revoke_key(
    key_id: str, principal: Principal = Depends(require_perm("keys:write")),
) -> dict:
    svc = services()
    if not svc.key_store.revoke_by_key_id(key_id):
        raise HTTPException(404, f"key_id 없음: {key_id}")
    svc.audit.record(principal.subject, "key.revoke", target=key_id)
    return {"revoked": key_id}

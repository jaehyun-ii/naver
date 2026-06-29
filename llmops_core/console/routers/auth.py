"""RBAC 토큰 관리 — 역할 부여 API 토큰 발급/조회/폐기 (admin 전용).

발급된 토큰은 `Authorization: Bearer <token>`으로 사용하며, 부여된 역할에 따라 권한이 결정된다.
master key는 별도(admin)로 동작하므로, 운영에서는 master key 대신 역할 토큰 사용을 권장한다.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from llmops_core.common.security import ALL_ROLES, Principal
from llmops_core.console.security import require_perm
from llmops_core.console.services import services

router = APIRouter(prefix="/api/auth", tags=["auth"],
                   dependencies=[Depends(require_perm("*"))])


class IssueTokenBody(BaseModel):
    subject: str  # 사용자/서비스 식별자
    roles: list[str] = Field(default_factory=list)  # admin|operator|approver|viewer


@router.get("/roles")
def list_roles() -> dict:
    return {"roles": sorted(ALL_ROLES)}


@router.get("/tokens")
def list_tokens() -> list[dict]:
    return services().tokens.list()


@router.post("/tokens")
def issue_token(
    body: IssueTokenBody, principal: Principal = Depends(require_perm("*")),
) -> dict:
    invalid = [r for r in body.roles if r not in ALL_ROLES]
    if invalid:
        raise HTTPException(422, f"알 수 없는 역할: {invalid} (가능: {sorted(ALL_ROLES)})")
    svc = services()
    raw = svc.tokens.issue(body.subject, body.roles)
    svc.audit.record(principal.subject, "token.issue", target=body.subject,
                     detail={"roles": body.roles})
    return {"token": raw, "subject": body.subject, "roles": body.roles,
            "note": "토큰은 1회만 노출됩니다. Authorization: Bearer <token> 으로 사용."}


@router.delete("/tokens/{token_id}")
def revoke_token(
    token_id: str, principal: Principal = Depends(require_perm("*")),
) -> dict:
    svc = services()
    if not svc.tokens.revoke(token_id):
        raise HTTPException(404, f"token_id 없음: {token_id}")
    svc.audit.record(principal.subject, "token.revoke", target=token_id)
    return {"revoked": token_id}

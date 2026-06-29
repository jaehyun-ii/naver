"""콘솔 인증·인가 — RBAC(역할 기반) + 하위호환 master key.

인증: `X-Master-Key`(=admin) 또는 `Authorization: Bearer <token>`(역할 토큰).
인가: 엔드포인트가 요구하는 permission을 principal의 역할이 보유하는지 검사.
챗 플레이그라운드는 테넌트 가상키로 별도 인증(라우터 내부).
"""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException

from llmops_core.common.config import get_settings
from llmops_core.common.security import Principal


def _resolve_principal(
    x_master_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> Principal:
    """요청 헤더 → Principal. master key=admin, Bearer 토큰=역할. 실패 시 401."""
    if x_master_key and x_master_key == get_settings().gateway.master_key:
        return Principal(subject="master", roles=["admin"])
    if authorization and authorization.lower().startswith("bearer "):
        from llmops_core.console.services import services

        token = authorization.split(" ", 1)[1].strip()
        p = services().tokens.resolve(token)
        if p is not None:
            return p
    raise HTTPException(401, "인증 필요 (X-Master-Key 또는 Bearer 토큰)")


def require_perm(perm: str):
    """permission을 요구하는 의존성 팩토리. principal을 반환(핸들러에서 actor로 사용)."""

    def dep(principal: Principal = Depends(_resolve_principal)) -> Principal:
        if not principal.has_perm(perm):
            raise HTTPException(
                403, f"권한 없음: '{perm}' 필요 (roles={principal.roles})")
        return principal

    return dep


# 하위호환: 기존 master 전용 동작 = admin 권한 요구
def require_master(principal: Principal = Depends(require_perm("*"))) -> Principal:
    return principal

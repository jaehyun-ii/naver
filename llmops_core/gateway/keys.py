"""가상키 발급/검증 — 멀티테넌시 제어 평면의 자체 구현.

LiteLLM Proxy 제품의 키 관리 기능을 쓰지 않고, 우리가 직접 키→테넌트 정책을 보유한다.
저장소는 인터페이스(VirtualKeyStore)로 추상화: 개발은 인메모리, 운영은 Postgres 어댑터로 교체.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from abc import ABC, abstractmethod

from llmops_core.common.errors import AuthError
from llmops_core.common.schemas import TenantContext


def _hash(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


class VirtualKeyStore(ABC):
    """가상키 저장소 계약. 운영 시 Postgres 구현으로 교체."""

    @abstractmethod
    def issue(
        self,
        tenant_id: str,
        *,
        allowed_models: list[str] | None = None,
        monthly_budget_usd: float | None = None,
        rpm_limit: int | None = None,
        expires_at: float | None = None,
    ) -> str:
        """raw 가상키(평문)를 반환. 저장소에는 해시만 보관. expires_at(epoch초) 초과 시 무효."""

    @abstractmethod
    def verify(self, raw_key: str) -> TenantContext:
        """raw 키 → TenantContext. 실패 시 AuthError."""

    @abstractmethod
    def revoke(self, raw_key: str) -> None: ...

    @abstractmethod
    def list_keys(self) -> list[TenantContext]:
        """발급된 키의 테넌트 컨텍스트 목록(평문 키 제외, key_id만). 콘솔 관리용."""

    @abstractmethod
    def revoke_by_key_id(self, key_id: str) -> bool:
        """key_id로 폐기. 성공 시 True."""


class InMemoryKeyStore(VirtualKeyStore):
    """개발/테스트용 인메모리 구현."""

    def __init__(self) -> None:
        self._by_hash: dict[str, TenantContext] = {}
        self._expiry: dict[str, float] = {}  # key_hash → expires_at(epoch초)

    def issue(
        self,
        tenant_id: str,
        *,
        allowed_models: list[str] | None = None,
        monthly_budget_usd: float | None = None,
        rpm_limit: int | None = None,
        expires_at: float | None = None,
    ) -> str:
        raw = "sk-" + secrets.token_urlsafe(24)
        h = _hash(raw)
        self._by_hash[h] = TenantContext(
            tenant_id=tenant_id,
            key_id=h[:12],
            allowed_models=allowed_models or [],
            monthly_budget_usd=monthly_budget_usd,
            rpm_limit=rpm_limit,
        )
        if expires_at is not None:
            self._expiry[h] = expires_at
        return raw

    def verify(self, raw_key: str) -> TenantContext:
        h = _hash(raw_key)
        ctx = self._by_hash.get(h)
        if ctx is None:
            raise AuthError("유효하지 않은 가상키")
        exp = self._expiry.get(h)
        if exp is not None and time.time() >= exp:
            raise AuthError("만료된 가상키")
        return ctx

    def revoke(self, raw_key: str) -> None:
        h = _hash(raw_key)
        self._by_hash.pop(h, None)
        self._expiry.pop(h, None)

    def list_keys(self) -> list[TenantContext]:
        return list(self._by_hash.values())

    def revoke_by_key_id(self, key_id: str) -> bool:
        for h, ctx in list(self._by_hash.items()):
            if ctx.key_id == key_id:
                del self._by_hash[h]
                self._expiry.pop(h, None)
                return True
        return False


class PostgresKeyStore(VirtualKeyStore):
    """Postgres 영속 키 저장소 — 평문 키는 보관하지 않고 sha256 해시만 저장."""

    def __init__(self) -> None:
        from llmops_core.common.db import init_schema

        init_schema()

    def issue(
        self,
        tenant_id: str,
        *,
        allowed_models: list[str] | None = None,
        monthly_budget_usd: float | None = None,
        rpm_limit: int | None = None,
        expires_at: float | None = None,
    ) -> str:
        import json

        from llmops_core.common.db import cursor

        raw = "sk-" + secrets.token_urlsafe(24)
        h = _hash(raw)
        # NOTE(P8): expires_at 영속화는 virtual_keys.expires_at 컬럼 추가(Alembic 마이그레이션)
        # 이후 활성화. DDL(common/db.py)은 본 변경 범위 밖이라 현재는 파라미터만 수용한다.
        with cursor() as cur:
            cur.execute(
                "INSERT INTO virtual_keys "
                "(key_hash, key_id, tenant_id, allowed_models, monthly_budget_usd, rpm_limit) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (h, h[:12], tenant_id, json.dumps(allowed_models or []),
                 monthly_budget_usd, rpm_limit),
            )
        return raw

    def _row_to_ctx(self, row) -> TenantContext:
        key_id, tenant_id, allowed, budget, rpm = row
        return TenantContext(
            tenant_id=tenant_id, key_id=key_id,
            allowed_models=allowed or [], monthly_budget_usd=budget, rpm_limit=rpm,
        )

    def verify(self, raw_key: str) -> TenantContext:
        from llmops_core.common.db import cursor

        with cursor() as cur:
            cur.execute(
                "SELECT key_id, tenant_id, allowed_models, monthly_budget_usd, rpm_limit "
                "FROM virtual_keys WHERE key_hash=%s", (_hash(raw_key),),
            )
            row = cur.fetchone()
        if row is None:
            raise AuthError("유효하지 않은 가상키")
        return self._row_to_ctx(row)

    def revoke(self, raw_key: str) -> None:
        from llmops_core.common.db import cursor

        with cursor() as cur:
            cur.execute("DELETE FROM virtual_keys WHERE key_hash=%s", (_hash(raw_key),))

    def list_keys(self) -> list[TenantContext]:
        from llmops_core.common.db import cursor

        with cursor() as cur:
            cur.execute(
                "SELECT key_id, tenant_id, allowed_models, monthly_budget_usd, rpm_limit "
                "FROM virtual_keys ORDER BY created_at DESC")
            rows = cur.fetchall()
        return [self._row_to_ctx(r) for r in rows]

    def revoke_by_key_id(self, key_id: str) -> bool:
        from llmops_core.common.db import cursor

        with cursor() as cur:
            cur.execute("DELETE FROM virtual_keys WHERE key_id=%s", (key_id,))
            return cur.rowcount > 0

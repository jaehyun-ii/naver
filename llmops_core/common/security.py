"""RBAC·감사 — 역할 기반 접근통제 + 토큰 스토어 + 감사로그 (제어평면 보안 하드닝).

- Principal: 호출 주체(subject + roles). master key는 admin으로 매핑(하위호환).
- 역할/권한: admin·operator·approver·viewer. 엔드포인트는 permission을 요구.
- 토큰: sha256 해시만 보관(평문 미보관). backend=postgres면 영속.
- 감사로그: 권한 동작(키 발급/폐기, 승인/반려, 배포)을 actor·action·target으로 기록.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass, field

from llmops_core.common.config import get_settings

# ── 역할 → 권한 ──
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "admin": {"*"},
    "operator": {"read", "data:write", "pipeline:run", "prompts:write",
                 "tuning:run", "keys:read"},
    "approver": {"read", "releases:read", "releases:approve"},
    "viewer": {"read", "releases:read"},
}
ALL_ROLES = set(ROLE_PERMISSIONS)


@dataclass
class Principal:
    subject: str
    roles: list[str] = field(default_factory=list)

    def has_perm(self, perm: str) -> bool:
        for r in self.roles:
            granted = ROLE_PERMISSIONS.get(r, set())
            if "*" in granted or perm in granted:
                return True
        # 'read'는 모든 인증 주체에 허용(최소 권한 조회)
        return perm == "read" and bool(self.roles)


def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


# ── 토큰 스토어 ──
class TokenStore:
    """API 토큰 → (subject, roles). 평문 미보관."""

    def issue(self, subject: str, roles: list[str]) -> str:
        raise NotImplementedError

    def resolve(self, raw: str) -> Principal | None:
        raise NotImplementedError

    def list(self) -> list[dict]:
        raise NotImplementedError

    def revoke(self, token_id: str) -> bool:
        raise NotImplementedError


class InMemoryTokenStore(TokenStore):
    def __init__(self) -> None:
        self._by_hash: dict[str, dict] = {}

    def issue(self, subject: str, roles: list[str]) -> str:
        raw = "tok-" + secrets.token_urlsafe(24)
        h = _hash(raw)
        self._by_hash[h] = {"token_id": h[:12], "subject": subject, "roles": roles}
        return raw

    def resolve(self, raw: str) -> Principal | None:
        rec = self._by_hash.get(_hash(raw))
        return Principal(rec["subject"], rec["roles"]) if rec else None

    def list(self) -> list[dict]:
        return [{"token_id": r["token_id"], "subject": r["subject"], "roles": r["roles"]}
                for r in self._by_hash.values()]

    def revoke(self, token_id: str) -> bool:
        for h, r in list(self._by_hash.items()):
            if r["token_id"] == token_id:
                del self._by_hash[h]
                return True
        return False


class PostgresTokenStore(TokenStore):
    def __init__(self) -> None:
        from llmops_core.common.db import init_schema

        init_schema()

    def issue(self, subject: str, roles: list[str]) -> str:
        import json

        from llmops_core.common.db import cursor

        raw = "tok-" + secrets.token_urlsafe(24)
        h = _hash(raw)
        with cursor() as cur:
            cur.execute(
                "INSERT INTO auth_tokens (token_hash, token_id, subject, roles) "
                "VALUES (%s,%s,%s,%s)", (h, h[:12], subject, json.dumps(roles)))
        return raw

    def resolve(self, raw: str) -> Principal | None:
        from llmops_core.common.db import cursor

        with cursor() as cur:
            cur.execute("SELECT subject, roles FROM auth_tokens WHERE token_hash=%s", (_hash(raw),))
            row = cur.fetchone()
        return Principal(row[0], row[1] or []) if row else None

    def list(self) -> list[dict]:
        from llmops_core.common.db import cursor

        with cursor() as cur:
            cur.execute("SELECT token_id, subject, roles FROM auth_tokens ORDER BY created_at DESC")
            rows = cur.fetchall()
        return [{"token_id": r[0], "subject": r[1], "roles": r[2] or []} for r in rows]

    def revoke(self, token_id: str) -> bool:
        from llmops_core.common.db import cursor

        with cursor() as cur:
            cur.execute("DELETE FROM auth_tokens WHERE token_id=%s", (token_id,))
            return cur.rowcount > 0


# ── 감사로그 ──
class AuditLog:
    def record(self, actor: str, action: str, *, target: str | None = None,
               result: str = "ok", detail: dict | None = None) -> None:
        raise NotImplementedError

    def list(self, limit: int = 100) -> list[dict]:
        raise NotImplementedError


class InMemoryAuditLog(AuditLog):
    def __init__(self) -> None:
        self._items: list[dict] = []

    def record(self, actor, action, *, target=None, result="ok", detail=None) -> None:
        self._items.insert(0, {"ts": time.time(), "actor": actor, "action": action,
                               "target": target, "result": result, "detail": detail or {}})

    def list(self, limit: int = 100) -> list[dict]:
        return self._items[:limit]


class PostgresAuditLog(AuditLog):
    def __init__(self) -> None:
        from llmops_core.common.db import init_schema

        init_schema()

    def record(self, actor, action, *, target=None, result="ok", detail=None) -> None:
        import json

        from llmops_core.common.db import cursor

        with cursor() as cur:
            cur.execute(
                "INSERT INTO audit_log (actor, action, target, result, detail) "
                "VALUES (%s,%s,%s,%s,%s)",
                (actor, action, target, result, json.dumps(detail or {})))

    def list(self, limit: int = 100) -> list[dict]:
        from llmops_core.common.db import cursor

        with cursor() as cur:
            cur.execute(
                "SELECT ts, actor, action, target, result, detail FROM audit_log "
                "ORDER BY ts DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
        return [{"ts": str(r[0]), "actor": r[1], "action": r[2], "target": r[3],
                 "result": r[4], "detail": r[5]} for r in rows]


# ── 팩토리(backend 선택) ──
def make_token_store() -> TokenStore:
    return (PostgresTokenStore() if get_settings().store.backend == "postgres"
            else InMemoryTokenStore())


def make_audit_log() -> AuditLog:
    return (PostgresAuditLog() if get_settings().store.backend == "postgres"
            else InMemoryAuditLog())

"""Postgres 연결 레이어 — 제어평면 상태 영속화(영속·HA) 공통 기반.

스레드세이프 커넥션 풀(psycopg2.pool) + 컨텍스트 매니저. backend=memory면 사용하지 않는다.
운영에서 다중 콘솔/게이트웨이 인스턴스가 동일 Postgres를 공유 → 단일실패점 제거.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

from llmops_core.common.config import get_settings
from llmops_core.common.errors import OptionalDependencyError

_pool = None
_lock = threading.Lock()


def _get_pool():
    global _pool
    if _pool is None:
        with _lock:
            if _pool is None:
                try:
                    import time

                    from psycopg2 import pool
                except ImportError as exc:  # pragma: no cover
                    raise OptionalDependencyError("psycopg2-binary", "store") from exc
                dsn = get_settings().store.dsn
                last = None
                # 컨테이너 기동 시 Postgres 준비 레이스에 대비한 재시도
                for attempt in range(10):
                    try:
                        _pool = pool.ThreadedConnectionPool(minconn=1, maxconn=10, dsn=dsn)
                        break
                    except Exception as exc:  # noqa: BLE001
                        last = exc
                        time.sleep(min(1.0 * (attempt + 1), 5.0))
                if _pool is None:
                    raise RuntimeError(f"Postgres 연결 실패(DSN={dsn}): {last}")
    return _pool


@contextmanager
def connection() -> Iterator:
    """풀에서 커넥션을 빌려 commit/rollback 후 반납."""
    p = _get_pool()
    conn = p.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)


@contextmanager
def cursor() -> Iterator:
    """connection() + cursor 편의."""
    with connection() as conn:
        cur = conn.cursor()
        try:
            yield cur
        finally:
            cur.close()


# 제어평면 테이블 스키마 (단순 — 정규화보다 가용성·이식성 우선)
_SCHEMA = """
CREATE TABLE IF NOT EXISTS virtual_keys (
    key_hash         TEXT PRIMARY KEY,
    key_id           TEXT UNIQUE NOT NULL,
    tenant_id        TEXT NOT NULL,
    allowed_models   JSONB NOT NULL DEFAULT '[]',
    monthly_budget_usd DOUBLE PRECISION,
    rpm_limit        INTEGER,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS usage_ledger (
    id          BIGSERIAL PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    cost_usd    DOUBLE PRECISION NOT NULL,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_ledger_tenant_ts ON usage_ledger (tenant_id, ts);
CREATE TABLE IF NOT EXISTS release_requests (
    id          TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    payload     JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS kv_store (
    kind        TEXT NOT NULL,
    id          TEXT NOT NULL,
    payload     JSONB NOT NULL,
    created_at  DOUBLE PRECISION,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (kind, id)
);
CREATE TABLE IF NOT EXISTS auth_tokens (
    token_hash  TEXT PRIMARY KEY,
    token_id    TEXT UNIQUE NOT NULL,
    subject     TEXT NOT NULL,
    roles       JSONB NOT NULL DEFAULT '[]',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS audit_log (
    id          BIGSERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
    actor       TEXT NOT NULL,
    action      TEXT NOT NULL,
    target      TEXT,
    result      TEXT,
    detail      JSONB
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log (ts DESC);
"""


def init_schema() -> None:
    """제어평면 테이블 생성(멱등) — 개발 부트스트랩용.

    운영(prod)에서는 이 CREATE TABLE IF NOT EXISTS 대신 Alembic 마이그레이션을 사용한다:
        alembic -c alembic.ini upgrade head
    Alembic 초기 리비전은 아래 _SCHEMA 와 동일한 스키마를 반영하며, 이후 스키마 변경은
    마이그레이션으로 버전 관리된다(expires_at 컬럼 추가 등). init_schema()는 로컬/테스트
    편의를 위해 유지한다.
    """
    with cursor() as cur:
        cur.execute(_SCHEMA)


def healthcheck() -> bool:
    try:
        with cursor() as cur:
            cur.execute("SELECT 1")
            return cur.fetchone()[0] == 1
    except Exception:  # noqa: BLE001
        return False

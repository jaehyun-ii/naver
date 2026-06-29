"""스토어 팩토리 — 설정(store.backend)에 따라 memory/postgres 구현 선택.

게이트웨이·콘솔이 동일 팩토리를 써서 backend=postgres면 상태가 Postgres에 영속(영속·HA),
memory면 인메모리(개발·테스트). 단일 전환 지점.
"""

from __future__ import annotations

from typing import Callable

from llmops_core.common.config import get_settings


def _backend() -> str:
    return get_settings().store.backend


# ── 핵심 거버넌스/인증/과금 스토어 ──
def make_key_store():
    from llmops_core.gateway.keys import InMemoryKeyStore, PostgresKeyStore

    return PostgresKeyStore() if _backend() == "postgres" else InMemoryKeyStore()


def make_ledger():
    from llmops_core.gateway.policy import InMemoryLedger, PostgresLedger

    return PostgresLedger() if _backend() == "postgres" else InMemoryLedger()


def make_release_store():
    from llmops_core.governance.release import InMemoryReleaseStore, PostgresReleaseStore

    return PostgresReleaseStore() if _backend() == "postgres" else InMemoryReleaseStore()


# ── 범용 KV 레지스트리 (파이프라인 런·HPO·데이터셋) ──
class PostgresKVRegistry:
    """kv_store 테이블 기반 영속 레지스트리. id/created_at/직렬화 함수를 주입받는다."""

    def __init__(
        self, kind: str, *, id_of: Callable, to_payload: Callable,
        from_payload: Callable, created_of: Callable | None = None,
    ) -> None:
        from llmops_core.common.db import init_schema

        init_schema()
        self.kind = kind
        self._id_of = id_of
        self._to = to_payload
        self._from = from_payload
        self._created = created_of or (lambda o: None)

    def add(self, obj) -> None:
        self.save(obj)

    def save(self, obj) -> None:
        import json

        from llmops_core.common.db import cursor

        with cursor() as cur:
            cur.execute(
                "INSERT INTO kv_store (kind, id, payload, created_at) VALUES (%s,%s,%s,%s) "
                "ON CONFLICT (kind, id) DO UPDATE SET payload=EXCLUDED.payload, updated_at=now()",
                (self.kind, str(self._id_of(obj)), json.dumps(self._to(obj)), self._created(obj)),
            )

    def get(self, item_id: str):
        from llmops_core.common.db import cursor

        with cursor() as cur:
            cur.execute(
                "SELECT payload FROM kv_store WHERE kind=%s AND id=%s", (self.kind, item_id))
            row = cur.fetchone()
        return self._from(row[0]) if row else None

    def list(self) -> list:
        from llmops_core.common.db import cursor

        with cursor() as cur:
            cur.execute(
                "SELECT payload FROM kv_store WHERE kind=%s "
                "ORDER BY COALESCE(created_at,0) DESC, updated_at DESC", (self.kind,))
            rows = cur.fetchall()
        return [self._from(r[0]) for r in rows]


class WriteThroughRegistry:
    """라이브 인메모리 객체(백그라운드 변경 가시성) + Postgres 영속(재기동 생존).

    실행 중에는 인메모리 객체를 그대로 변경·조회하고, save()/add() 시 Postgres에 스냅샷한다.
    재기동 후 캐시에 없으면 Postgres에서 복원한다. 단일 인스턴스 HA(영속 재기동)에 충분.
    """

    def __init__(self, pg: PostgresKVRegistry, *, id_of: Callable,
                 created_of: Callable | None = None) -> None:
        self._pg = pg
        self._id_of = id_of
        self._created = created_of or (lambda o: 0)
        self._live: dict[str, object] = {}

    def add(self, obj) -> None:
        self._live[str(self._id_of(obj))] = obj
        self._pg.save(obj)

    def save(self, obj) -> None:
        self._live[str(self._id_of(obj))] = obj
        self._pg.save(obj)

    def get(self, item_id: str):
        if item_id in self._live:
            return self._live[item_id]
        obj = self._pg.get(item_id)  # 재기동 후 복원
        if obj is not None:
            self._live[item_id] = obj
        return obj

    def list(self) -> list:
        # 라이브 우선, 없는 것은 Postgres에서 보충
        out = dict(self._live)
        for obj in self._pg.list():
            k = str(self._id_of(obj))
            out.setdefault(k, obj)
        return sorted(out.values(), key=lambda o: self._created(o) or 0, reverse=True)


def make_registry(kind: str, *, id_of, to_payload, from_payload, created_of=None):
    """backend=postgres면 write-through(영속), 아니면 단순 인메모리 dict 레지스트리."""
    if _backend() == "postgres":
        pg = PostgresKVRegistry(kind, id_of=id_of, to_payload=to_payload,
                                from_payload=from_payload, created_of=created_of)
        return WriteThroughRegistry(pg, id_of=id_of, created_of=created_of)
    return _MemoryRegistry(id_of=id_of, created_of=created_of)


class _MemoryRegistry:
    """인메모리 dict 레지스트리(개발/테스트). add/save/get/list."""

    def __init__(self, *, id_of, created_of=None) -> None:
        self._id_of = id_of
        self._created = created_of or (lambda o: 0)
        self._items: dict[str, object] = {}

    def add(self, obj) -> None:
        self._items[str(self._id_of(obj))] = obj

    def save(self, obj) -> None:
        self._items[str(self._id_of(obj))] = obj

    def get(self, item_id: str):
        return self._items.get(item_id)

    def list(self) -> list:
        return sorted(self._items.values(), key=lambda o: self._created(o) or 0, reverse=True)

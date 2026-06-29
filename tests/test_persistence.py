"""영속화·HA 테스트 — Postgres 백엔드 스토어 라운드트립 + 재기동 복원.

Postgres 미가용 환경(CI 등)에서는 자동 스킵된다. 기본(memory) 동작은 다른 테스트가 커버.
"""

from __future__ import annotations

import os

import pytest

DSN = os.environ.get("LLMOPS_TEST_PG_DSN", "postgresql://mlflow:mlflow@localhost:5432/llmops")


@pytest.fixture()
def pg(monkeypatch):
    """store.backend=postgres로 전환하고 DB 가용 시에만 진행(아니면 skip)."""
    monkeypatch.setenv("LLMOPS_STORE__BACKEND", "postgres")
    monkeypatch.setenv("LLMOPS_STORE__DSN", DSN)
    from llmops_core.common import config

    config.get_settings.cache_clear()
    try:
        from llmops_core.common import db

        db._pool = None  # 풀 리셋(테스트 DSN 반영)
        if not db.healthcheck():
            pytest.skip("Postgres 미가용")
        db.init_schema()
        with db.cursor() as cur:
            cur.execute("TRUNCATE virtual_keys, usage_ledger, release_requests, kv_store")
    except Exception:  # noqa: BLE001
        pytest.skip("Postgres 연결 불가")
    yield
    from llmops_core.common import db as db2

    db2._pool = None
    config.get_settings.cache_clear()


def test_key_store_survives_restart(pg):
    from llmops_core.gateway.keys import PostgresKeyStore

    s1 = PostgresKeyStore()
    raw = s1.issue("shipyard-A", allowed_models=["hcx-seed-0_5b"], monthly_budget_usd=50)

    # "재기동": 새 스토어 인스턴스(인메모리 상태 없음)
    s2 = PostgresKeyStore()
    ctx = s2.verify(raw)
    assert ctx.tenant_id == "shipyard-A"
    assert ctx.allowed_models == ["hcx-seed-0_5b"]
    assert ctx.monthly_budget_usd == 50
    assert any(c.tenant_id == "shipyard-A" for c in s2.list_keys())


def test_ledger_month_to_date_persists(pg):
    from llmops_core.gateway.policy import PostgresLedger

    l1 = PostgresLedger()
    l1.add_cost("t1", 1.5)
    l1.add_cost("t1", 2.5)
    # 재기동 후에도 누적 보존
    l2 = PostgresLedger()
    assert l2.month_to_date("t1") == pytest.approx(4.0)


def test_release_store_roundtrip(pg):
    from llmops_core.common.schemas import EvalResult
    from llmops_core.governance.release import PostgresReleaseStore, ReleaseGateway

    gw = ReleaseGateway(store=PostgresReleaseStore())
    req = gw.request(EvalResult(suite="t", model_ref="m", passed=True), requested_by="tester")
    gw.approve(req.id, "approver-1")

    # 재기동: 새 게이트웨이/스토어가 승인 상태를 복원
    gw2 = ReleaseGateway(store=PostgresReleaseStore())
    restored = gw2.store.get(req.id)
    assert restored is not None
    assert restored.status.value == "Approved"
    assert restored.approver == "approver-1"

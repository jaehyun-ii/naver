"""핵심 글루 단위 테스트 — 무거운 의존성 없이 실행 가능(멀티테넌시·프롬프트·게이트)."""

from __future__ import annotations

import pytest

from llmops_core.common.errors import (
    AuthError,
    BudgetExceeded,
    EvalGateFailed,
    ModelNotAllowed,
    RateLimited,
)
from llmops_core.evaluation import GatePolicy
from llmops_core.gateway import InMemoryKeyStore, PolicyEngine
from llmops_core.prompts import GitPromptStore


def test_virtual_key_issue_and_verify():
    ks = InMemoryKeyStore()
    raw = ks.issue("tenantA", allowed_models=["m1"])
    ctx = ks.verify(raw)
    assert ctx.tenant_id == "tenantA"
    with pytest.raises(AuthError):
        ks.verify("sk-bad")


def test_policy_whitelist_rpm_budget():
    ks, pe = InMemoryKeyStore(), PolicyEngine()
    ctx = ks.verify(ks.issue("t", allowed_models=["m1"], monthly_budget_usd=1.0, rpm_limit=2))
    with pytest.raises(ModelNotAllowed):
        pe.authorize(ctx, "other")
    pe.authorize(ctx, "m1")
    pe.authorize(ctx, "m1")
    with pytest.raises(RateLimited):
        pe.authorize(ctx, "m1")

    ks2, pe2 = InMemoryKeyStore(), PolicyEngine()
    ctx2 = ks2.verify(ks2.issue("t2", monthly_budget_usd=1.0, rpm_limit=1000))
    pe2.record_spend(ctx2, 1.5)
    with pytest.raises(BudgetExceeded):
        pe2.authorize(ctx2, "m1")


def test_prompt_store_versions_labels_rollback(tmp_path):
    ps = GitPromptStore(repo_path=str(tmp_path))
    ps.create_version("p", "v1 {x}")
    ps.create_version("p", "v2 {x}")
    assert ps.versions("p") == [1, 2]
    ps.promote("p", 2, "prod")
    assert ps.by_label("p", "prod").version == 2
    assert ps.by_label("p", "prod").render(x="A") == "v2 A"
    ps.promote("p", 1, "prod")  # 롤백
    assert ps.by_label("p", "prod").version == 1


def test_eval_gate():
    gate = GatePolicy(thresholds={"faithfulness": 0.8})
    assert gate.passes({"faithfulness": 0.9})
    assert not gate.passes({"faithfulness": 0.5})
    with pytest.raises(EvalGateFailed):
        gate.evaluate({"faithfulness": 0.5})

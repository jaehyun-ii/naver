"""드리프트→재학습→배포 폐루프 테스트 (FakeExecutor 주입, GPU 불필요)."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from llmops_core.console.app import app

MASTER = {"X-Master-Key": "sk-master-changeme"}
_REF = ["선급증서 유효기간이 지나면 재검사를 신청한다.",
        "공장인수시험 불합격 시 시정조치 요구서를 발행한다.",
        "재료증명서가 없으면 입고를 보류한다."] * 10
_RETRAIN = [{"text": "선급증서 만료 시?", "response": "즉시 재검사를 신청해 갱신합니다."}]


@pytest.fixture()
def client():
    from llmops_core.console import services as svc_mod
    svc_mod._services = None
    return TestClient(app)


def _set_baseline(client):
    assert client.post("/api/drift/baseline", headers=MASTER,
                       json={"name": "prod", "texts": _REF}).status_code == 200


def test_no_drift_no_retrain(client):
    _set_baseline(client)
    r = client.post("/api/drift/auto-retrain", headers=MASTER, json={
        "baseline_name": "prod", "current_texts": _REF,  # 동일 분포
        "method": "sft", "labeled": _RETRAIN}).json()
    assert r["drift"] is False
    assert r["triggered"] is False
    assert "생략" in r["reason"]


def test_drift_triggers_retrain_and_deploy(client):
    _set_baseline(client)
    drifted = ["A completely different English sentence about cloud computing " * 3 for _ in range(30)]
    r = client.post("/api/drift/auto-retrain", headers=MASTER, json={
        "baseline_name": "prod", "current_texts": drifted,  # 분포 급변
        "method": "sft", "labeled": _RETRAIN, "auto_approve": True}).json()
    assert r["drift"] is True
    assert r["triggered"] is True
    assert r["run_id"]
    # 백그라운드 워처가 waiting→승인→배포 → 완료까지 폴링
    for _ in range(40):
        run = client.get(f"/api/pipeline/runs/{r['run_id']}", headers=MASTER).json()
        if run["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.5)
    assert run["status"] == "succeeded"


def test_auto_retrain_unknown_baseline_404(client):
    assert client.post("/api/drift/auto-retrain", headers=MASTER, json={
        "baseline_name": "none", "current_texts": ["x"], "labeled": _RETRAIN}).status_code == 404


def test_auto_retrain_records_audit(client):
    _set_baseline(client)
    client.post("/api/drift/auto-retrain", headers=MASTER, json={
        "baseline_name": "prod", "current_texts": _REF, "labeled": _RETRAIN})
    actions = [e["action"] for e in client.get("/api/audit", headers=MASTER).json()]
    assert "drift.auto_retrain" in actions

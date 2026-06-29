"""드리프트 콘솔 API 테스트 — 기준선 등록 + 검사(무드리프트/드리프트)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llmops_core.console.app import app

MASTER = {"X-Master-Key": "sk-master-changeme"}

_REF = [
    "선급증서 유효기간이 지나면 재검사를 신청한다.",
    "공장인수시험 불합격 시 시정조치 요구서를 발행한다.",
    "재료증명서가 없으면 입고를 보류한다.",
] * 10


@pytest.fixture()
def client():
    from llmops_core.console import services as svc_mod

    svc_mod._services = None
    return TestClient(app)


def test_baseline_then_check_no_drift(client):
    r = client.post("/api/drift/baseline", headers=MASTER,
                    json={"name": "ship-qa", "texts": _REF})
    assert r.status_code == 200 and r.json()["n"] == 30
    chk = client.post("/api/drift/check", headers=MASTER,
                      json={"name": "ship-qa", "texts": _REF}).json()
    assert chk["drift"] is False


def test_check_detects_drift(client):
    client.post("/api/drift/baseline", headers=MASTER, json={"name": "ship-qa", "texts": _REF})
    drifted = ["A totally different English long sentence about cloud infra " * 3 for _ in range(30)]
    chk = client.post("/api/drift/check", headers=MASTER,
                      json={"name": "ship-qa", "texts": drifted}).json()
    assert chk["drift"] is True


def test_check_unknown_baseline_404(client):
    assert client.post("/api/drift/check", headers=MASTER,
                       json={"name": "nope", "texts": ["x"]}).status_code == 404


def test_baseline_requires_write_perm(client):
    tok = client.post("/api/auth/tokens", headers=MASTER,
                      json={"subject": "v", "roles": ["viewer"]}).json()["token"]
    r = client.post("/api/drift/baseline", headers={"Authorization": f"Bearer {tok}"},
                    json={"name": "x", "texts": ["a"]})
    assert r.status_code == 403  # viewer는 기준선 등록 불가

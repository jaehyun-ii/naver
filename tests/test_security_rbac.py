"""보안 하드닝 테스트 — RBAC(역할 토큰)·감사로그·직무분리(SoD). 인메모리 백엔드."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llmops_core.console.app import app

MASTER = {"X-Master-Key": "sk-master-changeme"}


@pytest.fixture()
def client():
    from llmops_core.console import services as svc_mod

    svc_mod._services = None
    return TestClient(app)


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _issue_token(client, subject, roles) -> str:
    r = client.post("/api/auth/tokens", headers=MASTER,
                    json={"subject": subject, "roles": roles})
    assert r.status_code == 200, r.text
    return r.json()["token"]


def test_unauthenticated_is_401(client):
    assert client.get("/api/keys").status_code == 401
    assert client.get("/api/keys", headers={"X-Master-Key": "wrong"}).status_code == 401


def test_viewer_can_read_not_write(client):
    tok = _issue_token(client, "viewer-1", ["viewer"])
    # 읽기 허용
    assert client.get("/api/infra", headers=_bearer(tok)).status_code == 200
    # 키 발급(쓰기)은 금지 → 403
    r = client.post("/api/keys", headers=_bearer(tok), json={"tenant_id": "t1"})
    assert r.status_code == 403


def test_operator_can_run_pipeline_not_issue_keys(client):
    tok = _issue_token(client, "op-1", ["operator"])
    # 파이프라인 실행 허용
    r = client.post("/api/pipeline/run", headers=_bearer(tok),
                    json={"method": "sft"})
    assert r.status_code == 200
    # 키 발급은 admin 전용 → 403
    assert client.post("/api/keys", headers=_bearer(tok),
                       json={"tenant_id": "t1"}).status_code == 403


def _wait_release_id(client, run_id, headers):
    """백그라운드 파이프라인이 승인 대기(release_id 생성)에 도달할 때까지 폴링."""
    import time
    for _ in range(40):
        run = client.get(f"/api/pipeline/runs/{run_id}", headers=headers).json()
        if run.get("release_id"):
            return run["release_id"]
        time.sleep(0.5)
    raise AssertionError("release_id 미생성")


def test_approver_can_approve_operator_cannot(client):
    op = _issue_token(client, "op-1", ["operator"])
    appr = _issue_token(client, "approver-1", ["approver"])
    # operator가 파이프라인 실행 → 승인 요청 생성(대기)
    run = client.post("/api/pipeline/run", headers=_bearer(op),
                      json={"method": "sft"}).json()
    rel = _wait_release_id(client, run["id"], _bearer(op))
    # operator는 승인 권한 없음 → 403
    assert client.post(f"/api/releases/{rel}/approve", headers=_bearer(op),
                       json={"approver": "op-1"}).status_code == 403
    # approver는 승인 가능
    r = client.post(f"/api/releases/{rel}/approve", headers=_bearer(appr),
                    json={"approver": "ignored"})
    assert r.status_code == 200
    assert r.json()["approver"] == "approver-1"  # 승인자는 principal로 강제


def test_separation_of_duties_blocks_self_approval(client):
    # requested_by와 동일 주체가 승인하면 차단
    rel = client.post("/api/releases", headers=MASTER,
                      json={"model_ref": "m", "passed": True,
                            "requested_by": "alice"}).json()
    tok = _issue_token(client, "alice", ["approver"])
    r = client.post(f"/api/releases/{rel['id']}/approve", headers=_bearer(tok),
                    json={"approver": "alice"})
    assert r.status_code == 403
    assert "직무분리" in r.json()["detail"]


def test_audit_log_records_privileged_actions(client):
    client.post("/api/keys", headers=MASTER, json={"tenant_id": "audit-t"})
    log = client.get("/api/audit", headers=MASTER).json()
    actions = [e["action"] for e in log]
    assert "key.issue" in actions


def test_audit_requires_admin(client):
    tok = _issue_token(client, "viewer-1", ["viewer"])
    assert client.get("/api/audit", headers=_bearer(tok)).status_code == 403

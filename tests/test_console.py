"""콘솔 API 단위 테스트 — TestClient (인프라/모델 없이 동작, 에코 백엔드)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llmops_core.console.app import app

MASTER = {"X-Master-Key": "sk-master-changeme"}  # config 기본값


@pytest.fixture()
def client():
    # 모듈 싱글톤 상태를 테스트 간 격리
    from llmops_core.console import services as svc_mod

    svc_mod._services = None
    return TestClient(app)


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_master_key_required(client):
    # 인증 실패 → 401 (RBAC: 인증 누락/오류)
    assert client.get("/api/keys").status_code == 401  # 자격증명 누락
    assert client.get("/api/keys", headers={"X-Master-Key": "wrong"}).status_code == 401


def test_key_lifecycle_and_chat(client):
    # 발급
    r = client.post(
        "/api/keys",
        headers=MASTER,
        json={"tenant_id": "acme", "allowed_models": ["hcx-seed-3b"]},
    )
    assert r.status_code == 200
    issued = r.json()
    assert issued["virtual_key"].startswith("sk-")
    key_id = issued["key_id"]

    # 목록
    keys = client.get("/api/keys", headers=MASTER).json()
    assert any(k["key_id"] == key_id for k in keys)

    # 챗(에코 백엔드) — 허용 모델
    chat = client.post(
        "/api/chat",
        json={
            "virtual_key": issued["virtual_key"],
            "model": "hcx-seed-3b",
            "messages": [{"role": "user", "content": "안녕"}],
        },
    )
    assert chat.status_code == 200
    body = chat.json()
    assert body["backend"] == "echo" and "안녕" in body["content"]

    # 미허용 모델 → 429
    blocked = client.post(
        "/api/chat",
        json={
            "virtual_key": issued["virtual_key"],
            "model": "not-allowed",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert blocked.status_code == 429

    # 폐기
    assert client.delete(f"/api/keys/{key_id}", headers=MASTER).status_code == 200
    assert not client.get("/api/keys", headers=MASTER).json()


def test_releases_flow(client):
    created = client.post(
        "/api/releases",
        headers=MASTER,
        json={"model_ref": "hcx-seed-3b", "passed": True, "metrics": {"faithfulness": 0.9}},
    ).json()
    rid = created["id"]
    assert created["status"] == "Pending"

    approved = client.post(
        f"/api/releases/{rid}/approve", headers=MASTER, json={"approver": "alice"}
    ).json()
    assert approved["status"] == "Approved"

    # 자동 게이트 미통과 → 요청 차단(422)
    blocked = client.post(
        "/api/releases", headers=MASTER, json={"model_ref": "m", "passed": False}
    )
    assert blocked.status_code == 422


def test_data_validate_and_build(client):
    # 검증 통과
    recs = [{"id": str(i), "text": f"유효한 문장 {i} 입니다"} for i in range(10)]
    report = client.post("/api/data/validate", headers=MASTER, json={"records": recs}).json()
    assert report["passed"]

    # 데이터셋 빌드 → 매니페스트 등록
    labeled = [{"text": f"질문{i}", "response": f"답변{i}"} for i in range(50)]
    manifest = client.post(
        "/api/data/build", headers=MASTER, json={"name": "demo", "labeled": labeled}
    ).json()
    assert manifest["num_train"] + manifest["num_val"] + manifest["num_test"] == 50
    assert len(manifest["fingerprint"]) == 64

    datasets = client.get("/api/data/datasets", headers=MASTER).json()
    assert datasets[0]["name"] == "demo"


def test_models_list(client):
    r = client.get("/api/models", headers=MASTER)
    assert r.status_code == 200
    assert "models" in r.json()


def test_pipeline_run_approve_resume(client):
    # 내장 샘플로 1-클릭 실행 → release-gate까지 진행 후 승인 대기
    run = client.post("/api/pipeline/run", headers=MASTER, json={}).json()
    assert run["status"] == "waiting"
    rid, relid = run["id"], run["release_id"]
    assert relid
    # 데이터/평가 단계는 실제 실행되어 성공
    by = {s["name"]: s for s in run["stages"]}
    assert by["data-quality"]["status"] == "succeeded"
    assert by["evaluate"]["status"] == "succeeded"
    assert by["release-approval"]["status"] == "waiting"

    # 승인 전 재개 → 여전히 대기
    assert client.post(f"/api/pipeline/runs/{rid}/resume", headers=MASTER).json()["status"] == "waiting"

    # 승인 후 재개 → 완료
    client.post(f"/api/releases/{relid}/approve", headers=MASTER, json={"approver": "alice"})
    done = client.post(f"/api/pipeline/runs/{rid}/resume", headers=MASTER).json()
    assert done["status"] == "succeeded"
    assert all(s["status"] == "succeeded" for s in done["stages"])

    # 목록 조회
    runs = client.get("/api/pipeline/runs", headers=MASTER).json()
    assert any(r["id"] == rid for r in runs)


def test_pipeline_reject_skips_deploy(client):
    run = client.post("/api/pipeline/run", headers=MASTER, json={}).json()
    rid, relid = run["id"], run["release_id"]
    client.post(f"/api/releases/{relid}/reject", headers=MASTER, json={"approver": "bob", "reason": "회귀"})
    done = client.post(f"/api/pipeline/runs/{rid}/resume", headers=MASTER).json()
    assert done["status"] == "failed"
    by = {s["name"]: s for s in done["stages"]}
    assert by["deploy"]["status"] == "skipped"

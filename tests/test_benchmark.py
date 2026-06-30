"""벤치마크 평가 — 순수 러너 + 콘솔 라우터(등록·실행·리더보드, 모델 모킹)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llmops_core.console.app import app
from llmops_core.evaluation.benchmark import run_benchmark

MASTER = {"X-Master-Key": "sk-master-changeme"}
_CASES = [
    {"question": "환불 정책?", "expected": "14일 이내 전액 환불"},
    {"question": "배송 기간?", "expected": "평일 2~3일 소요"},
]


def test_run_benchmark_scores_generation():
    # 정답을 그대로 생성 → answer_match=1.0
    out = run_benchmark(lambda q: {"환불 정책?": "14일 이내 전액 환불",
                                   "배송 기간?": "평일 2~3일 소요"}[q], _CASES)
    assert out["num_cases"] == 2
    assert out["errors"] == 0
    assert out["metrics"]["answer_match"] == 1.0


def test_run_benchmark_graceful_on_generate_error():
    def boom(q):
        raise RuntimeError("model down")
    out = run_benchmark(boom, _CASES)
    assert out["errors"] == 2
    assert out["metrics"]["answer_match"] == 0.0


@pytest.fixture()
def client():
    from llmops_core.console import services as svc_mod
    svc_mod._services = None
    return TestClient(app)


def test_register_list_run_leaderboard(client, monkeypatch):
    # 모델 클라이언트 모킹 — 정답을 그대로 생성하는 fake
    class FakeClient:
        def complete(self, model, messages, **kw):
            q = messages[-1]["content"]
            return {"환불 정책?": "14일 이내 전액 환불",
                    "배송 기간?": "평일 2~3일 소요"}.get(q, "모름")
    monkeypatch.setattr("llmops_core.common.model_client.get_model_client",
                        lambda: FakeClient())

    # 등록
    r = client.post("/api/benchmark", headers=MASTER,
                    json={"name": "cs-qa", "cases": _CASES})
    assert r.status_code == 200 and r.json()["num_cases"] == 2
    # 목록
    assert any(b["name"] == "cs-qa" for b in client.get("/api/benchmark", headers=MASTER).json())
    # 실행
    run = client.post("/api/benchmark/run", headers=MASTER,
                      json={"name": "cs-qa", "model": "hcx-seed-0_5b"}).json()
    assert run["metrics"]["answer_match"] == 1.0
    # 리더보드
    res = client.get("/api/benchmark/results?name=cs-qa", headers=MASTER).json()
    assert "cs-qa" in res["leaderboard"]
    assert res["leaderboard"]["cs-qa"][0]["model"] == "hcx-seed-0_5b"


def test_run_unknown_benchmark_404(client):
    assert client.post("/api/benchmark/run", headers=MASTER,
                       json={"name": "none", "model": "m"}).status_code == 404

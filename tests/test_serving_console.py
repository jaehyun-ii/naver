"""서빙 카나리 콘솔 API 테스트 — 롤아웃 상태 조회(가중 라우팅) + 백엔드 선택."""

from __future__ import annotations

import pytest
import yaml
from fastapi.testclient import TestClient

from llmops_core.serving import canary

MASTER = {"X-Master-Key": "sk-master-changeme"}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # 카나리 엔트리가 든 임시 model_list로 config_path 교체
    ml = canary.add_canary(
        canary.set_stable([], "hcx-seed-tuned", "http://host.docker.internal:8020/v1"),
        "hcx-seed-tuned", stable_base="http://host.docker.internal:8020/v1",
        canary_base="http://host.docker.internal:8021/v1", weight=10)
    cfg = tmp_path / "model_list.yaml"
    cfg.write_text(yaml.safe_dump({"model_list": ml}, allow_unicode=True))
    monkeypatch.setenv("LLMOPS_GATEWAY__CONFIG_PATH", str(cfg))
    from llmops_core.common import config

    config.get_settings.cache_clear()
    from llmops_core.console import services as svc_mod

    svc_mod._services = None
    yield TestClient(__import__("llmops_core.console.app", fromlist=["app"]).app)
    config.get_settings.cache_clear()


def test_rollout_status_shows_canary_split(client):
    r = client.get("/api/serving/rollout/hcx-seed-tuned", headers=MASTER)
    assert r.status_code == 200
    d = r.json()
    assert d["stable"]["weight"] == 90
    assert d["canary"]["weight"] == 10
    assert d["canary"]["api_base"].endswith(":8021/v1")


def test_rollout_requires_auth(client):
    assert client.get("/api/serving/rollout/hcx-seed-tuned").status_code == 401


def test_serving_backend_spec_selects_vllm(monkeypatch):
    from llmops_core.console.executor import ExecutorConfig, RealExecutor

    monkeypatch.setenv("LLMOPS_EXEC__SERVE_BACKEND", "vllm")
    ex = RealExecutor(ExecutorConfig())
    image, args = ex._serving_spec("m")
    assert "vllm" in image
    assert "--served-model-name" in args  # vLLM 서버 인자

    monkeypatch.setenv("LLMOPS_EXEC__SERVE_BACKEND", "transformers")
    ex2 = RealExecutor(ExecutorConfig())
    image2, args2 = ex2._serving_spec("m")
    assert "--served-name" in args2  # transformers(hf_server) 인자

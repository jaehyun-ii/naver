"""테스트 공용 — 파이프라인 실행기를 docker 없는 Fake로 대체.

운영 파이프라인은 docker로 학습/병합/배포를 수행하지만, 단위 테스트는 배선 로직만 보면 되므로
`pipeline_engine._executor`만 FakeExecutor를 반환하도록 monkeypatch한다(autouse). 실행기 클래스
자체(RealExecutor)는 건드리지 않으므로 serving 등 실행기 직접 테스트에는 영향이 없다.
"""

from __future__ import annotations

import pytest


class FakeExecutor:
    """docker 없이 학습/평가/병합/배포 결과를 즉시 모사. 게이트를 통과하는 메트릭 반환."""

    def __init__(self) -> None:
        self.last_loss = [{"step": 1, "loss": 0.5}, {"step": 2, "loss": 0.3}]

    def finetune(self, run_id, train_rows, **kw):
        self.last_loss = [{"step": 1, "loss": 0.5}, {"step": 2, "loss": 0.3}]
        return f"/fake/runs/{run_id}/adapter"

    def evaluate(self, run_id, cases, *, task="reference", **kw):
        if task == "preference":
            metrics = {"preference_accuracy": 1.0, "preference_margin": 0.5}
        else:
            metrics = {"answer_match": 1.0, "reference_f1": 0.9}
        return {"metrics": metrics, "num_cases": len(cases), "prompt": None, "rag": None}

    def benchmark_preference(self, model, cases, **kw):
        return {"metrics": {"preference_accuracy": 1.0, "preference_margin": 0.5},
                "num_cases": len(cases)}

    def merge(self, run_id, **kw):
        return f"/fake/runs/{run_id}/merged"

    def deploy(self, run_id, served, **kw):
        return f"http://fake/{served}/v1"


@pytest.fixture(autouse=True)
def _fake_executor(monkeypatch):
    # 파이프라인을 동기 실행(테스트가 즉시 결과를 보도록) + docker 없는 Fake 실행기
    monkeypatch.setenv("LLMOPS_PIPELINE_BACKGROUND", "0")
    monkeypatch.setattr(
        "llmops_core.console.pipeline_engine._executor", lambda svc=None: FakeExecutor())

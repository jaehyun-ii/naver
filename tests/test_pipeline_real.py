"""파이프라인 배선 단위 테스트 — docker 없이 순수 로직만 검증(FakeExecutor 주입).

실제 학습/병합/배포(docker)는 통합 검증(README 절차)에서 다루고, 여기서는
단계 구성·데이터 변환·실행기 설정 등을 동기 실행(background=False)으로 본다.
"""

from __future__ import annotations

from llmops_core.console.executor import ExecutorConfig, docker_available
from llmops_core.console.pipeline_engine import _message_rows, _new_run


def test_message_rows_from_labeled():
    rows = _message_rows([{"text": "환불?", "response": "14일 이내"}])
    assert rows == [
        {"messages": [
            {"role": "user", "content": "환불?"},
            {"role": "assistant", "content": "14일 이내"},
        ]}
    ]


def test_message_rows_skips_incomplete():
    # text/response 한쪽만 있는 행은 변환에서 제외
    assert _message_rows([{"text": "x"}, {"response": "y"}]) == []


def test_new_run_builds_all_stages():
    run = _new_run("t")
    names = {s.name for s in run.stages}
    assert {"data-ingest", "finetune", "evaluate", "deploy"} <= names
    assert all(s.status == "pending" for s in run.stages)


def test_executor_config_env_override(monkeypatch):
    monkeypatch.setenv("LLMOPS_EXEC__SERVE_PORT", "9999")
    monkeypatch.setenv("LLMOPS_EXEC__TRAIN_IMAGE", "custom/train:dev")
    cfg = ExecutorConfig()
    assert cfg.serve_port == 9999
    assert cfg.train_image == "custom/train:dev"


def test_docker_available_returns_bool():
    assert isinstance(docker_available(), bool)


def test_dpo_pipeline_builds_preference_and_completes():
    # docker 없이(FakeExecutor) DPO 분기(데이터 유형=Preference)가 끝까지 도는지 — 동기 실행
    from llmops_core.console.pipeline_engine import resume_run, start_run
    from llmops_core.console.schemas import RunPipelineBody
    from llmops_core.console.services import ConsoleServices

    svc = ConsoleServices()
    body = RunPipelineBody(
        method="dpo", pet="dora",
        preference=[{"prompt": "q", "chosen": "좋은 답", "rejected": "나쁜 답"}],
    )
    run = start_run(svc, body, background=False)
    assert run.status == "waiting"
    by = {s.name: s for s in run.stages}
    assert by["data-build"].detail.startswith("preference")
    assert "DPO/DoRA" in by["finetune"].detail
    svc.releases.approve(run.release_id, "tester")
    done = resume_run(svc, run.id, background=False)
    assert done.status == "succeeded"


def test_reference_metrics_judge_free():
    from llmops_core.common.schemas import EvalCase
    from llmops_core.evaluation.harness import run_reference_metrics

    cases = [
        EvalCase(id="1", question="환불?", answer="14일 이내 전액 환불됩니다",
                 expected="14일 이내 전액 환불됩니다"),  # 완전일치 → f1=1
        EvalCase(id="2", question="배송?", answer="전혀 다른 답",
                 expected="평일 2~3일 소요"),  # 불일치 → 낮음
    ]
    m = run_reference_metrics(cases)
    assert 0.0 <= m["reference_f1"] <= 1.0
    assert m["answer_match"] == 0.5  # 1건만 핵심부 포함

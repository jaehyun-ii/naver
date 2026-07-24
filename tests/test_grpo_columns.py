"""GRPO 추가 컬럼 pass-through — 정규화·로더가 gold 메타데이터를 보존하는지."""
import json

from llmops_core.training.run import _load_prompt_rows
from llmops_core.training.sft import grpo_rows


def test_prompt_only_rows_backward_compat():
    rows = grpo_rows(["질문 1", {"prompt": "질문 2"}, "", {"no_prompt": 1}])
    assert rows == [{"prompt": "질문 1"}, {"prompt": "질문 2"}]


def test_heterogeneous_rows_unified_schema():
    """일부 행에만 있는 메타 컬럼도 전 행 스키마에 유지(None 채움) —
    datasets.from_list의 선두 행 스키마 탈락으로 컬럼이 사라지던 버그 회귀."""
    rows = grpo_rows([{"prompt": "a"},
                      {"prompt": "b", "condition_results": [{"field": "x"}],
                       "program_label": "APPLICABLE"}])
    assert set(rows[0]) == set(rows[1])
    assert rows[0]["condition_results"] is None
    assert rows[1]["condition_results"] == [{"field": "x"}]


def test_extra_columns_preserved():
    src = [{"prompt": "판정하십시오", "gold_label": "uncertain",
            "required_conditions": ["ship_type"], "rule_id": "KR-1"}]
    rows = grpo_rows(src)
    assert rows[0]["gold_label"] == "uncertain"
    assert rows[0]["required_conditions"] == ["ship_type"]
    assert rows[0]["rule_id"] == "KR-1"
    rows[0]["prompt"] = "변경"           # 원본 dict 비파괴(복사) 확인
    assert src[0]["prompt"] == "판정하십시오"


def test_metadata_alignment_order_preserved():
    src = [{"prompt": f"q{i}", "gold_label": f"L{i}"} for i in range(5)]
    rows = grpo_rows(src)
    assert [r["prompt"] for r in rows] == [f"q{i}" for i in range(5)]
    assert [r["gold_label"] for r in rows] == [f"L{i}" for i in range(5)]
    # TRL은 행 i의 추가 컬럼을 num_generations회 반복해 completions와 정렬해 전달한다.
    # 우리 쪽 계약: 행 순서·컬럼 짝이 어긋나지 않으면 정렬이 유지된다(위 두 검증).


def test_loader_mixed_formats_and_metadata(tmp_path):
    p = tmp_path / "grpo.jsonl"
    p.write_text("\n".join([
        json.dumps({"prompt": "규정 판정", "gold_label": "compliant",
                    "evidence_quote": "…하여야 한다", "junk_meta": [1, 2]}),
        json.dumps({"text": "text 형식 프롬프트"}),
        json.dumps({"messages": [{"role": "system", "content": "s"},
                                 {"role": "user", "content": "메시지 프롬프트"}],
                    "rule_id": "KR-2"}),
        json.dumps({"no_prompt_field": True}),
    ]), encoding="utf-8")
    rows = _load_prompt_rows(str(p))
    assert len(rows) == 3
    assert rows[0] == {"prompt": "규정 판정", "gold_label": "compliant",
                       "evidence_quote": "…하여야 한다", "junk_meta": [1, 2]}
    assert rows[1] == {"prompt": "text 형식 프롬프트"}
    assert rows[2] == {"prompt": "메시지 프롬프트", "rule_id": "KR-2"}  # messages 원본은 제거


def test_reward_func_receives_columns_contract():
    """TRL kwargs 전달 계약 시뮬레이션 — 행 컬럼이 reward에 정렬되어 도달."""
    rows = grpo_rows([{"prompt": "q1", "gold_label": "A"},
                      {"prompt": "q2", "gold_label": "B"}])
    num_generations = 2
    # TRL GRPOTrainer 동작 재현: 각 행을 num_generations회 반복해 배치 구성
    prompts = [r["prompt"] for r in rows for _ in range(num_generations)]
    gold_label = [r["gold_label"] for r in rows for _ in range(num_generations)]
    completions = [f"답 {i}" for i in range(len(prompts))]

    def label_accuracy_reward(prompts, completions, gold_label=None, **kwargs):
        assert gold_label is not None and len(gold_label) == len(completions)
        return [1.0 if g == "A" else 0.0 for g in gold_label]

    out = label_accuracy_reward(prompts, completions, gold_label=gold_label)
    assert out == [1.0, 1.0, 0.0, 0.0]

"""데이터 형태 변환기 단위 테스트 — RAG/Reasoning/판정/도구호출 (GPU 불필요)."""

from __future__ import annotations

from llmops_core.dataset import (
    examples_to_rows,
    to_dialogue_examples,
    to_rag_examples,
    to_reasoning_examples,
    to_summary_examples,
    to_tool_call_examples,
    to_translation_examples,
    to_verdict_examples,
)


def test_rag_injects_context_as_system():
    ex = to_rag_examples([
        {"question": "환불 기한?", "contexts": ["환불은 14일 이내"], "answer": "14일 이내"}
    ])
    roles = [m.role for m in ex[0].messages]
    assert roles == ["system", "user", "assistant"]
    assert "14일 이내" in ex[0].messages[0].content  # 컨텍스트 주입


def test_rag_skips_without_answer():
    assert to_rag_examples([{"question": "q"}]) == []


def test_reasoning_wraps_think_tag():
    ex = to_reasoning_examples([
        {"question": "왜?", "reasoning": "근거 단계", "answer": "결론"}
    ])
    body = ex[0].messages[-1].content
    assert "<think>근거 단계</think>" in body and "결론" in body


def test_verdict_uses_verdict_field():
    ex = to_verdict_examples([
        {"question": "규정 적합?", "verdict": "부적합", "reasoning": "조항 위반"}
    ])
    assert "부적합" in ex[0].messages[-1].content


def test_tool_call_builds_assistant_tool_turns():
    ex = to_tool_call_examples([
        {"question": "선급증서 조회", "tool_name": "lookup_cert",
         "arguments": {"imo": "1234567"}, "result": {"status": "valid"},
         "answer": "유효한 증서입니다"}
    ])
    msgs = ex[0].messages
    assert msgs[1].role == "assistant" and msgs[1].tool_calls
    assert msgs[1].tool_calls[0]["function"]["name"] == "lookup_cert"
    assert msgs[2].role == "tool" and msgs[2].tool_call_id == msgs[1].tool_calls[0]["id"]
    assert msgs[-1].content == "유효한 증서입니다"


def test_translation_sets_direction_instruction():
    ex = to_translation_examples([
        {"source": "Hull inspection report", "target": "선체 검사 보고서",
         "src_lang": "영어", "tgt_lang": "한국어"}
    ])
    assert ex[0].messages[0].role == "system"
    assert "영어" in ex[0].messages[0].content and "한국어" in ex[0].messages[0].content
    assert ex[0].messages[-1].content == "선체 검사 보고서"


def test_summary_builds_doc_to_summary():
    ex = to_summary_examples([{"document": "긴 사양서 본문…", "summary": "핵심 요약"}])
    assert ex[0].messages[1].content.startswith("긴 사양서")
    assert ex[0].messages[-1].content == "핵심 요약"


def test_dialogue_keeps_multiturn():
    ex = to_dialogue_examples([{"turns": [
        {"role": "user", "content": "선급 문의"},
        {"role": "assistant", "content": "어떤 선급인가요?"},
        {"role": "user", "content": "DNV"},
        {"role": "assistant", "content": "DNV 규칙을 안내합니다."},
    ]}])
    assert len(ex[0].messages) == 4
    assert ex[0].messages[0].role == "user" and ex[0].messages[-1].role == "assistant"


def test_dialogue_requires_user_and_assistant():
    assert to_dialogue_examples([{"turns": [{"role": "user", "content": "x"}]}]) == []


def test_examples_to_rows_drops_none_fields():
    rows = examples_to_rows(to_rag_examples([
        {"question": "q", "contexts": ["c"], "answer": "a"}
    ]))
    assert rows[0]["messages"][0]["role"] == "system"
    # exclude_none → tool_calls 같은 None 필드는 직렬화에서 빠짐
    assert "tool_calls" not in rows[0]["messages"][1]

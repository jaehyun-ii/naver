"""데이터 형태별 변환기 — QA 외 RAG·판정·4E(Reasoning)·도구호출을 학습 chat 포맷으로.

조선 도메인 라벨링 산출물(다양한 형태)을 학습기(trl)가 그대로 소비하는 conversational
포맷({"messages":[{role,content,...}]})으로 통일한다. QA/SFT 변환은 dataset.build에 있고,
여기서는 컨텍스트·추론·판정·도구호출 형태를 다룬다. 순수 함수(의존성 없음).
"""

from __future__ import annotations

import json

from llmops_core.common.schemas import ChatMessage, SFTExample

# RAG 컨텍스트 주입 시스템 프롬프트(평가 predict와 동일 규약 유지)
_RAG_SYSTEM = "다음 컨텍스트를 근거로 답하라:\n{context}"
# 4E 추론(Reason→Evidence→Evaluate→Execute 등) 사고 유도
_REASON_SYSTEM = "단계적으로 추론한 뒤 결론을 제시하라."


def to_rag_examples(
    rows: list[dict], *, system: str | None = None, source: str | None = None
) -> list[SFTExample]:
    """{question, contexts:[...], answer} → 컨텍스트 주입 SFT 예제.

    contexts를 system에 근거로 넣어 'RAG 형태'를 학습한다(검색은 서빙/평가 시 retriever가 담당).
    """
    out: list[SFTExample] = []
    for row in rows:
        q = row.get("question") or row.get("text")
        a = row.get("answer") or row.get("response")
        if not q or not a:
            continue
        ctx = row.get("contexts") or []
        ctx_text = "\n\n".join(ctx) if isinstance(ctx, list) else str(ctx)
        msgs: list[ChatMessage] = []
        sys_text = system or (_RAG_SYSTEM.format(context=ctx_text) if ctx else None)
        if sys_text:
            msgs.append(ChatMessage(role="system", content=sys_text))
        msgs.append(ChatMessage(role="user", content=q))
        msgs.append(ChatMessage(role="assistant", content=a))
        out.append(SFTExample(messages=msgs, source=source))
    return out


def to_translation_examples(
    rows: list[dict], *, source: str | None = None
) -> list[SFTExample]:
    """{source, target, src_lang?, tgt_lang?} → 번역 SFT 예제.

    조선 도메인 다국어 문서(영문 사양서↔국문 등) 번역. instruction을 system에 명시.
    """
    out: list[SFTExample] = []
    for row in rows:
        src = row.get("source") or row.get("text")
        tgt = row.get("target") or row.get("response")
        if not src or not tgt:
            continue
        sl, tl = row.get("src_lang"), row.get("tgt_lang")
        instr = f"다음 텍스트를 {sl or '원문'}에서 {tl or '대상 언어'}로 번역하라." if (sl or tl) \
            else "다음 텍스트를 번역하라."
        msgs = [
            ChatMessage(role="system", content=instr),
            ChatMessage(role="user", content=src),
            ChatMessage(role="assistant", content=tgt),
        ]
        out.append(SFTExample(messages=msgs, source=source))
    return out


def to_summary_examples(
    rows: list[dict], *, source: str | None = None
) -> list[SFTExample]:
    """{document, summary} → 요약 SFT 예제. 긴 사양서·보고서의 핵심 요약 학습."""
    out: list[SFTExample] = []
    for row in rows:
        doc = row.get("document") or row.get("text")
        summ = row.get("summary") or row.get("response")
        if not doc or not summ:
            continue
        msgs = [
            ChatMessage(role="system", content="다음 문서를 핵심만 간결히 요약하라."),
            ChatMessage(role="user", content=doc),
            ChatMessage(role="assistant", content=summ),
        ]
        out.append(SFTExample(messages=msgs, source=source))
    return out


def to_dialogue_examples(
    rows: list[dict], *, system: str | None = None, source: str | None = None
) -> list[SFTExample]:
    """멀티턴 대화 → SFT 예제. {turns:[{role,content},...]} 또는 {messages:[...]} 수용.

    여러 차례 주고받는 상담형 대화(선급 문의 등)를 그대로 학습한다.
    """
    out: list[SFTExample] = []
    for row in rows:
        turns = row.get("turns") or row.get("messages") or []
        norm: list[ChatMessage] = []
        if system:
            norm.append(ChatMessage(role="system", content=system))
        for t in turns:
            role = t.get("role")
            content = t.get("content")
            if role in ("system", "user", "assistant", "tool") and content is not None:
                norm.append(ChatMessage(role=role, content=content))
        # 최소 1 user + 1 assistant 턴이 있어야 유효
        roles = {m.role for m in norm}
        if "user" in roles and "assistant" in roles:
            out.append(SFTExample(messages=norm, source=source))
    return out


def to_reasoning_examples(
    rows: list[dict], *, source: str | None = None, tag_thinking: bool = True
) -> list[SFTExample]:
    """{question, reasoning, answer} → 추론(4E/CoT) 학습 예제.

    reasoning을 답변 앞에 결합한다. tag_thinking이면 <think>…</think>로 감싸 분리 학습.
    판정(verdict) 형태도 reasoning+answer로 동일하게 표현 가능.
    """
    out: list[SFTExample] = []
    for row in rows:
        q = row.get("question") or row.get("text")
        a = row.get("answer") or row.get("response")
        reasoning = row.get("reasoning") or row.get("rationale") or ""
        if not q or not a:
            continue
        if reasoning:
            body = (f"<think>{reasoning}</think>\n{a}" if tag_thinking
                    else f"{reasoning}\n\n{a}")
        else:
            body = a
        msgs = [
            ChatMessage(role="system", content=_REASON_SYSTEM),
            ChatMessage(role="user", content=q),
            ChatMessage(role="assistant", content=body),
        ]
        out.append(SFTExample(messages=msgs, source=source))
    return out


def to_verdict_examples(rows: list[dict], *, source: str | None = None) -> list[SFTExample]:
    """{question, verdict(적합/부적합/…), reasoning?} → 판정 학습 예제.

    규정 적합성 검토·부적합 보고 등 조선 도메인 '판정' 형태. reasoning 있으면 근거 포함.
    """
    norm = []
    for row in rows:
        v = row.get("verdict") or row.get("answer") or row.get("response")
        if not (row.get("question") or row.get("text")) or not v:
            continue
        norm.append({
            "question": row.get("question") or row.get("text"),
            "answer": str(v),
            "reasoning": row.get("reasoning") or row.get("rationale") or "",
        })
    return to_reasoning_examples(norm, source=source, tag_thinking=False)


def to_tool_call_examples(rows: list[dict], *, source: str | None = None) -> list[SFTExample]:
    """{question, tool_name, arguments(dict), result?, answer?} → 도구호출 학습 예제.

    assistant가 함수를 호출(tool_calls)하고, 선택적으로 tool 결과를 받아 최종 답을 생성하는
    멀티턴을 구성한다. OpenAI 호환 tool_calls/tool_call_id 사용.
    """
    out: list[SFTExample] = []
    for i, row in enumerate(rows):
        q = row.get("question") or row.get("text")
        name = row.get("tool_name") or row.get("name")
        if not q or not name:
            continue
        args = row.get("arguments") or row.get("args") or {}
        call_id = row.get("tool_call_id") or f"call_{i}"
        msgs: list[ChatMessage] = [ChatMessage(role="user", content=q)]
        msgs.append(ChatMessage(
            role="assistant", content="",
            tool_calls=[{
                "id": call_id, "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }],
        ))
        result = row.get("result")
        if result is not None:
            msgs.append(ChatMessage(
                role="tool", name=name, tool_call_id=call_id,
                content=result if isinstance(result, str)
                else json.dumps(result, ensure_ascii=False),
            ))
            answer = row.get("answer") or row.get("response")
            if answer:
                msgs.append(ChatMessage(role="assistant", content=answer))
        out.append(SFTExample(messages=msgs, source=source))
    return out


def examples_to_rows(examples: list[SFTExample]) -> list[dict]:
    """SFTExample 리스트 → trl conversational 행(dict). None 필드는 제거."""
    return [
        {"messages": [m.model_dump(exclude_none=True) for m in ex.messages]}
        for ex in examples
    ]

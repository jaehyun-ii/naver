"""예산 기반 단계적 문맥 확장 (small-to-big, graded) — noksan_ax expansion 동일 정책.

검색 히트(leaf: 항/호/목/세목)를 상위 문맥으로 확장하되, **컨텍스트 예산**을 넘는
거대 조(예: LR 8.5k자, KR 부록 INTRO 수십만 자)는 통째로 붙이지 않고 중간 단위로
단계 하강한다. 계층이 필드(paragraph_no/item_no)로 인코딩돼 있어 별도 저장 없이
어떤 중간 granularity 든 형제 재조립으로 만들 수 있다:

    조(article) 전체  ≤ 예산 → 그대로 (기본, 조의 99%+)
      └ 항(paragraph) 단위: 같은 조 + 같은 paragraph_no 형제 재조립
          └ 호(item) 단위: + 같은 item_no
              └ leaf 단독 (이미 경로·항 lead로 문맥화돼 있음)

입력은 저장소 중립 — 호출측(리트리버)이 사이드카(SQL)나 Qdrant payload에서
``parent_chunk_id`` 로 조의 child 들을 로드해 넘긴다. 청크는 평면 dict(우리 청커
JSONL 행)와 {"content","metadata"} 형태 모두 수용한다.
"""

from __future__ import annotations

from typing import Any

# 기본 예산: ≈2k 토큰(한글/2·영문/4 기준). 조 크기 p99≈3k자라 대부분 article 레벨로 끝난다.
DEFAULT_BUDGET_CHARS = 4000

_LEVEL_KEYS = (
    ("paragraph", ("paragraph_no",)),
    ("item", ("paragraph_no", "item_no")),
)


def expand_context(
    leaf: Any,
    parent: Any | None,
    siblings: list[Any],
    *,
    budget_chars: int = DEFAULT_BUDGET_CHARS,
) -> tuple[str, str]:
    """(문맥 텍스트, 사용된 레벨) 반환. 레벨: article | paragraph | item | leaf.

    siblings 는 같은 조(parent_chunk_id 동일)의 child 들(순서 무관 — chunk_order로 정렬).
    """
    # 1) 조 전체 — 예산 안이면 그대로 (기본 경로)
    ptext = _content(parent) if parent is not None else ""
    if ptext and len(ptext) <= budget_chars:
        return ptext, "article"
    # 1b) 표 병합으로 조가 예산을 넘긴 경우 — 본문이 예산 안이면 표만 잘라 동반.
    #     (표 값은 Qdrant payload에 없어 하강 재조립로는 복구 불가 — 여기서 지켜야 한다)
    body, tables = _split_tables(ptext)
    if body and len(body) <= budget_chars:
        return _with_tables(body, tables, budget_chars), "article"

    # 2) 항 → 3) 호 단위로 하강 — 남는 예산에 표를 동반
    lmeta = _meta(leaf)
    for level, keys in _LEVEL_KEYS:
        text = _assemble(leaf, siblings, keys)
        if text and len(text) <= budget_chars:
            return _with_tables(text, tables, budget_chars), level
        # leaf 에 해당 키가 없으면(무번호 도입부 등) 더 내려가도 같음 → 중단
        if not lmeta.get(keys[-1]):
            break

    # 4) leaf 단독 (content 가 이미 경로+항 lead 로 문맥화돼 있음)
    return _content(leaf), "leaf"


_TABLE_MARK = "\n\n[인용된 표]\n"


def _split_tables(text: str) -> tuple[str, str]:
    """parent content를 (본문, 표 블록)으로 분리 — 사이드카가 병합한 표 섹션 기준."""
    if _TABLE_MARK not in (text or ""):
        return text or "", ""
    body, tables = text.split(_TABLE_MARK, 1)
    return body.rstrip(), tables


def _with_tables(text: str, tables: str, budget_chars: int) -> str:
    """남는 예산에 표 블록을 동반 — 200자 미만이면 무의미해 생략."""
    room = budget_chars - len(text) - len(_TABLE_MARK)
    if not tables or room < 200:
        return text
    if len(tables) > room:
        tables = tables[:room].rstrip() + "\n(표 일부 생략 — 분량 제한)"
    return text + _TABLE_MARK + tables


def _assemble(leaf: Any, siblings: list[Any], keys: tuple[str, ...]) -> str:
    """leaf 와 같은 (항[/호]) 좌표의 text 형제들을 원문 순서로 재조립.

    child content 는 `경로줄\n[헤딩]\n본문` 형태 — 경로줄은 그룹 헤더로 1회만 남긴다.
    """
    lmeta = _meta(leaf)
    group = [
        s for s in siblings
        if _meta(s).get("chunk_type") == "text"
        and all((_meta(s).get(k) or "") == (lmeta.get(k) or "") for k in keys)
    ]
    if not group:
        return ""
    group.sort(key=lambda s: _meta(s).get("chunk_order") or 0)
    header = _content(group[0]).split("\n", 1)[0]          # 경로줄(전 형제 동일)
    bodies = []
    for s in group:
        parts = _content(s).split("\n", 1)
        bodies.append(parts[1] if len(parts) > 1 else parts[0])
    return "\n".join([header, *bodies]).strip()


def _content(c: Any) -> str:
    if isinstance(c, dict):
        return (c.get("content") or c.get("text") or "")
    return getattr(c, "content", "") or ""


def _meta(c: Any) -> dict:
    if isinstance(c, dict):
        m = c.get("metadata")
        return m if isinstance(m, dict) else c    # 평면 JSONL 행이면 행 자체가 메타
    return getattr(c, "metadata", None) or {}

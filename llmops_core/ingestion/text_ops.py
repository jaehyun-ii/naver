"""원천 텍스트 정규화·중복제거 — 의존성 없는 순수 로직 (Spark/단건 공용).

L1 적재 직후의 1차 정제. 무거운 Spark 잡과 경량 경로가 동일한 규칙을 쓰도록
핵심 함수를 여기 두고 양쪽에서 재사용한다(규칙 일관성 보장). 단위 테스트 대상.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

from llmops_core.common.schemas import TextRecord

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """유니코드 NFKC 정규화 + 공백 축약 + 양끝 정리."""
    text = unicodedata.normalize("NFKC", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def content_hash(text: str) -> str:
    """정규화된 내용 기반 해시 (정확 중복 판정 키)."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def normalize_records(records: list[TextRecord]) -> list[TextRecord]:
    """각 레코드의 text를 정규화한 새 리스트 반환."""
    return [r.model_copy(update={"text": normalize_text(r.text)}) for r in records]


def exact_dedup(records: list[TextRecord]) -> tuple[list[TextRecord], int]:
    """정확 중복 제거(최초 등장 유지). (남은 레코드, 제거 수) 반환."""
    seen: set[str] = set()
    kept: list[TextRecord] = []
    for r in records:
        h = content_hash(r.text)
        if h in seen:
            continue
        seen.add(h)
        kept.append(r)
    return kept, len(records) - len(kept)

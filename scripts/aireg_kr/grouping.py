"""canonical_rule_group_id — 전 트랙(판정·스위트 QA·파생 학습 파일) 공통 규칙 그룹 식별자.

판정 트랙은 rule_uid, 스위트 QA는 evidence chunk_id를 그룹 키로 쓰면 같은 실제
조항에서 파생된 데이터가 서로 다른 split에 갈 수 있다(특히 비-KR: rule_uid가
``publisher__file__chunk`` 접두라 chunk_id와 문자열이 다름). 여기서 정규화 규칙과
우선순위를 한 곳에 고정해 동일 조항 → 동일 그룹 → 동일 split을 보장한다.

ID는 정규화 키의 sha256(``crg_`` 접두 16자)이며, 디버깅용으로 원본 키와 생성
기준(source)을 함께 반환해 Master Dataset에 저장한다.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

# rule_uid가 이미 publisher/document를 접두한 형태(비-KR: PUB__file__chunk) —
# 키 조립 전에 접두를 벗겨 chunk 좌표만 남긴다(표현 차이로 인한 충돌 방지).
_UID_PREFIX = re.compile(r"^[a-z0-9]+__[^_].*?__", re.IGNORECASE)
_SEPARATORS = re.compile(r"\s*(?:>|/|::|—|–)\s*")
_MULTI_SPACE = re.compile(r"\s+")


def normalize_key_part(value: str | None) -> str:
    """그룹 키 구성 요소 정규화 — 의미 변환 없이 표현 차이만 흡수.

    유니코드 NFKC, 소문자화, 앞뒤 공백 제거, 연속 공백 축소, 경로 구분자(>,/,::)를
    '-'로 통일. '제5편'↔'Part 5' 같은 의미 매핑은 하지 않는다(임의 변환 금지).
    """
    if not value:
        return ""
    s = unicodedata.normalize("NFKC", str(value)).strip().lower()
    s = _SEPARATORS.sub("-", s)
    s = _MULTI_SPACE.sub(" ", s)
    return s.strip("- ")


def _strip_uid_prefix(rule_uid: str) -> str:
    return _UID_PREFIX.sub("", rule_uid, count=1)


def build_canonical_rule_group_id(
    *,
    publisher: str | None,
    document_id: str | None,
    edition: str | None,
    rule_uid: str | None = None,
    article_id: str | None = None,
    section_path: str | None = None,
    chunk_id: str | None = None,
    parent_rule_id: str | None = None,
) -> tuple[str, str, str]:
    """(canonical_id, source, source_key) 반환.

    우선순위: rule_uid → article_id → section_path → parent_rule_id → chunk_id.
    앞 4개는 publisher|document_id|edition 접두로 조립하고, chunk_id는 최후
    fallback(문서 문맥 없이 단독) — 호출부는 fallback 비율을 통계로 감시한다.
    """
    pub = normalize_key_part(publisher) or "unknown-publisher"
    doc = normalize_key_part(document_id) or "unknown-document"
    ed = normalize_key_part(edition) or "unknown-edition"
    prefix = f"{pub}|{doc}|{ed}"

    candidates: list[tuple[str, str]] = []
    if rule_uid:
        candidates.append(("rule_uid", normalize_key_part(_strip_uid_prefix(rule_uid))))
    if article_id:
        candidates.append(("article_id", normalize_key_part(article_id)))
    if section_path:
        candidates.append(("section_path", normalize_key_part(section_path)))
    if parent_rule_id:
        candidates.append(("parent_rule_id", normalize_key_part(parent_rule_id)))
    for source, part in candidates:
        if part:
            key = f"{prefix}|{part}"
            return _digest(key), source, key
    if chunk_id:
        key = normalize_key_part(chunk_id)
        return _digest(key), "chunk_id", key
    raise ValueError("그룹 키를 만들 수 있는 입력이 없습니다")


def _digest(key: str) -> str:
    return "crg_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def group_id_for_sample(sample: dict) -> tuple[str, str, str]:
    """Master 샘플(dict) → canonical 그룹 ID. build_master의 단일 진입점.

    판정 트랙은 rule.rule_id(=rule_uid), 스위트 QA는 evidence chunk_id가
    rule.rule_id로 들어오지만, KR은 둘 다 RULE_* 원문 좌표라 동일 키로 수렴하고
    비-KR은 _strip_uid_prefix가 접두를 벗겨 chunk 좌표로 수렴한다.
    """
    src = sample.get("source") or {}
    rule = sample.get("rule") or {}
    return build_canonical_rule_group_id(
        publisher=src.get("publisher"),
        document_id=src.get("source_file") or src.get("doc_title"),
        edition=src.get("edition"),
        rule_uid=rule.get("rule_id"),
        section_path=src.get("section_path"),
        chunk_id=src.get("chunk_id") or sample.get("sample_id"),
    )

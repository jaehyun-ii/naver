"""L1 정제 — PII 마스킹(Presidio) · 언어감지(fastText) · 근사중복(datasketch).

전부 선택적 의존성(lazy). "쓰레기 입력, 쓰레기 모델"을 상류에서 차단하는 1차 정제.
규칙성 정확중복은 ingestion.text_ops.exact_dedup, 의미 근사중복은 여기 MinHashLSH가 담당.
"""

from __future__ import annotations

import re

from llmops_core.common.config import get_settings
from llmops_core.common.errors import OptionalDependencyError
from llmops_core.common.schemas import TextRecord

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


# ── PII 마스킹 (Presidio, MIT) ──
class PIIMasker:
    """Presidio 기반 PII 탐지/익명화. 데이터 주권·개인정보 보호의 상류 게이트."""

    def __init__(self, lang: str = "en") -> None:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
        except ImportError as exc:  # pragma: no cover
            raise OptionalDependencyError("presidio-analyzer", "quality") from exc
        self.lang = lang
        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()

    def mask(self, text: str) -> str:
        results = self._analyzer.analyze(text=text, language=self.lang)
        return self._anonymizer.anonymize(text=text, analyzer_results=results).text


def mask_records(records: list[TextRecord], lang: str = "en") -> list[TextRecord]:
    """각 레코드 텍스트의 PII를 마스킹한 새 리스트."""
    masker = PIIMasker(lang=lang)
    return [r.model_copy(update={"text": masker.mask(r.text)}) for r in records]


# ── 언어감지 (fastText, MIT) ──
def detect_language(text: str, model_path: str | None = None) -> str:
    """fastText lid 모델로 언어코드 반환(예: 'ko'). 모델 경로 미설정 시 예외."""
    try:
        import fasttext
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("fasttext", "quality") from exc

    path = model_path or get_settings().data_quality.fasttext_model_path
    if not path:
        raise OptionalDependencyError("fasttext lid model", "quality")
    model = fasttext.load_model(path)
    label = model.predict(text.replace("\n", " "))[0][0]  # "__label__ko"
    return label.replace("__label__", "")


def annotate_language(records: list[TextRecord]) -> list[TextRecord]:
    """각 레코드에 감지 언어를 부여(.lang). 모델 미설정이면 원본 그대로 반환."""
    if not get_settings().data_quality.fasttext_model_path:
        return records
    return [r.model_copy(update={"lang": detect_language(r.text)}) for r in records]


def filter_by_language(
    records: list[TextRecord], allowed: list[str]
) -> tuple[list[TextRecord], int]:
    """허용 언어만 통과(.lang 미설정 레코드는 보존). (남은 것, 제거 수)."""
    kept = [r for r in records if r.lang is None or r.lang in allowed]
    return kept, len(records) - len(kept)


# ── 근사중복 (datasketch MinHashLSH, MIT) ──
def _minhash(text: str, num_perm: int):
    from datasketch import MinHash

    m = MinHash(num_perm=num_perm)
    for tok in set(_TOKEN_RE.findall(text.lower())):
        m.update(tok.encode("utf-8"))
    return m


def drop_near_duplicates(
    records: list[TextRecord], threshold: float = 0.85, num_perm: int = 128
) -> tuple[list[TextRecord], int]:
    """MinHash Jaccard ≥ threshold 인 근사중복을 제거(최초 유지). (남은 것, 제거 수)."""
    try:
        from datasketch import MinHashLSH
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("datasketch", "quality") from exc

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[TextRecord] = []
    for r in records:
        mh = _minhash(r.text, num_perm)
        if lsh.query(mh):  # 유사 항목이 이미 존재
            continue
        lsh.insert(r.id, mh)
        kept.append(r)
    return kept, len(records) - len(kept)

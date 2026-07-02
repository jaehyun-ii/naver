"""L2 품질검증 게이트 — 네이티브 규칙(기본) + Great Expectations(선택).

검증 미달 시 DataQualityFailed로 파이프라인을 차단한다(학습 진입 금지).
GE API는 버전 변동이 커서, 의존성 없이 항상 동작하는 네이티브 규칙을 기본 게이트로 두고,
GE는 보강 옵션으로 둔다(둘 다 동일한 DataQualityReport 계약을 따른다).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llmops_core.common.config import get_settings
from llmops_core.common.errors import DataQualityFailed, OptionalDependencyError
from llmops_core.common.schemas import DataQualityReport, TextRecord
from llmops_core.ingestion.text_ops import content_hash


@dataclass
class QualityRules:
    """SFT 원천 텍스트 품질 규칙(임계값). 미설정 시 공통 설정에서 채운다."""

    min_chars: int | None = None
    max_chars: int | None = None
    allowed_langs: list[str] | None = None
    max_dup_ratio: float | None = None
    near_dup_threshold: float | None = None
    required_metadata: list[str] = field(default_factory=list)

    def resolved(self) -> "QualityRules":
        s = get_settings().data_quality
        return QualityRules(
            min_chars=self.min_chars if self.min_chars is not None else s.min_chars,
            max_chars=self.max_chars if self.max_chars is not None else s.max_chars,
            allowed_langs=self.allowed_langs or s.allowed_langs,
            max_dup_ratio=(
                self.max_dup_ratio if self.max_dup_ratio is not None else s.max_dup_ratio
            ),
            near_dup_threshold=(
                self.near_dup_threshold if self.near_dup_threshold is not None
                else s.near_dup_threshold
            ),
            required_metadata=self.required_metadata,
        )


def validate_records(
    records: list[TextRecord],
    rules: QualityRules | None = None,
    *,
    suite: str = "sft_quality",
    raise_on_fail: bool = True,
) -> DataQualityReport:
    """네이티브 규칙으로 검증한 리포트 반환. 미달 시 DataQualityFailed(옵션)."""
    r = (rules or QualityRules()).resolved()
    failures: dict[str, str] = {}
    n = len(records)

    if n == 0:
        failures["non_empty"] = "레코드가 0건"

    # 언어감지(fasttext lid 설정 시) — .lang을 채워 아래 화이트리스트 검사를 실효화.
    # 모델 미설정/미설치 환경(dev)에서는 원본을 그대로 두고 스킵(graceful).
    if r.allowed_langs:
        try:
            from llmops_core.quality.clean import annotate_language

            records = annotate_language(records)
        except OptionalDependencyError:
            pass

    # 길이 경계
    too_short = sum(1 for x in records if len(x.text) < (r.min_chars or 0))
    too_long = sum(1 for x in records if r.max_chars and len(x.text) > r.max_chars)
    if too_short:
        failures["min_chars"] = f"{too_short}건이 {r.min_chars}자 미만"
    if too_long:
        failures["max_chars"] = f"{too_long}건이 {r.max_chars}자 초과"

    # 언어 화이트리스트(.lang 부여된 레코드만 검사)
    if r.allowed_langs:
        bad_lang = sum(1 for x in records if x.lang and x.lang not in r.allowed_langs)
        if bad_lang:
            failures["allowed_langs"] = f"{bad_lang}건이 허용 언어 {r.allowed_langs} 밖"

    # 정확중복 비율
    dup_ratio = 0.0
    if n:
        uniq = len({content_hash(x.text) for x in records})
        dup_ratio = round((n - uniq) / n, 4)
        if r.max_dup_ratio is not None and dup_ratio > r.max_dup_ratio:
            failures["max_dup_ratio"] = f"중복비율 {dup_ratio} > {r.max_dup_ratio}"

    # 근사중복(MinHash LSH) — Jaccard ≥ near_dup_threshold 인 유사 레코드 비율.
    # datasketch 미설치(dev) 시 스킵. 근사중복 비율도 max_dup_ratio 상한으로 게이트.
    near_dup_ratio = 0.0
    if n and r.near_dup_threshold:
        try:
            from llmops_core.quality.clean import drop_near_duplicates

            _, removed = drop_near_duplicates(records, threshold=r.near_dup_threshold)
            near_dup_ratio = round(removed / n, 4)
            if r.max_dup_ratio is not None and near_dup_ratio > r.max_dup_ratio:
                failures["near_dup_ratio"] = (
                    f"근사중복비율 {near_dup_ratio} > {r.max_dup_ratio} "
                    f"(임계 {r.near_dup_threshold})")
        except OptionalDependencyError:
            pass

    # 필수 메타데이터
    for field_name in r.required_metadata:
        missing = sum(1 for x in records if field_name not in x.metadata)
        if missing:
            failures[f"meta:{field_name}"] = f"{missing}건에 '{field_name}' 누락"

    report = DataQualityReport(
        suite=suite,
        num_records=n,
        passed=not failures,
        failures=failures,
        stats={"dup_ratio": dup_ratio, "near_dup_ratio": near_dup_ratio,
               "num_records": float(n)},
    )
    if not report.passed and raise_on_fail:
        raise DataQualityFailed(failures)
    return report


def validate_with_great_expectations(
    records: list[TextRecord], *, suite: str = "sft_suite"
) -> DataQualityReport:
    """Great Expectations로 보강 검증(선택). pandas DataFrame 기반 Expectation Suite."""
    try:
        import great_expectations as gx
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("great-expectations", "quality") from exc

    df = pd.DataFrame([{"id": x.id, "text": x.text, "lang": x.lang} for x in records])
    ctx = gx.get_context()
    batch = ctx.sources.add_pandas("sft").read_dataframe(df)
    rules = get_settings().data_quality
    val = batch.validate(
        expectations=[
            gx.expectations.ExpectColumnValuesToNotBeNull(column="text"),
            gx.expectations.ExpectColumnValueLengthsToBeBetween(
                column="text", min_value=rules.min_chars, max_value=rules.max_chars
            ),
            gx.expectations.ExpectColumnValuesToBeUnique(column="id"),
        ]
    )
    failures = {} if val.success else {"great_expectations": "Expectation Suite 미충족"}
    return DataQualityReport(
        suite=suite, num_records=len(records), passed=val.success, failures=failures
    )

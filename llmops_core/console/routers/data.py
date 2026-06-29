"""데이터·데이터셋 — 품질검증(L2) + SFT 데이터셋 빌드(L3). 전부 오프라인 동작.

S3 적재 없이 인메모리로 검증/빌드해 매니페스트를 레지스트리에 등록(콘솔 데모/검토용).
실제 적재는 파이프라인 스텝(data_quality/data_build)이 담당한다.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from llmops_core.common.errors import DataQualityFailed
from llmops_core.common.schemas import DatasetManifest, DataQualityReport, TextRecord
from llmops_core.console.schemas import BuildDatasetBody, ValidateBody
from llmops_core.console.security import require_perm
from llmops_core.console.services import services
from llmops_core.dataset.build import SplitConfig, filter_sft_by_length, split_sft, to_sft_examples
from llmops_core.dataset.versioning import fingerprint
from llmops_core.quality import QualityRules, validate_records

router = APIRouter(prefix="/api/data", tags=["data"], dependencies=[Depends(require_perm("data:write"))])


@router.post("/validate")
def validate(body: ValidateBody) -> DataQualityReport:
    records = [TextRecord(**r) for r in body.records]
    rules = QualityRules(
        min_chars=body.min_chars, max_chars=body.max_chars, max_dup_ratio=body.max_dup_ratio
    )
    try:
        return validate_records(records, rules)
    except DataQualityFailed:
        # 게이트 실패도 리포트로 돌려준다(차단 사유 표시용)
        return validate_records(records, rules, raise_on_fail=False)


@router.post("/build")
def build(body: BuildDatasetBody) -> DatasetManifest:
    examples = filter_sft_by_length(to_sft_examples(body.labeled, system=body.system))
    if not examples:
        raise HTTPException(422, "유효한 (text, response) 쌍이 없습니다")
    train, val, test = split_sft(
        examples, SplitConfig(val_ratio=body.val_ratio, test_ratio=body.test_ratio)
    )
    manifest = DatasetManifest(
        name=body.name,
        kind="sft",
        fingerprint=fingerprint(train + val + test),
        num_train=len(train),
        num_val=len(val),
        num_test=len(test),
    )
    services().datasets.add(manifest)
    return manifest


@router.get("/datasets")
def list_datasets() -> list[DatasetManifest]:
    return services().datasets.list()

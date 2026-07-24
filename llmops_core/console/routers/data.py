"""데이터·데이터셋 — 품질검증(L2) + SFT 데이터셋 빌드(L3).

빌드한 스플릿(train/val/test JSONL)은 S3/MinIO(datasets 버킷)에 적재하고, 매니페스트에
s3_uri를 기록해 레지스트리에 등록한다. S3 미가용(dev) 시 적재를 건너뛰고 인메모리 등록만 한다.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException

from llmops_core.common.errors import DataQualityFailed
from llmops_core.common.schemas import DatasetManifest, DataQualityReport, TextRecord
from llmops_core.common.storage import ObjectStore
from llmops_core.console.schemas import BuildDatasetBody, ValidateBody
from llmops_core.console.security import require_perm
from llmops_core.console.services import services
from llmops_core.dataset.build import SplitConfig, filter_sft_by_length, split_sft, to_sft_examples
from llmops_core.dataset.versioning import fingerprint
from llmops_core.quality import QualityRules, validate_records

router = APIRouter(prefix="/api/data", tags=["data"], dependencies=[Depends(require_perm("data:write"))])


def _to_jsonl(examples: list) -> bytes:
    """SFTExample 리스트를 JSONL 바이트로 직렬화(S3 적재용)."""
    return "\n".join(
        json.dumps(e.model_dump(), ensure_ascii=False) for e in examples
    ).encode("utf-8")


def _persist_dataset(fp: str, train: list, val: list, test: list) -> str | None:
    """스플릿을 datasets 버킷 `{fingerprint}/{split}.jsonl`에 적재. 실패 시 None(graceful)."""
    try:
        store = ObjectStore()
        store.ensure_bucket("datasets")
        for split, rows in (("train", train), ("val", val), ("test", test)):
            store.put_bytes("datasets", f"{fp}/{split}.jsonl", _to_jsonl(rows))
        return f"s3://{store.bucket_for('datasets')}/{fp}/"
    except Exception:  # noqa: BLE001 — S3 미가용(dev)이면 인메모리 등록만
        return None


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
    fp = fingerprint(train + val + test)
    manifest = DatasetManifest(
        name=body.name,
        kind="sft",
        fingerprint=fp,
        num_train=len(train),
        num_val=len(val),
        num_test=len(test),
        s3_uri=_persist_dataset(fp, train, val, test),
        samples=[e.model_dump() for e in train[:12]],  # 상세 미리보기용 대표 레코드
    )
    services().datasets.add(manifest)
    return manifest


@router.get("/datasets")
def list_datasets() -> list[DatasetManifest]:
    return services().datasets.list()

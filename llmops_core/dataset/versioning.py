"""L3 버전 고정 — 내용 핑거프린트 + DVC 연계 + S3 적재 + 매니페스트.

데이터 버전 해시 ↔ run ↔ 모델 버전 사슬(거버넌스)의 시작점.
핑거프린트는 DVC와 독립적인 내용 기반 sha256이라, DVC 미사용 환경에서도 안정 식별자가 된다.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel

from llmops_core.common.schemas import DatasetManifest
from llmops_core.ingestion.records import dumps_jsonl, write_records


def fingerprint(records: list[BaseModel]) -> str:
    """레코드 집합의 내용 기반 sha256 (순서 무관)."""
    h = hashlib.sha256()
    for line in sorted(r.model_dump_json(exclude_none=True) for r in records):
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def dvc_url(path: str, repo: str | None = None) -> str:
    """DVC가 가리키는 데이터 버전 URL/해시(tracking 모듈 재사용)."""
    from llmops_core.tracking import data_version

    return data_version(path, repo=repo)


def publish_split(
    name: str,
    kind: str,
    train: list[BaseModel],
    val: list[BaseModel],
    test: list[BaseModel],
    *,
    purpose: str = "datasets",
    dvc_path: str | None = None,
) -> DatasetManifest:
    """분할을 S3에 JSONL로 적재하고 버전 매니페스트를 생성한다.

    s3 경로: {bucket}/{name}/{fingerprint}/{split}.jsonl — 핑거프린트로 불변 버전 고정.
    """
    all_records = train + val + test
    fp = fingerprint(all_records)
    base = f"{name}/{fp}"
    for split_name, recs in (("train", train), ("val", val), ("test", test)):
        write_records(purpose, f"{base}/{split_name}.jsonl", recs)

    return DatasetManifest(
        name=name,
        kind=kind,  # type: ignore[arg-type]
        fingerprint=fp,
        dvc_url=dvc_url(dvc_path) if dvc_path else None,
        num_train=len(train),
        num_val=len(val),
        num_test=len(test),
        s3_uri=f"s3://{base}",
    )


def manifest_bytes(manifest: DatasetManifest) -> bytes:
    """매니페스트를 JSONL 1줄로 직렬화(레지스트리/MLflow 아티팩트 기록용)."""
    return dumps_jsonl([manifest])

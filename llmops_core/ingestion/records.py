"""JSONL ↔ S3 레코드 입출력 — 모든 영속은 S3 seam(ObjectStore) 경유.

상류 단계 간 데이터 교환 포맷은 JSONL(한 줄=한 pydantic 레코드)로 고정한다.
로컬 FS 직접 접근 대신 ObjectStore를 사용해 MinIO→SeaweedFS→NCP 전환에 무손실.
"""

from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel

from llmops_core.common.storage import ObjectStore

M = TypeVar("M", bound=BaseModel)


def loads_jsonl(data: bytes, model_cls: type[M]) -> list[M]:
    """JSONL 바이트 → 모델 리스트."""
    out: list[M] = []
    for line in data.decode("utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(model_cls(**json.loads(line)))
    return out


def dumps_jsonl(records: list[BaseModel]) -> bytes:
    """모델 리스트 → JSONL 바이트."""
    body = "\n".join(r.model_dump_json(exclude_none=True) for r in records)
    return (body + "\n").encode("utf-8") if body else b""


def read_records(
    purpose: str, key: str, model_cls: type[M], *, store: ObjectStore | None = None
) -> list[M]:
    store = store or ObjectStore()
    return loads_jsonl(store.get_bytes(purpose, key), model_cls)


def write_records(
    purpose: str, key: str, records: list[BaseModel], *, store: ObjectStore | None = None
) -> str:
    store = store or ObjectStore()
    return store.put_bytes(purpose, key, dumps_jsonl(records))

"""문서 등록 서비스 — 원본 PDF DB의 쓰기 진입점.

register_document()가 곧 "생성/수정" 이벤트다: MinIO(documents-raw)에 블롭 업로드 +
레지스트리 upsert(sha256 dedup·버전) + 큐 enqueue. 내용이 동일(sha256)하고 이미
processed면 무동작(멱등). 내용이 바뀌면 새 버전으로 재처리 트리거.
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

from llmops_core.chunking import FAMILIES as _FAMILIES
from llmops_core.common.storage import ObjectStore

from .models import Document, STATUS_PENDING, STATUS_PROCESSED
from .pipeline import RAW
from .queue import queue
from .registry import registry


def _slug(name: str) -> str:
    s = re.sub(r"[^\w.-]+", "_", name.strip()).strip("._")
    return s or "document"


def register_document(source: str | Path | bytes, *, name: str | None = None,
                      family: str = "auto", enqueue: bool = True) -> Document:
    """PDF를 원본 DB에 등록/갱신하고 처리 잡을 큐에 넣는다."""
    if isinstance(source, (str, Path)):
        data = Path(source).read_bytes()
        name = name or Path(source).stem
    else:
        data = source
        if not name:
            raise ValueError("bytes 등록 시 name 필수")
    doc_id = _slug(name)
    sha = hashlib.sha256(data).hexdigest()

    reg = registry()
    existing = reg.get(doc_id)
    if existing and existing.sha256 == sha and existing.status == STATUS_PROCESSED:
        return existing  # 동일 내용·처리완료 → 멱등 무동작

    store = ObjectStore()
    store.ensure_bucket(RAW)
    raw_key = f"{doc_id}/{sha}.pdf"
    store.put_bytes(RAW, raw_key, data)

    now = time.time()
    version = (existing.version + 1) if (existing and existing.sha256 != sha) else (existing.version if existing else 1)
    doc = Document(
        id=doc_id, name=name, sha256=sha, raw_key=raw_key,
        family=family if family in ({"auto"} | set(_FAMILIES)) else "auto",
        version=version, status=STATUS_PENDING,
        created_at=existing.created_at if existing else now, updated_at=now,
    )
    reg.save(doc)
    if enqueue:
        queue().enqueue(doc_id)
    return doc

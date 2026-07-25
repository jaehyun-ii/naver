"""문서 처리 파이프라인 — 원본 PDF 1건의 e2e 체인.

    raw PDF(MinIO) → ETL(MinerU@서버 + 후처리) → content_list 적재(processed 버킷)
      → 도메인 청킹(kr_rule|abs_guide) → chunks 적재 → Qdrant 벡터 적재 → 레지스트리 갱신

각 단계 산출물은 MinIO documents-processed 버킷에 `<name>/v<version>/` 프리픽스로 적재.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from llmops_core.common.storage import ObjectStore

from .models import Document, STATUS_FAILED, STATUS_PROCESSED, STATUS_PROCESSING
from .registry import registry
from .vectorize import vectorize_chunks
from llmops_core.chunking import FAMILIES, chunk_document, write_jsonl

RAW = "documents-raw"
PROCESSED = "documents-processed"


def _endpoint() -> str:
    import os
    return os.environ.get("PARSER_ENDPOINT", "http://localhost:8009")


def _collection() -> str:
    try:
        from llmops_core.common.config import get_settings
        s = get_settings()
        return f"{s.env}-{s.domain}-docchunks"
    except Exception:  # noqa: BLE001
        return "dev-llmops-docchunks"


def process_document(doc_id: str) -> Document:
    reg = registry()
    doc = reg.get(doc_id)
    if doc is None:
        raise ValueError(f"unknown document: {doc_id}")
    store = ObjectStore()
    doc.status = STATUS_PROCESSING
    doc.error = None
    doc.updated_at = time.time()
    reg.save(doc)

    try:
        with tempfile.TemporaryDirectory(prefix=f"ingest-{doc.id}-") as tmp:
            work = Path(tmp)
            # 1) raw PDF 내려받기
            pdf = work / f"{doc.name}.pdf"
            pdf.write_bytes(store.get_bytes(RAW, doc.raw_key))

            # 2) ETL: MinerU(서버) + pdftext 후처리
            from llmops_core.etl import EtlConfig, run_etl
            cfg = EtlConfig.from_env()
            if not cfg.endpoint:
                from dataclasses import replace
                cfg = replace(cfg, endpoint=_endpoint())
            status = run_etl(pdf, work / "etl", cfg, name=doc.name)
            content_list = Path(status["content_list"])
            prefix = f"{doc.name}/v{doc.version}"

            # 3) content_list + md + images 적재
            store.ensure_bucket(PROCESSED)
            cl_key = f"{prefix}/{doc.name}_content_list.json"
            store.upload_file(PROCESSED, str(content_list), cl_key)
            doc.content_list_key = cl_key
            etl_dir = content_list.parent
            md = etl_dir / f"{doc.name}.md"
            if md.exists():
                store.upload_file(PROCESSED, str(md), f"{prefix}/{doc.name}.md")
            img_dir = etl_dir / "images"
            if img_dir.is_dir():
                for img in img_dir.iterdir():
                    if img.is_file():
                        store.upload_file(PROCESSED, str(img), f"{prefix}/images/{img.name}")

            # 4) 도메인 청킹 (llmops_core.chunking 패키지 — 선급별 특화 청커)
            requested = doc.family if doc.family in FAMILIES else "auto"
            family, chunk_list = chunk_document(
                content_list, family=requested, source_file=content_list.name
            )
            doc.family = family
            chunks = work / f"{doc.name}_chunks.jsonl"
            write_jsonl(chunk_list, chunks)
            doc.n_chunks = len(chunk_list)

            # 5) chunks 적재
            ck_key = f"{prefix}/{doc.name}_chunks.jsonl"
            store.upload_file(PROCESSED, str(chunks), ck_key)
            doc.chunks_key = ck_key

            # 6) Qdrant 벡터 적재(child·표·그림만)
            collection = _collection()
            vres = vectorize_chunks(chunks, collection)
            doc.vector_collection = collection
            doc.n_vectors = vres["n_vectors"]

            # 6.5) parent-직접 인덱스(서빙 검색 스택 정합) — env 게이트
            import os as _os
            pcoll = _os.environ.get("INGEST_PARENT_COLLECTION")
            if pcoll:
                from .vectorize import vectorize_parents
                pres = vectorize_parents(
                    chunks, pcoll, parent_db=_os.environ.get("INGEST_PARENT_DB"))
                logger.info("parent 인덱스 %s += %d", pcoll, pres["n_parents"])

        doc.status = STATUS_PROCESSED
        doc.updated_at = time.time()
        reg.save(doc)
        return doc
    except Exception as exc:  # noqa: BLE001
        doc.status = STATUS_FAILED
        doc.error = str(exc)[:1000]
        doc.updated_at = time.time()
        reg.save(doc)
        raise

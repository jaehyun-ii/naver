"""인제스트 워커 — 큐에서 문서 잡을 받아 e2e 파이프라인 실행.

    python -m llmops_core.ingest worker           # 무한 소비
    python -m llmops_core.ingest worker --once     # 큐 빌 때까지만
"""

from __future__ import annotations

import logging

from .pipeline import process_document
from .queue import queue

logger = logging.getLogger("llmops.ingest.worker")


def run_worker(*, once: bool = False, idle_exits: int = 1) -> None:
    q = queue()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger.info("ingest worker started (queue=%s, depth=%d)", q.backend, q.depth())
    idle = 0
    while True:
        doc_id = q.dequeue(timeout=5)
        if doc_id is None:
            idle += 1
            if once and idle >= idle_exits:
                logger.info("queue drained → exit")
                return
            continue
        idle = 0
        logger.info("processing %s", doc_id)
        try:
            doc = process_document(doc_id)
            logger.info("✓ %s → status=%s chunks=%d vectors=%d coll=%s",
                        doc_id, doc.status, doc.n_chunks, doc.n_vectors, doc.vector_collection)
        except Exception as exc:  # noqa: BLE001
            logger.exception("✗ %s failed", doc_id)
            q.dead_letter(doc_id, str(exc))

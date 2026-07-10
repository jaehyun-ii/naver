"""문서 처리 잡 큐 — redis 기반(즉시 dequeue·재시도·dead-letter).

register 시 enqueue → 워커가 BRPOP으로 즉시 소비. redis 미가용 시 로컬 리스트로
폴백(단일프로세스 dev). 잡 페이로드는 문서 id 문자열.
"""

from __future__ import annotations

import os
from collections import deque

_QUEUE = "llmops:ingest:jobs"
_DEADLETTER = "llmops:ingest:jobs:dead"


def _redis_url() -> str | None:
    try:
        from llmops_core.common.config import get_settings
        url = get_settings().gateway.redis_url
    except Exception:  # noqa: BLE001
        url = None
    return url or os.environ.get("REDIS_URL") or "redis://localhost:6379/0"


class JobQueue:
    def __init__(self) -> None:
        self._r = None
        self._local: deque[str] = deque()
        url = _redis_url()
        if url:
            try:
                import redis
                self._r = redis.Redis.from_url(url, decode_responses=True)
                self._r.ping()
            except Exception:  # noqa: BLE001 — redis 없으면 로컬 폴백
                self._r = None

    @property
    def backend(self) -> str:
        return "redis" if self._r is not None else "memory"

    def enqueue(self, doc_id: str) -> None:
        if self._r is not None:
            self._r.lpush(_QUEUE, doc_id)
        else:
            self._local.appendleft(doc_id)

    def dequeue(self, timeout: int = 5) -> str | None:
        if self._r is not None:
            try:
                item = self._r.brpop(_QUEUE, timeout=timeout)
            except Exception:  # noqa: BLE001 — 소켓 타임아웃/일시 단절은 '없음'으로 취급
                return None
            return item[1] if item else None
        return self._local.pop() if self._local else None

    def dead_letter(self, doc_id: str, error: str) -> None:
        if self._r is not None:
            self._r.lpush(_DEADLETTER, f"{doc_id}\t{error[:500]}")

    def depth(self) -> int:
        return self._r.llen(_QUEUE) if self._r is not None else len(self._local)


_queue: JobQueue | None = None


def queue() -> JobQueue:
    global _queue
    if _queue is None:
        _queue = JobQueue()
    return _queue

"""마이크로배칭 — 동시 도착 요청을 짧은 윈도우로 모아 한 번에 generate (GPU 활용 향상).

transformers generate는 요청당 1회 호출이 기본이라 동시성에서 GPU가 논다. 본 큐는
짧은 시간(window_ms) 또는 max_batch개가 모이면 묶어서 배치 생성한다(좌측 패딩).
프로덕션 처리량은 vLLM(PagedAttention/연속배칭)이 상한이며, 본 모듈은 그 전 단계의
저비용 개선이다. 동일 생성 파라미터(max_tokens 등)끼리만 한 배치로 묶는다.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class _Req:
    messages: list
    max_tokens: int
    temperature: float
    top_p: float
    event: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None


class MicroBatcher:
    """백그라운드 워커가 큐를 비우며 배치 생성. submit()은 결과까지 블록."""

    def __init__(self, generate_batch, *, window_ms: int = 20, max_batch: int = 8) -> None:
        # generate_batch(list[_Req]) -> list[dict]  (호출자가 모델 의존성 주입)
        self._generate_batch = generate_batch
        self.window = window_ms / 1000.0
        self.max_batch = max_batch
        self._q: list[_Req] = []
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._stop = False
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def submit(self, req: _Req, timeout: float = 120.0) -> dict:
        with self._cv:
            self._q.append(req)
            self._cv.notify()
        if not req.event.wait(timeout):
            raise TimeoutError("배치 생성 타임아웃")
        return req.result

    def _take_batch(self) -> list[_Req]:
        # 첫 요청을 기다린다(락 보유)
        with self._cv:
            while not self._q and not self._stop:
                self._cv.wait()
            if self._stop:
                return []
        # 윈도우 동안 락을 풀어 동시 요청이 큐에 합류하도록 한다(배칭의 핵심)
        time.sleep(self.window)
        with self._cv:
            if not self._q:
                return []
            # 동일 max_tokens끼리 묶어 패딩 낭비 최소화
            head = self._q[0]
            batch, rest = [], []
            for r in self._q:
                if len(batch) < self.max_batch and r.max_tokens == head.max_tokens:
                    batch.append(r)
                else:
                    rest.append(r)
            self._q = rest
            return batch

    def _loop(self) -> None:
        while not self._stop:
            batch = self._take_batch()
            if not batch:
                continue
            try:
                results = self._generate_batch(batch)
            except Exception as exc:  # noqa: BLE001
                results = [{"error": str(exc)} for _ in batch]
            for r, out in zip(batch, results):
                r.result = out
                r.event.set()

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()


def make_request(messages, max_tokens, temperature, top_p) -> _Req:
    return _Req(messages=messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

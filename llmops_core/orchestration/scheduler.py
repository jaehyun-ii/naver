"""GPU 인지 잡 스케줄러 — 쿠버네티스 없이 Docker로 다중 GPU/노드 스케줄링.

각 노드(로컬 소켓 또는 원격 docker_host)의 GPU 디바이스를 풀로 관리하고, 잡이 요청한
GPU 수만큼 비어있는 디바이스를 한 노드에서 배정한다(없으면 대기 큐). 동시 실행 = 가용 GPU 수.

- 단일 H200×8 : 노드 1개(gpus=8) → 최대 8개 1-GPU 잡 또는 1개 8-GPU 잡 동시 실행.
- 2노드 H200×4 : 노드 2개(gpus=4) → 노드별로 배정, 원격 노드는 docker -H 로 디스패치.

프로세스 로컬 스케줄러(단일 오케스트레이터). 다중 오케스트레이터는 DB 락이 필요(향후).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from llmops_core.common.config import get_settings


@dataclass
class Lease:
    """배정 결과 — 어느 노드의 어느 디바이스를 점유했는지."""

    node: str
    docker_host: str | None
    devices: list[int]

    @property
    def device_arg(self) -> str:
        """docker --gpus 값. 예: device=0,1 (셸 없이 subprocess에 그대로 전달)."""
        return "device=" + ",".join(str(d) for d in self.devices)


@dataclass
class _Node:
    name: str
    docker_host: str | None
    free: list[int]  # 비어있는 디바이스 id
    total: int


class GpuScheduler:
    """노드별 GPU 디바이스 풀 + acquire/release. 스레드세이프, 부족 시 대기."""

    def __init__(self, nodes: list | None = None) -> None:
        cfg_nodes = nodes if nodes is not None else get_settings().orch.nodes
        self._nodes: list[_Node] = []
        for n in cfg_nodes:
            devs = list(n.devices) if getattr(n, "devices", None) else list(range(n.gpus))
            self._nodes.append(_Node(n.name, n.docker_host, devs, len(devs)))
        self._lock = threading.Condition()
        self._held: dict[str, Lease] = {}  # 장기 점유(서빙 등) tag→Lease

    @property
    def total_gpus(self) -> int:
        return sum(n.total for n in self._nodes)

    def _try_assign(self, n_gpus: int) -> Lease | None:
        for node in self._nodes:
            if len(node.free) >= n_gpus:
                devices = sorted(node.free)[:n_gpus]
                for d in devices:
                    node.free.remove(d)
                return Lease(node.name, node.docker_host, devices)
        return None

    def acquire(self, n_gpus: int = 1, *, timeout: float | None = None) -> Lease:
        """n_gpus개를 한 노드에서 배정. 없으면 대기(타임아웃 시 TimeoutError)."""
        if n_gpus < 1:
            raise ValueError("n_gpus는 1 이상")
        max_cap = max((n.total for n in self._nodes), default=0)
        if n_gpus > max_cap:
            raise ValueError(f"요청 GPU {n_gpus} > 최대 노드 용량 {max_cap}")
        timeout = get_settings().orch.acquire_timeout_s if timeout is None else timeout
        with self._lock:
            lease = self._try_assign(n_gpus)
            if lease is not None:
                return lease
            if not self._lock.wait_for(
                lambda: (self._peek(n_gpus)), timeout=timeout
            ):
                raise TimeoutError(f"GPU {n_gpus}개 대기 타임아웃({timeout}s)")
            lease = self._try_assign(n_gpus)
            assert lease is not None
            return lease

    def _peek(self, n_gpus: int) -> bool:
        return any(len(n.free) >= n_gpus for n in self._nodes)

    def release(self, lease: Lease) -> None:
        with self._lock:
            for node in self._nodes:
                if node.name == lease.node:
                    node.free.extend(lease.devices)
                    node.free = sorted(set(node.free))
                    break
            self._lock.notify_all()

    def hold(self, tag: str, n_gpus: int = 1) -> Lease:
        """장기 점유(서빙) — 같은 tag의 기존 lease를 해제하고 새로 배정·보관."""
        self.release_tag(tag)
        lease = self.acquire(n_gpus)
        with self._lock:
            self._held[tag] = lease
        return lease

    def release_tag(self, tag: str) -> None:
        with self._lock:
            lease = self._held.pop(tag, None)
        if lease is not None:
            self.release(lease)

    def get_held(self, tag: str) -> Lease | None:
        with self._lock:
            return self._held.get(tag)

    def snapshot(self) -> list[dict]:
        """대시보드용 노드별 점유 현황."""
        with self._lock:
            return [
                {"node": n.name, "docker_host": n.docker_host or "local",
                 "total": n.total, "free": len(n.free), "busy": n.total - len(n.free)}
                for n in self._nodes
            ]


_scheduler: GpuScheduler | None = None
_sched_lock = threading.Lock()


def scheduler() -> GpuScheduler:
    """프로세스 단일 스케줄러."""
    global _scheduler
    if _scheduler is None:
        with _sched_lock:
            if _scheduler is None:
                _scheduler = GpuScheduler()
    return _scheduler

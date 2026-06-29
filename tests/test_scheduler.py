"""GPU 스케줄러 테스트 — 다중 GPU/노드 할당·디바이스 핀닝·큐·서빙 영구점유 (GPU 불필요)."""

from __future__ import annotations

import threading
import time

import pytest

from llmops_core.common.config import GpuNode
from llmops_core.orchestration.scheduler import GpuScheduler


def test_single_node_8gpu_assigns_distinct_devices():
    # H200x8 단일 노드
    s = GpuScheduler([GpuNode(name="local", gpus=8)])
    assert s.total_gpus == 8
    leases = [s.acquire(1) for _ in range(8)]
    devs = sorted(d for ls in leases for d in ls.devices)
    assert devs == list(range(8))  # 8개 잡이 서로 다른 디바이스
    # 9번째는 즉시 불가 → 타임아웃
    with pytest.raises(TimeoutError):
        s.acquire(1, timeout=0.2)
    s.release(leases[0])
    again = s.acquire(1, timeout=0.5)  # 반납 후 재배정
    assert again.devices == leases[0].devices


def test_explicit_devices_for_shared_server():
    # 공용 서버: 빈 GPU 4,5,6만 사용
    s = GpuScheduler([GpuNode(name="local", devices=[4, 5, 6])])
    assert s.total_gpus == 3
    leases = [s.acquire(1) for _ in range(3)]
    assert sorted(d for ls in leases for d in ls.devices) == [4, 5, 6]
    assert leases[0].device_arg in ("device=4", "device=5", "device=6")
    with pytest.raises(TimeoutError):
        s.acquire(1, timeout=0.2)  # 4번째 불가


def test_multi_gpu_job_device_arg():
    s = GpuScheduler([GpuNode(name="local", gpus=8)])
    lease = s.acquire(4)  # 4-GPU 잡
    assert len(lease.devices) == 4
    assert lease.device_arg == "device=0,1,2,3"


def test_two_node_4gpu_each_routes_per_node():
    # 2노드 H200x4 (4+4)
    s = GpuScheduler([
        GpuNode(name="a", docker_host="tcp://10.0.0.1:2375", gpus=4),
        GpuNode(name="b", docker_host="tcp://10.0.0.2:2375", gpus=4),
    ])
    # 노드 a를 가득 채우면 다음은 노드 b로
    a = [s.acquire(1) for _ in range(4)]
    assert all(ls.node == "a" for ls in a)
    b = s.acquire(1)
    assert b.node == "b" and b.docker_host == "tcp://10.0.0.2:2375"


def test_request_exceeding_node_capacity_errors():
    s = GpuScheduler([GpuNode(name="local", gpus=4)])
    with pytest.raises(ValueError):
        s.acquire(8)  # 단일 노드 용량 초과


def test_queue_blocks_until_release():
    s = GpuScheduler([GpuNode(name="local", gpus=1)])
    held = s.acquire(1)
    got = {}

    def waiter():
        lease = s.acquire(1, timeout=5)  # 비워질 때까지 대기
        got["dev"] = lease.devices

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.2)
    assert "dev" not in got  # 아직 대기 중
    s.release(held)  # 반납 → 대기자 진행
    t.join(timeout=3)
    assert got.get("dev") == [0]


def test_serving_hold_persists_and_frees_pool():
    s = GpuScheduler([GpuNode(name="local", gpus=2)])
    s.hold("serve:tuned", 1)  # 서빙이 1개 영구 점유
    snap = {n["node"]: n for n in s.snapshot()}["local"]
    assert snap["busy"] == 1 and snap["free"] == 1
    # 재배포: 같은 tag → 이전 해제 후 재점유(순증 없음)
    s.hold("serve:tuned", 1)
    snap = {n["node"]: n for n in s.snapshot()}["local"]
    assert snap["busy"] == 1
    s.release_tag("serve:tuned")
    snap = {n["node"]: n for n in s.snapshot()}["local"]
    assert snap["busy"] == 0

"""인프라 대시보드 — GPU 자원·서비스 상태·온프렘/NCP 하이브리드 스토리지 (콘솔 UI).

GPU 자원 최적화/운영 가시성과 온프렘-NCP 하이브리드 전환 상태를 한 화면에 노출한다.
"""

from __future__ import annotations

import subprocess

from fastapi import APIRouter, Depends

from llmops_core.common.config import get_settings
from llmops_core.console.security import require_perm

router = APIRouter(prefix="/api/infra", tags=["infra"], dependencies=[Depends(require_perm("read"))])


def _gpus() -> list[dict]:
    """nvidia-smi 질의 → GPU 목록(이름·메모리·사용률·온도). 실패 시 빈 리스트."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
        )
        if out.returncode != 0:
            return []
        gpus = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "name": parts[0], "mem_used_mb": _num(parts[1]),
                    "mem_total_mb": _num(parts[2]), "util_pct": _num(parts[3]),
                    "temp_c": _num(parts[4]),
                })
        return gpus
    except Exception:  # noqa: BLE001
        return []


def _num(s: str):
    try:
        return float(s) if "." in s else int(s)
    except ValueError:
        return s


def _docker_services() -> list[dict]:
    """관련 컨테이너 상태(있으면). docker 미접근 시 빈 리스트."""
    try:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}"],
            capture_output=True, text=True, timeout=8,
        )
        if out.returncode != 0:
            return []
        rows = []
        for line in out.stdout.strip().splitlines():
            if "\t" not in line:
                continue
            name, status = line.split("\t", 1)
            if any(k in name for k in ("llmops", "minio", "mlflow", "qdrant", "jaeger",
                                       "postgres", "serving", "otel")):
                rows.append({"name": name, "status": status})
        return rows
    except Exception:  # noqa: BLE001
        return []


def _scheduler_state() -> list[dict]:
    try:
        from llmops_core.orchestration.scheduler import scheduler

        return scheduler().snapshot()
    except Exception:  # noqa: BLE001
        return []


@router.get("")
def infra() -> dict:
    s = get_settings()
    return {
        "gpus": _gpus(),
        "scheduler": _scheduler_state(),
        "services": _docker_services(),
        "storage": {
            "provider": s.s3.provider,
            "endpoint": s.s3.effective_endpoint(),
            "hybrid_options": ["minio", "ncp"],
        },
        "env": {"env": s.env, "domain": s.domain},
        "optimization": {
            "precision": "bf16", "pet": ["lora", "dora"],
            "multi_lora_serving": True, "micro_batching": True,
            "throughput_path": "vLLM(NGC) — seam 동일",
        },
    }

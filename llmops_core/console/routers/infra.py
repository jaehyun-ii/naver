"""인프라 대시보드 — GPU 자원·서비스 상태·온프렘/NCP 하이브리드 스토리지 (콘솔 UI).

GPU 자원 최적화/운영 가시성과 온프렘-NCP 하이브리드 전환 상태를 한 화면에 노출한다.
"""

from __future__ import annotations

import logging
import subprocess

from fastapi import APIRouter, Depends

from llmops_core.common.config import get_settings
from llmops_core.console.security import require_perm

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/infra", tags=["infra"], dependencies=[Depends(require_perm("read"))])

# 주의(prod 경로): nvidia-smi / `docker ps` 스크레이프는 콘솔이 GPU 호스트에 로컬일 때만 유효한
# 데모/온호스트 편의 경로다. 운영에서는 DCGM/Prometheus(노드 exporter)로 GPU 텔레메트리를,
# 오케스트레이터 API(스케줄러/쿠버네티스 등)로 서비스 상태를 수집하는 것이 정식 경로다.


def _gpus() -> tuple[list[dict], str | None]:
    """nvidia-smi 질의 → (GPU 목록, 에러사유|None). 실패를 조용히 [] 로 삼키지 않고 플래그로 노출."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
        )
        if out.returncode != 0:
            err = (out.stderr or "").strip() or f"nvidia-smi rc={out.returncode}"
            logger.warning("nvidia-smi 실패: %s", err)
            return [], err
        gpus = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "name": parts[0], "mem_used_mb": _num(parts[1]),
                    "mem_total_mb": _num(parts[2]), "util_pct": _num(parts[3]),
                    "temp_c": _num(parts[4]),
                })
        return gpus, None
    except FileNotFoundError:
        # GPU 없는 호스트(콘솔 전용 노드) — 정상 상황이므로 info 수준.
        logger.info("nvidia-smi 미설치 — GPU 텔레메트리 없음")
        return [], "nvidia-smi not found"
    except Exception as exc:  # noqa: BLE001
        logger.warning("GPU 조회 실패: %s", exc, exc_info=True)
        return [], str(exc)


def _num(s: str):
    try:
        return float(s) if "." in s else int(s)
    except ValueError:
        return s


def _docker_services() -> tuple[list[dict], str | None]:
    """관련 컨테이너 상태 → (목록, 에러사유|None). docker 미접근을 조용히 [] 로 삼키지 않는다."""
    try:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}"],
            capture_output=True, text=True, timeout=8,
        )
        if out.returncode != 0:
            err = (out.stderr or "").strip() or f"docker ps rc={out.returncode}"
            logger.warning("docker ps 실패: %s", err)
            return [], err
        rows = []
        for line in out.stdout.strip().splitlines():
            if "\t" not in line:
                continue
            name, status = line.split("\t", 1)
            if any(k in name for k in ("llmops", "minio", "mlflow", "qdrant", "jaeger",
                                       "postgres", "serving", "otel")):
                rows.append({"name": name, "status": status})
        return rows, None
    except FileNotFoundError:
        logger.info("docker CLI 미설치 — 컨테이너 상태 수집 불가")
        return [], "docker not found"
    except Exception as exc:  # noqa: BLE001
        logger.warning("docker ps 조회 실패: %s", exc, exc_info=True)
        return [], str(exc)


def _scheduler_state() -> list[dict]:
    try:
        from llmops_core.orchestration.scheduler import scheduler

        return scheduler().snapshot()
    except Exception:  # noqa: BLE001
        return []


@router.get("")
def infra() -> dict:
    s = get_settings()
    gpus, gpus_error = _gpus()
    services, services_error = _docker_services()
    return {
        "gpus": gpus,
        "gpus_error": gpus_error,  # None이면 정상. 값이 있으면 스크레이프 실패(빈 목록과 구분).
        "scheduler": _scheduler_state(),
        "services": services,
        "services_error": services_error,
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

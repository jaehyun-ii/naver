"""MLflow 통합 뷰 — 실험·런·메트릭·모델 레지스트리를 메인 콘솔에서 조회(읽기전용 프록시).

MLflow UI를 따로 띄우지 않고 MlflowClient로 핵심 뷰를 콘솔 스타일로 렌더. 미도달 시 graceful.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from llmops_core.common.config import get_settings
from llmops_core.console.security import require_perm

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tracking", tags=["tracking"],
                   dependencies=[Depends(require_perm("read"))])


def _client():
    import mlflow

    mlflow.set_tracking_uri(get_settings().mlflow.tracking_uri)
    return mlflow.tracking.MlflowClient()


@router.get("/experiments")
def experiments() -> dict:
    try:
        c = _client()
        exps = c.search_experiments()
        return {"available": True, "tracking_uri": get_settings().mlflow.tracking_uri,
                "experiments": [{"id": e.experiment_id, "name": e.name,
                                 "stage": e.lifecycle_stage} for e in exps]}
    except Exception as exc:  # noqa: BLE001
        # 백엔드 미도달/오류를 "데이터 없음"으로 착각하지 않도록 로깅 + error 사유 노출.
        logger.warning("MLflow experiments 조회 실패: %s", exc, exc_info=True)
        return {"available": False, "error": str(exc)}


@router.get("/runs")
def runs(experiment_id: str, limit: int = 20) -> dict:
    try:
        c = _client()
        rs = c.search_runs([experiment_id], max_results=limit,
                           order_by=["attributes.start_time DESC"])
        out = []
        for r in rs:
            out.append({
                "run_id": r.info.run_id, "status": r.info.status,
                "start_time": r.info.start_time,
                "metrics": dict(r.data.metrics), "params": dict(r.data.params),
                "tags": {k: v for k, v in r.data.tags.items()
                         if not k.startswith("mlflow.")},
            })
        return {"available": True, "runs": out}
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow runs 조회 실패(experiment=%s): %s", experiment_id, exc,
                       exc_info=True)
        return {"available": False, "error": str(exc)}


@router.get("/models")
def models() -> dict:
    try:
        c = _client()
        out = []
        for m in c.search_registered_models():
            versions = []
            for v in (m.latest_versions or []):
                versions.append({"version": v.version, "stage": v.current_stage,
                                 "run_id": v.run_id})
            aliases = dict(getattr(m, "aliases", {}) or {})
            out.append({"name": m.name, "versions": versions, "aliases": aliases})
        return {"available": True, "models": out}
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow registered models 조회 실패: %s", exc, exc_info=True)
        return {"available": False, "error": str(exc)}

"""실험추적·데이터버전·레지스트리 — mlflow client + dvc.api 임베드.

MLflow는 클라이언트만 임베드(Tracking 서버는 Service). 핵심 글루:
데이터 버전 해시 ↔ run ↔ 모델 버전을 한 사슬로 연결하고, 승격 정책을 보유한다.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from llmops_core.common.config import get_settings
from llmops_core.common.errors import OptionalDependencyError
from llmops_core.common.schemas import EvalResult, ModelStage, ModelVersionRef


def _mlflow():
    try:
        import mlflow

        mlflow.set_tracking_uri(get_settings().mlflow.tracking_uri)
        return mlflow
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("mlflow", "tracking") from exc


def data_version(path: str, repo: str | None = None) -> str:
    """DVC로 추적되는 데이터셋의 버전 식별자(URL/해시)를 가져온다."""
    try:
        import dvc.api
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("dvc", "tracking") from exc
    return dvc.api.get_url(path, repo=repo)


class ExperimentTracker:
    """학습/평가 run 기록 + 모델 레지스트리 승격 정책."""

    def __init__(self, experiment: str = "llmops") -> None:
        self.mlflow = _mlflow()
        self.mlflow.set_experiment(experiment)

    @contextmanager
    def run(self, params: dict, *, data_ver: str | None = None) -> Iterator:
        """run 컨텍스트. 데이터 버전 해시를 태그로 박아 사슬을 만든다."""
        with self.mlflow.start_run() as active:
            self.mlflow.log_params(params)
            if data_ver:
                self.mlflow.set_tag("data_version", data_ver)
            yield active

    def log_eval(self, result: EvalResult) -> None:
        """평가 하니스 결과를 메트릭으로 기록 (데이터 버전별 비교 가능)."""
        for name, value in result.metrics.items():
            self.mlflow.log_metric(f"eval.{name}", value)
        self.mlflow.set_tag("eval.passed", str(result.passed))
        if result.data_version:
            self.mlflow.set_tag("data_version", result.data_version)

    def register(
        self, model_uri: str, name: str, *, tenant: str | None = None
    ) -> ModelVersionRef:
        mv = self.mlflow.register_model(model_uri, name)
        client = self.mlflow.tracking.MlflowClient()
        if tenant:
            client.set_model_version_tag(name, mv.version, "tenant", tenant)
        return ModelVersionRef(
            name=name, version=str(mv.version), stage=ModelStage.NONE, tenant=tenant
        )

    def promote(self, ref: ModelVersionRef, stage: ModelStage) -> ModelVersionRef:
        """승격 정책(Staging→Production). 평가 게이트 통과가 전제."""
        client = self.mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(ref.name, ref.version, stage.value)
        return ref.model_copy(update={"stage": stage})

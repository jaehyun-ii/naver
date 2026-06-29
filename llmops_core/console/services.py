"""콘솔 서비스 컨테이너 — 제어평면 상태의 단일 출처(프로세스 싱글톤).

게이트웨이 코어 컴포넌트(키스토어·정책·릴리스 게이트웨이·프롬프트·데이터셋)를 한 곳에 모아
콘솔 API가 공유한다. 개발/데모는 인메모리, 운영은 각 스토어를 Postgres/Redis로 교체.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from llmops_core.common.config import get_settings
from llmops_core.common.schemas import DatasetManifest
from llmops_core.gateway.keys import InMemoryKeyStore, VirtualKeyStore
from llmops_core.gateway.policy import PolicyEngine
from llmops_core.governance import ReleaseGateway
from llmops_core.prompts import GitPromptStore


class DatasetRegistry:
    """L3 데이터셋 매니페스트의 경량 인메모리 레지스트리(콘솔 조회용)."""

    def __init__(self) -> None:
        self._items: list[DatasetManifest] = []

    def add(self, manifest: DatasetManifest) -> None:
        self._items.insert(0, manifest)  # 최신 우선

    def list(self) -> list[DatasetManifest]:
        return list(self._items)


def load_logical_models() -> list[str]:
    """config/model_list.yaml에서 논리 모델명 로드 (litellm 불필요)."""
    path = Path(get_settings().gateway.config_path)
    if not path.exists():
        return []
    data: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    return sorted({m["model_name"] for m in data.get("model_list", [])})


class ConsoleServices:
    """제어평면 상태 컨테이너. backend=postgres면 키·승인·예산·런이 Postgres에 영속(HA)."""

    def __init__(self) -> None:
        from llmops_core.common.schemas import DatasetManifest
        from llmops_core.common.security import make_audit_log, make_token_store
        from llmops_core.common.stores import (
            make_key_store,
            make_ledger,
            make_registry,
            make_release_store,
        )
        from llmops_core.console.schemas import PipelineRun

        self.key_store: VirtualKeyStore = make_key_store()
        self.policy = PolicyEngine(ledger=make_ledger())
        self.releases = ReleaseGateway(store=make_release_store())
        self.prompts = GitPromptStore()
        self.tokens = make_token_store()  # RBAC API 토큰
        self.audit = make_audit_log()  # 감사로그
        # 영속 가능한 레지스트리(파이프라인 런·HPO·데이터셋)
        self.pipelines = make_registry(
            "pipeline",
            id_of=lambda r: r.id, created_of=lambda r: r.created_at,
            to_payload=lambda r: r.model_dump(),
            from_payload=lambda p: PipelineRun.model_validate(p),
        )
        self.hpo = make_registry(
            "hpo",
            id_of=lambda d: d["id"], created_of=lambda d: d.get("created_at"),
            to_payload=lambda d: d, from_payload=lambda p: p,
        )
        self.datasets = make_registry(
            "dataset",
            id_of=lambda m: m.fingerprint, created_of=lambda m: None,
            to_payload=lambda m: m.model_dump(),
            from_payload=lambda p: DatasetManifest.model_validate(p),
        )
        # 드리프트 기준선(분포 요약) — 이름 키, dict 영속
        self.drift_baselines = make_registry(
            "drift_baseline",
            id_of=lambda d: d["name"], created_of=lambda d: d.get("created_at"),
            to_payload=lambda d: d, from_payload=lambda p: p,
        )


_services: ConsoleServices | None = None


def services() -> ConsoleServices:
    global _services
    if _services is None:
        _services = ConsoleServices()
    return _services

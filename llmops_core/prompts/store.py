"""Git-backed 경량 프롬프트 스토어 (옵션 B — 자체 구현).

프롬프트를 코드가 아니라 버전 자산으로 관리한다: 버전·라벨(prod/stg)·롤백.
- 각 프롬프트: prompts 디렉토리 아래 `<name>/v<N>.txt` (불변 버전)
- 라벨: `<name>/labels.json` 이 라벨→버전 매핑 (prod/stg 포인터)
애플리케이션은 라벨만 참조하므로 코드 배포 없이 무중단 프롬프트 교체가 가능하다.

옵션 A(Langfuse 제품) 채택 시 이 모듈 대신 langfuse Prompt API를 쓰면 되며,
인터페이스(get/by_label)는 동일하게 유지해 교체 가능하게 한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from llmops_core.common.config import get_settings
from llmops_core.common.errors import LLMOpsError


class PromptNotFound(LLMOpsError):
    pass


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    label: str | None
    template: str

    def render(self, **vars: object) -> str:
        """단순 `{var}` 치환 (langchain_core.prompts와 호환되는 입력으로도 사용 가능)."""
        return self.template.format(**vars)


class GitPromptStore:
    """파일시스템(Git 워킹트리) 기반 프롬프트 스토어.

    커밋/푸시는 운영자가 Git으로 수행 → 변경 이력·롤백이 Git에 남는다.
    """

    def __init__(self, repo_path: str | None = None) -> None:
        self.root = Path(repo_path or get_settings().prompts.repo_path)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, name: str) -> Path:
        return self.root / name

    def _labels_file(self, name: str) -> Path:
        return self._dir(name) / "labels.json"

    def _read_labels(self, name: str) -> dict[str, int]:
        f = self._labels_file(name)
        return json.loads(f.read_text()) if f.exists() else {}

    def _write_labels(self, name: str, labels: dict[str, int]) -> None:
        self._labels_file(name).write_text(json.dumps(labels, indent=2))

    def versions(self, name: str) -> list[int]:
        d = self._dir(name)
        if not d.exists():
            return []
        return sorted(
            int(p.stem[1:]) for p in d.glob("v*.txt") if p.stem[1:].isdigit()
        )

    def create_version(self, name: str, template: str) -> Prompt:
        """새 불변 버전을 추가. 라벨은 건드리지 않는다(검증 후 promote)."""
        d = self._dir(name)
        d.mkdir(parents=True, exist_ok=True)
        next_v = (self.versions(name)[-1] + 1) if self.versions(name) else 1
        (d / f"v{next_v}.txt").write_text(template)
        return Prompt(name=name, version=next_v, label=None, template=template)

    def get(self, name: str, version: int) -> Prompt:
        f = self._dir(name) / f"v{version}.txt"
        if not f.exists():
            raise PromptNotFound(f"{name} v{version} 없음")
        return Prompt(name, version, None, f.read_text())

    def by_label(self, name: str, label: str = "prod") -> Prompt:
        """애플리케이션 런타임 진입점: 라벨로 프롬프트를 가져온다."""
        labels = self._read_labels(name)
        if label not in labels:
            raise PromptNotFound(f"{name} 라벨 '{label}' 없음")
        v = labels[label]
        p = self.get(name, v)
        return Prompt(p.name, p.version, label, p.template)

    def promote(self, name: str, version: int, label: str) -> None:
        """버전을 라벨에 연결 (예: stg 검증 통과 → prod 승격). 롤백도 같은 호출로."""
        if version not in self.versions(name):
            raise PromptNotFound(f"{name} v{version} 없음")
        labels = self._read_labels(name)
        labels[label] = version
        self._write_labels(name, labels)

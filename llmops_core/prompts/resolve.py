"""프롬프트 해석기 — 이름으로 GitPromptStore의 prod 버전을 가져오되, 없으면 기본값.

가드레일·RAG 등 시스템 프롬프트도 코드 하드코딩이 아니라 버전 자산으로 관리하기 위함.
스토어에 해당 이름/라벨이 등록돼 있으면 그것을, 없으면 내장 기본 템플릿을 쓴다(graceful).
"""

from __future__ import annotations


def resolve_template(name: str | None, default: str, label: str = "prod") -> str:
    if not name:
        return default
    try:
        from llmops_core.prompts.store import GitPromptStore

        return GitPromptStore().by_label(name, label).template
    except Exception:  # noqa: BLE001 — 미등록/스토어 부재 시 기본값
        return default

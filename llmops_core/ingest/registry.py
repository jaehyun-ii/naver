"""원본 PDF 레지스트리 — 문서 메타의 단일 소스.

`common/stores.make_registry` 위에 구축(backend=postgres면 영속, dev면 인메모리).
문서 블롭 자체는 MinIO(documents-raw)에, 여기서는 메타·상태만 관리한다.
"""

from __future__ import annotations

from llmops_core.common.stores import make_registry

from .models import Document


class DocumentRegistry:
    def __init__(self) -> None:
        self._reg = make_registry(
            "document",
            id_of=lambda d: d.id,
            to_payload=lambda d: d.to_payload(),
            from_payload=Document.from_payload,
            created_of=lambda d: d.created_at,
        )

    def get(self, doc_id: str) -> Document | None:
        return self._reg.get(doc_id)

    def save(self, doc: Document) -> None:
        self._reg.save(doc)

    def list(self) -> list[Document]:
        return self._reg.list()


_registry: DocumentRegistry | None = None


def registry() -> DocumentRegistry:
    global _registry
    if _registry is None:
        _registry = DocumentRegistry()
    return _registry

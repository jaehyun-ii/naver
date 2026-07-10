"""문서 인제스트 도메인 모델 — 원본 PDF 레지스트리 레코드.

원본 PDF DB의 한 행. MinIO(블롭) + 레지스트리(메타)로 구성되며, 생성/수정은
sha256 기준 dedup·버전 관리한다. status 전이: pending → processing →
processed | failed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_PROCESSED = "processed"
STATUS_FAILED = "failed"


@dataclass
class Document:
    id: str                       # 안정 식별자(이름 기반) — 버전 넘어 동일
    name: str                     # 사람이 읽는 문서명(= 산출물 프리픽스)
    sha256: str                   # 원본 PDF 내용 해시(변경 감지 키)
    raw_key: str                  # MinIO documents-raw 내 키
    family: str = "auto"          # kr_rule | abs_guide | auto (청커 디스패치)
    version: int = 1              # sha256 바뀔 때마다 증가
    status: str = STATUS_PENDING
    content_list_key: str | None = None   # 후처리된 content_list.json (processed 버킷)
    chunks_key: str | None = None         # chunks.jsonl (processed 버킷)
    vector_collection: str | None = None  # Qdrant 컬렉션명
    n_chunks: int = 0
    n_vectors: int = 0
    error: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_payload(self) -> dict:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict) -> "Document":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})

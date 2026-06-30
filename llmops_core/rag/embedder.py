"""임베더 — 텍스트→벡터. bge-m3(캐시됨) 또는 순수파이썬 해시 폴백.

해시 임베더는 의존성·GPU 없이 결정적 벡터를 만든다(테스트·오프라인 베이스라인).
실서비스 품질은 bge-m3(sentence-transformers, HF 캐시에 존재)를 쓴다.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from llmops_core.common.config import get_settings
from llmops_core.common.errors import OptionalDependencyError

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> list[list[float]]: ...


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class HashingEmbedder:
    """해싱 트릭 임베더 — 토큰을 dim 버킷에 해시·집계 후 L2 정규화. 순수파이썬·결정적."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            vec = [0.0] * self.dim
            for tok in _TOKEN_RE.findall(t.lower()):
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 8) & 1 else -1.0  # 부호 해싱(충돌 완화)
                vec[idx] += sign
            out.append(_l2_normalize(vec))
        return out


class SentenceTransformerEmbedder:
    """bge-m3 등 sentence-transformers 모델 임베더(lazy). 정규화 임베딩 반환."""

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise OptionalDependencyError("sentence-transformers", "rag") from exc
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


def make_embedder() -> Embedder:
    """설정(rag.embedder)에 따라 임베더 생성. 기본은 무의존 해시 임베더."""
    cfg = get_settings().rag
    if cfg.embedder == "bge-m3":
        return SentenceTransformerEmbedder(cfg.embedding_model)
    return HashingEmbedder(cfg.dim)

"""Qdrant 실검색 헬퍼 — RAFT distractor·no-golden 실측용 (build_training에서 사용).

실서빙과 같은 인덱스(classification_rules_full: data_chunks_full 기반 bge-m3
child 청크, cosine)를 질의해 **조(parent) 단위**로 집계한다. 학습 입력의 검색
문서 분포를 서빙 리트리버와 일치시키는 것이 목적.

동등 요건 제외: 타 선급의 동일 요건 조항이 distractor로 들어가면 no-golden
라벨이 거짓이 된다. cluster_map.jsonl은 구 청킹 id(무접두 RULE_*)라 현 코퍼스와
조인 불가 — 대신 골든 조항 텍스트로 재검색해 유사도가 클러스터 임계값
(cluster_rules thr 0.85)을 넘는 조를 제외한다(인덱스와 항상 정합).

임베더(bge-m3)는 lazy 싱글턴 — Qdrant 미가동·의존성 부재 시 available()이
False를 반환하고 호출측(build_training)은 휴리스틱 폴백을 쓴다.
"""
from __future__ import annotations

import os
from functools import lru_cache

import requests

QDRANT_URL = os.environ.get("AIREG_QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("AIREG_QDRANT_COLLECTION", "classification_rules_full")
# 동등 요건 제외 임계값. 클러스터링(cluster_rules)은 0.85(0.80은 연쇄 병합 붕괴)지만,
# 여기는 골든 기준 pairwise 판단이라 연쇄 문제가 없고 과제외 비용이 distractor 후보
# 몇 개 손실뿐이다 — 실측 준동등(주조타장치 능력 쌍 0.834)을 걸러내도록 0.80.
EQUIV_THR = float(os.environ.get("AIREG_EQUIV_THR", "0.80"))
# child 오버페치 배수 — 한 조의 child 여러 개가 상위를 점유해도 조 수를 채우도록
CHILD_OVERSAMPLE = 4

_embedder = None


def _get_embedder():
    """bge-m3 직접 로드 — llmops_core 설정 체인(pydantic) 의존을 피한다."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("BAAI/bge-m3")
    return _embedder


def embed(text: str) -> list[float]:
    v = _get_embedder().encode([text[:4000]], normalize_embeddings=True)[0]
    return [float(x) for x in v]


@lru_cache(maxsize=1)
def available() -> bool:
    """Qdrant 컬렉션 존재 + sentence-transformers 임포트 가능 여부."""
    try:
        r = requests.get(f"{QDRANT_URL}/collections/{COLLECTION}", timeout=5)
        if r.status_code != 200:
            return False
        import sentence_transformers  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def search_parents(query: str, k: int = 20, *, publisher: str | None = None,
                   query_vec: list[float] | None = None) -> list[dict]:
    """child 검색 → parent 조 단위 최고점 집계. [{parent_chunk_id, score, ...}]."""
    vec = query_vec if query_vec is not None else embed(query)
    body: dict = {
        "vector": vec, "limit": k * CHILD_OVERSAMPLE,
        "with_payload": ["parent_chunk_id", "chunk_id", "publisher", "doc_id"],
    }
    if publisher:
        body["filter"] = {"must": [{"key": "publisher", "match": {"value": publisher}}]}
    r = requests.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
                      json=body, timeout=30)
    r.raise_for_status()
    out: list[dict] = []
    seen: set[str] = set()
    for p in r.json()["result"]:
        pl = p.get("payload") or {}
        pid = pl.get("parent_chunk_id") or pl.get("chunk_id")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append({"parent_chunk_id": pid, "score": p["score"],
                    "publisher": pl.get("publisher"), "doc_id": pl.get("doc_id")})
        if len(out) >= k:
            break
    return out


def equivalent_parents(article_text: str, self_parent_id: str,
                       k: int = 40) -> set[str]:
    """골든 조항과 동등(유사도 ≥ EQUIV_THR)한 조의 parent_chunk_id 집합.

    골든 원문 자체를 질의로 써서 인덱스에서 근접 조를 찾는다 — 자기 자신 포함.
    """
    hits = search_parents(article_text, k=k)
    return {h["parent_chunk_id"] for h in hits
            if h["score"] >= EQUIV_THR} | {self_parent_id}

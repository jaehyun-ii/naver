"""서빙 API 실검색 헬퍼 — RAFT distractor·no-golden 실측용 (build_training에서 사용).

2026-07-26: 로컬 bge-m3+child 컬렉션 → 콘솔 서빙 API 호출로 전환. 임베더·컬렉션·
리랭커가 서빙과 자동 정합되고(transformers 충돌 회피), 동등 임계 0.80은 Nemotron
점수 스케일에서도 유효함을 실측(자기일치 0.93~0.97 / 동등 0.81~0.95 / 무관 ≤0.64).

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
# 2026-07-26 서빙 전환 정합: parent-직접 컬렉션(Nemotron) 기본값 —
# RAFT distractor는 실서빙 리트리버가 실제로 노출할 문서 분포여야 한다.
COLLECTION = os.environ.get("AIREG_QDRANT_COLLECTION", "regs_parents_nem1b")
EMBED_MODEL = os.environ.get("AIREG_EMBED_MODEL", "nvidia/Nemotron-3-Embed-1B-BF16")
# 동등 요건 제외 임계값. 클러스터링(cluster_rules)은 0.85(0.80은 연쇄 병합 붕괴)지만,
# 여기는 골든 기준 pairwise 판단이라 연쇄 문제가 없고 과제외 비용이 distractor 후보
# 몇 개 손실뿐이다 — 실측 준동등(주조타장치 능력 쌍 0.834)을 걸러내도록 0.80.
EQUIV_THR = float(os.environ.get("AIREG_EQUIV_THR", "0.80"))
# child 오버페치 배수 — 한 조의 child 여러 개가 상위를 점유해도 조 수를 채우도록
CHILD_OVERSAMPLE = 4

RAG_API = os.environ.get("AIREG_RAG_API", "http://localhost:4100")
RAG_KEY = os.environ.get("AIREG_RAG_KEY", "sk-master-changeme")


@lru_cache(maxsize=1)
def available() -> bool:
    """콘솔 서빙 API 가용 여부 — 서빙과 동일 스택(Nemotron+리랭커) 사용."""
    try:
        r = requests.get(f"{RAG_API}/api/health", timeout=5)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def search_parents(query: str, k: int = 20, *, publisher: str | None = None,
                   query_vec: list[float] | None = None) -> list[dict]:
    """서빙 API(/api/rag/query) 호출 — 실서빙 리트리버(Nemotron parent-직접
    + VL 리랭커)가 실제로 노출하는 순서 그대로의 조 목록.

    publisher 필터는 무시한다 — 실서빙이 무필터(전 발행처)이므로 학습
    distractor도 같은 분포여야 한다. query_vec 인자는 하위 호환용(미사용).
    """
    r = requests.post(f"{RAG_API}/api/rag/query",
                      json={"query": query[:4000], "k": k},
                      headers={"X-Master-Key": RAG_KEY}, timeout=300)
    r.raise_for_status()
    out = []
    for h in r.json().get("hits", []):
        md = h.get("metadata") or {}
        out.append({"parent_chunk_id": h["id"], "score": h["score"],
                    "publisher": md.get("publisher"), "doc_id": None})
    return out


def equivalent_parents(article_text: str, self_parent_id: str,
                       k: int = 40) -> set[str]:
    """골든 조항과 동등(유사도 ≥ EQUIV_THR)한 조의 parent_chunk_id 집합.

    골든 원문 자체를 질의로 써서 인덱스에서 근접 조를 찾는다 — 자기 자신 포함.
    """
    hits = search_parents(article_text, k=k)
    return {h["parent_chunk_id"] for h in hits
            if h["score"] >= EQUIV_THR} | {self_parent_id}

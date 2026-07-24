#!/usr/bin/env python3
"""RAG 검색 평가 — 골든셋으로 recall@k · MRR 측정.

각 질의를 bge-m3로 임베딩→Qdrant 검색, top-k 안에 **정답 조**(parent_chunk_id 일치)가
있는지로 조-레벨 재현율/MRR을 계산한다.

    python scripts/eval_retrieval.py [collection]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from llmops_core.rag.embedder import SentenceTransformerEmbedder

COLL = sys.argv[1] if len(sys.argv) > 1 else "classification_rules"
KS = [1, 3, 5, 10]


def main() -> int:
    gold = [json.loads(l) for l in Path("data_chunks/goldenset.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    emb = SentenceTransformerEmbedder("BAAI/bge-m3")
    cl = QdrantClient(url="http://localhost:6333")

    hits = {k: 0 for k in KS}
    rr = 0.0
    per_soc: dict = {}
    misses = []
    for g in gold:
        v = emb.encode([g["query"]])[0]
        pts = cl.query_points(collection_name=COLL, query=v, limit=10, with_payload=True).points
        rank = None
        for i, p in enumerate(pts, 1):
            if (p.payload or {}).get("parent_chunk_id") == g["relevant_article_id"]:
                rank = i
                break
        for k in KS:
            if rank and rank <= k:
                hits[k] += 1
        rr += (1.0 / rank) if rank else 0.0
        s = per_soc.setdefault(g["soc"], [0, 0])
        s[1] += 1
        if rank and rank <= 5:
            s[0] += 1
        if not rank:
            misses.append((g["soc"], g["query"][:40], g["article_no"], g["article_title"][:16]))

    n = len(gold)
    print(f"골든셋 {n}개 · 컬렉션 '{COLL}'\n")
    print("전체:")
    for k in KS:
        print(f"  recall@{k:<2} = {hits[k]/n:.3f}  ({hits[k]}/{n})")
    print(f"  MRR     = {rr/n:.3f}")
    print("\n선급별 recall@5:")
    for soc, (h, t) in per_soc.items():
        print(f"  {soc:9} {h}/{t} = {h/t:.2f}")
    print(f"\n미검색(10위 내 정답 없음) {len(misses)}건:")
    for soc, q, no, ti in misses[:12]:
        print(f"  [{soc}] '{q}' → 정답 {no} {ti}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""청킹 결과 → bge-m3 임베딩 → Qdrant 적재 (클린·중복제거).

이전 문제: doc_id가 파일마다 고유하지 않아 멱등 삭제가 다른 문서 포인트를 지웠고
(적재 절반 손실), 통합본↔개별편 내용 중복이 그대로 색인됐다. 이를 고친다:
  · 컬렉션을 새로 만든다(초기화)
  · **내용 해시로 중복 청크 스킵**(통합본=개별편 중복 제거)
  · point id = uuid5(파일스템::chunk_id) → 파일 간 충돌 없음

    python scripts/embed_to_qdrant.py [collection] [--src data_chunks]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import time
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from llmops_core.ingest.vectorize import _embed_text, _payload
from llmops_core.rag.embedder import SentenceTransformerEmbedder

NS = uuid.UUID("1b671a64-40d5-491e-99b0-da01ff1f3341")
_ap = argparse.ArgumentParser()
_ap.add_argument("collection", nargs="?", default="classification_rules")
_ap.add_argument("--src", default="data_chunks")
_args = _ap.parse_args()
COLL = _args.collection
SRC = _args.src
URL = "http://localhost:6333"


def main() -> int:
    print("임베더 로드(BAAI/bge-m3)…", flush=True)
    emb = SentenceTransformerEmbedder("BAAI/bge-m3")
    cl = QdrantClient(url=URL)
    if COLL in {c.name for c in cl.get_collections().collections}:
        cl.delete_collection(COLL)
    cl.create_collection(COLL, vectors_config=VectorParams(size=emb.dim, distance=Distance.COSINE))
    print(f"컬렉션 '{COLL}' 초기화(dim={emb.dim})\n", flush=True)

    seen: set = set()
    buf: list = []          # (uid, embed_text, payload)
    total = dup = 0
    t0 = time.time()

    def flush():
        nonlocal total
        if not buf:
            return
        vecs = emb.encode([b[1] for b in buf])
        pts = [PointStruct(id=str(uuid.uuid5(NS, b[0])), vector=list(map(float, v)), payload=b[2])
               for b, v in zip(buf, vecs)]
        for i in range(0, len(pts), 256):
            cl.upsert(collection_name=COLL, points=pts[i:i + 256])
        total += len(pts)
        buf.clear()

    files = sorted(glob.glob(f"{SRC}/**/*.jsonl", recursive=True))
    for fi, f in enumerate(files, 1):
        stem = Path(f).stem
        for order, l in enumerate(open(f, encoding="utf-8")):
            c = json.loads(l)
            if c.get("chunk_level") != "child":
                continue
            txt = _embed_text(c)
            if not txt.strip():
                continue
            key = (c.get("content") or c.get("retrieval_text") or "").strip()
            h = hashlib.md5(key.encode()).hexdigest()
            if h in seen:           # 내용 중복(통합본=개별편) 스킵
                dup += 1
                continue
            seen.add(h)
            buf.append((f"{stem}::{c['chunk_id']}", txt, _payload(c, order)))
            if len(buf) >= 512:
                flush()
        if fi % 20 == 0 or fi == len(files):
            print(f"[{fi:3}/{len(files)}] 적재 {total:,} · 중복스킵 {dup:,} · {(time.time()-t0)/60:.1f}분",
                  flush=True)
    flush()
    print(f"\n완료: {total:,} 벡터(중복 {dup:,} 제거) → '{COLL}' ({(time.time()-t0)/60:.1f}분)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

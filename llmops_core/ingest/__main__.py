"""인제스트 CLI — 원본 PDF 등록 / 워커 / 조회.

    python -m llmops_core.ingest register FILE.pdf [--name N] [--family auto|kr_rule|nk_rule|lr_code|dnv_cg|bv_rule|iacs|abs_guide]
    python -m llmops_core.ingest worker [--once]
    python -m llmops_core.ingest list
    python -m llmops_core.ingest status DOC_ID
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(prog="llmops_core.ingest")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_reg = sub.add_parser("register", help="원본 PDF 등록/갱신 → 처리 큐")
    p_reg.add_argument("pdf", type=Path)
    p_reg.add_argument("--name", default=None)
    from llmops_core.chunking import FAMILIES
    p_reg.add_argument("--family", default="auto", choices=["auto", *FAMILIES])
    p_reg.add_argument("--no-enqueue", action="store_true")

    p_w = sub.add_parser("worker", help="큐 소비 → e2e 파이프라인")
    p_w.add_argument("--once", action="store_true")

    sub.add_parser("list", help="등록 문서 목록")
    p_s = sub.add_parser("status", help="문서 상태 조회")
    p_s.add_argument("doc_id")

    args = ap.parse_args()

    if args.cmd == "register":
        from .service import register_document
        doc = register_document(args.pdf, name=args.name, family=args.family, enqueue=not args.no_enqueue)
        print(f"registered {doc.id} v{doc.version} sha={doc.sha256[:12]} status={doc.status} family={doc.family}")
        return 0
    if args.cmd == "worker":
        from .worker import run_worker
        run_worker(once=args.once)
        return 0
    if args.cmd == "list":
        from .registry import registry
        for d in registry().list():
            print(f"{d.id:24} v{d.version} {d.status:11} chunks={d.n_chunks:<5} vectors={d.n_vectors:<5} {d.family}")
        return 0
    if args.cmd == "status":
        from .registry import registry
        d = registry().get(args.doc_id)
        if not d:
            print("not found", file=sys.stderr)
            return 1
        import json
        from dataclasses import asdict
        print(json.dumps(asdict(d), ensure_ascii=False, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

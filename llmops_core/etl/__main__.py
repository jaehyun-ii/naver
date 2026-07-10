"""CLI: PDF → data/<name>/ (MinerU hybrid ETL).

    python -m llmops_core.etl INPUT.pdf -o data [--endpoint http://host:8002]
                              [--effort high] [--parse-method auto]
                              [--no-image-analysis] [--no-formula] [--no-table]

여러 PDF를 한 번에: 디렉터리를 주면 하위 *.pdf 전부 처리.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from .config import EtlConfig
from .pipeline import run_etl


def main() -> int:
    ap = argparse.ArgumentParser(description="MinerU hybrid ETL: PDF → content_list.json")
    ap.add_argument("input", type=Path, help="PDF 파일 또는 PDF들이 든 디렉터리")
    ap.add_argument("-o", "--out", type=Path, default=Path("data"), help="출력 루트(기본 data/)")
    ap.add_argument("--endpoint", default=None, help="MinerU VLM 서버(hybrid) 주소. 미지정 시 PARSER_ENDPOINT")
    ap.add_argument("--effort", default=None, choices=["medium", "high"])
    ap.add_argument("--parse-method", default=None, choices=["auto", "ocr", "txt"])
    ap.add_argument("--no-image-analysis", action="store_true")
    ap.add_argument("--no-formula", action="store_true")
    ap.add_argument("--no-table", action="store_true")
    ap.add_argument("--name", default=None, help="출력 폴더/프리픽스 이름(기본: PDF stem)")
    args = ap.parse_args()

    cfg = EtlConfig.from_env()
    overrides = {}
    if args.endpoint:
        overrides["endpoint"] = args.endpoint
    if args.effort:
        overrides["effort"] = args.effort
    if args.parse_method:
        overrides["parse_method"] = args.parse_method
    if args.no_image_analysis:
        overrides["image_analysis"] = False
    if args.no_formula:
        overrides["formula_enabled"] = False
    if args.no_table:
        overrides["table_enabled"] = False
    if overrides:
        cfg = replace(cfg, **overrides)

    pdfs = sorted(args.input.glob("*.pdf")) if args.input.is_dir() else [args.input]
    if not pdfs:
        print(f"PDF 없음: {args.input}", file=sys.stderr)
        return 2

    rc = 0
    for pdf in pdfs:
        try:
            status = run_etl(pdf, args.out, cfg, name=args.name if len(pdfs) == 1 else None)
            print(f"✓ {pdf.name} → {status['content_list']}")
        except Exception as exc:  # noqa: BLE001
            print(f"✗ {pdf.name}: {exc}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

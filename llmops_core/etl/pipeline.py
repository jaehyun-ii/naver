"""MinerU hybrid ETL 파이프라인 — PDF → data/<name>/ 산출물.

noksan_ax의 서비스 경로(_run_mineru_hybrid_http_client → postprocess_mineru_output_dir
→ _collect_mineru_hybrid_outputs)에서 FastAPI/DB/큐를 벗기고 동일 순서를 재현한다:

    1. MinerU CLI(hybrid-http-client)로 raw 산출물 생성
    2. raw content_list 스냅샷 보존(raw/)
    3. pdftext 병합 후처리(결정적) — content_list.json 덮어쓰기 + .md 재생성
    4. 산출물을 <name>_ 프리픽스로 상위에 수집(우리 청커 입력)

결과 레이아웃(기준 data/ 예시와 동일한 형태):
    <out>/<name>/
        <name>_content_list.json      # 후처리된 hybrid 결과 (청커 입력)
        <name>_content_list_v2.json
        <name>_middle.json / _model.json / _layout.pdf / <name>.md
        images/
        raw/                          # 후처리 전 raw content_list 스냅샷
        etl_status.json               # 실행 매니페스트
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .config import EtlConfig
from .mineru import run_mineru
from .postprocess import postprocess_mineru_output_dir

# (탐색 suffix, 수집 파일명 접미사)
_ARTIFACTS = [
    ("content_list.json", "content_list.json", ("content_list_v2.json",)),
    ("content_list_v2.json", "content_list_v2.json", ()),
    ("middle.json", "middle.json", ()),
    ("model.json", "model.json", ()),
    ("layout.pdf", "layout.pdf", ()),
    (".md", ".md", ()),
]


def _find(files: list[Path], suffix: str, exclude: tuple[str, ...]) -> Path | None:
    matches = [
        p for p in files
        if p.name.lower().endswith(suffix) and not any(p.name.lower().endswith(x) for x in exclude)
    ]
    # 가장 얕고 짧은 이름 우선(원본 _find_mineru_output과 동일 취지)
    return sorted(matches, key=lambda p: (len(p.parts), len(p.name), str(p)))[0] if matches else None


def run_etl(pdf_path: Path, out_root: Path, cfg: EtlConfig | None = None, *, name: str | None = None) -> dict:
    cfg = cfg or EtlConfig.from_env()
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    name = name or pdf_path.stem
    doc_dir = Path(out_root) / name
    doc_dir.mkdir(parents=True, exist_ok=True)

    # 1) MinerU CLI (raw 산출물 → doc_dir/<stem>/hybrid_auto/)
    command = run_mineru(pdf_path, doc_dir, cfg)

    files = [p for p in doc_dir.rglob("*") if p.is_file()]
    raw_cl = _find(files, "content_list.json", ("content_list_v2.json",))
    if raw_cl is None:
        raise RuntimeError(f"MinerU가 content_list.json을 생성하지 않았습니다: {doc_dir}")
    mineru_out = raw_cl.parent

    # 2) 후처리 전 raw 스냅샷 보존
    raw_snapshot = doc_dir / "raw"
    raw_snapshot.mkdir(parents=True, exist_ok=True)
    for suffix in ("content_list.json", "content_list_v2.json"):
        src = _find(files, suffix, ("content_list_v2.json",) if suffix == "content_list.json" else ())
        if src:
            shutil.copy2(src, raw_snapshot / f"{name}_raw_{suffix}")

    # 3) pdftext 병합 후처리(결정적) — content_list.json 덮어쓰기 + .md
    report = postprocess_mineru_output_dir(input_path=pdf_path, output_dir=mineru_out, doc_name=name)

    # 3.5) 수식 재인식 패스 — 비-LaTeX equation 블록을 bbox 크롭으로 VLM 재인식
    #      (실측: 1차 추출 후 음역 잔존 수식 복원, 실패 시 formula_reco_failed 마크)
    if cfg.formula_enabled:
        import json as _json

        from .formula_fix import rerecognize_equations

        cl_path = _find([p for p in mineru_out.rglob("*") if p.is_file()],
                        "content_list.json", ("content_list_v2.json",))
        if cl_path:
            cl = _json.loads(cl_path.read_text(encoding="utf-8"))
            from .formula_fix import normalize_equations_spacing

            from .formula_fix import rerecognize_scrambled_text

            fx = rerecognize_equations(pdf_path, cl, endpoint=cfg.endpoint,
                                       model=cfg.model)
            fx["scramble"] = rerecognize_scrambled_text(
                pdf_path, cl, endpoint=cfg.endpoint, model=cfg.model)
            fx["spacing_normalized"] = normalize_equations_spacing(cl)
            report["formula_fix"] = fx
            tx = {}
            if cfg.table_enabled:
                from .table_fix import recover_lost_tables

                tx = recover_lost_tables(pdf_path, cl, endpoint=cfg.endpoint,
                                         model=cfg.model,
                                         images_dir=cl_path.parent / "images")
                report["table_fix"] = tx
            gx = {}
            if cfg.image_analysis:
                from .image_fix import reanalyze_images

                gx = reanalyze_images(pdf_path, cl, endpoint=cfg.endpoint,
                                      model=cfg.model)
                report["image_fix"] = gx
            if (fx["fixed"] or fx["failed"] or fx.get("spacing_normalized") or tx.get("lost")
                    or gx.get("fixed") or gx.get("failed")
                    or gx.get("caption_cleaned") or gx.get("merged")
                    or gx.get("noise_dropped")):
                cl_path.write_text(_json.dumps(cl, ensure_ascii=False),
                                   encoding="utf-8")

    # 4) 산출물을 <name>_ 프리픽스로 상위 doc_dir에 수집
    files = [p for p in mineru_out.rglob("*") if p.is_file()]
    collected: dict[str, str] = {}
    for suffix, out_suffix, exclude in _ARTIFACTS:
        src = _find(files, suffix, exclude)
        if not src:
            continue
        dst = doc_dir / (f"{name}{out_suffix}" if out_suffix.startswith(".") else f"{name}_{out_suffix}")
        shutil.copy2(src, dst)
        collected[out_suffix] = str(dst)
    # 이미지 디렉터리 수집
    img_src = mineru_out / "images"
    if img_src.is_dir():
        img_dst = doc_dir / "images"
        if img_dst.exists():
            shutil.rmtree(img_dst)
        shutil.copytree(img_src, img_dst)
        collected["images"] = str(img_dst)

    status = {
        "name": name,
        "source_pdf": str(pdf_path),
        "out_dir": str(doc_dir),
        "content_list": collected.get("content_list.json"),
        "mineru_command": command,
        "config": {
            "endpoint": cfg.endpoint, "model": cfg.model, "effort": cfg.effort,
            "parse_method": cfg.parse_method, "image_analysis": cfg.image_analysis,
            "formula_enabled": cfg.formula_enabled, "table_enabled": cfg.table_enabled,
        },
        "postprocess": report,
        "collected": collected,
    }
    (doc_dir / "etl_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return status

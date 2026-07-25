"""표 소실 폴백 — table_body·img_path 모두 없는 표 블록을 bbox 크롭으로 복구.

MinerU가 전면 크기 표에서 파싱 실패하면 본문도 이미지도 없는 빈 블록만 남는다
(1편_2025 실측: 37개 영역, 해당 페이지 텍스트 볼륨 ≈0 — 부기상세·검사항목표·
취약지역 도해 통째 소실). 복구 순서:
  1. bbox 크롭 이미지를 images/에 저장해 img_path 확보(최소한 시각 자료 보존)
  2. VLM content_extract(type="table")로 표 본문(HTML/마크다운) 재인식 시도
  3. 재인식 실패 시 `table_parse_failed` 마크(이미지 폴백은 유지)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_PAD = 4
_MIN_BODY = 30  # 표 본문으로 인정할 최소 길이


def recover_lost_tables(pdf_path: Path, content_list: list[dict], *,
                        endpoint: str, model: str, images_dir: Path,
                        zoom: float = 3.0) -> dict:
    """빈 표 블록 복구(in-place). 통계 반환."""
    lost = [b for b in content_list
            if b.get("type") == "table"
            and not str(b.get("table_body") or "").strip()
            and not str(b.get("img_path") or "").strip()
            and b.get("bbox")]
    stats = {"tables": sum(1 for b in content_list if b.get("type") == "table"),
             "lost": len(lost), "image_saved": 0, "body_recovered": 0, "failed": 0}
    if not lost:
        return stats
    import pymupdf
    from PIL import Image
    from mineru_vl_utils import MinerUClient

    client = MinerUClient(backend="http-client", server_url=endpoint.rstrip("/"),
                          model_name=model)
    images_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(str(pdf_path))
    for b in lost:
        try:
            page = doc[b["page_idx"]]
            r = pymupdf.Rect(b["bbox"])
            r = pymupdf.Rect(max(0, r.x0 - _PAD), max(0, r.y0 - _PAD),
                             r.x1 + _PAD, r.y1 + _PAD)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=r)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            tag = hashlib.md5(
                f"{pdf_path.name}:{b['page_idx']}:{b['bbox']}".encode()).hexdigest()[:16]
            fname = f"tablefix_p{b['page_idx']}_{tag}.png"
            img.save(images_dir / fname)
            b["img_path"] = f"images/{fname}"
            b["table_recovered"] = "image"
            stats["image_saved"] += 1
        except Exception:  # noqa: BLE001 — 크롭 실패 시 본문 재인식도 불가
            b["table_parse_failed"] = True
            stats["failed"] += 1
            continue
        try:
            body = str(client.content_extract(img, type="table") or "").strip()
        except Exception:  # noqa: BLE001
            body = ""
        if len(body) >= _MIN_BODY and ("|" in body or "<tr" in body or "\n" in body):
            b["table_body"] = body
            b["table_recovered"] = "image+body"
            stats["body_recovered"] += 1
        else:
            b["table_parse_failed"] = True
            stats["failed"] += 1
    doc.close()
    return stats

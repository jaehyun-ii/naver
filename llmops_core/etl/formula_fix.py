"""수식 재인식 패스 — LaTeX 마커 없는 equation 블록을 bbox 크롭으로 VLM 재인식.

MinerU 1차 추출에서 일부 수식(짧은 식·저해상도 영역)이 LaTeX가 아닌 음역
텍스트("2 75.1 t C aH …")로 떨어진다(실측: 11편 수식 17개 중 6개). 이 블록만
페이지에서 bbox를 잘라 같은 VLM 서버(content_extract, type="equation")로
재인식해 교체한다. 재인식도 실패하면 원문을 유지하되 `formula_reco_failed`
마크를 남겨 하류(청커·데이터 생성)가 신뢰 불가 수식임을 알 수 있게 한다.
"""

from __future__ import annotations

import re
from pathlib import Path

_LATEX_MARK = re.compile(r"[\\$]")
_PAD = 6  # bbox 여유(px, 페이지 포인트 기준) — 경계 잘림으로 인한 오인식 방지


def _is_latex(text: str) -> bool:
    return bool(_LATEX_MARK.search(text or ""))


def _balanced(text: str) -> bool:
    """중괄호 균형 — 잘린 재인식 결과({} 공백 분모 등) 걸러내는 최소 검증."""
    return text.count("{") == text.count("}") and "{}" not in text.replace("{ }", "{}")


def rerecognize_equations(pdf_path: Path, content_list: list[dict], *,
                          endpoint: str, model: str, zoom: float = 3.0) -> dict:
    """content_list의 비-LaTeX equation 블록을 재인식(in-place). 통계 반환."""
    bad = [b for b in content_list
           if b.get("type") == "equation" and not _is_latex(b.get("text", ""))]
    stats = {"equations": sum(1 for b in content_list if b.get("type") == "equation"),
             "bad": len(bad), "fixed": 0, "failed": 0}
    if not bad:
        return stats
    import pymupdf
    from PIL import Image
    from mineru_vl_utils import MinerUClient

    client = MinerUClient(backend="http-client", server_url=endpoint.rstrip("/"),
                          model_name=model)
    doc = pymupdf.open(str(pdf_path))
    for b in bad:
        try:
            page = doc[b["page_idx"]]
            r = pymupdf.Rect(b["bbox"])
            r = pymupdf.Rect(max(0, r.x0 - _PAD), max(0, r.y0 - _PAD),
                             r.x1 + _PAD, r.y1 + _PAD)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=r)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            out = str(client.content_extract(img, type="equation") or "").strip()
        except Exception:  # noqa: BLE001 — 블록 단위 실패는 마크 후 계속
            out = ""
        if out and _is_latex(out) and _balanced(out):
            b["text"] = out
            b["text_format"] = "latex"
            stats["fixed"] += 1
        else:
            b["formula_reco_failed"] = True
            stats["failed"] += 1
    doc.close()
    return stats

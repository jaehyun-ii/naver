"""그림 재분석 패스 — 분석 설명이 빈약한 image 블록을 bbox 크롭으로 VLM 재분석.

MinerU image-analysis가 특정 유형(선도·다이어그램·소형 삽화)에서 빈/캡션 반복
출력을 내는 비율이 문서당 15~30%로 실측됐다. 캡션 제외 설명이 15자 미만인
image 블록만 페이지에서 bbox를 잘라 재분석한다:
  1차: MinerU content_extract(type="image") — 파싱 전용 태스크
  2차: 같은 VLM 서버에 OpenAI vision chat으로 기술 문서 맥락 설명 요청
둘 다 빈약하면 `image_analysis_failed` 마크를 남겨 하류가 신뢰 불가임을 알게 한다.
"""

from __future__ import annotations

import base64
import io
import json
import urllib.request
from pathlib import Path

_PAD = 6
_MIN_DESC = 15

_CHAT_PROMPT = (
    "이 그림은 선급 규정(선박 기술 기준) 문서의 삽화입니다. 그림이 보여주는 "
    "내용(구조·부재 명칭, 축·기호, 치수 관계, 흐름 등)을 한국어 2~4문장으로 "
    "설명하십시오. 그림에 실제로 보이는 정보만 기술하고 추정하지 마십시오."
)


def _caption_of(block: dict) -> str:
    cap = block.get("image_caption")
    if isinstance(cap, list):
        return " ".join(cap).strip()
    return str(cap or "").strip()


def _desc_len(block: dict) -> int:
    cap = _caption_of(block)
    content = str(block.get("content") or "").strip()
    return len(content.replace(cap, "").strip())


def _chat_describe(endpoint: str, model: str, png: bytes, timeout: int = 120) -> str:
    b64 = base64.b64encode(png).decode()
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": _CHAT_PROMPT},
        ]}],
        "max_tokens": 300, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(f"{endpoint.rstrip('/')}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=timeout))
    return (out["choices"][0]["message"]["content"] or "").strip()


def reanalyze_images(pdf_path: Path, content_list: list[dict], *,
                     endpoint: str, model: str, zoom: float = 3.0) -> dict:
    """설명 빈약 image 블록 재분석(in-place). 통계 반환."""
    weak = [b for b in content_list
            if b.get("type") == "image" and _desc_len(b) < _MIN_DESC]
    stats = {"images": sum(1 for b in content_list if b.get("type") == "image"),
             "weak": len(weak), "fixed": 0, "failed": 0}
    if not weak:
        return stats
    import pymupdf
    from PIL import Image
    from mineru_vl_utils import MinerUClient

    client = MinerUClient(backend="http-client", server_url=endpoint.rstrip("/"),
                          model_name=model)
    doc = pymupdf.open(str(pdf_path))
    for b in weak:
        desc = ""
        try:
            page = doc[b["page_idx"]]
            r = pymupdf.Rect(b["bbox"])
            r = pymupdf.Rect(max(0, r.x0 - _PAD), max(0, r.y0 - _PAD),
                             r.x1 + _PAD, r.y1 + _PAD)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=r)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            desc = str(client.content_extract(img, type="image") or "").strip()
            if len(desc) < _MIN_DESC:  # 파싱 태스크 실패 → vision chat 폴백
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                desc = _chat_describe(endpoint, model, buf.getvalue())
        except Exception:  # noqa: BLE001 — 블록 단위 실패는 마크 후 계속
            desc = ""
        cap = _caption_of(b)
        if len(desc.replace(cap, "").strip()) >= _MIN_DESC:
            b["content"] = f"{cap}\n{desc}".strip() if cap else desc
            stats["fixed"] += 1
        else:
            b["image_analysis_failed"] = True
            stats["failed"] += 1
    doc.close()
    return stats

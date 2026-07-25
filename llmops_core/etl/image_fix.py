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
import re
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


_MERMAID = re.compile(r"```\s*mermaid|graph\s+(TD|LR|RL|BT)\b|flowchart\s+(TD|LR)")


def _is_confabulated(text: str) -> bool:
    """VLM이 도면과 무관한 mermaid 그래프를 창작한 출력(실측 68건) 감지."""
    t = text or ""
    return bool(_MERMAID.search(t)) or t.count("-->") >= 2


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


# 그림 '제목' 마커 — "그림 N" 뒤에 조사가 붙으면 본문 참조("그림 2.1.13과 같이"),
# 조사 없이 이어지면 제목("그림 2.1.12 이음매 없는 강관")으로 구분한다.
_TITLE_MARK = re.compile(r"(그림|Fig\.?)\s*[\d.]+(?![과와을를에의은는이가로0-9.])")
_SENT_END = re.compile(r"(한다|된다|이다|같다|시오)\s*\.?\s*$")


def sanitize_captions(content_list: list[dict]) -> int:
    """캡션에 흡수된 본문 문장 분리 — 제목 마커 앞의 서술형 텍스트를 캡션에서
    떼어 image 블록 직전의 text 블록으로 되돌린다(읽기 순서 보존). 정화 수 반환."""
    cleaned = 0
    i = 0
    while i < len(content_list):
        b = content_list[i]
        if b.get("type") != "image":
            i += 1
            continue
        cap = _caption_of(b)
        m = _TITLE_MARK.search(cap)
        if m and m.start() >= 3:
            pre = cap[:m.start()].strip()
            if pre and (_SENT_END.search(pre) or len(pre) > 40):
                b["image_caption"] = [cap[m.start():].strip()]
                content_list.insert(i, {"type": "text", "text": pre,
                                        "page_idx": b.get("page_idx", 0),
                                        "restored_from_caption": True})
                cleaned += 1
                i += 1  # 삽입한 text 블록 건너뜀
        i += 1
    return cleaned


# 캡션 결속 패턴 — 그림 제목(아래형) / 번호형 표제(위형, 부록 1-14) / 표 머리글 오염
_FIG_TITLE = re.compile(r"^(그림|Fig\.?)\s*[\d.\-]+")
_NUM_HEAD = re.compile(r"^\d{1,3}[.)]\s*\S")
_TBL_TITLE = re.compile(r"^표\s*[\d\-.]+\s*\S")      # 표 스크린샷 이미지의 제목
_EN_TITLE = re.compile(r"^[A-Z][A-Z0-9 /&'()\-]{4,50}$")  # SURVEY PROGRAMME 류
_TBL_HEAD = re.compile(r"^표\s*[\d\-.]+[^그림]{0,60}(\(계속\))?\s*")


def bind_captions(content_list: list[dict]) -> dict:
    """빈 캡션 이미지에 인접 text 블록 표제 결속 + 표 머리글 오염 제거 + 과병합 이관.

    실측(1편_2025): 빈 캡션 80/152 중 다수는 PDF에 캡션 실재 —
    ① 그림 '아래' 「그림 N …」 제목 미결속  ② 그림 '위' 번호형 표제(부록 1-14) 미결속
    ③ 표 프레임 머리글("표 3-1 …")이 캡션에 흡수  ④ 첫 그림이 다음 그림 표제까지 과병합.
    """
    stats = {"bound_below": 0, "bound_above": 0, "table_head_stripped": 0,
             "title_redistributed": 0}
    n = len(content_list)
    for i, b in enumerate(content_list):
        if b.get("type") != "image":
            continue
        cap = _caption_of(b)
        # ③ 표 머리글 오염 제거 — "표 N-N …"으로 시작하면 그 구간을 벗겨낸다
        m_t = _TBL_HEAD.match(cap)
        if m_t and m_t.end() > 4:
            rest = cap[m_t.end():].strip()
            b["image_caption"] = [rest] if rest else []
            cap = rest
            stats["table_head_stripped"] += 1
        if cap:
            continue
        pg = b.get("page_idx", 0)
        # ① 아래 「그림 N …」 제목 결속
        for j in range(i + 1, min(i + 3, n)):
            nb = content_list[j]
            if nb.get("page_idx", 0) != pg or nb.get("type") == "image":
                break
            t = (nb.get("text") or "").strip()
            if nb.get("type") == "text" and t and len(t) <= 90 and _FIG_TITLE.match(t):
                b["image_caption"] = [t]
                nb["_consumed_as_caption"] = True
                stats["bound_below"] += 1
                break
        if _caption_of(b):
            continue
        # ② 위 번호형 표제 결속(짧고 문장형이 아닌 직전 text)
        for j in range(i - 1, max(i - 3, -1), -1):
            pb = content_list[j]
            if pb.get("page_idx", 0) != pg or pb.get("type") == "image":
                break
            t = (pb.get("text") or "").strip()
            if (pb.get("type") == "text" and t and len(t) <= 70
                    and (_NUM_HEAD.match(t) or _FIG_TITLE.match(t)
                         or _TBL_TITLE.match(t) or _EN_TITLE.match(t))
                    and not _SENT_END.search(t)):
                b["image_caption"] = [t]
                pb["_consumed_as_caption"] = True
                stats["bound_above"] += 1
                break
    # ④ 과병합 이관 — 캡션에 제목 마커 2개 + 다음 이미지 캡션이 빈 경우 분배
    imgs = [b for b in content_list if b.get("type") == "image"]
    for a, c in zip(imgs, imgs[1:]):
        if a.get("page_idx") != c.get("page_idx") and \
           (a.get("page_idx", 0) + 1) != c.get("page_idx", 0):
            continue
        cap = _caption_of(a)
        marks = list(_TITLE_MARK.finditer(cap))
        if len(marks) >= 2 and not _caption_of(c):
            a["image_caption"] = [cap[:marks[1].start()].strip()]
            c["image_caption"] = [cap[marks[1].start():].strip()]
            stats["title_redistributed"] += 1
    # 캡션으로 소비된 text 블록 제거
    content_list[:] = [b for b in content_list if not b.get("_consumed_as_caption")]
    return stats


# 노이즈 그림 필터 — 로고·장식·sliver (실측: 닻 기호 19×13px, 파형 조각 등)
_NOISE_MIN_W = 22
_NOISE_MIN_H = 16
_NOISE_MIN_AREA = 900.0


def drop_noise_images(content_list: list[dict]) -> int:
    """극소 bbox image 블록 제거(캡션 있는 것은 보존). 제거 수 반환."""
    drop = []
    for b in content_list:
        if b.get("type") != "image" or not b.get("bbox") or _caption_of(b):
            continue
        x0, y0, x1, y1 = b["bbox"]
        w, h = x1 - x0, y1 - y0
        if w < _NOISE_MIN_W or h < _NOISE_MIN_H or w * h < _NOISE_MIN_AREA:
            drop.append(id(b))
    if drop:
        content_list[:] = [b for b in content_list if id(b) not in set(drop)]
    return len(drop)


_MERGE_GAP = 18      # 세로 인접 판단 간격(pt)
_MERGE_XOVL = 0.6    # 가로 겹침 최소 비율


def merge_fragmented_images(content_list: list[dict]) -> int:
    """세로로 쪼개진 그림 조각 병합 — 큰 그림이 여러 image 블록으로 나뉘고
    캡션이 마지막 조각에만 붙는 문제(코퍼스 실측: 복수 그림 조의 6%) 대응.

    같은 페이지에서 세로로 인접(간격 ≤18pt)하고 가로 겹침 ≥60%인 image 블록
    체인을 하나로 합친다: bbox 합집합, 캡션·분석 텍스트 이어붙임. 좌우 병렬
    배치(별개 소도면)는 병합하지 않는다. 병합 수 반환."""
    from collections import defaultdict

    by_pg: dict[int, list[dict]] = defaultdict(list)
    for b in content_list:
        if b.get("type") == "image" and b.get("bbox"):
            by_pg[b.get("page_idx", 0)].append(b)
    merged = 0
    drop: set[int] = set()
    for bs in by_pg.values():
        bs.sort(key=lambda x: x["bbox"][1])
        i = 0
        while i < len(bs) - 1:
            a, c = bs[i], bs[i + 1]
            ax0, ay0, ax1, ay1 = a["bbox"]
            cx0, cy0, cx1, cy1 = c["bbox"]
            xovl = max(0.0, min(ax1, cx1) - max(ax0, cx0))
            xden = max(1.0, min(ax1 - ax0, cx1 - cx0))
            if cy0 - ay1 <= _MERGE_GAP and xovl / xden >= _MERGE_XOVL:
                a["bbox"] = [min(ax0, cx0), ay0, max(ax1, cx1), cy1]
                cap_a, cap_c = _caption_of(a), _caption_of(c)
                a["image_caption"] = [" ".join(x for x in (cap_a, cap_c) if x)]
                cont = " ".join(x for x in (str(a.get("content") or "").strip(),
                                            str(c.get("content") or "").strip()) if x)
                a["content"] = cont
                a["merged_fragments"] = a.get("merged_fragments", 1) + 1
                drop.add(id(c))
                bs.pop(i + 1)
                merged += 1
            else:
                i += 1
    if drop:
        content_list[:] = [b for b in content_list if id(b) not in drop]
    return merged


def reanalyze_images(pdf_path: Path, content_list: list[dict], *,
                     endpoint: str, model: str, zoom: float = 3.0) -> dict:
    """캡션 정화·결속 → 노이즈 제거 → 조각 병합 → 빈약 설명 재분석(in-place)."""
    n_clean = sanitize_captions(content_list)
    bind = bind_captions(content_list)
    n_noise = drop_noise_images(content_list)
    n_merged = merge_fragmented_images(content_list)
    weak = [b for b in content_list
            if b.get("type") == "image"
            and (_desc_len(b) < _MIN_DESC
                 or _is_confabulated(str(b.get("content") or "")))]
    stats = {"images": sum(1 for b in content_list if b.get("type") == "image"),
             "caption_cleaned": n_clean, "caption_bound": bind, "noise_dropped": n_noise, "merged": n_merged, "weak": len(weak), "fixed": 0, "failed": 0}
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
        if _is_confabulated(desc):
            desc = ""  # 재분석도 허구 그래프면 폐기
        if len(desc.replace(cap, "").strip()) >= _MIN_DESC:
            b["content"] = f"{cap}\n{desc}".strip() if cap else desc
            stats["fixed"] += 1
        else:
            b["image_analysis_failed"] = True
            stats["failed"] += 1
    # 합성 캡션 — 결속·원본 캡션이 없지만 분석 설명이 있는 그림은 설명 첫
    # 문장(≤60자)을 캡션으로 부여(caption_synthesized 마크). 양식 스크린샷·
    # 무제 도판도 하류(그림 주입·figure_qa)에서 지칭 가능해진다.
    for b in content_list:
        if b.get("type") != "image" or _caption_of(b):
            continue
        content = str(b.get("content") or "").strip()
        if len(content) < _MIN_DESC:
            continue
        first = re.split(r"(?<=[.다])\s", content)[0][:60].strip()
        if len(first) >= 8:
            b["image_caption"] = [first]
            b["caption_synthesized"] = True
            stats["caption_synth"] = stats.get("caption_synth", 0) + 1
    doc.close()
    return stats

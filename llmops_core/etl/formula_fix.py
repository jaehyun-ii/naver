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


_HANGUL = re.compile(r"[가-힣]$")
_HANGUL_S = re.compile(r"^[가-힣]")
_CMD_END = re.compile(r"\\[a-zA-Z]+$")
_MATH_SPAN = re.compile(r"\$\$.*?\$\$|\$[^$\n]+\$", re.S)
_TEXT_GRP = re.compile(r"(\\text\s*\{)([^{}]*)(\})")


def _join_single_runs(s: str) -> str:
    """단일 문자 토큰 연속을 병합 — 'a s - b u i l t' → 'as-built', '1 . 8' → '1.8'."""
    toks = s.split(" ")
    out: list[str] = []
    run: list[str] = []
    for t in toks:
        if len(t) == 1 and t.strip():
            run.append(t)
        else:
            out.append("".join(run)) if len(run) >= 2 else out.extend(run)
            run = []
            if t:
                out.append(t)
    out.append("".join(run)) if len(run) >= 2 else out.extend(run)
    return " ".join(out)


def _tighten_math(seg: str) -> str:
    """수학 모드 공백 제거 — LaTeX 명령 뒤 문자, 한글 사이 공백만 보존."""
    toks = seg.split()
    out = ""
    for i, t in enumerate(toks):
        out += t
        if i + 1 < len(toks):
            nxt = toks[i + 1]
            if (_CMD_END.search(t) and re.match(r"[a-zA-Z]", nxt)) or \
                    (_HANGUL.search(t) and _HANGUL_S.match(nxt)):
                out += " "
    return out


def normalize_math_spacing(text: str) -> str:
    """수식 스팬($$…$$·$…$)의 글자별 공백 아티팩트 제거(렌더 불변·검색성 복원).

    MinerU 산출 't _ {a s - b u i l t}', '1 . 8', '1 0 ^ {6}' 류는 렌더는 정상이나
    "as-built"·"1.8" 문자열 검색이 전부 깨진다. \\text{} 내부는 단일 문자 연속만
    병합(정상 단어 간 공백 보존), 외부는 명령·한글 경계만 남기고 공백을 제거한다.
    """
    def _fix_span(m: re.Match) -> str:
        span = m.group(0)
        marker = "$$" if span.startswith("$$") else "$"
        inner = span[len(marker):-len(marker)]
        # \text{} 그룹은 보호하며 내부 단일 문자 연속만 병합
        protected: list[str] = []

        def _guard(tm: re.Match) -> str:
            protected.append("\\text{" + _join_single_runs(tm.group(2)) + tm.group(3))
            return f"\x00{len(protected) - 1}\x00"

        inner = _TEXT_GRP.sub(_guard, inner)
        inner = _tighten_math(_join_single_runs(inner))
        for i, p in enumerate(protected):
            inner = inner.replace(f"\x00{i}\x00", p)
        return f"{marker} {inner} {marker}" if marker == "$$" else f"{marker}{inner}{marker}"

    return _MATH_SPAN.sub(_fix_span, text)


def normalize_equations_spacing(content_list: list[dict]) -> int:
    """equation 블록·본문 인라인 수식의 공백 정규화(in-place). 변경 블록 수 반환."""
    changed = 0
    for b in content_list:
        t = b.get("text")
        if not t or b.get("type") not in ("equation", "text", "list"):
            continue
        if b.get("type") == "equation":
            if "$" in t:
                new = normalize_math_spacing(t)
            elif _LATEX_MARK.search(t):
                new = _tighten_math(_join_single_runs(t))
            else:
                new = t
        else:
            new = normalize_math_spacing(t)
        if new != t:
            b["text"] = new
            changed += 1
    return changed


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

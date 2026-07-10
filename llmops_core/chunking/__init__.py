"""도메인 인지 청킹 패키지 — 선급별 특화 청커 + 자동 디스패치.

선급마다 문서 구조가 근본적으로 달라 기관별 **자립 모듈**로 특화한다:

    kr_rule    한국선급(KR)         편>장>절>조>항>호>목, 규칙/지침 교차링크
    nk_rule    ClassNK(NK)          Chapter>N.M>N.M.K, 규정/지침 교차링크
    lr_code    Lloyd's Register     제목내장 Chapter>Section>N.M
    dnv_cg     DNV                  SECTION>N/N.M (섹션별 재시작)
    bv_rule    Bureau Veritas       Part>Chapter>Section>[n]/n.m
    iacs       IACS                 REC(평면) / CSR(대형 구조규칙)
    abs_guide  ABS                  SECTION>clause 십진

모든 모듈이 **동일 JSONL 스키마**(parent/child/table/figure/table_row/note)를
내므로 하나의 뷰어·벡터라이저가 전 선급을 처리한다. ``detect_family()`` 가
content_list 내용(발행기관 시그니처)으로 기관을 자동 판별한다.

    from llmops_core.chunking import chunk_document, detect_family, write_jsonl
    family, chunks = chunk_document("data/KR_2_2025/KR_2_2025_content_list.json")
    write_jsonl(chunks, "data_chunks/kr_2.jsonl")
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import abs_guide, bv_rule, dnv_cg, iacs, kr_rule, lr_code, nk_rule

CHUNKERS = {
    "kr_rule": kr_rule.Chunker,
    "nk_rule": nk_rule.Chunker,
    "lr_code": lr_code.Chunker,
    "dnv_cg": dnv_cg.Chunker,
    "bv_rule": bv_rule.Chunker,
    "iacs": iacs.Chunker,
    "abs_guide": abs_guide.Chunker,
}
FAMILIES = tuple(CHUNKERS)

# 발행기관 시그니처(우선순위 순). 각 선급의 *발행처* 식별 — 타 선급 인용과
# 구분하려면 발행기관명(머리글/바닥글에 매 페이지 반복)을 본다. IACS(Rec/UR/UI)는
# 단일 선급 발행처명이 없으므로 **최후순위**로 두어, 어느 선급이든 iacs보다 우선한다.
# (예: ClassNK Part CS는 IACS CSR을 채택해 "common structural rules"를 담지만
#  머리글에 ClassNK 명이 있어 nk_rule로 올바로 라우팅된다.)
_SIGNATURES: list[tuple[str, re.Pattern]] = [
    ("nk_rule", re.compile(r"nippon kaiji kyokai|\bclass\s?nk\b", re.I)),
    # 곧은/곱슬 아포스트로피 모두 허용(문서는 U+2019 "Lloyd's"를 씀) + LR-CO/RU/FR 코드
    ("lr_code", re.compile(r"lloyd['’]?s register|\bLR-(?:CO|RU|FR)-\d", re.I)),
    ("bv_rule", re.compile(r"bureau veritas|\bN[RI]\s?\d{3}\b|\b\d{3}\s?-?\s?N[RI]\b", re.I)),
    # DNV는 발행 컨텍스트에서만(문서번호/법인/저작권) — 맨 "DNV" 인용은 제외
    ("dnv_cg", re.compile(r"det norske veritas|©\s*DNV|\bDNV(?:GL)?[\s-](?:AS|CG|RU|RP|ST|OS|SE)\b", re.I)),
    ("abs_guide", re.compile(r"american bureau of shipping|\bABS\b\s+(?:guide|rules|guidance)", re.I)),
    ("kr_rule", re.compile(r"한국선급|선급 및 강선규칙|규칙\s*제\s*\d+\s*편|korean register|적용지침", re.I)),
    ("iacs", re.compile(r"\bIACS\b|international association of classification societies|common structural rules|recommendation\s+no", re.I)),
]


def _signature_text(items: list[dict], n_text: int = 150, cap: int = 40000) -> str:
    """판별용 텍스트: 문서 전체의 머리글/바닥글(발행기관명이 매 페이지 반복 — 가장
    신뢰도 높은 신호)을 **먼저**, 그다음 앞부분 본문. 조밀한 본문이 헤더를 밀어내
    잘리지 않도록 순서를 고정한다."""
    body, running = [], []
    for b in items:
        t = b.get("text")
        if not t:
            continue
        typ = b.get("type")
        if typ in ("header", "footer", "page_number", "aside_text"):
            running.append(t)
        elif typ == "text" and len(body) < n_text:
            body.append(t)
    return ("\n".join(dict.fromkeys(running)) + "\n" + "\n".join(body))[:cap]


def _load(content_list) -> list[dict]:
    if isinstance(content_list, list):
        return content_list
    return json.loads(Path(content_list).read_text(encoding="utf-8"))


def detect_family(content_list) -> str:
    """content_list 내용으로 발행 선급을 판별. 미상이면 한글비중으로 kr/iacs 폴백."""
    text = _signature_text(_load(content_list))
    for family, rx in _SIGNATURES:          # 우선순위 순(iacs 최후), 첫 일치 채택
        if rx.search(text):
            return family
    return "kr_rule" if len(re.findall(r"[가-힣]", text)) > 40 else "iacs"


# LLM 보강 전용 자리표시자 필드 — 구조적 청킹에선 안 채워지므로 출력 스키마에서 제외.
# (의미 태그·그림 이해는 별도 보강 파이프라인의 몫. notation은 `notations` 필드에 별도 보존.)
# visual_summary는 제외하지 않는다 — MinerU image-analysis가 ETL에서 채우는 실데이터
# (이미지 아이템 content: text_image=OCR, natural_image=시각 설명).
_ENRICH_FIELDS = ("summary", "topic", "keywords", "entities", "ocr_text")


def _strip_enrich(chunks: list[dict]) -> list[dict]:
    for c in chunks:
        for k in _ENRICH_FIELDS:
            c.pop(k, None)
    return chunks


def chunk_document(content_list, family: str = "auto", source_file: str | None = None):
    """content_list → (family, chunks). family='auto'면 detect_family로 판별."""
    items = _load(content_list)
    if family in ("auto", None):
        family = detect_family(items)
    if family not in CHUNKERS:
        raise ValueError(f"unknown family '{family}'; choose from {FAMILIES} or 'auto'")
    if source_file is None:
        source_file = ("content_list.json" if isinstance(content_list, list)
                       else Path(content_list).name)
    return family, _strip_enrich(CHUNKERS[family](items, source_file).run())


def write_jsonl(chunks: list[dict], out_path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _strip_enrich(chunks)                       # 직접 호출 경로도 보강 자리표시자 제외 보장
    with p.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return p

"""학습셋 질문 스타일 증강 — build_training 앞단 전처리.

시험 문형(하라체·조항 경로 명시·완전 명세)으로 생성된 스위트 질문을 실사용자
레지스터로 변형한다. 골든 답변·evidence·라벨은 불변, [상황]/[발췌] 블록은 문서
시뮬레이션이므로 그대로 두고 **질문 래퍼만** 재작성한다.

스타일 3종(항목 인덱스로 결정적 순환):
- polite : 존댓말 실무 질문(~나요?/~ㄹ까요?), 조항 경로는 자연스럽게 유지 가능
- field  : 현장 구어체 — 짧고 직설, 조항 경로 제거·주제어로 대체
- no_path: 존댓말 + 조항 경로 제거(검색이 실제로 일하게)

의미 보존 게이트(프로그램):
① 새 수치 금지 — 변형 래퍼의 내용 수치는 원 질문에 있던 것만 허용
② 결론 누출 금지 — 단정형(적용된다/부적합하다 등) 금지, 의문형 유지
③ 항·호 참조 보존 — 원 래퍼의 "N항 (n)호"류 토큰은 변형에도 있어야 함
④ 경로 제거 스타일이면 「」 부재 + 조항 주제어(제목 실사) 1개 이상 포함
⑤ 한국어·물음표 종결·길이 하한

사용: python -m scripts.aireg_kr.style_augment --src <dir> --out <dir> --limit 5
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

from scripts.aireg_kr.common import chat_json, log
from scripts.aireg_kr.prompts import STYLE_AUGMENT, VERSIONS

STYLES = ("polite", "field", "no_path")
STYLE_DESC = {
    "polite": "존댓말 실무 질문체(~나요?/~는지요?/~ㄹ까요?). 선급 검사·설계 실무자가 "
              "동료나 상담 창구에 묻는 자연스러운 문장. 조항 경로를 언급해도 되지만 "
              "「」 괄호식 인용이 아니라 자연스러운 문장으로 녹일 것.",
    "field": "조선소 현장 구어체. 짧고 직설적(~되나요?/~해야 해요?/~맞아요?). "
             "조항 경로 표기(「규칙 / …」·[대상 조항] 줄)는 쓰지 말고 조항의 "
             "주제를 일상 용어로 언급. 다만 항·호 번호(예: 2항 (1)호)가 원 "
             "질문에 있으면 유지.",
    "no_path": "존댓말 질문체. 조항 경로 표기(「규칙 / …」·[대상 조항] 줄)를 완전히 "
               "제거하고, 무엇에 관한 규정인지 주제로만 언급(검색 시스템이 조항을 "
               "찾아야 하는 상황). 항·호 번호가 원 질문에 있으면 유지.",
}

# 질문 래퍼와 블록 분리 — 첫 "[xxx" 블록 헤더 직전까지가 래퍼
_BLOCK_START = re.compile(r"\n\s*\[(?:상황|발췌|선박/설비 개요)")

# 내용 수치(구조 참조 제외): 뒤에 조항 구조 접미가 붙는 수는 참조로 간주
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_STRUCT_SUFFIX = re.compile(r"^\s*(?:편|장|절|조|항|호|목|류|차|년|판|\.)")

# 단정형 결론(누출) — 의문형("적용되는가/되나요"), 부정 의문은 허용
_CONCL_LEAK = re.compile(
    r"(?:적용된다|적용되지 않는다|적용 대상이 아니다|적합하다|부적합하다|"
    r"충족한다|충족하지 못한다|위반이다)(?![가-힣]*\?)")

_ART_REF = re.compile(r"\d+\s*항(?:\s*\(\d+\)\s*호)?|\(\d+\)\s*호|\([가-힣]\)\s*목")


def split_question(q: str) -> tuple[str, str]:
    m = _BLOCK_START.search(q)
    if not m:
        return q, ""
    return q[:m.start()].rstrip(), q[m.start():]


def _content_nums(text: str) -> set[str]:
    out = set()
    for m in _NUM.finditer(text):
        if m.group(0).count(".") >= 2:  # 다단 번호(1.5.4)는 조항 참조 — 수치 아님
            continue
        if not _STRUCT_SUFFIX.match(text[m.end():m.end() + 3]):
            out.add(m.group(0).replace(",", ""))
    return out


# 조항 제목 범용어 — 주제 식별력이 없어 앵커로 부적합(정의 조항 등에서 빈발)
_GENERIC_TITLE = {"정의", "일반", "일반사항", "총칙", "적용", "적용범위", "기타",
                  "용어", "통칙"}


def _title_words(row: dict) -> list[str]:
    """evidence section_path 말단 제목의 실사(숫자·조사·범용어 제외) — 주제어 앵커."""
    words: list[str] = []
    for ev in row.get("evidence") or []:
        sp = ev.get("section_path") or ""
        tail = sp.split(" / ")[-1] if isinstance(sp, str) else ""
        tail = re.sub(r"^[\d.\-\s]+", "", tail)
        words += [w for w in re.split(r"[\s,()·]+", tail)
                  if len(w) >= 2 and w not in _GENERIC_TITLE]
    return words


# 조 앵커: "703." / "1.5.4"(CSR 다단 번호) / "703조" 또는 조 제목 실사
_ART_NUM_ANCHOR = re.compile(r"\d{3,4}\.(?!\d)|\d+\.\d+(?:\.\d+)+|\d+\s*조")


def _has_article_anchor(row: dict, text: str) -> bool:
    if _ART_NUM_ANCHOR.search(text):
        return True
    return any(w in text for w in _title_words(row))


def _scenario_absorbed(blocks: str, wrapper: str) -> bool:
    """[상황]/[개요] 블록 사실이 래퍼에 충분히 흡수됐는가([발췌]는 흡수 불가).

    수치는 전량 보존 + 실사(어미 변형 감안, 접두 매칭) 커버리지 ≥ 0.6이면 흡수로
    간주해 블록을 제거한다 — 실사용자는 상황을 문장에 녹여 물어보기 때문.
    """
    head = blocks.strip()
    if not (head.startswith("[상황]") or head.startswith("[선박/설비 개요]")):
        return False
    if "[발췌" in blocks:
        return False
    if not _content_nums(blocks) <= _content_nums(wrapper):
        return False
    words = [w for w in re.findall(r"[가-힣]{2,}", blocks) if w not in _GENERIC_TITLE]
    if not words:
        return False
    covered = sum(1 for w in words
                  if w in wrapper or w[:max(2, len(w) - 2)] in wrapper)
    return covered / len(words) >= 0.6


def style_gate_issues(row: dict, orig_wrapper: str, new_wrapper: str,
                      style: str) -> list[str]:
    issues = []
    nw = new_wrapper.strip()
    if len(nw) < 15:
        issues.append("변형 질문이 너무 짧음")
    if not re.search(r"[가-힣]", nw):
        issues.append("한국어가 아님")
    if not re.search(r"[?？]\s*$", nw):
        issues.append("물음표로 끝나지 않음")
    extra = _content_nums(nw) - _content_nums(row.get("question") or "")
    if extra:
        issues.append(f"원 질문에 없는 수치 추가: {sorted(extra)}")
    if _CONCL_LEAK.search(nw):
        issues.append("단정형 결론 누출(의문형이어야 함)")
    missing_refs = [r for r in _ART_REF.findall(orig_wrapper)
                    if re.sub(r"\s+", "", r) not in re.sub(r"\s+", "", nw)]
    if missing_refs:
        issues.append(f"항·호 참조 유실: {missing_refs}")
    # 항·호는 조가 특정된 뒤에야 의미가 생김 — KR 원문 관례상 호 단독 인용은
    # 같은 조 내부에서만 유효하므로, 질문에는 조 앵커(번호 또는 제목 주제어) 필수
    if _ART_REF.search(nw) and not _has_article_anchor(row, nw):
        issues.append("항·호 참조에 조 앵커 부재 — 조 번호(예: 703.)나 조항 "
                      "주제(예: 콘 커플링 조항)를 같은 질문에 함께 쓸 것")
    if style in ("field", "no_path"):
        if "「" in nw or "[대상 조항]" in nw or "규칙 / " in nw:
            issues.append("조항 경로 표기(「」·[대상 조항]·경로)가 남아 있음")
        anchors = _title_words(row)
        if anchors and not any(w in nw for w in anchors):
            # 폴백: 원 래퍼와 실사(2자 이상 한글 어절) 공유가 있으면 주제 유지로 간주
            orig_words = {w for w in re.findall(r"[가-힣]{2,}", orig_wrapper)
                          if w not in _GENERIC_TITLE}
            if not (orig_words & set(re.findall(r"[가-힣]{2,}", nw))):
                issues.append(f"조항 주제어 부재(후보: {anchors[:6]})")
    return issues


# 블록 포장 자연화 — 실서빙(채팅 직접 붙여넣기)에는 "[발췌: …]" 대괄호 라벨이
# 없다. 본문은 프로그램이 축자 이동하므로 내용 불변이 구성상 보장된다.
_BLOCK_SPLIT = re.compile(r"\n\s*\[(발췌[^\]]*|상황|선박/설비 개요)\]\s*\n")

# (스타일 레지스터, 블록 종류) → 도입문. 변형 1은 라벨 없는 맨 붙여넣기.
_INTROS = {
    ("polite", "발췌"): "제출문서에는 이렇게 되어 있습니다:",
    ("field", "발췌"): "설계서엔 이렇게 돼 있어요:",
    ("polite", "상황"): "상황을 설명드리면 이렇습니다:",
    ("field", "상황"): "상황은 이래요:",
}


def repackage_blocks(blocks: str, style: str, qid: str) -> str:
    """대괄호 라벨 블록 → 자연 도입문/맨 붙여넣기(질문 id 기반 결정적 선택)."""
    import zlib
    parts = _BLOCK_SPLIT.split(blocks)
    out = []
    for i in range(1, len(parts), 2):
        header, body = parts[i], parts[i + 1].strip()
        kind = "발췌" if header.startswith("발췌") else "상황"
        reg = "field" if style == "field" else "polite"
        if zlib.crc32(f"{qid}:{header}".encode()) % 2 == 0:
            out.append(_INTROS[(reg, kind)] + "\n" + body)
        else:
            out.append(body)
    return "\n\n" + "\n\n".join(out) if out else ""


def naturalize_question(q: str, qid: str) -> str:
    """구조 라벨([발췌]/[상황]/[개요]) 제거 → 맨 붙여넣기 포장. 프로그램 전용.

    학습 조립(build_training)의 기본 변환 — 실서빙(채팅 직접 붙여넣기)에는
    대괄호 라벨이 없다. 래퍼·본문은 축자 그대로, 라벨만 제거해 빈 줄로 잇는다
    (도입문은 래퍼 어체와 충돌할 수 있어 원본 변환에는 쓰지 않는다 — 어체
    다양화는 스타일 변형본 담당). 원본 파일은 라벨 유지(검수·게이트 파싱)."""
    wrapper, blocks = split_question(q)
    if not blocks:
        return q
    parts = _BLOCK_SPLIT.split(blocks)
    bodies = [parts[i + 1].strip() for i in range(1, len(parts), 2)
              if parts[i + 1].strip()]
    if not bodies:
        return q
    return wrapper + "\n\n" + "\n\n".join(bodies)


def augment_row(row: dict, style: str, retries: int = 2) -> dict | None:
    """질문 래퍼를 style로 재작성한 행 사본을 반환. 게이트 통과 실패 시 None."""
    orig_q = row["question"]
    wrapper, blocks = split_question(orig_q)
    feedback = ""
    for _ in range(retries + 1):
        article = "; ".join(
            (ev.get("section_path") or "").split(" / ")[-1]
            for ev in row.get("evidence") or [] if ev.get("section_path"))
        prompt = STYLE_AUGMENT.format(
            style_desc=STYLE_DESC[style], article=article or "(불명)",
            wrapper=wrapper, blocks=blocks or "(없음)", feedback=feedback)
        try:
            out = chat_json(prompt, max_tokens=900)
        except Exception as e:  # noqa: BLE001
            log(f"  !! LLM 실패: {str(e)[:60]}")
            return None
        nw = (out.get("question") or "").strip()
        issues = style_gate_issues(row, wrapper, nw, style)
        if not issues:
            new = json.loads(json.dumps(row, ensure_ascii=False))
            absorbed = bool(blocks) and _scenario_absorbed(blocks, nw)
            if absorbed or not blocks:
                new["question"] = nw
            else:
                packed = repackage_blocks(blocks, style, row["question_id"])
                # 본문 축자 보존 검증 — 포장만 바뀌어야 한다
                for i, part in enumerate(_BLOCK_SPLIT.split(blocks)):
                    if i % 2 == 0 and part.strip():
                        assert part.strip() in packed, "블록 본문 유실"
                new["question"] = nw + packed
            meta = new.setdefault("metadata", {})
            meta["style_variant"] = style
            meta["style_prompt"] = f"STYLE_AUGMENT/{VERSIONS['STYLE_AUGMENT']}"
            meta["question_orig"] = orig_q
            meta["block_absorbed"] = absorbed
            return new
        feedback = ("\n[이전 시도 문제점 — 반드시 고치시오]\n"
                    + "\n".join(f"- {i}" for i in issues))
    log(f"  게이트 미통과({style}): {issues}")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    src, out = pathlib.Path(args.src), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(src.glob("*_qa.jsonl")):
        for l in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if r.get("status") == "ACCEPT":
                rows.append((p.name, r))
    # 트랙 다양성: 파일 라운드로빈 순서로 limit 적용
    by_file: dict[str, list[dict]] = {}
    for fname, r in rows:
        by_file.setdefault(fname, []).append(r)
    picked, i = [], 0
    while (not args.limit or len(picked) < args.limit) and any(by_file.values()):
        for fname in sorted(by_file):
            if by_file[fname]:
                picked.append((fname, by_file[fname].pop(0)))
                if args.limit and len(picked) >= args.limit:
                    break
        i += 1
        if i > 100:
            break
    ok = 0
    with (out / "styled_qa.jsonl").open("w", encoding="utf-8") as f:
        for idx, (fname, r) in enumerate(picked):
            style = STYLES[idx % len(STYLES)]
            log(f"({idx + 1}/{len(picked)}) {fname[:14]} [{style}] "
                f"{r['question_id'][:-7].split('__')[-1][-30:]}")
            new = augment_row(r, style)
            if new is not None:
                f.write(json.dumps(new, ensure_ascii=False) + "\n")
                ok += 1
                log(f"  → {split_question(new['question'])[0][:80]}")
    log(f"== 스타일 증강: {ok}/{len(picked)} 통과 ==")


if __name__ == "__main__":
    main()

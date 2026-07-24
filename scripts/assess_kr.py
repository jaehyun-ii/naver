#!/usr/bin/env python3
"""KR 문서별로 kr_rule 청커가 의도대로 동작하는지 자동 판정.

표준 KR 규칙 구조 = 편>장>절>조(101. 형)>항(1.)>호((1))>목((가)). 청커는 이 구조
전용이므로, content_list의 구조 신호를 보고 정상/문제를 분류한다:
  · 영문판/비한글  → 한글 마커 없음 → 실패
  · 십진구조(CSR류: 1.1.1) → 조(NNN.)가 거의 없음 → 부적합
  · 조 미검출/0 청크 → 구조 파싱 실패
"""
from __future__ import annotations
import glob, json, re, unicodedata, os

RE_JO   = re.compile(r"^\d{3,4}\.\s*\S")      # 조 101.
RE_DEC  = re.compile(r"^\d+\.\d+")            # 십진 1.1 / 1.1.1
RE_JANG = re.compile(r"^제\s*\d+\s*장")
RE_HANGUL = re.compile(r"[가-힣]")


def load(p):
    return json.loads(open(p, encoding="utf-8").read())


def signals(items):
    texts = [(b.get("text") or "").strip() for b in items if b.get("type") == "text"]
    head = "".join(texts[:400])
    kr = len(RE_HANGUL.findall(head))
    ratio = kr / max(1, len(head))
    n_jo = sum(1 for t in texts if RE_JO.match(t))
    n_dec = sum(1 for t in texts if RE_DEC.match(t))
    n_jang = sum(1 for t in texts if RE_JANG.match(t))
    pages = max((b.get("page_idx", 0) or 0 for b in items), default=0) + 1
    return dict(kr_ratio=ratio, n_jo=n_jo, n_dec=n_dec, n_jang=n_jang,
                pages=pages, blocks=len(items))


def main():
    import sys
    sys.path.insert(0, "/home/jaehyun/Dev/naver")
    from llmops_core.chunking import chunk_document

    cls = [p for p in glob.glob("sample_20/**/*content_list.json", recursive=True)
           if "v2" not in p and "KR" in unicodedata.normalize("NFC", p).split("/")[1]]
    rows = []
    for p in cls:
        name = unicodedata.normalize("NFC", os.path.basename(os.path.dirname(p)))
        items = load(p)
        s = signals(items)
        try:
            _, chunks = chunk_document(p, family="kr_rule")
        except Exception as e:
            rows.append((name, s, 0, 0, f"청킹 예외: {type(e).__name__}", "FAIL"))
            continue
        parents = [c for c in chunks if c["chunk_level"] == "parent"]
        arts_nnn = sum(1 for c in parents if re.fullmatch(r"\d{3,4}", str(c.get("article_no") or "")))
        arts_dec = sum(1 for c in parents if "." in str(c.get("article_no") or ""))
        arts = arts_nnn + arts_dec
        # 판정 — NNN.(표준) 또는 십진(CSR 모드) clause가 잡히면 정상
        if s["kr_ratio"] < 0.20:
            verdict, why = "FAIL", "영문판 — 영문 청커 필요"
        elif not parents or arts < 3:
            verdict, why = "FAIL", f"조/clause 미검출 ({arts})"
        elif arts < max(3, 0.3 * len(parents)):
            verdict, why = "REVIEW", f"조 비율 낮음 {arts}/{len(parents)}"
        else:
            mode = "십진" if arts_dec > arts_nnn else "표준"
            verdict, why = "OK", (f"[{mode}]" if mode == "십진" else "")
        rows.append((name, s, len(parents), arts, why, verdict))

    order = {"OK": 0, "REVIEW": 1, "FAIL": 2}
    rows.sort(key=lambda r: (order[r[5]], unicodedata.normalize("NFC", r[0])))
    print(f"{'문서':40}{'p':>4}{'조패턴':>5}{'십진':>5}{'장':>4}{'한글%':>6}{'parent':>7}{'NNN조':>6}  판정")
    print("─" * 100)
    cnt = {"OK": 0, "REVIEW": 0, "FAIL": 0}
    for name, s, npar, nnn, why, v in rows:
        cnt[v] += 1
        mark = {"OK": "✅", "REVIEW": "⚠️ ", "FAIL": "❌"}[v]
        print(f"{name[:39]:40}{s['pages']:>4}{s['n_jo']:>5}{s['n_dec']:>5}{s['n_jang']:>4}"
              f"{s['kr_ratio']*100:>5.0f}%{npar:>7}{nnn:>6}  {mark}{why}")
    print("─" * 100)
    print(f"정상 {cnt['OK']} · 검토필요 {cnt['REVIEW']} · 실패 {cnt['FAIL']}  (총 {len(rows)})")


if __name__ == "__main__":
    main()

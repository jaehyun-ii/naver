"""입력 데이터 드리프트 탐지 — PSI(Population Stability Index) + 텍스트 분포 (순수 파이썬).

학습/기준 데이터의 분포 요약(summarize)을 기준선으로 저장하고, 라이브/평가 텍스트의 요약과
비교해 분포 변화를 정량화한다. LLM judge·GPU 불필요. MLOps 표준 PSI 임계:
  PSI < 0.1 안정 · 0.1~0.25 경미 · > 0.25 유의(드리프트).

특징(feature):
- length     : 텍스트 길이 분포(고정 경계 히스토그램) → PSI
- vocab      : 상위 토큰 분포 → Jensen-Shannon divergence
- lang/char  : 한글/영문/숫자/기타 문자 비율 → 절대차 합

조선 도메인: 학습 분포(예: 선급/사양서 용어)와 다른 입력(타 도메인 유입)을 조기 탐지.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

# 길이 히스토그램 고정 경계(상한). 기준선·현재가 같은 경계를 써야 PSI 비교 가능.
_LENGTH_BINS = [10, 50, 100, 200, 500, 1000]
_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")
_EPS = 1e-6


def _length_hist(texts: list[str]) -> list[int]:
    counts = [0] * (len(_LENGTH_BINS) + 1)
    for t in texts:
        n = len(t)
        placed = False
        for i, hi in enumerate(_LENGTH_BINS):
            if n < hi:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    return counts


def _char_ratios(texts: list[str]) -> dict[str, float]:
    total = ko = en = dig = other = 0
    for t in texts:
        for ch in t:
            if ch.isspace():
                continue
            total += 1
            if "가" <= ch <= "힣":
                ko += 1
            elif ch.isascii() and ch.isalpha():
                en += 1
            elif ch.isdigit():
                dig += 1
            else:
                other += 1
    total = total or 1
    return {"ko": ko / total, "en": en / total, "digit": dig / total, "other": other / total}


def _vocab(texts: list[str], top: int = 50) -> dict[str, int]:
    c: Counter = Counter()
    for t in texts:
        c.update(m.lower() for m in _TOKEN_RE.findall(t))
    return dict(c.most_common(top))


def summarize(texts: list[str], *, top_vocab: int = 50) -> dict:
    """텍스트 집합 → 분포 요약(기준선/현재 공통 포맷)."""
    texts = [t for t in texts if t]
    return {
        "n": len(texts),
        "length_hist": _length_hist(texts),
        "char_ratios": _char_ratios(texts),
        "vocab": _vocab(texts, top_vocab),
        "avg_len": (sum(len(t) for t in texts) / len(texts)) if texts else 0.0,
    }


def _psi(ref: list[int], cur: list[int]) -> float:
    """Population Stability Index — 두 히스토그램(같은 bin)의 분포 안정성."""
    rs, cs = sum(ref) or 1, sum(cur) or 1
    total = 0.0
    for r, c in zip(ref, cur):
        rp = max(r / rs, _EPS)
        cp = max(c / cs, _EPS)
        total += (cp - rp) * math.log(cp / rp)
    return total


def _js_divergence(ref: dict[str, float], cur: dict[str, float]) -> float:
    """Jensen-Shannon divergence(0~1, log2). 어휘 분포 변화."""
    keys = set(ref) | set(cur)
    if not keys:
        return 0.0
    rs = sum(ref.values()) or 1.0
    cs = sum(cur.values()) or 1.0

    def kl(p, q):
        s = 0.0
        for k in keys:
            pp = p.get(k, 0.0)
            if pp <= 0:
                continue
            s += pp * math.log2(pp / max(q.get(k, 0.0), _EPS))
        return s

    p = {k: ref.get(k, 0.0) / rs for k in keys}
    q = {k: cur.get(k, 0.0) / cs for k in keys}
    m = {k: 0.5 * (p[k] + q[k]) for k in keys}
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


@dataclass
class DriftReport:
    drift: bool
    scores: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    n_reference: int = 0
    n_current: int = 0
    details: dict = field(default_factory=dict)


# 기본 임계값(MLOps 관례)
DEFAULT_THRESHOLDS = {"length_psi": 0.25, "vocab_js": 0.30, "char_shift": 0.20}


def compute_drift(
    reference: dict, current_texts: list[str], *, thresholds: dict | None = None,
) -> DriftReport:
    """기준선 요약 vs 현재 텍스트 → 드리프트 점수·플래그."""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    cur = summarize(current_texts, top_vocab=len(reference.get("vocab", {})) or 50)

    length_psi = _psi(reference.get("length_hist", []), cur["length_hist"])
    vocab_js = _js_divergence(reference.get("vocab", {}), cur["vocab"])
    rc = reference.get("char_ratios", {})
    cc = cur["char_ratios"]
    char_shift = sum(abs(cc.get(k, 0.0) - rc.get(k, 0.0)) for k in {"ko", "en", "digit", "other"})

    scores = {"length_psi": round(length_psi, 4), "vocab_js": round(vocab_js, 4),
              "char_shift": round(char_shift, 4)}
    drift = any(scores[k] > th[k] for k in scores)
    return DriftReport(
        drift=drift, scores=scores, thresholds=th,
        n_reference=reference.get("n", 0), n_current=cur["n"],
        details={"avg_len_ref": reference.get("avg_len"), "avg_len_cur": cur["avg_len"]},
    )

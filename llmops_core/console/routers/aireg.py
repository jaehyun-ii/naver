"""AIReg 벤치·학습 뷰어 — 데이터 생성/학습/판정 산출물을 실시간 조회.

파일 기반 read-only 라우터: data_aireg/<suite>/ 의 트랙 QA·배치 로그·분할·
생성 평가(judge 점수/이유)를 그대로 노출한다. 배치가 도는 중에는 파일이
append되므로 offset 폴링으로 라이브 테일이 된다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from llmops_core.console.security import require_master

router = APIRouter(prefix="/api/aireg", tags=["aireg"],
                   dependencies=[Depends(require_master)])

# 컨테이너에선 코드가 /app 아래 마운트되고 레포는 호스트 경로 그대로 마운트됨
# — 존재하는 후보를 선택 (review.py와 동일 관례).
_CANDIDATES = [Path(__file__).resolve().parents[3] / "data_aireg",
               Path("/home/jaehyun/Dev/naver/data_aireg")]
DATA = next((p for p in _CANDIDATES if p.is_dir()), _CANDIDATES[0])
TRACK_FILES = ["spec_qa.jsonl", "applicability_qa.jsonl", "crossref_qa.jsonl",
               "def_link_qa.jsonl", "precedence_qa.jsonl", "unit_convert_qa.jsonl",
               "table_lookup_qa.jsonl", "hierarchy_qa.jsonl"]
_LABEL = re.compile(r"판단\s*[:：]\s*\n?\s*(미적용|적용|조건부|부적합|적합|판단 불가)")


def _suite_dir(suite: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_]+", suite):
        raise HTTPException(422, "잘못된 suite 이름")
    d = DATA / suite
    if not d.is_dir():
        raise HTTPException(404, f"suite 없음: {suite}")
    return d


def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


@router.get("/suites")
def suites() -> list[str]:
    return sorted(p.name for p in DATA.iterdir()
                  if p.is_dir() and any((p / f).exists() for f in TRACK_FILES))


@router.get("/overview")
def overview(suite: str = "suite_v5") -> dict:
    d = _suite_dir(suite)
    tracks: dict[str, dict] = {}
    for f in TRACK_FILES:
        rows = _read_jsonl(d / f)
        if not rows:
            continue
        st: dict[str, int] = {}
        for r in rows:
            st[r.get("status", "?")] = st.get(r.get("status", "?"), 0) + 1
        tracks[f.replace("_qa.jsonl", "")] = {"total": len(rows), **st}
    split = None
    sp = d / "split_report.json"
    if sp.exists():
        split = json.loads(sp.read_text(encoding="utf-8"))
    batch_tail = []
    bl = d / "batch.log"
    if bl.exists():
        batch_tail = bl.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
    # 생성 평가 진행/judge 요약
    shards = d / "gen_shards"
    gen = {"tuned": 0, "base": 0}
    if shards.is_dir():
        for p in shards.glob("gt_p*.jsonl"):
            gen["tuned"] += sum(1 for l in p.read_text(encoding="utf-8").splitlines() if l.strip())
        for p in shards.glob("gb_p*.jsonl"):
            gen["base"] += sum(1 for l in p.read_text(encoding="utf-8").splitlines() if l.strip())
    judge = {}
    for j in _read_jsonl(d / "judge_scores.jsonl"):
        if j.get("score", -1) < 0:
            continue
        t = judge.setdefault(j["tag"], {"n": 0, "sum": 0.0, "ge4": 0})
        t["n"] += 1
        t["sum"] += j["score"]
        t["ge4"] += j["score"] >= 4
    judge_summary = {k: {"n": v["n"], "avg": round(v["sum"] / v["n"], 2),
                         "ge4_ratio": round(v["ge4"] / v["n"], 3)}
                     for k, v in judge.items() if v["n"]}
    summary = None
    sm = d / "summary.json"
    if sm.exists():
        summary = json.loads(sm.read_text(encoding="utf-8"))
    return {"suite": suite, "tracks": tracks, "split": split,
            "batch_tail": batch_tail, "gen_progress": gen,
            "judge_summary": judge_summary, "summary": summary}


@router.get("/qa")
def qa(suite: str = "suite_v5", track: str | None = None,
       status: str | None = None, offset: int = 0,
       limit: int = Query(50, le=200)) -> dict:
    d = _suite_dir(suite)
    files = ([f"{track}_qa.jsonl"] if track else TRACK_FILES)
    rows: list[dict] = []
    for f in files:
        for r in _read_jsonl(d / f):
            if status and r.get("status") != status:
                continue
            rows.append({
                "question_id": r.get("question_id", ""),
                "track": r.get("track", f.replace("_qa.jsonl", "")),
                "status": r.get("status"),
                "review_reasons": r.get("review_reasons") or [],
                "question": (r.get("question") or "")[:600],
                "gold_answer": (r.get("gold_answer") or "")[:600],
                "ts": (r.get("_gen") or {}).get("ts", ""),
            })
    rows.sort(key=lambda r: r["ts"], reverse=True)
    return {"total": len(rows), "rows": rows[offset:offset + limit]}


@router.get("/geneval")
def geneval(suite: str = "suite_v5", offset: int = 0,
            limit: int = Query(30, le=100), only: str = "all") -> dict:
    """생성 평가 뷰 — val 문항별 정답·튜닝·베이스 응답 + judge 점수/이유."""
    d = _suite_dir(suite)
    val_path = d / "sft_val.messages.jsonl"
    if not val_path.exists():
        raise HTTPException(404, "sft_val.messages.jsonl 없음")
    gens: dict[tuple[str, str], dict] = {}
    shards = d / "gen_shards"
    if shards.is_dir():
        for p in shards.glob("gt_p*.jsonl"):
            for r in _read_jsonl(p):
                gens[("tuned", r["source"])] = r
        for p in shards.glob("gb_p*.jsonl"):
            for r in _read_jsonl(p):
                gens[("base", r["source"])] = r
    judge: dict[tuple[str, str], dict] = {}
    for j in _read_jsonl(d / "judge_scores.jsonl"):
        if j.get("score", -1) >= 0:
            judge[(j["tag"], j["source"])] = j
    rows = []
    for v in _read_jsonl(val_path):
        src = v["source"]
        q = v["messages"][0]["content"]
        i = q.rfind("[질문]")
        gold = v["messages"][1]["content"]
        t, b = gens.get(("tuned", src)), gens.get(("base", src))
        jt, jb = judge.get(("tuned", src)), judge.get(("base", src))
        glab = _LABEL.search(gold)
        tlab = _LABEL.search(t["gen"]) if t else None
        row = {
            "source": src,
            "track": src.split(":")[0].replace("aireg_kr_", ""),
            "question": q[i + 4:].strip()[:800] if i >= 0 else q[-800:],
            "gold": gold[:1000],
            "tuned_gen": (t or {}).get("gen", "")[:1000],
            "base_gen": (b or {}).get("gen", "")[:1000],
            "tuned_score": (jt or {}).get("score"),
            "tuned_reason": (jt or {}).get("reason", ""),
            "base_score": (jb or {}).get("score"),
            "base_reason": (jb or {}).get("reason", ""),
            "gold_label": glab.group(1) if glab else None,
            "tuned_label": tlab.group(1) if tlab else None,
        }
        row["label_match"] = (row["gold_label"] == row["tuned_label"]
                              if row["gold_label"] else None)
        if only == "mismatch" and row["label_match"] is not False:
            continue
        if only == "scored" and row["tuned_score"] is None:
            continue
        rows.append(row)
    return {"total": len(rows), "rows": rows[offset:offset + limit]}


_CFG = {  # HPO 실험 설정 매핑 (H100 3라운드)
    "cfgA": "r16/α32 lr5e-5 3ep", "cfgB": "r32/α32 lr5e-5 3ep",
    "cfgC": "r16/α32 lr1e-4 2ep", "cfgD": "r16/α32 lr1e-5 3ep",
    "cfgE": "r16/α32 lr1.5e-4 2ep", "cfgF": "r16/α32 lr2e-4 2ep",
    "cfgG": "r16/α32 lr3e-4 2ep", "cfgH": "r16/α32 lr7e-5 3ep",
    "cfgI": "r16/α32 lr1e-4 3ep", "cfgJ": "r8/α16 lr1e-4 2ep",
    "cfgK": "r32/α64 lr1e-4 2ep", "cfgL": "r16/α64 lr1e-4 2ep",
    "cfgM": "r16/α32 lr5e-4 2ep", "cfgN": "r16/α32 lr7e-4 2ep",
    "cfgO": "r16/α32 lr3e-4 3ep", "cfgP": "r16/α64 lr3e-4 2ep",
    "cfgQ": "r16/α32 lr1e-3 2ep", "cfgR": "r16/α64 lr5e-4 3ep",
    "cfgS": "r16/α32 lr2e-3 2ep(파괴점)", "base32b": "베이스(무학습)",
    "main800": "본학습 r16/α64 lr2e-4 3ep · 800건",
}


@router.get("/train")
def train(suite: str = "suite_v5") -> dict:
    """학습 실험(HPO+본학습) 결과와 로그 — H100 산출물 동기화본."""
    d = _suite_dir(suite) / "train_runs"
    exps = []
    for r in _read_jsonl(d / "experiments.jsonl"):
        exps.append({"name": r["name"], "config": _CFG.get(r["name"], ""),
                     "val_loss": (r.get("val") or {}).get("val_loss"),
                     "n": (r.get("val") or {}).get("n")})
    exps.sort(key=lambda x: (x["val_loss"] is None, x["val_loss"] or 9))
    logs_text = ""
    lt = d / "train_logs.txt"
    if lt.exists():
        logs_text = lt.read_text(encoding="utf-8", errors="replace")[-20000:]
    return {"experiments": exps, "logs": logs_text}


@router.get("/retrieval")
def retrieval_bench(suite: str = "suite_v5") -> dict:
    """검색 벤치 산출물 전부 — 임베더/리랭커/최종스택/무앵커."""
    d = _suite_dir(suite) / "retrieval_bench"
    out: dict[str, object] = {}
    if d.is_dir():
        for p in sorted(d.glob("*.json")):
            try:
                out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return out


@router.get("/logs")
def logs(suite: str = "suite_v5", name: str = "batch",
         lines: int = Query(80, le=400)) -> dict:
    d = _suite_dir(suite)
    safe = {"batch": "batch.log", "split": "split_stdout.log"}
    fn = safe.get(name)
    if not fn:
        raise HTTPException(422, "지원 로그: batch, split")
    p = d / fn
    if not p.exists():
        return {"lines": []}
    return {"lines": p.read_text(encoding="utf-8", errors="replace")
            .splitlines()[-lines:]}

"""AIReg-Bench 선급 변형 파이프라인 공용 유틸.

- LLM 호출: OpenAI 호환 엔드포인트(기본: 로컬 hf_server의 HCX SEED Think 14B).
  Think 계열이므로 chat_template_kwargs={"skip_reasoning": True}로 사고 블록을 끈다.
- 조항 소스: data_chunks/KR parent 청크(조 단위 전문 + section_path).
- 모든 산출물은 JSONL append + id 기반 resume.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import sys
import time
import unicodedata
from fnmatch import fnmatch
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent.parent

# 저장소 루트 .env(gitignore 대상) 자동 로드 — 이미 설정된 환경변수가 우선.
# 검증기 분리(AIREG_VERIFY_*)·API 키를 코드/셸 밖에서 관리하기 위함.
if (REPO / ".env").exists():
    for _line in (REPO / ".env").read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())
    del _line, _k, _v
# 생성 모델별 데이터셋 분리(혼합 방지)·비교 실험용으로 출력 디렉터리를 env로 전환 가능
OUT_DIR = Path(os.environ.get("AIREG_OUT_DIR", REPO / "data_aireg"))
# 청크 코퍼스 — 재청킹으로 디렉터리가 교체될 수 있어 env 우선, 없으면 존재하는 후보
_CHUNK_CANDIDATES = [REPO / "data_chunks", REPO / "data_chunks_full"]
CHUNK_DIR = (Path(os.environ["AIREG_CHUNK_DIR"]) if os.environ.get("AIREG_CHUNK_DIR")
             else next((p for p in _CHUNK_CANDIDATES if p.exists()), _CHUNK_CANDIDATES[0]))
# 하위 호환(기존 스크립트가 참조) — KR 기본 파일
CHUNKS = CHUNK_DIR / "KR" / "5편_2025_chunks.jsonl"

# 기본 teacher: 사내망 MLX 서버 (DeepSeek-V4-Flash) — Ollama gemma4에서 교체(2026-07-16).
# 생성·검증이 같은 모델이지만 프로그램 평가기(제3축)·정적 게이트가 자기편향을 견제한다.
LLM_BASE = os.environ.get("AIREG_LLM_BASE", "http://192.168.0.29:8080/v1")
LLM_MODEL = os.environ.get("AIREG_LLM_MODEL", "mlx-community/DeepSeek-V4-Flash-4bit")
LLM_API_KEY = os.environ.get("AIREG_LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
# 독립 검증자(verify.py)는 생성 모델과 분리 가능 — 미지정 시 동일 엔드포인트를 쓰되
# 프롬프트·온도가 다르므로 최소한의 분리는 유지된다.
VERIFY_LLM_BASE = os.environ.get("AIREG_VERIFY_LLM_BASE", LLM_BASE)
VERIFY_LLM_MODEL = os.environ.get("AIREG_VERIFY_LLM_MODEL", LLM_MODEL)
# min_tokens·chat_template_kwargs는 로컬 hf_server 전용 확장 필드
IS_LOCAL = "localhost" in LLM_BASE or "127.0.0.1" in LLM_BASE
# 기본은 force_reasoning — 사고 헤더(`assistant/think\n`)를 완성해서 줘야 안정적이다.
# skip_reasoning(빈 think)이나 기본 헤더(개행 없는 `assistant`)에서는 "첫 줄은 X로
# 시작" 류 지시가 헤더 개행을 밀어내 모델이 조기 종료(<|stop|>)하는 문제가 있었다.
# hf_server가 사고/답변을 분리해 content에는 최종 답변만 온다. (HCX Think 로컬 전용)
FORCE_REASONING = IS_LOCAL and os.environ.get("AIREG_FORCE_REASONING", "1") == "1"


def chat(prompt: str, *, max_tokens: int = 700, temperature: float = 0.5,
         retries: int = 3, timeout: int = 300, system: str | None = None,
         min_chars: int = 30, must_contain: str | None = None,
         min_tokens: int = 200, base: str | None = None,
         model: str | None = None, reasoning_effort: str | None = None) -> str:
    """단일 user 턴 호출. 실패·비정상 응답 시 재시도.

    base/model로 엔드포인트를 오버라이드할 수 있다(독립 검증자 분리용).

    Think 계열의 두 가지 비정상 모드를 걸러낸다:
    - 답변 헤더 직후 조기 종료(<|stop|>) → min_chars 미만이면 재생성
    - 사고가 답변으로 전환되지 않고 통째로 content가 되는 경우
      → must_contain(형식 앵커 문자열)이 없으면 재생성
    """
    use_base = base or LLM_BASE
    is_local = "localhost" in use_base or "127.0.0.1" in use_base
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}]
    body = {
        "model": model or LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if is_local:
        body["min_tokens"] = min_tokens
        if FORCE_REASONING:
            body["chat_template_kwargs"] = {"force_reasoning": True}
    else:
        # 원격 reasoning 모델은 사고 토큰이 max_tokens를 잠식해 content가 비는 경우가
        # 있다(hy3는 단답에도 사고 ~700토큰) — 한도를 넉넉히 잡고 사고량을 낮춘다.
        # 검증 등 판단 품질이 중요한 호출은 reasoning_effort로 개별 상향한다
        # (실측: gpt-oss-120b 검증이 low에서 결함 0/4 검출, high에서 3/4).
        body["max_tokens"] = max(max_tokens * 3, 6000)
        body["reasoning"] = {"effort": reasoning_effort
                             or os.environ.get("AIREG_REASONING_EFFORT", "low")}
    # API 키는 외부(https) 엔드포인트에만 전송 — 사내 http 서버로 키 유출 방지
    headers = ({"Authorization": f"Bearer {LLM_API_KEY}"}
               if LLM_API_KEY and use_base.startswith("https://") else {})
    last = None
    for i in range(retries):
        try:
            r = requests.post(f"{use_base}/chat/completions", json=body,
                              headers=headers, timeout=timeout)
            if r.status_code == 429:  # 원격 무료 티어 rate limit — 길게 대기 후 재시도
                last = f"429 rate limited: {r.text[:200]}"
                time.sleep(30 * (i + 1))
                continue
            r.raise_for_status()
            content = (r.json()["choices"][0]["message"].get("content") or "").strip()
            if len(content) >= min_chars and (not must_contain or must_contain in content):
                return content
            last = f"비정상 응답({len(content)}자, 앵커 {must_contain!r} 부재): {content[:80]!r}"
            body["temperature"] = min(1.0, temperature + 0.15 * (i + 1))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"LLM 호출 실패({retries}회): {last}")


def extract_json(text: str) -> dict | None:
    """응답에서 첫 번째 균형 잡힌 JSON 오브젝트를 추출."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


JSON_SYSTEM = ("최종 답변은 유효한 JSON 오브젝트 하나여야 한다. "
               "최종 답변에 설명·서론·코드블록을 넣지 않는다.")


def chat_json(prompt: str, *, max_tokens: int = 1800, retries: int = 3,
              base: str | None = None, model: str | None = None,
              reasoning_effort: str | None = None,
              timeout: int = 300) -> dict:
    """JSON 응답 강제 호출 — 파싱 실패 시 온도·토큰 한도를 올려 재생성.

    force_reasoning 모드라 사고 분량까지 포함해 max_tokens를 넉넉히 잡고,
    재시도마다 더 늘린다(잘림 대비).
    """
    for i in range(retries):
        # 재시도 시 토큰 한도 증가는 1회분(+600)까지만 — 서버 정체 시 점점 큰
        # 요청을 던져 큐 교착을 악화시키는 악순환 차단(2026-07-21 MLX 정체 실측)
        ans = chat(prompt, max_tokens=max_tokens + 600 * min(i, 1),
                   temperature=0.3 + 0.2 * i, system=JSON_SYSTEM, min_tokens=120,
                   base=base, model=model, reasoning_effort=reasoning_effort,
                   timeout=timeout)
        obj = extract_json(ans)
        if obj:
            return obj
    raise RuntimeError(f"JSON 파싱 실패({retries}회). 마지막 응답:\n{ans[:500]}")


def gen_meta(prompt_id: str, *, model: str | None = None) -> dict:
    """산출 행에 붙일 provenance — 프롬프트 id/버전·모델·시각.

    검수·재현 추적용: 어떤 프롬프트 버전과 모델이 이 행을 만들었는지 남긴다.
    """
    from . import prompts
    return {
        "prompt": f"{prompt_id}/{prompts.VERSIONS.get(prompt_id, 'v1')}",
        "model": model or LLM_MODEL,
        # 타임존 명시(외부 검토 반영) — "+09:00" 오프셋 포함 RFC3339
        "ts": __import__("datetime").datetime.now().astimezone().isoformat(
            timespec="seconds"),
    }


# ── JSONL I/O (id 기반 resume, 병렬 워커 안전) ──────────────────────────
# append는 flock 배타 잠금(4096B 초과 행의 교차 기록 방지), 읽기는 다른 워커가
# 쓰는 중일 수 있어 손상 행(마지막 미완성 행)을 건너뛴다 — 해당 행은 그 행을
# 쓰던 워커의 소유라 resume 정합성에 영향 없다.
def _iter_rows(path: Path):
    for l in path.open(encoding="utf-8"):
        if not l.strip():
            continue
        try:
            yield json.loads(l)
        except json.JSONDecodeError:
            continue


def load_done(path: Path, key: str = "id") -> set[str]:
    if not path.exists():
        return set()
    return {r[key] for r in _iter_rows(path)}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(_iter_rows(path))


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def pmap(fn, items, workers: int = 1) -> None:
    """items를 워커 스레드로 병렬 처리(순서 무관, IO-bound LLM 호출용).

    MLX 서버가 동시 요청을 배치 처리함을 실측(6병렬 ≈ 처리량 2.4×) — 파이프라인
    병목은 클라이언트 직렬화였다. 예외는 fn 안에서 처리(기존 루프와 동일 규약)."""
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        for it in items:
            fn(it)
        return
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(fn, items))


# ── 대상 조항 선정 ──────────────────────────────────────────────────────
# 적합성 판정 대상이 아닌 조(용어 정의·적용 범위 선언·제출서류 목록·목차 등)는 제외.
# 비-KR 선급은 영어 제목이므로 영어 비요건 제목도 함께 거른다.
NON_REQUIREMENT_TITLES = re.compile(
    r"정의|용어|일반사항$|적용$|제출도면"
    r"|^(Scope|General|Objective|Definitions?|Application|Introduction|Contents"
    r"|Abbreviations?|References?|Terminology)\s*$",
    re.IGNORECASE)


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def glob_docs(directory: Path, pattern: str) -> list[Path]:
    """한글 파일명의 NFC/NFD 차이를 흡수하는 글롭.

    data_chunks_full의 파일명이 자모 분해(NFD, 맥 복사본)라 소스의 NFC 패턴
    ("5편_*" 등)과 Path.glob이 매칭되지 않는다 — 양쪽을 NFC로 정규화해 비교."""
    pat = unicodedata.normalize("NFC", pattern)
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and fnmatch(unicodedata.normalize("NFC", p.name), pat))


def _load_publisher(publisher: str, doc_glob: str) -> list[dict]:
    rows: list[dict] = []
    for f in glob_docs(CHUNK_DIR / publisher, doc_glob):
        for l in f.open(encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            r["_source_file"] = nfc(f.stem)
            rows.append(r)
    return rows


def _is_requirement_parent(r: dict, publisher: str, min_tokens: int) -> bool:
    if r.get("chunk_level") != "parent" or r.get("content_tokens", 0) < min_tokens:
        return False
    if NON_REQUIREMENT_TITLES.search(r.get("article_title") or ""):
        return False
    if publisher == "KR":
        # GUIDE/지침은 규칙 본문의 미러 중복 — 규칙 세그먼트만 사용.
        # 신 청커는 document_type("rule")을 제공, 구 산출물은 RULE_ 접두로 폴백.
        if r.get("document_type"):
            return r["document_type"] == "rule"
        return r["chunk_id"].startswith("RULE_")
    # 비-KR: 목차/서문 청크, 문자 없는 제목("(2009)" 등 파싱 잔재) 제외
    title = r.get("article_title") or ""
    return (not r["chunk_id"].endswith(("INTRO", "TOC"))
            and bool(re.search(r"[A-Za-z가-힣]", title)))


def _attach_tables(parents: list[dict], tables_by_id: dict[str, dict],
                   cap_chars: int = 12000) -> None:
    """조 parent의 content 뒤에 링크된 표(table_html)를 이어붙인다.

    parent content에는 "표 N에 따른다" 참조 문장만 있고 값은 표 청크의
    table_html에만 있다(2026-07-16 실측: KR 5편 표 참조 134건 중 129건 부재).
    카드 생성 LLM이 표 속 기준치를 봐야 경계값 트랙 수율이 살고, evidence
    부분문자열 검증도 article_text 기준이라 표 인용이 유효해진다.
    거대 표 폭주 방지로 조당 cap_chars에서 자른다(잘림은 명시)."""
    for p in parents:
        tids = list(dict.fromkeys((p.get("linked_table_chunk_ids") or [])
                                  + (p.get("linked_tables") or [])))
        blocks, total = [], 0
        for tid in tids:
            t = tables_by_id.get(tid)
            if not t:
                continue
            cap = (t.get("content") or "").strip()
            html = (t.get("table_html") or "").strip()
            if not html:
                continue
            block = f"{cap}\n{html}" if cap and cap not in html else html
            if total + len(block) > cap_chars:
                blocks.append("(이하 표 생략 — 분량 제한)")
                break
            blocks.append(block)
            total += len(block)
        if blocks:
            p["content"] = p["content"].rstrip() + "\n\n[인용된 표]\n" + "\n\n".join(blocks)


def select_rules(chapters: str = "7,2", n: int = 30, min_tokens: int = 150,
                 publisher: str = "KR", doc_glob: str | None = None) -> list[dict]:
    """실질 요건이 있는 규칙 조(parent)를 n개 선정.

    - KR: doc_glob 기본 "5편_*.jsonl", chapters(예: "7,2")는 장 우선순위.
    - 비-KR: doc_glob 기본 전체, chapters는 지정 시에만 필터(문서 구조가 다양해
      기본은 문서·장 순서대로 채운다).
    """
    if doc_glob is None:
        doc_glob = "5편_*_chunks.jsonl" if publisher == "KR" else "*_chunks.jsonl"
    rows = _load_publisher(publisher, doc_glob)
    parents = [r for r in rows if _is_requirement_parent(r, publisher, min_tokens)]
    for r in parents:
        r["publisher"] = publisher
    picked: list[dict] = []
    chapter_list = [c.strip() for c in chapters.split(",")] if chapters else [None]
    for chapter in chapter_list:
        cand = [r for r in parents if chapter is None or str(r.get("chapter_no")) == chapter]
        cand.sort(key=lambda r: (r["_source_file"], str(r.get("section_no")), str(r.get("article_no"))))
        picked.extend(c for c in cand if c not in picked)
        if len(picked) >= n:
            break
    picked = picked[:n]
    _attach_tables(picked, {r["chunk_id"]: r for r in rows
                            if r.get("chunk_type") == "table"})
    return picked


ALL_PUBLISHERS = "KR,ClassNK,DNV,ABS,BV,IACS,LR"


def select_one_per_doc(publishers: str = ALL_PUBLISHERS, min_tokens: int = 150,
                       prefer_max_tokens: int = 1500) -> list[dict]:
    """문서(파일)당 대표 요건 조 1개 선정 — 문서당 1문항 판정 트랙용.

    실질 요건 조 중 내용이 가장 풍부한 것을 고르되, 거대 표 조항을 피하기 위해
    prefer_max_tokens 이하에서 최대를 우선한다(없으면 그중 최소 초과분).
    """
    picked: list[dict] = []
    for publisher in [p.strip() for p in publishers.split(",")]:
        for f in glob_docs(CHUNK_DIR / publisher, "*_chunks.jsonl"):
            cands, tables = [], {}
            for l in f.open(encoding="utf-8"):
                if '"parent"' not in l and '"table"' not in l:
                    continue
                r = json.loads(l)
                if r.get("chunk_type") == "table":
                    tables[r["chunk_id"]] = r
                elif _is_requirement_parent(r, publisher, min_tokens):
                    r["publisher"] = publisher
                    r["_source_file"] = nfc(f.stem)
                    cands.append(r)
            if not cands:
                continue
            small = [c for c in cands if c["content_tokens"] <= prefer_max_tokens]
            best = (max(small, key=lambda c: c["content_tokens"]) if small
                    else min(cands, key=lambda c: c["content_tokens"]))
            _attach_tables([best], tables)
            picked.append(best)
    return picked


def rule_uid(chunk: dict) -> str:
    """전역 유일 id. KR RULE_*은 자체 유일, 비-KR은 파일별 chunk_id 중복이
    가능(예: ABS 문서마다 ABS_S1_1)해 발행처·파일명을 접두한다."""
    if chunk["chunk_id"].startswith("RULE_"):
        return chunk["chunk_id"]
    return f"{chunk.get('publisher', '?')}__{chunk.get('_source_file', '?')}__{chunk['chunk_id']}"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    sys.stdout.flush()

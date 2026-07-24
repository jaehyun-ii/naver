"""prompts.py의 프롬프트 원문을 PROMPTS_FULL.md로 추출.

모듈을 임포트해 **실효 텍스트**(LANG_RULE 부착 후)를 쓰므로 모델에 실제로
전송되는 내용과 일치한다. 프롬프트 수정 후 재실행하면 문서가 갱신된다:

    PYTHONPATH=. python3 scripts/aireg_kr/build_prompts_doc.py
"""
from __future__ import annotations

import re
from pathlib import Path

from scripts.aireg_kr import prompts

SRC = Path(prompts.__file__)
OUT = SRC.parent / "PROMPTS_FULL.md"

# 정의 순서 보존 — 소스에서 모듈 수준 상수 등장 순서를 취한다
names = [m.group(1) for m in
         re.finditer(r"^([A-Z][A-Z0-9_]*) = ", SRC.read_text(encoding="utf-8"),
                     re.M)
         if isinstance(getattr(prompts, m.group(1), None), str)]

lines = [
    "# AIReg-KR 프롬프트 원문 전체",
    "",
    f"`{SRC.name}`에서 자동 추출(`build_prompts_doc.py`). **직접 수정 금지** — ",
    "원본 수정 후 스크립트를 재실행할 것. LANG_RULE이 부착되는 프롬프트는 부착된",
    "실효 텍스트 그대로이며, `{placeholder}`는 실행 시 채워지는 포맷 슬롯이다.",
    "",
    "요약·설계 배경은 [PROMPTS.md](PROMPTS.md) 참조.",
    "",
    "## 목차",
    "",
]
for n in names:
    ver = prompts.VERSIONS.get(n)
    lines.append(f"- [{n}](#{n.lower().replace('_', '-')})"
                 + (f" ({ver})" if ver else ""))
lines.append("")

for n in names:
    ver = prompts.VERSIONS.get(n)
    lines += [f"## {n}" + (f" ({ver})" if ver else ""), "", "````text",
              getattr(prompts, n).strip("\n"), "````", ""]

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"{OUT.name}: 프롬프트 {len(names)}개, {OUT.stat().st_size:,} bytes")

"""서빙 가드레일 — 입력 프롬프트인젝션·출력 모더레이션·PII 누출 차단.

게이트웨이 요청 경로에 끼워 입력(messages)과 출력(생성 텍스트)을 검사한다.
기본은 순수 휴리스틱(추가 의존성 0). mask_output_pii=True면 Presidio(quality extra)로 출력 PII 마스킹.
LiteLLM promptguard 등 외부 가드는 이 인터페이스 뒤에서 교체 가능.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from llmops_core.common.config import get_settings

# 프롬프트 인젝션·탈옥 패턴(한/영). 보수적으로 흔한 공격 표현만.
_INJECTION_PATTERNS = [
    r"ignore (all |the |your )?(previous|above|prior) (instructions|prompts?)",
    r"disregard (all |the )?(previous|above|prior)",
    r"forget (all |the |your )?(previous|above|prior|earlier)",
    r"system prompt",
    r"reveal (your |the )?(system )?(prompt|instructions)",
    r"developer mode",
    r"\bDAN\b",
    r"jailbreak",
    r"이전\s*(지시|명령|프롬프트)\s*(무시|잊)",
    r"시스템\s*프롬프트",
    r"규칙\s*무시",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str | None = None
    text: str | None = None  # 마스킹/정제된 텍스트(있으면 대체 사용)
    flags: list[str] = field(default_factory=list)


class GuardrailEngine:
    """입력/출력 가드레일. 설정(guardrails.*)으로 동작 제어."""

    def __init__(self, settings=None) -> None:
        self.cfg = (settings or get_settings()).guardrails
        self._banned = [b.lower() for b in self.cfg.banned_terms]
        self._masker = None
        if self.cfg.mask_output_pii:
            try:
                from llmops_core.quality.clean import PIIMasker

                self._masker = PIIMasker(lang="en")
            except Exception:  # noqa: BLE001  # pragma: no cover
                self._masker = None  # Presidio 미설치면 마스킹 비활성(차단은 유지)

    # ── 입력: 프롬프트 인젝션 ──
    def check_input(self, messages: list[dict]) -> GuardrailResult:
        if not self.cfg.enabled:
            return GuardrailResult(allowed=True)
        text = "\n".join(str(m.get("content", "")) for m in messages)
        hits = [p.pattern for p in _INJECTION_RE if p.search(text)]
        if hits and self.cfg.block_on_injection:
            return GuardrailResult(allowed=False, reason="prompt_injection_detected",
                                   flags=hits)
        return GuardrailResult(allowed=True, flags=hits)

    # ── 출력: 금칙어·PII ──
    def check_output(self, text: str) -> GuardrailResult:
        if not self.cfg.enabled:
            return GuardrailResult(allowed=True, text=text)
        low = text.lower()
        banned_hit = [b for b in self._banned if b in low]
        if banned_hit and self.cfg.block_on_banned:
            return GuardrailResult(allowed=False, reason="banned_term_in_output",
                                   flags=banned_hit)
        out = text
        flags: list[str] = []
        if self._masker is not None:
            masked = self._masker.mask(text)
            if masked != text:
                out, flags = masked, ["pii_masked"]
        return GuardrailResult(allowed=True, text=out, flags=flags)

"""L3 데이터셋 가공 — SFT/Preference 포맷 변환·필터·결정적 분할 (의존성 없음).

라벨링 결과(또는 정제된 레코드)를 학습기가 그대로 소비하는 chat 포맷으로 변환하고,
재현 가능한 train/val/test 분할을 만든다. 분할은 난수가 아닌 해시 버킷이라
같은 입력 → 항상 같은 분할(데이터 버전별 비교의 전제).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from llmops_core.common.schemas import ChatMessage, PreferenceExample, SFTExample


@dataclass
class SplitConfig:
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: str = "v1"  # 분할 해시 솔트(버전 바꾸면 분할도 재배치)


# ── 포맷 변환 ──
def to_sft_examples(
    labeled: list[dict], *, system: str | None = None, source: str | None = None
) -> list[SFTExample]:
    """{text(user), response(assistant)} 라벨링 결과 → SFT chat 예제."""
    out: list[SFTExample] = []
    for row in labeled:
        user = row.get("text")
        assistant = row.get("response")
        if not user or not assistant:
            continue
        msgs: list[ChatMessage] = []
        if system:
            msgs.append(ChatMessage(role="system", content=system))
        msgs.append(ChatMessage(role="user", content=user))
        msgs.append(ChatMessage(role="assistant", content=assistant))
        out.append(SFTExample(messages=msgs, source=source))
    return out


def to_preference_examples(rows: list[dict], *, source: str | None = None) -> list[PreferenceExample]:
    """{prompt, chosen, rejected} → 선호 정렬(DPO) 예제."""
    out: list[PreferenceExample] = []
    for row in rows:
        if row.get("prompt") and row.get("chosen") and row.get("rejected"):
            out.append(
                PreferenceExample(
                    prompt=row["prompt"],
                    chosen=row["chosen"],
                    rejected=row["rejected"],
                    source=source,
                )
            )
    return out


# ── 필터 ──
def filter_sft_by_length(
    examples: list[SFTExample], *, min_chars: int = 1, max_chars: int = 20000
) -> list[SFTExample]:
    """assistant 응답 길이 기준 필터(빈/과도 응답 제거)."""
    kept = []
    for ex in examples:
        ans = next((m.content for m in reversed(ex.messages) if m.role == "assistant"), "")
        if min_chars <= len(ans) <= max_chars:
            kept.append(ex)
    return kept


# ── 결정적 분할 ──
def _bucket(key: str, seed: str) -> float:
    """key를 [0,1) 실수로 안정 매핑(해시 기반)."""
    h = hashlib.md5(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def stable_split(items: list, key_of, cfg: SplitConfig) -> tuple[list, list, list]:
    """해시 버킷으로 train/val/test 분할. (train, val, test) 반환."""
    test_hi = cfg.test_ratio
    val_hi = cfg.test_ratio + cfg.val_ratio
    train, val, test = [], [], []
    for it in items:
        b = _bucket(str(key_of(it)), cfg.seed)
        if b < test_hi:
            test.append(it)
        elif b < val_hi:
            val.append(it)
        else:
            train.append(it)
    return train, val, test


def split_sft(examples: list[SFTExample], cfg: SplitConfig | None = None):
    """SFT 예제를 사용자 프롬프트 기준으로 결정적 분할."""
    cfg = cfg or SplitConfig()

    def key_of(ex: SFTExample) -> str:
        return next((m.content for m in ex.messages if m.role == "user"), "")

    return stable_split(examples, key_of, cfg)

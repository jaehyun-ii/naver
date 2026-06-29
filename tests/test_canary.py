"""카나리/블루그린 배포 라우팅 테스트 — model_list 가중 변환 (순수 로직)."""

from __future__ import annotations

from llmops_core.serving import canary


def _weights(ml, name):
    return {m["model_info"]["rollout"]: m["litellm_params"]["weight"]
            for m in ml if m["model_name"] == name}


def test_set_stable_single_entry_100():
    ml = canary.set_stable([], "m", "http://stable:8000/v1")
    assert len(ml) == 1
    assert _weights(ml, "m") == {"stable": 100}


def test_add_canary_splits_weight():
    ml = canary.set_stable([], "m", "http://stable:8000/v1")
    ml = canary.add_canary(ml, "m", stable_base="http://stable:8000/v1",
                           canary_base="http://canary:8001/v1", weight=10)
    # 같은 model_name 두 엔트리, 90/10
    assert _weights(ml, "m") == {"stable": 90, "canary": 10}
    assert sum(_weights(ml, "m").values()) == 100


def test_set_weight_ramps_canary():
    ml = canary.add_canary(canary.set_stable([], "m", "s"), "m",
                           stable_base="s", canary_base="c", weight=10)
    ml = canary.set_weight(ml, "m", weight=30)  # 증량
    assert _weights(ml, "m") == {"stable": 70, "canary": 30}


def test_promote_makes_canary_the_only_stable():
    ml = canary.add_canary(canary.set_stable([], "m", "s"), "m",
                           stable_base="s", canary_base="c", weight=10)
    ml = canary.promote(ml, "m", canary_base="c")
    assert _weights(ml, "m") == {"stable": 100}
    entry = next(m for m in ml if m["model_name"] == "m")
    assert entry["litellm_params"]["api_base"] == "c"  # 카나리 base가 stable로


def test_rollback_removes_canary_keeps_stable():
    ml = canary.add_canary(canary.set_stable([], "m", "s"), "m",
                           stable_base="s", canary_base="c", weight=10)
    ml = canary.rollback(ml, "m")
    assert _weights(ml, "m") == {"stable": 100}
    entry = next(m for m in ml if m["model_name"] == "m")
    assert entry["litellm_params"]["api_base"] == "s"  # 기존 stable 유지


def test_does_not_touch_other_models():
    base = [{"model_name": "other", "litellm_params": {"model": "x", "api_base": "o"}}]
    ml = canary.add_canary(canary.set_stable(base, "m", "s"), "m",
                           stable_base="s", canary_base="c", weight=20)
    assert any(m["model_name"] == "other" for m in ml)


def test_invalid_weight_rejected():
    import pytest

    with pytest.raises(ValueError):
        canary.add_canary([], "m", stable_base="s", canary_base="c", weight=0)
    with pytest.raises(ValueError):
        canary.add_canary([], "m", stable_base="s", canary_base="c", weight=100)

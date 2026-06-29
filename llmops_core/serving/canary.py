"""카나리/블루그린 배포 — model_list 가중 라우팅으로 안전한 점진 롤아웃 (순수 변환).

같은 논리 모델명(model_name)에 stable·canary 두 백엔드를 두고 weight로 트래픽을 분할한다.
게이트웨이(litellm.Router)가 weight 비율로 라우팅하므로, 신모델을 소량(예: 10%)만 노출하고
지표 확인 후 promote(100%) 또는 rollback(0%)한다.

본 모듈은 model_list(dict) 변환만 담당(부작용 없음) — 컨테이너 기동/정리는 executor가 수행.
"""

from __future__ import annotations

from copy import deepcopy


def _entry(name: str, api_base: str, weight: int, tag: str) -> dict:
    return {
        "model_name": name,
        "litellm_params": {
            "model": f"openai/{name}", "api_base": api_base,
            "api_key": "dummy", "weight": weight,
        },
        "model_info": {"rollout": tag},  # stable | canary 식별
    }


def _without(model_list: list, name: str) -> list:
    return [m for m in model_list if m.get("model_name") != name]


def set_stable(model_list: list, name: str, api_base: str) -> list:
    """카나리 없이 단일 stable 배포(weight 100)."""
    out = _without(model_list, name)
    out.append(_entry(name, api_base, 100, "stable"))
    return out


def add_canary(model_list: list, name: str, *, stable_base: str, canary_base: str,
               weight: int) -> list:
    """stable은 유지하고 canary를 weight%로 추가. (stable=100-weight, canary=weight)."""
    if not 0 < weight < 100:
        raise ValueError("canary weight는 1~99")
    out = _without(model_list, name)
    out.append(_entry(name, stable_base, 100 - weight, "stable"))
    out.append(_entry(name, canary_base, weight, "canary"))
    return out


def set_weight(model_list: list, name: str, *, weight: int) -> list:
    """기존 stable/canary 엔트리의 weight만 조정(점진 증량). canary 존재 가정."""
    ml = deepcopy(model_list)
    has_canary = any(m.get("model_name") == name
                     and m.get("model_info", {}).get("rollout") == "canary" for m in ml)
    if not has_canary:
        raise ValueError(f"{name} 카나리가 없습니다")
    for m in ml:
        if m.get("model_name") != name:
            continue
        roll = m.get("model_info", {}).get("rollout")
        m["litellm_params"]["weight"] = weight if roll == "canary" else 100 - weight
    return ml


def promote(model_list: list, name: str, *, canary_base: str) -> list:
    """카나리를 stable로 승격(canary_base가 100% stable이 됨, 이전 stable 제거)."""
    return set_stable(model_list, name, canary_base)


def rollback(model_list: list, name: str) -> list:
    """카나리 제거, stable만 100%로 복귀."""
    ml = _without(model_list, name)
    stable = next((m for m in model_list
                   if m.get("model_name") == name
                   and m.get("model_info", {}).get("rollout") != "canary"), None)
    if stable is not None:
        base = stable["litellm_params"]["api_base"]
        ml.append(_entry(name, base, 100, "stable"))
    return ml


def rollout_status(model_list: list, name: str) -> dict:
    """현재 stable/canary weight 현황."""
    out = {"name": name, "stable": None, "canary": None}
    for m in model_list:
        if m.get("model_name") != name:
            continue
        roll = m.get("model_info", {}).get("rollout", "stable")
        out[roll] = {"api_base": m["litellm_params"]["api_base"],
                     "weight": m["litellm_params"].get("weight", 100)}
    return out

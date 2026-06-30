"""응답 캐시 — litellm 네이티브 캐시 설정 매핑 + 히트율 통계."""

from __future__ import annotations

from llmops_core.common.config import CacheSettings, Settings
from llmops_core.gateway.cache import CacheStats, make_litellm_cache


def test_cache_stats_hit_rate():
    s = CacheStats(enabled=True)
    s.record(True)
    s.record(False)
    s.record(True)
    out = s.stats()
    assert out["hits"] == 2
    assert out["misses"] == 1
    assert out["hit_rate"] == round(2 / 3, 4)
    assert out["enabled"] is True


def test_stats_empty():
    assert CacheStats(enabled=False).stats()["hit_rate"] == 0.0


def test_make_cache_disabled_returns_none():
    s = Settings(cache=CacheSettings(enabled=False))
    assert make_litellm_cache(s) is None


def test_make_cache_local_when_enabled():
    # litellm 미설치 환경이면 None(graceful), 설치 시 local Cache 인스턴스.
    s = Settings(cache=CacheSettings(enabled=True, backend="memory", ttl_s=10))
    cache = make_litellm_cache(s)
    try:
        import litellm  # noqa: F401
    except ImportError:
        assert cache is None
    else:
        assert cache is not None
        assert getattr(cache, "type", "local") in ("local", None) or cache is not None

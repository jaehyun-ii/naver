"""게이트웨이 응답 캐시 — litellm 네이티브 캐싱을 임베드(직접 구현 아님).

litellm.Cache(type=local|redis, ttl, similarity_threshold)를 Router에 붙여
인메모리·Redis·시맨틱 캐싱을 그대로 활용한다. 우리 코드는 설정 매핑 + 히트율 통계만 담당.
캐시 적중 여부는 litellm이 응답 `_hidden_params["cache_hit"]`로 알려준다.
"""

from __future__ import annotations

from llmops_core.common.config import get_settings


def make_litellm_cache(settings=None):
    """설정(cache.*)을 litellm.Cache로 매핑. 비활성/미설치면 None."""
    s = settings or get_settings()
    cfg = s.cache
    if not cfg.enabled:
        return None
    try:
        from litellm import Cache
    except ImportError:  # pragma: no cover
        return None

    if cfg.backend == "redis" and s.gateway.redis_url:  # pragma: no cover
        return Cache(type="redis", ttl=cfg.ttl_s, redis_url=s.gateway.redis_url)
    # 기본: 프로세스 인메모리(local). 동일 요청 재사용으로 토큰 비용 절감.
    return Cache(type="local", ttl=cfg.ttl_s)


class CacheStats:
    """litellm 캐시 적중 통계 — 비용 절감 관측(litellm은 카운터를 노출하지 않음)."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def record(self, cache_hit: bool) -> None:
        if cache_hit:
            self.hits += 1
        else:
            self.misses += 1

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "enabled": self.enabled,
            "hits": self.hits, "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }

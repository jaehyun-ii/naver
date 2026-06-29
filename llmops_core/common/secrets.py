"""시크릿 추상화 — 환경변수(기본) 또는 HashiCorp Vault에서 비밀값 조회.

코드/이미지에 비밀을 박지 않고 단일 인터페이스(get_secret)로 조회한다. provider 전환만으로
개발(env)↔운영(Vault) 전환. master key·DB DSN·S3 크레덴셜 등 민감값의 조회 지점.

설정(env):
    LLMOPS_SECRETS__PROVIDER = env | vault
    LLMOPS_SECRETS__VAULT_ADDR, LLMOPS_SECRETS__VAULT_TOKEN, LLMOPS_SECRETS__VAULT_MOUNT, _PATH
"""

from __future__ import annotations

import os
from functools import lru_cache

from llmops_core.common.errors import OptionalDependencyError


def _provider() -> str:
    return os.environ.get("LLMOPS_SECRETS__PROVIDER", "env")


@lru_cache(maxsize=128)
def _vault_secrets() -> dict:
    """Vault KV v2에서 시크릿 맵을 1회 로드(캐시)."""
    try:
        import hvac
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("hvac", "secrets") from exc

    client = hvac.Client(
        url=os.environ.get("LLMOPS_SECRETS__VAULT_ADDR", "http://localhost:8200"),
        token=os.environ.get("LLMOPS_SECRETS__VAULT_TOKEN"),
    )
    mount = os.environ.get("LLMOPS_SECRETS__VAULT_MOUNT", "secret")
    path = os.environ.get("LLMOPS_SECRETS__VAULT_PATH", "llmops")
    resp = client.secrets.kv.v2.read_secret_version(path=path, mount_point=mount)
    return resp["data"]["data"]


def get_secret(name: str, default: str | None = None) -> str | None:
    """비밀값 조회. provider=env면 환경변수, vault면 Vault KV에서. 없으면 default.

    name은 env 키(예: LLMOPS_GATEWAY__MASTER_KEY) 또는 Vault 키.
    """
    if _provider() == "vault":
        val = _vault_secrets().get(name)
        if val is not None:
            return val
    return os.environ.get(name, default)


def require_secret(name: str) -> str:
    val = get_secret(name)
    if val is None:
        raise RuntimeError(f"필수 시크릿 누락: {name} (provider={_provider()})")
    return val

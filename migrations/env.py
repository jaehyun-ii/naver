"""Alembic 마이그레이션 환경.

DB URL 은 애플리케이션 설정(get_settings().store.dsn)에서 읽어 앱과 단일 소스로 통일한다.
우선순위: 환경변수 LLMOPS_ALEMBIC_URL > alembic.ini sqlalchemy.url > store.dsn.

스키마는 원시 SQL(op.create_table)로 정의하므로 SQLAlchemy 메타데이터 오토젠은 사용하지 않는다
(target_metadata=None). 각 리비전이 스키마 변경을 명시적으로 기술한다.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _resolve_url() -> str:
    env_url = os.getenv("LLMOPS_ALEMBIC_URL")
    if env_url:
        return env_url
    ini_url = config.get_main_option("sqlalchemy.url")
    if ini_url:
        return ini_url
    # 앱 설정에서 DSN 을 가져온다(게이트웨이/콘솔과 동일 소스).
    from llmops_core.common.config import get_settings

    return get_settings().store.dsn


target_metadata = None


def run_migrations_offline() -> None:
    context.configure(
        url=_resolve_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _resolve_url()
    connectable = engine_from_config(
        section, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

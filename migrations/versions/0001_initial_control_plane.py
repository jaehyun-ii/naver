"""initial control-plane schema (db.py init_schema 반영)

Revision ID: 0001_initial
Revises:
Create Date: 2026-07-02

llmops_core/common/db.py 의 _SCHEMA(CREATE TABLE IF NOT EXISTS ...)와 동일한 스키마를
Alembic 버전 관리로 이관한 초기 리비전. 이후 스키마 변경은 새 리비전으로 추가한다.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

_TS = sa.TIMESTAMP(timezone=True)


def upgrade() -> None:
    op.create_table(
        "virtual_keys",
        sa.Column("key_hash", sa.Text(), primary_key=True),
        sa.Column("key_id", sa.Text(), nullable=False, unique=True),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("allowed_models", postgresql.JSONB(), nullable=False,
                  server_default=sa.text("'[]'::jsonb")),
        sa.Column("monthly_budget_usd", sa.Float(), nullable=True),
        sa.Column("rpm_limit", sa.Integer(), nullable=True),
        sa.Column("created_at", _TS, nullable=False, server_default=sa.text("now()")),
    )

    op.create_table(
        "usage_ledger",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
        sa.Column("ts", _TS, nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("idx_ledger_tenant_ts", "usage_ledger", ["tenant_id", "ts"])

    op.create_table(
        "release_requests",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", _TS, nullable=False, server_default=sa.text("now()")),
    )

    op.create_table(
        "kv_store",
        sa.Column("kind", sa.Text(), primary_key=True, nullable=False),
        sa.Column("id", sa.Text(), primary_key=True, nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=True),
        sa.Column("updated_at", _TS, nullable=False, server_default=sa.text("now()")),
    )

    op.create_table(
        "auth_tokens",
        sa.Column("token_hash", sa.Text(), primary_key=True),
        sa.Column("token_id", sa.Text(), nullable=False, unique=True),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("roles", postgresql.JSONB(), nullable=False,
                  server_default=sa.text("'[]'::jsonb")),
        sa.Column("created_at", _TS, nullable=False, server_default=sa.text("now()")),
    )

    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("ts", _TS, nullable=False, server_default=sa.text("now()")),
        sa.Column("actor", sa.Text(), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("target", sa.Text(), nullable=True),
        sa.Column("result", sa.Text(), nullable=True),
        sa.Column("detail", postgresql.JSONB(), nullable=True),
    )
    op.create_index("idx_audit_ts", "audit_log", [sa.text("ts DESC")])


def downgrade() -> None:
    op.drop_index("idx_audit_ts", table_name="audit_log")
    op.drop_table("audit_log")
    op.drop_table("auth_tokens")
    op.drop_table("kv_store")
    op.drop_table("release_requests")
    op.drop_index("idx_ledger_tenant_ts", table_name="usage_ledger")
    op.drop_table("usage_ledger")
    op.drop_table("virtual_keys")

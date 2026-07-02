"""add expires_at to virtual_keys and auth_tokens (P1 deferred)

Revision ID: 0002_expires_at
Revises: 0001_initial
Create Date: 2026-07-02

가상키·인증토큰의 만료(expires_at)를 도입. NULL 이면 무기한(기존 동작 유지)이며,
값이 있으면 해당 시각 이후 검증에서 만료 처리한다. 하위호환을 위해 nullable.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_expires_at"
down_revision = "0001_initial"
branch_labels = None
depends_on = None

_TS = sa.TIMESTAMP(timezone=True)


def upgrade() -> None:
    op.add_column("virtual_keys", sa.Column("expires_at", _TS, nullable=True))
    op.add_column("auth_tokens", sa.Column("expires_at", _TS, nullable=True))


def downgrade() -> None:
    op.drop_column("auth_tokens", "expires_at")
    op.drop_column("virtual_keys", "expires_at")

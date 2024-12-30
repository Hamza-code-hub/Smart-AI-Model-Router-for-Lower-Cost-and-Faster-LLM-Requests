"""add_router_telemetry_columns

Revision ID: c3f4d5e6a7b8
Revises: b2e3c4d5f6a7
Create Date: 2026-06-16

Adds tier_fired, routed_category, conversation_id to `requests` so the Phase
1.6 admin/routing/stats endpoint can aggregate per-tier hit rates, per-category
counts, latency by tier, and per-conversation escalation rate from existing
request logs.

Additive only — see memory/feedback_permissions_mode.md.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c3f4d5e6a7b8"
down_revision: Union[str, Sequence[str], None] = "a1d2f3b4c5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("requests", sa.Column("tier_fired", sa.Text(), nullable=True))
    op.add_column("requests", sa.Column("routed_category", sa.Text(), nullable=True))
    op.add_column("requests", sa.Column("conversation_id", sa.Text(), nullable=True))
    op.create_index(
        "idx_requests_conversation",
        "requests",
        ["conversation_id", sa.text("created_at DESC")],
        postgresql_where=sa.text("conversation_id IS NOT NULL"),
    )


def downgrade() -> None:
    raise NotImplementedError(
        "Migrations are additive only — see memory/feedback_permissions_mode.md"
    )

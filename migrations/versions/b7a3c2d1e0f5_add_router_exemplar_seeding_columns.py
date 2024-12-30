"""add_router_exemplar_seeding_columns

Revision ID: b7a3c2d1e0f5
Revises: e5f6a7b8c9d0
Create Date: 2026-06-19

Backfills the auto-seeder columns on router_exemplars for databases that
already applied a1d2f3b4c5e6 before its CREATE TABLE was extended in-place.

The same columns live in a1d2f3b4c5e6's CREATE for fresh databases. This
ALTER uses IF NOT EXISTS so both paths converge to the same schema without
double-application errors.

Columns added:
  source             TEXT  NOT NULL DEFAULT 'curated'    -- curated vs haiku-seeded provenance
  source_confidence  NUMERIC(4,3)                         -- Haiku's self-reported confidence at seed
  match_count        INTEGER NOT NULL DEFAULT 0           -- bumped on tier-2 hit
  last_matched_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()   -- key for future LRU eviction

Additive only — see memory/feedback_permissions_mode.md.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "b7a3c2d1e0f5"
down_revision: Union[str, Sequence[str], None] = "e5f6a7b8c9d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE router_exemplars
            ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'curated',
            ADD COLUMN IF NOT EXISTS source_confidence NUMERIC(4, 3),
            ADD COLUMN IF NOT EXISTS match_count INTEGER NOT NULL DEFAULT 0,
            ADD COLUMN IF NOT EXISTS last_matched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_router_exemplars_last_matched "
        "ON router_exemplars (last_matched_at)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_router_exemplars_source "
        "ON router_exemplars (source)"
    )


def downgrade() -> None:
    raise NotImplementedError(
        "Migrations are additive only — see memory/feedback_permissions_mode.md"
    )

"""create_router_exemplars_table

Revision ID: a1d2f3b4c5e6
Revises: 7c4a1b89d2e3
Create Date: 2026-06-16

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a1d2f3b4c5e6"
down_revision: Union[str, Sequence[str], None] = "7c4a1b89d2e3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS "vector"')

    op.execute(
        """
        CREATE TABLE router_exemplars (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            query_text TEXT NOT NULL,
            category TEXT NOT NULL,
            embedding VECTOR(384) NOT NULL,
            text_hash TEXT NOT NULL UNIQUE,
            notes TEXT,
            source TEXT NOT NULL DEFAULT 'curated',
            source_confidence NUMERIC(4, 3),
            match_count INTEGER NOT NULL DEFAULT 0,
            last_matched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )

    op.execute(
        "CREATE INDEX idx_router_exemplars_embedding "
        "ON router_exemplars USING hnsw (embedding vector_cosine_ops)"
    )

    op.create_index(
        "idx_router_exemplars_category",
        "router_exemplars",
        ["category"],
    )

    op.create_index(
        "idx_router_exemplars_last_matched",
        "router_exemplars",
        ["last_matched_at"],
    )

    op.create_index(
        "idx_router_exemplars_source",
        "router_exemplars",
        ["source"],
    )


def downgrade() -> None:
    # Hobby project: migrations are additive only. See memory/feedback_permissions_mode.md.
    raise NotImplementedError("router_exemplars migration is non-destructive — no downgrade")

"""Isolate PostgreSQL vector collections by embedding space.

Revision ID: 0002_embedding_space_isolation
Revises: 0001_pgvector_foundation
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002_embedding_space_isolation"
down_revision: str | Sequence[str] | None = "0001_pgvector_foundation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

LEGACY_SPACE_ID = "legacy-unidentified"


def upgrade() -> None:
    """Add explicit vector-space identity without guessing old model identity."""
    op.execute(
        """
        ALTER TABLE ai_platform.vector_collections
        ADD COLUMN space_id text
        """
    )

    op.execute(
        f"""
        UPDATE ai_platform.vector_collections
        SET space_id = '{LEGACY_SPACE_ID}'
        WHERE space_id IS NULL
        """
    )

    op.execute(
        """
        ALTER TABLE ai_platform.vector_collections
        ALTER COLUMN space_id SET NOT NULL
        """
    )

    op.execute(
        """
        ALTER TABLE ai_platform.vector_collections
        ADD CONSTRAINT ck_vector_collections_space_id
        CHECK (
            btrim(space_id) <> ''
        )
        """
    )

    op.execute(
        """
        ALTER TABLE ai_platform.vector_collections
        DROP CONSTRAINT uq_vector_collections_namespace
        """
    )

    op.execute(
        """
        ALTER TABLE ai_platform.vector_collections
        ADD CONSTRAINT uq_vector_collections_namespace_space_id
        UNIQUE NULLS NOT DISTINCT (
            namespace,
            space_id
        )
        """
    )


def downgrade() -> None:
    """Return to namespace-only identity only when that collapse is lossless."""
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM ai_platform.vector_collections
                GROUP BY namespace
                HAVING count(*) > 1
            ) THEN
                RAISE EXCEPTION
                    'cannot downgrade embedding-space isolation: '
                    'multiple vector spaces exist for one namespace';
            END IF;
        END
        $$
        """
    )

    op.execute(
        """
        ALTER TABLE ai_platform.vector_collections
        DROP CONSTRAINT uq_vector_collections_namespace_space_id
        """
    )

    op.execute(
        """
        ALTER TABLE ai_platform.vector_collections
        DROP CONSTRAINT ck_vector_collections_space_id
        """
    )

    op.execute(
        """
        ALTER TABLE ai_platform.vector_collections
        DROP COLUMN space_id
        """
    )

    op.execute(
        """
        ALTER TABLE ai_platform.vector_collections
        ADD CONSTRAINT uq_vector_collections_namespace
        UNIQUE NULLS NOT DISTINCT (namespace)
        """
    )

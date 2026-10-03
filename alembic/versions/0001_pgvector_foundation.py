"""Create the PostgreSQL pgvector persistence foundation.

Revision ID: 0001_pgvector_foundation
Revises:
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0001_pgvector_foundation"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create extension, schema, vector tables, and runtime grants."""
    op.execute("CREATE EXTENSION vector WITH SCHEMA public")

    op.execute("CREATE SCHEMA ai_platform")

    op.execute(
        """
        CREATE TABLE ai_platform.vector_collections (
            collection_id bigint
                GENERATED ALWAYS AS IDENTITY
                PRIMARY KEY,
            namespace text NULL,
            dimensions integer NOT NULL,

            CONSTRAINT ck_vector_collections_namespace
                CHECK (
                    namespace IS NULL
                    OR btrim(namespace) <> ''
                ),

            CONSTRAINT ck_vector_collections_dimensions
                CHECK (
                    dimensions BETWEEN 1 AND 16000
                ),

            CONSTRAINT uq_vector_collections_namespace
                UNIQUE NULLS NOT DISTINCT (namespace),

            CONSTRAINT uq_vector_collections_id_dimensions
                UNIQUE (collection_id, dimensions)
        )
        """
    )

    op.execute(
        """
        CREATE TABLE ai_platform.vector_records (
            collection_id bigint NOT NULL,
            record_id text NOT NULL,
            embedding public.vector NOT NULL,

            dimensions integer
                GENERATED ALWAYS AS (
                    public.vector_dims(embedding)
                ) STORED,

            text text NULL,

            metadata jsonb NOT NULL
                DEFAULT '[]'::jsonb,

            CONSTRAINT pk_vector_records
                PRIMARY KEY (
                    collection_id,
                    record_id
                ),

            CONSTRAINT ck_vector_records_record_id
                CHECK (
                    btrim(record_id) <> ''
                ),

            CONSTRAINT ck_vector_records_text
                CHECK (
                    text IS NULL
                    OR btrim(text) <> ''
                ),

            CONSTRAINT ck_vector_records_metadata
                CHECK (
                    jsonb_typeof(metadata) = 'array'
                ),

            CONSTRAINT fk_vector_records_collection_dimensions
                FOREIGN KEY (
                    collection_id,
                    dimensions
                )
                REFERENCES ai_platform.vector_collections (
                    collection_id,
                    dimensions
                )
                ON DELETE CASCADE
        )
        """
    )

    op.execute(
        """
        GRANT USAGE
        ON SCHEMA ai_platform
        TO ai_platform_runtime
        """
    )

    op.execute(
        """
        GRANT SELECT, INSERT
        ON TABLE ai_platform.vector_collections
        TO ai_platform_runtime
        """
    )

    op.execute(
        """
        GRANT USAGE
        ON SEQUENCE
            ai_platform.vector_collections_collection_id_seq
        TO ai_platform_runtime
        """
    )

    op.execute(
        """
        GRANT SELECT, INSERT, UPDATE, DELETE
        ON TABLE ai_platform.vector_records
        TO ai_platform_runtime
        """
    )


def downgrade() -> None:
    """Remove project-owned persistence objects."""
    op.execute("DROP TABLE ai_platform.vector_records")

    op.execute("DROP TABLE ai_platform.vector_collections")

    op.execute("DROP SCHEMA ai_platform")

    op.execute("DROP EXTENSION vector")

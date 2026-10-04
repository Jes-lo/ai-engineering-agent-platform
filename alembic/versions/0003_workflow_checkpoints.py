"""Add durable project-owned workflow checkpoints.

Revision ID: 0003_workflow_checkpoints
Revises: 0002_embedding_space_isolation
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003_workflow_checkpoints"
down_revision: str | Sequence[str] | None = "0002_embedding_space_isolation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create bounded durable workflow checkpoint storage."""
    op.execute(
        """
        CREATE TABLE ai_platform.workflow_checkpoints (
            run_id text PRIMARY KEY,
            workflow_id text NOT NULL,
            workflow_version text NOT NULL,
            run_status text NOT NULL,
            checkpoint_format integer NOT NULL,
            event_count integer NOT NULL,
            checkpoint_payload text NOT NULL,
            checkpoint_sha256 text NOT NULL,

            created_at timestamptz NOT NULL
                DEFAULT clock_timestamp(),

            updated_at timestamptz NOT NULL
                DEFAULT clock_timestamp(),

            CONSTRAINT ck_workflow_checkpoints_run_id
                CHECK (
                    btrim(run_id) <> ''
                    AND length(run_id) <= 80
                ),

            CONSTRAINT ck_workflow_checkpoints_workflow_id
                CHECK (
                    btrim(workflow_id) <> ''
                    AND length(workflow_id) <= 80
                ),

            CONSTRAINT ck_workflow_checkpoints_workflow_version
                CHECK (
                    btrim(workflow_version) <> ''
                    AND length(workflow_version) <= 80
                ),

            CONSTRAINT ck_workflow_checkpoints_run_status
                CHECK (
                    run_status IN (
                        'running',
                        'completed',
                        'failed'
                    )
                ),

            CONSTRAINT ck_workflow_checkpoints_format
                CHECK (
                    checkpoint_format = 1
                ),

            CONSTRAINT ck_workflow_checkpoints_event_count
                CHECK (
                    event_count >= 1
                ),

            CONSTRAINT ck_workflow_checkpoints_payload_size
                CHECK (
                    octet_length(checkpoint_payload)
                    BETWEEN 1 AND 1048576
                ),

            CONSTRAINT ck_workflow_checkpoints_sha256
                CHECK (
                    checkpoint_sha256
                    ~ '^[0-9a-f]{64}$'
                )
        )
        """
    )

    op.execute(
        """
        GRANT SELECT, INSERT, UPDATE
        ON TABLE ai_platform.workflow_checkpoints
        TO ai_platform_runtime
        """
    )


def downgrade() -> None:
    """Remove durable workflow checkpoint storage."""
    op.execute("DROP TABLE ai_platform.workflow_checkpoints")

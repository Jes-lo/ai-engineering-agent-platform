"""Allow durable workflow approval-pause checkpoints.

Revision ID: 0006_workflow_awaiting_approval
Revises: 0005_agent_continuations
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0006_workflow_awaiting_approval"
down_revision: str | None = "0005_agent_continuations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Permit approval-pause state and checkpoint format v2."""
    op.execute(
        """
        ALTER TABLE ai_platform.workflow_checkpoints
        DROP CONSTRAINT ck_workflow_checkpoints_run_status
        """
    )

    op.execute(
        """
        ALTER TABLE ai_platform.workflow_checkpoints
        ADD CONSTRAINT ck_workflow_checkpoints_run_status
        CHECK (
            run_status IN (
                'running',
                'awaiting_approval',
                'completed',
                'failed'
            )
        )
        """
    )

    op.execute(
        """
        ALTER TABLE ai_platform.workflow_checkpoints
        DROP CONSTRAINT ck_workflow_checkpoints_format
        """
    )

    op.execute(
        """
        ALTER TABLE ai_platform.workflow_checkpoints
        ADD CONSTRAINT ck_workflow_checkpoints_format
        CHECK (
            checkpoint_format IN (1, 2)
        )
        """
    )


def downgrade() -> None:
    """Restore v1 only when no v2 state would be lost."""
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM ai_platform.workflow_checkpoints
                WHERE
                    run_status = 'awaiting_approval'
                    OR checkpoint_format = 2
            ) THEN
                RAISE EXCEPTION
                    'cannot downgrade while workflow v2 '
                    'or approval-pause checkpoints exist';
            END IF;
        END
        $$;
        """
    )

    op.execute(
        """
        ALTER TABLE ai_platform.workflow_checkpoints
        DROP CONSTRAINT ck_workflow_checkpoints_run_status
        """
    )

    op.execute(
        """
        ALTER TABLE ai_platform.workflow_checkpoints
        ADD CONSTRAINT ck_workflow_checkpoints_run_status
        CHECK (
            run_status IN (
                'running',
                'completed',
                'failed'
            )
        )
        """
    )

    op.execute(
        """
        ALTER TABLE ai_platform.workflow_checkpoints
        DROP CONSTRAINT ck_workflow_checkpoints_format
        """
    )

    op.execute(
        """
        ALTER TABLE ai_platform.workflow_checkpoints
        ADD CONSTRAINT ck_workflow_checkpoints_format
        CHECK (
            checkpoint_format = 1
        )
        """
    )

"""Add durable authenticated human approval requests.

Revision ID: 0004_approval_requests
Revises: 0003_workflow_checkpoints
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0004_approval_requests"
down_revision: str | Sequence[str] | None = "0003_workflow_checkpoints"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create bounded durable authenticated approval storage."""
    op.execute(
        """
        CREATE TABLE ai_platform.approval_requests (
            approval_id text PRIMARY KEY,
            run_id text NOT NULL,
            workflow_id text NOT NULL,
            workflow_version text NOT NULL,
            step_id text NOT NULL,
            call_id text NOT NULL,
            tool_name text NOT NULL,
            approval_status text NOT NULL,
            approval_format integer NOT NULL,
            state_version integer NOT NULL,
            approval_payload text NOT NULL,
            approval_sha256 text NOT NULL,
            requested_at timestamptz NOT NULL,
            expires_at timestamptz NOT NULL,

            created_at timestamptz NOT NULL
                DEFAULT clock_timestamp(),

            updated_at timestamptz NOT NULL
                DEFAULT clock_timestamp(),

            CONSTRAINT ck_approval_requests_approval_id
                CHECK (
                    btrim(approval_id) <> ''
                    AND length(approval_id) <= 80
                ),

            CONSTRAINT ck_approval_requests_run_id
                CHECK (
                    btrim(run_id) <> ''
                    AND length(run_id) <= 80
                ),

            CONSTRAINT ck_approval_requests_workflow_id
                CHECK (
                    btrim(workflow_id) <> ''
                    AND length(workflow_id) <= 80
                ),

            CONSTRAINT ck_approval_requests_workflow_version
                CHECK (
                    btrim(workflow_version) <> ''
                    AND length(workflow_version) <= 80
                ),

            CONSTRAINT ck_approval_requests_step_id
                CHECK (
                    btrim(step_id) <> ''
                    AND length(step_id) <= 80
                ),

            CONSTRAINT ck_approval_requests_call_id
                CHECK (
                    btrim(call_id) <> ''
                    AND length(call_id) <= 512
                ),

            CONSTRAINT ck_approval_requests_tool_name
                CHECK (
                    btrim(tool_name) <> ''
                    AND length(tool_name) <= 160
                ),

            CONSTRAINT ck_approval_requests_status
                CHECK (
                    approval_status IN (
                        'pending',
                        'approved',
                        'rejected',
                        'consumed'
                    )
                ),

            CONSTRAINT ck_approval_requests_format
                CHECK (
                    approval_format = 1
                ),

            CONSTRAINT ck_approval_requests_state_version
                CHECK (
                    state_version >= 1
                ),

            CONSTRAINT ck_approval_requests_payload_size
                CHECK (
                    octet_length(approval_payload)
                    BETWEEN 1 AND 262144
                ),

            CONSTRAINT ck_approval_requests_sha256
                CHECK (
                    approval_sha256
                    ~ '^[0-9a-f]{64}$'
                ),

            CONSTRAINT ck_approval_requests_expiration
                CHECK (
                    expires_at > requested_at
                )
        )
        """
    )

    op.execute(
        """
        CREATE INDEX ix_approval_requests_run_status
        ON ai_platform.approval_requests (
            run_id,
            approval_status
        )
        """
    )

    op.execute(
        """
        GRANT SELECT, INSERT, UPDATE
        ON TABLE ai_platform.approval_requests
        TO ai_platform_runtime
        """
    )


def downgrade() -> None:
    """Remove durable authenticated approval storage."""
    op.execute("DROP TABLE ai_platform.approval_requests")

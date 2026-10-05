"""Persist active and consumed agent continuations."""

from collections.abc import Sequence

from alembic import op

revision: str = "0005_agent_continuations"
down_revision: str | None = "0004_approval_requests"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create fail-closed durable agent continuation storage."""
    op.execute(
        """
        CREATE TABLE ai_platform.agent_continuations (
            continuation_key text PRIMARY KEY,
            continuation_kind text NOT NULL,
            identity_json text NOT NULL,
            continuation_status text NOT NULL,
            continuation_format integer NOT NULL,
            state_version integer NOT NULL,
            continuation_payload text NOT NULL,
            continuation_sha256 text NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            consumed_at timestamptz NULL,

            CONSTRAINT agent_continuation_key_format
                CHECK (
                    continuation_key ~ '^[0-9a-f]{64}$'
                ),

            CONSTRAINT agent_continuation_kind_valid
                CHECK (
                    continuation_kind IN ('turn', 'loop')
                ),

            CONSTRAINT agent_continuation_status_valid
                CHECK (
                    continuation_status IN ('active', 'consumed')
                ),

            CONSTRAINT agent_continuation_format_valid
                CHECK (
                    continuation_format = 1
                ),

            CONSTRAINT agent_continuation_state_version_valid
                CHECK (
                    state_version >= 1
                ),

            CONSTRAINT agent_continuation_identity_size
                CHECK (
                    octet_length(identity_json) <= 8192
                ),

            CONSTRAINT agent_continuation_payload_size
                CHECK (
                    octet_length(continuation_payload) <= 1048576
                ),

            CONSTRAINT agent_continuation_sha256_format
                CHECK (
                    continuation_sha256 ~ '^[0-9a-f]{64}$'
                ),

            CONSTRAINT agent_continuation_consumed_shape
                CHECK (
                    (
                        continuation_status = 'active'
                        AND consumed_at IS NULL
                    )
                    OR
                    (
                        continuation_status = 'consumed'
                        AND consumed_at IS NOT NULL
                    )
                ),

            CONSTRAINT agent_continuation_identity_unique
                UNIQUE (
                    continuation_kind,
                    identity_json
                )
        )
        """
    )

    op.execute(
        """
        CREATE INDEX agent_continuations_status_idx
        ON ai_platform.agent_continuations (
            continuation_status,
            continuation_kind
        )
        """
    )

    op.execute(
        """
        GRANT SELECT, INSERT, UPDATE
        ON TABLE ai_platform.agent_continuations
        TO ai_platform_runtime
        """
    )


def downgrade() -> None:
    """Drop durable agent continuation storage."""
    op.execute("DROP TABLE IF EXISTS ai_platform.agent_continuations")

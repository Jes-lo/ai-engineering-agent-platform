"""PostgreSQL adapter for durable workflow checkpoints."""

import hmac
import re
from hashlib import sha256

from psycopg import AsyncConnection
from psycopg.rows import TupleRow
from psycopg_pool import AsyncConnectionPool

from ai_engineering_agent_platform.services.workflow_persistence import (
    SUPPORTED_WORKFLOW_CHECKPOINT_FORMAT_VERSIONS,
    WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    WorkflowCheckpoint,
    WorkflowCheckpointCodec,
    WorkflowPersistenceConflictError,
    WorkflowPersistenceError,
    WorkflowPersistenceIntegrityError,
    WorkflowStateStore,
)

type WorkflowPostgresPool = AsyncConnectionPool[AsyncConnection[TupleRow]]


_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,79}$")

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


_SAVE_SQL = """
INSERT INTO ai_platform.workflow_checkpoints (
    run_id,
    workflow_id,
    workflow_version,
    run_status,
    checkpoint_format,
    event_count,
    checkpoint_payload,
    checkpoint_sha256
)
VALUES (
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s
)
ON CONFLICT (run_id)
DO UPDATE
SET
    run_status = EXCLUDED.run_status,
    checkpoint_format = EXCLUDED.checkpoint_format,
    event_count = EXCLUDED.event_count,
    checkpoint_payload = EXCLUDED.checkpoint_payload,
    checkpoint_sha256 = EXCLUDED.checkpoint_sha256,
    updated_at = clock_timestamp()
WHERE
    workflow_checkpoints.workflow_id
        = EXCLUDED.workflow_id
    AND workflow_checkpoints.workflow_version
        = EXCLUDED.workflow_version
    AND (
        workflow_checkpoints.checkpoint_payload
            = EXCLUDED.checkpoint_payload
        OR (
            workflow_checkpoints.run_status IN ('running', 'awaiting_approval')
            AND workflow_checkpoints.event_count
                < EXCLUDED.event_count
        )
    )
RETURNING run_id
"""


_LOAD_SQL = """
SELECT
    workflow_id,
    workflow_version,
    run_status,
    checkpoint_format,
    event_count,
    checkpoint_payload,
    checkpoint_sha256
FROM ai_platform.workflow_checkpoints
WHERE run_id = %s
"""


class PostgreSQLWorkflowStateStore(WorkflowStateStore):
    """Persist canonical workflow checkpoints in PostgreSQL."""

    def __init__(
        self,
        pool: WorkflowPostgresPool,
        *,
        codec: WorkflowCheckpointCodec | None = None,
    ) -> None:
        """Store an injected PostgreSQL pool and checkpoint codec."""
        if pool is None:
            raise ValueError("pool must not be None")

        self._pool = pool
        self._codec = WorkflowCheckpointCodec() if codec is None else codec

    async def save(
        self,
        checkpoint: WorkflowCheckpoint,
    ) -> None:
        """Atomically insert or monotonically advance one checkpoint."""
        payload = self._codec.dumps(checkpoint)

        digest = sha256(payload.encode("utf-8")).hexdigest()

        state = checkpoint.state

        parameters = (
            state.run_id,
            state.workflow_id,
            state.workflow_version,
            state.status.value,
            WORKFLOW_CHECKPOINT_FORMAT_VERSION,
            len(checkpoint.events),
            payload,
            digest,
        )

        try:
            async with self._pool.connection() as connection:
                cursor = await connection.execute(
                    _SAVE_SQL,
                    parameters,
                )

                row = await cursor.fetchone()

        except WorkflowPersistenceError:
            raise

        except Exception as exc:
            raise WorkflowPersistenceError(
                "PostgreSQL workflow checkpoint save failed"
            ) from exc

        if row is None:
            raise WorkflowPersistenceConflictError(
                "workflow checkpoint update rejected by identity or monotonicity guard"
            )

        if (
            not isinstance(
                row,
                tuple,
            )
            or len(row) != 1
            or row[0] != state.run_id
        ):
            raise WorkflowPersistenceIntegrityError(
                "unexpected PostgreSQL workflow save result"
            )

    async def load(
        self,
        run_id: str,
    ) -> WorkflowCheckpoint | None:
        """Load and fully revalidate one durable workflow checkpoint."""
        if not isinstance(
            run_id,
            str,
        ) or not _RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError("run_id must be a portable workflow identifier")

        try:
            async with self._pool.connection() as connection:
                cursor = await connection.execute(
                    _LOAD_SQL,
                    (run_id,),
                )

                row = await cursor.fetchone()

        except WorkflowPersistenceError:
            raise

        except Exception as exc:
            raise WorkflowPersistenceError(
                "PostgreSQL workflow checkpoint load failed"
            ) from exc

        if row is None:
            return None

        if (
            not isinstance(
                row,
                tuple,
            )
            or len(row) != 7
        ):
            raise WorkflowPersistenceIntegrityError(
                "unexpected PostgreSQL workflow checkpoint shape"
            )

        (
            workflow_id,
            workflow_version,
            run_status,
            checkpoint_format,
            event_count,
            payload,
            stored_digest,
        ) = row

        if (
            not isinstance(
                workflow_id,
                str,
            )
            or not isinstance(
                workflow_version,
                str,
            )
            or not isinstance(
                run_status,
                str,
            )
            or type(checkpoint_format) is not int
            or type(event_count) is not int
            or not isinstance(
                payload,
                str,
            )
            or not isinstance(
                stored_digest,
                str,
            )
        ):
            raise WorkflowPersistenceIntegrityError(
                "invalid PostgreSQL workflow checkpoint field types"
            )

        if checkpoint_format not in SUPPORTED_WORKFLOW_CHECKPOINT_FORMAT_VERSIONS:
            raise WorkflowPersistenceIntegrityError(
                "stored workflow checkpoint format differs"
            )

        if event_count < 1:
            raise WorkflowPersistenceIntegrityError(
                "stored workflow event count is invalid"
            )

        if not _SHA256_PATTERN.fullmatch(stored_digest):
            raise WorkflowPersistenceIntegrityError(
                "stored workflow checkpoint digest is invalid"
            )

        actual_digest = sha256(payload.encode("utf-8")).hexdigest()

        if not hmac.compare_digest(
            stored_digest,
            actual_digest,
        ):
            raise WorkflowPersistenceIntegrityError(
                "stored workflow checkpoint checksum mismatch"
            )

        try:
            checkpoint = self._codec.loads(payload)

        except WorkflowPersistenceError as exc:
            raise WorkflowPersistenceIntegrityError(
                "stored workflow checkpoint payload is invalid"
            ) from exc

        state = checkpoint.state

        if (
            state.run_id != run_id
            or state.workflow_id != workflow_id
            or state.workflow_version != workflow_version
            or state.status.value != run_status
            or len(checkpoint.events) != event_count
        ):
            raise WorkflowPersistenceIntegrityError(
                "stored workflow checkpoint metadata mismatch"
            )

        canonical = self._codec.dumps(
            checkpoint,
            format_version=checkpoint_format,
        )

        if canonical != payload:
            raise WorkflowPersistenceIntegrityError(
                "stored workflow checkpoint is not canonical"
            )

        return checkpoint

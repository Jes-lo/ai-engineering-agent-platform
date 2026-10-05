"""Unit tests for PostgreSQL durable workflow checkpoint storage."""

from contextlib import AbstractAsyncContextManager
from dataclasses import replace
from hashlib import sha256
from typing import cast

import pytest

from ai_engineering_agent_platform.adapters.postgres.workflow_state_store import (
    PostgreSQLWorkflowStateStore,
    WorkflowPostgresPool,
)
from ai_engineering_agent_platform.domain.workflow import (
    WorkflowEventType,
    WorkflowExecutionEvent,
    WorkflowRunState,
    WorkflowRunStatus,
    WorkflowStepState,
    WorkflowStepStatus,
)
from ai_engineering_agent_platform.services.workflow_persistence import (
    WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    WorkflowCheckpoint,
    WorkflowCheckpointCodec,
    WorkflowPersistenceConflictError,
    WorkflowPersistenceError,
    WorkflowPersistenceIntegrityError,
)


class FakeCursor:
    def __init__(
        self,
        row: tuple[object, ...] | None,
    ) -> None:
        self.row = row

    async def fetchone(
        self,
    ) -> tuple[object, ...] | None:
        return self.row


class FakeConnection:
    def __init__(
        self,
        rows: list[tuple[object, ...] | None],
    ) -> None:
        self.rows = rows
        self.calls: list[
            tuple[
                str,
                tuple[object, ...],
            ]
        ] = []
        self.error: Exception | None = None

    async def execute(
        self,
        query: str,
        parameters: tuple[object, ...],
    ) -> FakeCursor:
        if self.error is not None:
            raise self.error

        self.calls.append(
            (
                query,
                parameters,
            )
        )

        if not self.rows:
            raise AssertionError("no fake database row configured")

        return FakeCursor(self.rows.pop(0))


class FakeConnectionContext(AbstractAsyncContextManager[FakeConnection]):
    def __init__(
        self,
        connection: FakeConnection,
    ) -> None:
        self.connection = connection

    async def __aenter__(
        self,
    ) -> FakeConnection:
        return self.connection

    async def __aexit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        return None


class FakePool:
    def __init__(
        self,
        connection: FakeConnection,
    ) -> None:
        self.connection_value = connection

    def connection(
        self,
    ) -> FakeConnectionContext:
        return FakeConnectionContext(self.connection_value)


def _event(
    sequence: int,
    event_type: WorkflowEventType,
) -> WorkflowExecutionEvent:
    return WorkflowExecutionEvent(
        sequence=sequence,
        run_id="workflow-run:postgres",
        workflow_id="workflow-postgres",
        workflow_version="1",
        event_type=event_type,
    )


def _running_checkpoint() -> WorkflowCheckpoint:
    return WorkflowCheckpoint(
        state=WorkflowRunState(
            run_id="workflow-run:postgres",
            workflow_id="workflow-postgres",
            workflow_version="1",
            status=WorkflowRunStatus.RUNNING,
            inputs=(),
            steps=(),
        ),
        events=(
            _event(
                1,
                WorkflowEventType.RUN_STARTED,
            ),
        ),
    )


def _completed_checkpoint() -> WorkflowCheckpoint:
    running = _running_checkpoint()

    return WorkflowCheckpoint(
        state=replace(
            running.state,
            status=WorkflowRunStatus.COMPLETED,
        ),
        events=(
            *running.events,
            _event(
                2,
                WorkflowEventType.RUN_COMPLETED,
            ),
        ),
    )


def _failed_checkpoint() -> WorkflowCheckpoint:
    running = _running_checkpoint()

    failed_step = WorkflowStepState(
        step_id="failed_step",
        executor_name="synthetic",
        status=WorkflowStepStatus.FAILED,
    )

    failed_state = replace(
        running.state,
        status=WorkflowRunStatus.FAILED,
        steps=(failed_step,),
    )

    failed_event = WorkflowExecutionEvent(
        sequence=2,
        run_id=failed_state.run_id,
        workflow_id=failed_state.workflow_id,
        workflow_version=failed_state.workflow_version,
        event_type=WorkflowEventType.STEP_FAILED,
        step_id=failed_step.step_id,
        executor_name=failed_step.executor_name,
    )

    return WorkflowCheckpoint(
        state=failed_state,
        events=(
            *running.events,
            failed_event,
            _event(
                3,
                WorkflowEventType.RUN_FAILED,
            ),
        ),
    )


def _stored_row(
    checkpoint: WorkflowCheckpoint,
) -> tuple[object, ...]:
    codec = WorkflowCheckpointCodec()

    payload = codec.dumps(checkpoint)

    return (
        checkpoint.state.workflow_id,
        checkpoint.state.workflow_version,
        checkpoint.state.status.value,
        WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        len(checkpoint.events),
        payload,
        sha256(payload.encode("utf-8")).hexdigest(),
    )


def _store(
    connection: FakeConnection,
) -> PostgreSQLWorkflowStateStore:
    return PostgreSQLWorkflowStateStore(
        cast(
            WorkflowPostgresPool,
            FakePool(connection),
        )
    )


@pytest.mark.anyio
async def test_save_uses_single_atomic_upsert() -> None:
    checkpoint = _running_checkpoint()

    connection = FakeConnection(
        [
            (checkpoint.state.run_id,),
        ]
    )

    await _store(connection).save(checkpoint)

    assert len(connection.calls) == 1

    query, parameters = connection.calls[0]

    assert "ON CONFLICT (run_id)" in query
    assert "workflow_checkpoints.workflow_id" in query
    assert "workflow_checkpoints.workflow_version" in query
    assert "workflow_checkpoints.event_count" in query
    assert "workflow_checkpoints.run_status" in query
    assert "RETURNING run_id" in query

    assert parameters[0] == checkpoint.state.run_id

    assert parameters[5] == 1


@pytest.mark.anyio
async def test_save_rejects_stale_or_identity_conflict() -> None:
    connection = FakeConnection(
        [
            None,
        ]
    )

    with pytest.raises(
        WorkflowPersistenceConflictError,
        match="monotonicity",
    ):
        await _store(connection).save(_running_checkpoint())


@pytest.mark.anyio
async def test_load_missing_checkpoint_returns_none() -> None:
    connection = FakeConnection(
        [
            None,
        ]
    )

    assert await _store(connection).load("workflow-run:postgres") is None


@pytest.mark.anyio
async def test_load_reconstructs_valid_checkpoint() -> None:
    checkpoint = _completed_checkpoint()

    connection = FakeConnection(
        [
            _stored_row(checkpoint),
        ]
    )

    restored = await _store(connection).load(checkpoint.state.run_id)

    assert restored == checkpoint


@pytest.mark.anyio
async def test_load_reconstructs_failed_checkpoint() -> None:
    checkpoint = _failed_checkpoint()

    connection = FakeConnection(
        [
            _stored_row(checkpoint),
        ]
    )

    restored = await _store(connection).load(checkpoint.state.run_id)

    assert restored == checkpoint


@pytest.mark.anyio
async def test_load_detects_checksum_corruption() -> None:
    checkpoint = _running_checkpoint()

    row = list(_stored_row(checkpoint))

    row[6] = "0" * 64

    connection = FakeConnection(
        [
            tuple(row),
        ]
    )

    with pytest.raises(
        WorkflowPersistenceIntegrityError,
        match="checksum mismatch",
    ):
        await _store(connection).load(checkpoint.state.run_id)


@pytest.mark.anyio
async def test_load_detects_metadata_mismatch() -> None:
    checkpoint = _running_checkpoint()

    row = list(_stored_row(checkpoint))

    row[0] = "different-workflow"

    connection = FakeConnection(
        [
            tuple(row),
        ]
    )

    with pytest.raises(
        WorkflowPersistenceIntegrityError,
        match="metadata mismatch",
    ):
        await _store(connection).load(checkpoint.state.run_id)


@pytest.mark.anyio
async def test_load_detects_event_count_mismatch() -> None:
    checkpoint = _running_checkpoint()

    row = list(_stored_row(checkpoint))

    row[4] = 99

    connection = FakeConnection(
        [
            tuple(row),
        ]
    )

    with pytest.raises(
        WorkflowPersistenceIntegrityError,
        match="metadata mismatch",
    ):
        await _store(connection).load(checkpoint.state.run_id)


@pytest.mark.anyio
async def test_load_rejects_bad_digest_shape() -> None:
    checkpoint = _running_checkpoint()

    row = list(_stored_row(checkpoint))

    row[6] = "not-a-sha256"

    connection = FakeConnection(
        [
            tuple(row),
        ]
    )

    with pytest.raises(
        WorkflowPersistenceIntegrityError,
        match="digest is invalid",
    ):
        await _store(connection).load(checkpoint.state.run_id)


@pytest.mark.anyio
async def test_database_failure_is_normalized() -> None:
    connection = FakeConnection([])

    connection.error = RuntimeError("synthetic database outage")

    with pytest.raises(
        WorkflowPersistenceError,
        match="save failed",
    ):
        await _store(connection).save(_running_checkpoint())


@pytest.mark.anyio
async def test_load_database_failure_is_normalized() -> None:
    connection = FakeConnection([])

    connection.error = RuntimeError("synthetic database outage")

    with pytest.raises(
        WorkflowPersistenceError,
        match="load failed",
    ):
        await _store(connection).load("workflow-run:postgres")


@pytest.mark.anyio
async def test_invalid_run_id_fails_before_database_access() -> None:
    connection = FakeConnection([])

    with pytest.raises(
        ValueError,
        match="portable workflow identifier",
    ):
        await _store(connection).load("bad run id")

    assert connection.calls == []


@pytest.mark.anyio
async def test_load_reconstructs_legacy_v1_checkpoint() -> None:
    """A stored Feature-18 format-v1 row remains readable after v2."""
    checkpoint = _completed_checkpoint()

    payload = WorkflowCheckpointCodec().dumps(
        checkpoint,
        format_version=1,
    )

    row = (
        checkpoint.state.workflow_id,
        checkpoint.state.workflow_version,
        checkpoint.state.status.value,
        1,
        len(checkpoint.events),
        payload,
        sha256(payload.encode("utf-8")).hexdigest(),
    )

    connection = FakeConnection(
        [
            row,
        ]
    )

    restored = await _store(connection).load(checkpoint.state.run_id)

    assert restored == checkpoint

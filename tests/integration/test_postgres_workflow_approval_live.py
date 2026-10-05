"""Live PostgreSQL coverage for workflow checkpoint formats v1 and v2."""

from __future__ import annotations

import os
from hashlib import sha256
from uuid import uuid4

import pytest

from ai_engineering_agent_platform.adapters.postgres.workflow_state_store import (
    PostgreSQLWorkflowStateStore,
)
from ai_engineering_agent_platform.config import Settings
from ai_engineering_agent_platform.domain.workflow import (
    WorkflowApprovalPause,
    WorkflowEventType,
    WorkflowExecutionEvent,
    WorkflowRunState,
    WorkflowRunStatus,
    WorkflowStepState,
    WorkflowStepStatus,
)
from ai_engineering_agent_platform.runtime.postgres import postgres_pool_runtime
from ai_engineering_agent_platform.services.workflow_persistence import (
    WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    WorkflowCheckpoint,
    WorkflowCheckpointCodec,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("AI_PLATFORM_RUN_POSTGRES_INTEGRATION") != "1",
    reason="live PostgreSQL integration is opt-in",
)


def _awaiting_checkpoint(run_id: str) -> WorkflowCheckpoint:
    state = WorkflowRunState(
        run_id=run_id,
        workflow_id="workflow-approval-live",
        workflow_version="1",
        status=WorkflowRunStatus.AWAITING_APPROVAL,
        inputs=(),
        steps=(
            WorkflowStepState(
                step_id="agent-step",
                executor_name="agent",
                status=WorkflowStepStatus.AWAITING_APPROVAL,
                approval_pause=WorkflowApprovalPause(
                    continuation_identity=(f"call-{uuid4().hex}",),
                    approval_ids=(f"approval-{uuid4().hex}",),
                ),
            ),
        ),
    )

    return WorkflowCheckpoint(
        state=state,
        events=(
            WorkflowExecutionEvent(
                sequence=1,
                run_id=state.run_id,
                workflow_id=state.workflow_id,
                workflow_version=state.workflow_version,
                event_type=WorkflowEventType.RUN_STARTED,
            ),
            WorkflowExecutionEvent(
                sequence=2,
                run_id=state.run_id,
                workflow_id=state.workflow_id,
                workflow_version=state.workflow_version,
                event_type=WorkflowEventType.STEP_AWAITING_APPROVAL,
                step_id="agent-step",
                executor_name="agent",
            ),
        ),
    )


def _legacy_checkpoint(run_id: str) -> WorkflowCheckpoint:
    state = WorkflowRunState(
        run_id=run_id,
        workflow_id="workflow-v1-live",
        workflow_version="1",
        status=WorkflowRunStatus.RUNNING,
        inputs=(),
        steps=(),
    )

    return WorkflowCheckpoint(
        state=state,
        events=(
            WorkflowExecutionEvent(
                sequence=1,
                run_id=state.run_id,
                workflow_id=state.workflow_id,
                workflow_version=state.workflow_version,
                event_type=WorkflowEventType.RUN_STARTED,
            ),
        ),
    )


@pytest.mark.anyio
async def test_postgres_workflow_v2_pause_and_v1_compatibility() -> None:
    settings = Settings()

    assert settings.postgres_user == "ai_platform_runtime"
    assert WORKFLOW_CHECKPOINT_FORMAT_VERSION == 2

    awaiting = _awaiting_checkpoint(f"workflow-awaiting-{uuid4().hex}")
    legacy = _legacy_checkpoint(f"workflow-v1-{uuid4().hex}")

    codec = WorkflowCheckpointCodec()

    async with postgres_pool_runtime(settings) as pool:
        store = PostgreSQLWorkflowStateStore(pool)

        # V2 approval pause through the real adapter.
        await store.save(awaiting)

        restarted = PostgreSQLWorkflowStateStore(pool)

        assert await restarted.load(awaiting.state.run_id) == awaiting

        async with pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT
                    run_status,
                    checkpoint_format,
                    event_count
                FROM ai_platform.workflow_checkpoints
                WHERE run_id = %s
                """,
                (awaiting.state.run_id,),
            )

            assert await cursor.fetchone() == (
                "awaiting_approval",
                2,
                len(awaiting.events),
            )

            # Runtime workflow-table privileges remain least privilege.
            cursor = await connection.execute(
                """
                SELECT
                    has_table_privilege(
                        current_user,
                        'ai_platform.workflow_checkpoints',
                        'SELECT'
                    ),
                    has_table_privilege(
                        current_user,
                        'ai_platform.workflow_checkpoints',
                        'INSERT'
                    ),
                    has_table_privilege(
                        current_user,
                        'ai_platform.workflow_checkpoints',
                        'UPDATE'
                    ),
                    has_table_privilege(
                        current_user,
                        'ai_platform.workflow_checkpoints',
                        'DELETE'
                    )
                """
            )

            assert await cursor.fetchone() == (
                True,
                True,
                True,
                False,
            )

            # Persist one genuine canonical v1 payload to prove restart
            # compatibility with Feature-18 checkpoints.
            legacy_payload = codec.dumps(
                legacy,
                format_version=1,
            )
            legacy_digest = sha256(legacy_payload.encode("utf-8")).hexdigest()

            await connection.execute(
                """
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
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    legacy.state.run_id,
                    legacy.state.workflow_id,
                    legacy.state.workflow_version,
                    legacy.state.status.value,
                    1,
                    len(legacy.events),
                    legacy_payload,
                    legacy_digest,
                ),
            )

        restored_legacy = await restarted.load(legacy.state.run_id)

        assert restored_legacy == legacy

        assert (
            codec.dumps(
                restored_legacy,
                format_version=1,
            )
            == legacy_payload
        )

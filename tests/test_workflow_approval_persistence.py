"""Tests for workflow approval-pause checkpoint persistence."""

import json

import pytest

from ai_engineering_agent_platform.domain.workflow import (
    WorkflowApprovalPause,
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
    WorkflowSerializationError,
)


def _awaiting() -> WorkflowCheckpoint:
    state = WorkflowRunState(
        run_id="workflow-run",
        workflow_id="workflow",
        workflow_version="1",
        status=WorkflowRunStatus.AWAITING_APPROVAL,
        inputs=(),
        steps=(
            WorkflowStepState(
                step_id="agent-step",
                executor_name="agent",
                status=(WorkflowStepStatus.AWAITING_APPROVAL),
                approval_pause=WorkflowApprovalPause(
                    continuation_identity=("agent:workflow-run:tool:1:call",),
                    approval_ids=("approval-1",),
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
                event_type=(WorkflowEventType.RUN_STARTED),
            ),
            WorkflowExecutionEvent(
                sequence=2,
                run_id=state.run_id,
                workflow_id=state.workflow_id,
                workflow_version=state.workflow_version,
                event_type=(WorkflowEventType.STEP_AWAITING_APPROVAL),
                step_id="agent-step",
                executor_name="agent",
            ),
        ),
    )


def _running() -> WorkflowCheckpoint:
    state = WorkflowRunState(
        run_id="legacy-run",
        workflow_id="workflow",
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
                event_type=(WorkflowEventType.RUN_STARTED),
            ),
        ),
    )


def _canonical_json(
    value: object,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )


def test_v2_round_trip_preserves_pause_references() -> None:
    codec = WorkflowCheckpointCodec()
    checkpoint = _awaiting()

    payload = codec.dumps(checkpoint)

    raw = json.loads(payload)

    assert raw["format"] == WORKFLOW_CHECKPOINT_FORMAT_VERSION == 2

    assert raw["state"]["steps"][0]["approval_pause"] == {
        "approval_ids": [
            "approval-1",
        ],
        "continuation_identity": [
            "agent:workflow-run:tool:1:call",
        ],
    }

    decoded = codec.loads(payload)

    assert decoded == checkpoint
    assert codec.dumps(decoded) == payload


def test_v1_running_checkpoint_remains_canonical() -> None:
    codec = WorkflowCheckpointCodec()
    checkpoint = _running()

    payload = codec.dumps(
        checkpoint,
        format_version=1,
    )

    assert json.loads(payload)["format"] == 1

    decoded = codec.loads(payload)

    assert decoded == checkpoint

    assert (
        codec.dumps(
            decoded,
            format_version=1,
        )
        == payload
    )


def test_v1_cannot_encode_pause_state() -> None:
    with pytest.raises(
        WorkflowSerializationError,
        match="format 1 cannot encode approval pause",
    ):
        WorkflowCheckpointCodec().dumps(
            _awaiting(),
            format_version=1,
        )


def test_v1_tampered_to_awaiting_fails_closed() -> None:
    codec = WorkflowCheckpointCodec()

    raw = json.loads(
        codec.dumps(
            _running(),
            format_version=1,
        )
    )

    raw["state"]["status"] = "awaiting_approval"

    with pytest.raises(
        WorkflowSerializationError,
        match="format 1 cannot encode approval pause",
    ):
        codec.loads(_canonical_json(raw))


def test_v2_requires_explicit_approval_pause_key() -> None:
    codec = WorkflowCheckpointCodec()

    raw = json.loads(codec.dumps(_awaiting()))

    del raw["state"]["steps"][0]["approval_pause"]

    with pytest.raises(
        WorkflowSerializationError,
        match="checkpoint step fields differ from schema",
    ):
        codec.loads(_canonical_json(raw))


def test_v2_rejects_empty_approval_ids() -> None:
    codec = WorkflowCheckpointCodec()

    raw = json.loads(codec.dumps(_awaiting()))

    raw["state"]["steps"][0]["approval_pause"]["approval_ids"] = []

    with pytest.raises(
        WorkflowSerializationError,
        match="workflow domain invariants",
    ):
        codec.loads(_canonical_json(raw))


def test_awaiting_checkpoint_requires_matching_pause_event() -> None:
    checkpoint = _awaiting()

    with pytest.raises(
        ValueError,
        match="must end with matching pause event",
    ):
        WorkflowCheckpoint(
            state=checkpoint.state,
            events=(checkpoint.events[0],),
        )


def test_awaiting_checkpoint_rejects_terminal_event() -> None:
    checkpoint = _awaiting()

    with pytest.raises(
        ValueError,
        match="must not contain terminal event",
    ):
        WorkflowCheckpoint(
            state=checkpoint.state,
            events=(
                *checkpoint.events,
                WorkflowExecutionEvent(
                    sequence=3,
                    run_id=(checkpoint.state.run_id),
                    workflow_id=(checkpoint.state.workflow_id),
                    workflow_version=(checkpoint.state.workflow_version),
                    event_type=(WorkflowEventType.RUN_FAILED),
                ),
            ),
        )

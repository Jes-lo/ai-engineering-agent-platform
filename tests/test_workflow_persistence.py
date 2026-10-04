"""Tests for durable workflow checkpoint contracts and serialization."""

import json
from dataclasses import replace

import pytest

from ai_engineering_agent_platform.domain.workflow import (
    WorkflowEventType,
    WorkflowExecutionEvent,
    WorkflowInput,
    WorkflowRunState,
    WorkflowRunStatus,
    WorkflowStepExecution,
    WorkflowStepState,
    WorkflowStepStatus,
)
from ai_engineering_agent_platform.services.workflow_persistence import (
    WorkflowCheckpoint,
    WorkflowCheckpointCodec,
    WorkflowSerializationError,
)


def _event(
    sequence: int,
    event_type: WorkflowEventType,
    *,
    step_id: str | None = None,
    executor_name: str | None = None,
) -> WorkflowExecutionEvent:
    return WorkflowExecutionEvent(
        sequence=sequence,
        run_id="workflow-run:durable",
        workflow_id="durable-workflow",
        workflow_version="1",
        event_type=event_type,
        step_id=step_id,
        executor_name=executor_name,
    )


def _running_checkpoint() -> WorkflowCheckpoint:
    execution = WorkflowStepExecution(
        step_id="collect",
        executor_name="collect",
        output={
            "count": 3,
            "nested": (
                "ok",
                1.5,
                True,
                None,
            ),
        },
    )

    state = WorkflowRunState(
        run_id="workflow-run:durable",
        workflow_id="durable-workflow",
        workflow_version="1",
        status=WorkflowRunStatus.RUNNING,
        inputs=(
            WorkflowInput(
                name="region",
                value="mx",
            ),
            WorkflowInput(
                name="limit",
                value=3,
            ),
        ),
        steps=(
            WorkflowStepState(
                step_id="collect",
                executor_name="collect",
                status=WorkflowStepStatus.COMPLETED,
                execution=execution,
            ),
            WorkflowStepState(
                step_id="optional",
                executor_name="rag",
                status=WorkflowStepStatus.SKIPPED,
            ),
        ),
    )

    return WorkflowCheckpoint(
        state=state,
        events=(
            _event(
                1,
                WorkflowEventType.RUN_STARTED,
            ),
            _event(
                2,
                WorkflowEventType.STEP_COMPLETED,
                step_id="collect",
                executor_name="collect",
            ),
            _event(
                3,
                WorkflowEventType.STEP_SKIPPED,
                step_id="optional",
                executor_name="rag",
            ),
        ),
    )


def test_checkpoint_round_trip_preserves_exact_types() -> None:
    codec = WorkflowCheckpointCodec()
    checkpoint = _running_checkpoint()

    encoded = codec.dumps(checkpoint)
    restored = codec.loads(encoded)

    assert restored == checkpoint

    output = restored.state.steps[0].execution

    assert output is not None
    assert isinstance(
        output.output,
        dict,
    )

    nested = output.output["nested"]

    assert isinstance(
        nested,
        tuple,
    )
    assert type(nested[1]) is float
    assert type(nested[2]) is bool


def test_checkpoint_encoding_is_canonical() -> None:
    codec = WorkflowCheckpointCodec()
    checkpoint = _running_checkpoint()

    first = codec.dumps(checkpoint)

    second = codec.dumps(codec.loads(first))

    assert second == first


def test_unsupported_executor_output_fails_closed() -> None:
    codec = WorkflowCheckpointCodec()

    execution = WorkflowStepExecution(
        step_id="collect",
        executor_name="collect",
        output=object(),
    )

    checkpoint = WorkflowCheckpoint(
        state=WorkflowRunState(
            run_id="workflow-run:durable",
            workflow_id="durable-workflow",
            workflow_version="1",
            status=WorkflowRunStatus.RUNNING,
            inputs=(),
            steps=(
                WorkflowStepState(
                    step_id="collect",
                    executor_name="collect",
                    status=WorkflowStepStatus.COMPLETED,
                    execution=execution,
                ),
            ),
        ),
        events=(
            _event(
                1,
                WorkflowEventType.RUN_STARTED,
            ),
            _event(
                2,
                WorkflowEventType.STEP_COMPLETED,
                step_id="collect",
                executor_name="collect",
            ),
        ),
    )

    with pytest.raises(
        WorkflowSerializationError,
        match="unsupported durable value type",
    ):
        codec.dumps(checkpoint)


def test_running_checkpoint_rejects_terminal_event() -> None:
    checkpoint = _running_checkpoint()

    with pytest.raises(
        ValueError,
        match="running checkpoint must not contain terminal event",
    ):
        WorkflowCheckpoint(
            state=checkpoint.state,
            events=(
                *checkpoint.events,
                _event(
                    4,
                    WorkflowEventType.RUN_COMPLETED,
                ),
            ),
        )


def test_unknown_format_version_fails_closed() -> None:
    codec = WorkflowCheckpointCodec()

    data = json.loads(codec.dumps(_running_checkpoint()))

    data["format"] = 2

    with pytest.raises(
        WorkflowSerializationError,
        match="unsupported checkpoint format version",
    ):
        codec.loads(json.dumps(data))


def test_unknown_top_level_field_fails_closed() -> None:
    codec = WorkflowCheckpointCodec()

    data = json.loads(codec.dumps(_running_checkpoint()))

    data["unexpected"] = True

    with pytest.raises(
        WorkflowSerializationError,
        match="checkpoint fields differ from schema",
    ):
        codec.loads(json.dumps(data))


def test_duplicate_json_keys_fail_closed() -> None:
    codec = WorkflowCheckpointCodec()

    with pytest.raises(
        WorkflowSerializationError,
        match="duplicate JSON object key",
    ):
        codec.loads('{"format":1,"format":1,"state":{},"events":[]}')


def test_completed_checkpoint_round_trip() -> None:
    codec = WorkflowCheckpointCodec()
    running = _running_checkpoint()

    completed = WorkflowCheckpoint(
        state=replace(
            running.state,
            status=WorkflowRunStatus.COMPLETED,
        ),
        events=(
            *running.events,
            _event(
                4,
                WorkflowEventType.RUN_COMPLETED,
            ),
        ),
    )

    assert codec.loads(codec.dumps(completed)) == completed


def test_integer_and_boolean_types_remain_distinct() -> None:
    restored = WorkflowCheckpointCodec().loads(
        WorkflowCheckpointCodec().dumps(_running_checkpoint())
    )

    execution = restored.state.steps[0].execution

    assert execution is not None
    assert isinstance(
        execution.output,
        dict,
    )

    assert type(execution.output["count"]) is int

    nested = execution.output["nested"]

    assert isinstance(
        nested,
        tuple,
    )

    assert type(nested[2]) is bool


def test_nonfinite_executor_output_fails_closed() -> None:
    checkpoint = _running_checkpoint()

    first = checkpoint.state.steps[0]

    assert first.execution is not None

    bad_execution = replace(
        first.execution,
        output=float("nan"),
    )

    bad_state = replace(
        checkpoint.state,
        steps=(
            replace(
                first,
                execution=bad_execution,
            ),
            *checkpoint.state.steps[1:],
        ),
    )

    with pytest.raises(
        WorkflowSerializationError,
        match="finite",
    ):
        WorkflowCheckpointCodec().dumps(
            WorkflowCheckpoint(
                state=bad_state,
                events=checkpoint.events,
            )
        )


def test_oversized_executor_string_fails_closed() -> None:
    checkpoint = _running_checkpoint()

    first = checkpoint.state.steps[0]

    assert first.execution is not None

    bad_execution = replace(
        first.execution,
        output="x" * 65_537,
    )

    bad_state = replace(
        checkpoint.state,
        steps=(
            replace(
                first,
                execution=bad_execution,
            ),
            *checkpoint.state.steps[1:],
        ),
    )

    with pytest.raises(
        WorkflowSerializationError,
        match="string exceeds",
    ):
        WorkflowCheckpointCodec().dumps(
            WorkflowCheckpoint(
                state=bad_state,
                events=checkpoint.events,
            )
        )


def test_checkpoint_event_identity_mismatch_fails_closed() -> None:
    checkpoint = _running_checkpoint()

    wrong_event = WorkflowExecutionEvent(
        sequence=2,
        run_id="workflow-run:other",
        workflow_id="durable-workflow",
        workflow_version="1",
        event_type=WorkflowEventType.STEP_COMPLETED,
        step_id="collect",
        executor_name="collect",
    )

    with pytest.raises(
        ValueError,
        match="identity mismatch",
    ):
        WorkflowCheckpoint(
            state=checkpoint.state,
            events=(
                checkpoint.events[0],
                wrong_event,
            ),
        )

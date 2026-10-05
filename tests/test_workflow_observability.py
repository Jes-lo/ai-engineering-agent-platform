"""Tests for workflow structural events and privacy-safe observability."""

from dataclasses import dataclass, field, fields

import pytest

from ai_engineering_agent_platform.domain.workflow import (
    WorkflowCondition,
    WorkflowConditionOperator,
    WorkflowDefinition,
    WorkflowEventType,
    WorkflowExecutionEvent,
    WorkflowExecutionTrace,
    WorkflowInput,
    WorkflowInputDefinition,
    WorkflowInputType,
    WorkflowRunResult,
    WorkflowRunStatus,
    WorkflowStepDefinition,
)
from ai_engineering_agent_platform.services.workflow import (
    WorkflowEngine,
    WorkflowExecutorRegistry,
    WorkflowStepContext,
    WorkflowStepExecutionError,
    WorkflowStepExecutor,
)
from ai_engineering_agent_platform.services.workflow_observability import (
    summarize_workflow_state,
    workflow_event_attributes,
)


@dataclass
class ObservabilityExecutor:
    """Synthetic executor returning deliberately sensitive-looking content."""

    name: str
    calls: list[WorkflowStepContext] = field(default_factory=list)

    async def execute(
        self,
        *,
        step: WorkflowStepDefinition,
        context: WorkflowStepContext,
    ) -> object:
        self.calls.append(context)

        return "TOP-SECRET-OUTPUT:" + step.step_id


@dataclass
class FailingObservabilityExecutor:
    """Synthetic failing executor with sensitive-looking exception text."""

    name: str
    calls: int = 0

    async def execute(
        self,
        *,
        step: WorkflowStepDefinition,
        context: WorkflowStepContext,
    ) -> object:
        del step
        del context

        self.calls += 1

        raise RuntimeError("TOP-SECRET-ERROR-TEXT")


def _engine(
    *executors: WorkflowStepExecutor,
) -> WorkflowEngine:
    return WorkflowEngine(
        WorkflowExecutorRegistry(executors),
        run_id_factory=lambda: "workflow-run:observability",
    )


@pytest.mark.anyio
async def test_completed_run_emits_ordered_structural_events() -> None:
    executor = ObservabilityExecutor("executor")

    definition = WorkflowDefinition(
        workflow_id="observed-workflow",
        version="1",
        inputs=(
            WorkflowInputDefinition(
                name="route",
                input_type=WorkflowInputType.STRING,
            ),
        ),
        steps=(
            WorkflowStepDefinition(
                step_id="first",
                executor_name="executor",
            ),
            WorkflowStepDefinition(
                step_id="skipped",
                executor_name="executor",
                condition=WorkflowCondition(
                    input_name="route",
                    operator=WorkflowConditionOperator.EQUALS,
                    expected_value="execute-skipped",
                ),
            ),
            WorkflowStepDefinition(
                step_id="final",
                executor_name="executor",
                depends_on=(
                    "first",
                    "skipped",
                ),
            ),
        ),
    )

    result = await _engine(executor).run(
        definition,
        inputs=(
            WorkflowInput(
                name="route",
                value="TOP-SECRET-INPUT",
            ),
        ),
    )
    assert isinstance(result, WorkflowRunResult)

    assert result.trace is not None

    assert tuple(event.event_type for event in result.trace.events) == (
        WorkflowEventType.RUN_STARTED,
        WorkflowEventType.STEP_COMPLETED,
        WorkflowEventType.STEP_SKIPPED,
        WorkflowEventType.STEP_COMPLETED,
        WorkflowEventType.RUN_COMPLETED,
    )

    assert tuple(event.sequence for event in result.trace.events) == (
        1,
        2,
        3,
        4,
        5,
    )

    assert tuple(event.event_id for event in result.trace.events) == (
        "workflow-run:observability:event:1",
        "workflow-run:observability:event:2",
        "workflow-run:observability:event:3",
        "workflow-run:observability:event:4",
        "workflow-run:observability:event:5",
    )

    rendered_trace = repr(result.trace)

    assert "TOP-SECRET-INPUT" not in rendered_trace

    assert "TOP-SECRET-OUTPUT" not in rendered_trace


@pytest.mark.anyio
async def test_failed_run_trace_contains_no_exception_text() -> None:
    first = ObservabilityExecutor("first")

    failing = FailingObservabilityExecutor("failing")

    definition = WorkflowDefinition(
        workflow_id="failed-observed-workflow",
        version="1",
        steps=(
            WorkflowStepDefinition(
                step_id="first-step",
                executor_name="first",
            ),
            WorkflowStepDefinition(
                step_id="failed-step",
                executor_name="failing",
                depends_on=("first-step",),
            ),
        ),
    )

    with pytest.raises(
        WorkflowStepExecutionError,
    ) as captured:
        await _engine(
            first,
            failing,
        ).run(definition)

    error = captured.value

    assert failing.calls == 1

    assert error.trace is not None

    assert tuple(event.event_type for event in error.trace.events) == (
        WorkflowEventType.RUN_STARTED,
        WorkflowEventType.STEP_COMPLETED,
        WorkflowEventType.STEP_FAILED,
        WorkflowEventType.RUN_FAILED,
    )

    rendered_trace = repr(error.trace)

    assert "TOP-SECRET-OUTPUT" not in rendered_trace

    assert "TOP-SECRET-ERROR-TEXT" not in rendered_trace


def test_workflow_event_schema_contains_only_structural_metadata() -> None:
    assert tuple(item.name for item in fields(WorkflowExecutionEvent)) == (
        "sequence",
        "run_id",
        "workflow_id",
        "workflow_version",
        "event_type",
        "step_id",
        "executor_name",
    )


def test_trace_rejects_non_contiguous_sequences() -> None:
    with pytest.raises(
        ValueError,
        match="contiguous",
    ):
        WorkflowExecutionTrace(
            run_id="workflow-run:trace",
            workflow_id="workflow",
            workflow_version="1",
            events=(
                WorkflowExecutionEvent(
                    sequence=1,
                    run_id="workflow-run:trace",
                    workflow_id="workflow",
                    workflow_version="1",
                    event_type=WorkflowEventType.RUN_STARTED,
                ),
                WorkflowExecutionEvent(
                    sequence=3,
                    run_id="workflow-run:trace",
                    workflow_id="workflow",
                    workflow_version="1",
                    event_type=WorkflowEventType.RUN_COMPLETED,
                ),
            ),
        )


@pytest.mark.anyio
async def test_safe_event_attributes_are_strictly_allowlisted() -> None:
    executor = ObservabilityExecutor("executor")

    definition = WorkflowDefinition(
        workflow_id="attributes",
        version="1",
        steps=(
            WorkflowStepDefinition(
                step_id="step",
                executor_name="executor",
            ),
        ),
    )

    result = await _engine(executor).run(definition)
    assert isinstance(result, WorkflowRunResult)

    assert result.trace is not None

    event = result.trace.events[1]

    attributes = workflow_event_attributes(event)

    assert dict(attributes) == {
        "workflow.event.id": ("workflow-run:observability:event:2"),
        "workflow.event.sequence": 2,
        "workflow.event.type": ("step_completed"),
        "workflow.run.id": ("workflow-run:observability"),
        "workflow.id": "attributes",
        "workflow.version": "1",
        "workflow.step.id": "step",
        "workflow.executor.name": ("executor"),
    }

    serialized = repr(attributes).lower()

    for forbidden in (
        "input.value",
        "prompt",
        "tool.argument",
        "rag.content",
        "output",
        "exception",
        "error.message",
    ):
        assert forbidden not in serialized


@pytest.mark.anyio
async def test_completed_observability_summary_uses_only_state_counts() -> None:
    executor = ObservabilityExecutor("executor")

    definition = WorkflowDefinition(
        workflow_id="summary",
        version="1",
        inputs=(
            WorkflowInputDefinition(
                name="enabled",
                input_type=WorkflowInputType.BOOLEAN,
            ),
        ),
        steps=(
            WorkflowStepDefinition(
                step_id="completed",
                executor_name="executor",
            ),
            WorkflowStepDefinition(
                step_id="skipped",
                executor_name="executor",
                condition=WorkflowCondition(
                    input_name="enabled",
                    operator=WorkflowConditionOperator.EQUALS,
                    expected_value=True,
                ),
            ),
        ),
    )

    result = await _engine(executor).run(
        definition,
        inputs=(
            WorkflowInput(
                name="enabled",
                value=False,
            ),
        ),
    )
    assert isinstance(result, WorkflowRunResult)

    assert result.state is not None
    assert result.trace is not None

    summary = summarize_workflow_state(
        state=result.state,
        trace=result.trace,
    )

    assert summary.run_status is WorkflowRunStatus.COMPLETED

    assert summary.completed_steps == 1
    assert summary.skipped_steps == 1
    assert summary.failed_steps == 0
    assert summary.event_count == 4


@pytest.mark.anyio
async def test_failed_observability_summary_is_structural() -> None:
    failing = FailingObservabilityExecutor("failing")

    definition = WorkflowDefinition(
        workflow_id="failed-summary",
        version="1",
        steps=(
            WorkflowStepDefinition(
                step_id="failed",
                executor_name="failing",
            ),
        ),
    )

    with pytest.raises(
        WorkflowStepExecutionError,
    ) as captured:
        await _engine(failing).run(definition)

    error = captured.value

    assert error.state is not None
    assert error.trace is not None

    summary = summarize_workflow_state(
        state=error.state,
        trace=error.trace,
    )

    assert summary.run_status is WorkflowRunStatus.FAILED

    assert summary.completed_steps == 0
    assert summary.skipped_steps == 0
    assert summary.failed_steps == 1
    assert summary.event_count == 3

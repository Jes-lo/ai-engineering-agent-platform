"""Privacy-safe projections for structured workflow execution events."""

from dataclasses import dataclass

from ai_engineering_agent_platform.domain.workflow import (
    WorkflowExecutionEvent,
    WorkflowExecutionTrace,
    WorkflowRunState,
    WorkflowRunStatus,
    WorkflowStepStatus,
)

type WorkflowEventAttributeValue = str | int


@dataclass(
    frozen=True,
    slots=True,
)
class WorkflowObservabilitySummary:
    """Low-content structural summary suitable for operational telemetry."""

    run_id: str
    workflow_id: str
    workflow_version: str
    run_status: WorkflowRunStatus
    completed_steps: int
    skipped_steps: int
    failed_steps: int
    event_count: int


def workflow_event_attributes(
    event: WorkflowExecutionEvent,
) -> tuple[
    tuple[
        str,
        WorkflowEventAttributeValue,
    ],
    ...,
]:
    """Return allowlisted metadata for one structural workflow event.

    No workflow input values, prompts, tool arguments, retrieval/generation
    content, executor outputs, or exception text are exported.
    """
    if not isinstance(
        event,
        WorkflowExecutionEvent,
    ):
        raise ValueError("event must be WorkflowExecutionEvent")

    attributes: list[
        tuple[
            str,
            WorkflowEventAttributeValue,
        ]
    ] = [
        (
            "workflow.event.id",
            event.event_id,
        ),
        (
            "workflow.event.sequence",
            event.sequence,
        ),
        (
            "workflow.event.type",
            event.event_type.value,
        ),
        (
            "workflow.run.id",
            event.run_id,
        ),
        (
            "workflow.id",
            event.workflow_id,
        ),
        (
            "workflow.version",
            event.workflow_version,
        ),
    ]

    if event.step_id is not None:
        attributes.append(
            (
                "workflow.step.id",
                event.step_id,
            )
        )

    if event.executor_name is not None:
        attributes.append(
            (
                "workflow.executor.name",
                event.executor_name,
            )
        )

    return tuple(attributes)


def summarize_workflow_state(
    *,
    state: WorkflowRunState,
    trace: WorkflowExecutionTrace,
) -> WorkflowObservabilitySummary:
    """Build a structural run summary without inspecting execution outputs."""
    if not isinstance(
        state,
        WorkflowRunState,
    ):
        raise ValueError("state must be WorkflowRunState")

    if not isinstance(
        trace,
        WorkflowExecutionTrace,
    ):
        raise ValueError("trace must be WorkflowExecutionTrace")

    if (
        trace.run_id != state.run_id
        or trace.workflow_id != state.workflow_id
        or trace.workflow_version != state.workflow_version
    ):
        raise ValueError("workflow state/trace identity mismatch")

    completed_steps = sum(
        item.status is WorkflowStepStatus.COMPLETED for item in state.steps
    )

    skipped_steps = sum(
        item.status is WorkflowStepStatus.SKIPPED for item in state.steps
    )

    failed_steps = sum(item.status is WorkflowStepStatus.FAILED for item in state.steps)

    return WorkflowObservabilitySummary(
        run_id=state.run_id,
        workflow_id=state.workflow_id,
        workflow_version=state.workflow_version,
        run_status=state.status,
        completed_steps=completed_steps,
        skipped_steps=skipped_steps,
        failed_steps=failed_steps,
        event_count=len(trace.events),
    )

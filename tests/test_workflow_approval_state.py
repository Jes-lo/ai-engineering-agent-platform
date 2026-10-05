"""Domain tests for durable workflow approval-pause state."""

from __future__ import annotations

import pytest

from ai_engineering_agent_platform.domain.workflow import (
    WorkflowApprovalPause,
    WorkflowEventType,
    WorkflowExecutionEvent,
    WorkflowRunState,
    WorkflowRunStatus,
    WorkflowStepExecution,
    WorkflowStepState,
    WorkflowStepStatus,
)


def _pause() -> WorkflowApprovalPause:
    return WorkflowApprovalPause(
        continuation_identity=("agent:workflow-run:tool:1:call",),
        approval_ids=("approval-1",),
    )


def _awaiting_step() -> WorkflowStepState:
    return WorkflowStepState(
        step_id="agent-step",
        executor_name="agent",
        status=WorkflowStepStatus.AWAITING_APPROVAL,
        approval_pause=_pause(),
    )


def test_approval_pause_requires_unique_bounded_references() -> None:
    with pytest.raises(
        ValueError,
        match="approval IDs values must be unique",
    ):
        WorkflowApprovalPause(
            continuation_identity=("call-1",),
            approval_ids=(
                "approval-1",
                "approval-1",
            ),
        )

    with pytest.raises(
        ValueError,
        match="continuation identity contains an invalid value",
    ):
        WorkflowApprovalPause(
            continuation_identity=(" call-1",),
            approval_ids=("approval-1",),
        )


def test_awaiting_step_requires_pause_and_forbids_execution() -> None:
    with pytest.raises(
        ValueError,
        match="requires approval pause",
    ):
        WorkflowStepState(
            step_id="agent-step",
            executor_name="agent",
            status=WorkflowStepStatus.AWAITING_APPROVAL,
        )

    with pytest.raises(
        ValueError,
        match="must not contain execution",
    ):
        WorkflowStepState(
            step_id="agent-step",
            executor_name="agent",
            status=WorkflowStepStatus.AWAITING_APPROVAL,
            execution=WorkflowStepExecution(
                step_id="agent-step",
                executor_name="agent",
                output="unexpected",
            ),
            approval_pause=_pause(),
        )


def test_completed_step_cannot_retain_approval_pause() -> None:
    with pytest.raises(
        ValueError,
        match="cannot contain approval pause",
    ):
        WorkflowStepState(
            step_id="agent-step",
            executor_name="agent",
            status=WorkflowStepStatus.COMPLETED,
            execution=WorkflowStepExecution(
                step_id="agent-step",
                executor_name="agent",
                output="done",
            ),
            approval_pause=_pause(),
        )


def test_awaiting_run_requires_exactly_one_final_awaiting_step() -> None:
    awaiting = _awaiting_step()

    state = WorkflowRunState(
        run_id="workflow-run",
        workflow_id="workflow",
        workflow_version="1",
        status=WorkflowRunStatus.AWAITING_APPROVAL,
        inputs=(),
        steps=(awaiting,),
    )

    assert state.status is WorkflowRunStatus.AWAITING_APPROVAL

    assert state.steps[-1].approval_pause == _pause()

    with pytest.raises(
        ValueError,
        match="exactly one awaiting step",
    ):
        WorkflowRunState(
            run_id="workflow-run",
            workflow_id="workflow",
            workflow_version="1",
            status=WorkflowRunStatus.AWAITING_APPROVAL,
            inputs=(),
            steps=(),
        )

    completed = WorkflowStepState(
        step_id="completed-step",
        executor_name="synthetic",
        status=WorkflowStepStatus.COMPLETED,
        execution=WorkflowStepExecution(
            step_id="completed-step",
            executor_name="synthetic",
            output="done",
        ),
    )

    with pytest.raises(
        ValueError,
        match="must be the last run step",
    ):
        WorkflowRunState(
            run_id="workflow-run",
            workflow_id="workflow",
            workflow_version="1",
            status=WorkflowRunStatus.AWAITING_APPROVAL,
            inputs=(),
            steps=(
                awaiting,
                completed,
            ),
        )


def test_nonawaiting_run_cannot_contain_awaiting_step() -> None:
    with pytest.raises(
        ValueError,
        match="non-awaiting run state",
    ):
        WorkflowRunState(
            run_id="workflow-run",
            workflow_id="workflow",
            workflow_version="1",
            status=WorkflowRunStatus.RUNNING,
            inputs=(),
            steps=(_awaiting_step(),),
        )


def test_awaiting_event_is_structural_step_event() -> None:
    event = WorkflowExecutionEvent(
        sequence=2,
        run_id="workflow-run",
        workflow_id="workflow",
        workflow_version="1",
        event_type=(WorkflowEventType.STEP_AWAITING_APPROVAL),
        step_id="agent-step",
        executor_name="agent",
    )

    assert event.event_type is WorkflowEventType.STEP_AWAITING_APPROVAL

    with pytest.raises(
        ValueError,
        match="requires step identity",
    ):
        WorkflowExecutionEvent(
            sequence=2,
            run_id="workflow-run",
            workflow_id="workflow",
            workflow_version="1",
            event_type=(WorkflowEventType.STEP_AWAITING_APPROVAL),
        )

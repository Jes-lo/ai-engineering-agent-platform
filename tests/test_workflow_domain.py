"""Tests for immutable workflow domain contracts."""

import pytest

from ai_engineering_agent_platform.domain.workflow import (
    MAX_WORKFLOW_DEFINITION_STEPS,
    WorkflowDefinition,
    WorkflowRunResult,
    WorkflowStepDefinition,
    WorkflowStepExecution,
)


def test_static_dag_supports_fan_out_and_fan_in() -> None:
    definition = WorkflowDefinition(
        workflow_id="support-analysis",
        version="1",
        steps=(
            WorkflowStepDefinition(
                step_id="collect",
                executor_name="collect",
            ),
            WorkflowStepDefinition(
                step_id="rag",
                executor_name="rag",
                depends_on=("collect",),
            ),
            WorkflowStepDefinition(
                step_id="agent",
                executor_name="agent",
                depends_on=("collect",),
            ),
            WorkflowStepDefinition(
                step_id="finalize",
                executor_name="finalize",
                depends_on=(
                    "rag",
                    "agent",
                ),
            ),
        ),
    )

    assert definition.ordered_step_ids == (
        "collect",
        "rag",
        "agent",
        "finalize",
    )


def test_duplicate_step_ids_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="step IDs must be unique",
    ):
        WorkflowDefinition(
            workflow_id="duplicate",
            version="1",
            steps=(
                WorkflowStepDefinition(
                    step_id="same",
                    executor_name="first",
                ),
                WorkflowStepDefinition(
                    step_id="same",
                    executor_name="second",
                ),
            ),
        )


def test_dependency_must_reference_earlier_step() -> None:
    with pytest.raises(
        ValueError,
        match="earlier declared steps",
    ):
        WorkflowDefinition(
            workflow_id="invalid-order",
            version="1",
            steps=(
                WorkflowStepDefinition(
                    step_id="first",
                    executor_name="first",
                    depends_on=("later",),
                ),
                WorkflowStepDefinition(
                    step_id="later",
                    executor_name="later",
                ),
            ),
        )


def test_duplicate_dependency_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="dependency step IDs must be unique",
    ):
        WorkflowStepDefinition(
            step_id="second",
            executor_name="second",
            depends_on=(
                "first",
                "first",
            ),
        )


def test_self_dependency_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="cannot depend on itself",
    ):
        WorkflowStepDefinition(
            step_id="loop",
            executor_name="loop",
            depends_on=("loop",),
        )


def test_definition_has_hard_size_ceiling() -> None:
    steps = tuple(
        WorkflowStepDefinition(
            step_id=f"step-{index}",
            executor_name="executor",
        )
        for index in range(MAX_WORKFLOW_DEFINITION_STEPS + 1)
    )

    with pytest.raises(
        ValueError,
        match="exceeds maximum definition size",
    ):
        WorkflowDefinition(
            workflow_id="too-large",
            version="1",
            steps=steps,
        )


def test_completed_run_rejects_duplicate_execution_ids() -> None:
    execution = WorkflowStepExecution(
        step_id="one",
        executor_name="executor",
        output="ok",
    )

    with pytest.raises(
        ValueError,
        match="execution step IDs must be unique",
    ):
        WorkflowRunResult(
            run_id="workflow-run:test",
            workflow_id="workflow",
            workflow_version="1",
            executions=(
                execution,
                execution,
            ),
        )

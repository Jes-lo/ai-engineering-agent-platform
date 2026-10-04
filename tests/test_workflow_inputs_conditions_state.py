"""Tests for typed workflow inputs, conditions, and process-local run state."""

from dataclasses import dataclass, field

import pytest

from ai_engineering_agent_platform.domain.workflow import (
    WorkflowCondition,
    WorkflowConditionOperator,
    WorkflowDefinition,
    WorkflowInput,
    WorkflowInputDefinition,
    WorkflowInputType,
    WorkflowRunStatus,
    WorkflowStepDefinition,
    WorkflowStepStatus,
)
from ai_engineering_agent_platform.services.workflow import (
    WorkflowEngine,
    WorkflowExecutorRegistry,
    WorkflowInputError,
    WorkflowStepContext,
    WorkflowStepExecutionError,
)


@dataclass
class StateRecordingExecutor:
    """Synthetic executor capturing typed inputs/dependency state."""

    name: str
    calls: list[WorkflowStepContext] = field(default_factory=list)

    async def execute(
        self,
        *,
        step: WorkflowStepDefinition,
        context: WorkflowStepContext,
    ) -> object:
        self.calls.append(context)

        return "completed:" + step.step_id


@dataclass
class StateFailingExecutor:
    """Synthetic executor that fails exactly once when invoked."""

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

        raise RuntimeError("synthetic state failure")


def _engine(
    *executors: object,
) -> WorkflowEngine:
    return WorkflowEngine(
        WorkflowExecutorRegistry(
            executors  # type: ignore[arg-type]
        ),
        max_steps=16,
        run_id_factory=lambda: "workflow-run:state",
    )


def test_condition_must_reference_declared_input() -> None:
    with pytest.raises(
        ValueError,
        match="declared input",
    ):
        WorkflowDefinition(
            workflow_id="condition-schema",
            version="1",
            steps=(
                WorkflowStepDefinition(
                    step_id="conditional",
                    executor_name="executor",
                    condition=WorkflowCondition(
                        input_name="missing",
                        operator=WorkflowConditionOperator.EQUALS,
                        expected_value=True,
                    ),
                ),
            ),
        )


def test_condition_value_must_match_declared_type() -> None:
    with pytest.raises(
        ValueError,
        match="match declared input type",
    ):
        WorkflowDefinition(
            workflow_id="condition-type",
            version="1",
            inputs=(
                WorkflowInputDefinition(
                    name="enabled",
                    input_type=WorkflowInputType.BOOLEAN,
                ),
            ),
            steps=(
                WorkflowStepDefinition(
                    step_id="conditional",
                    executor_name="executor",
                    condition=WorkflowCondition(
                        input_name="enabled",
                        operator=WorkflowConditionOperator.EQUALS,
                        expected_value=1,
                    ),
                ),
            ),
        )


@pytest.mark.anyio
async def test_missing_inputs_fail_before_executor_side_effect() -> None:
    executor = StateRecordingExecutor("executor")

    definition = WorkflowDefinition(
        workflow_id="required-input",
        version="1",
        inputs=(
            WorkflowInputDefinition(
                name="mode",
                input_type=WorkflowInputType.STRING,
            ),
        ),
        steps=(
            WorkflowStepDefinition(
                step_id="step",
                executor_name="executor",
            ),
        ),
    )

    with pytest.raises(
        WorkflowInputError,
        match="exactly match",
    ):
        await _engine(executor).run(definition)

    assert executor.calls == []


@pytest.mark.anyio
async def test_extra_inputs_fail_before_executor_side_effect() -> None:
    executor = StateRecordingExecutor("executor")

    definition = WorkflowDefinition(
        workflow_id="extra-input",
        version="1",
        steps=(
            WorkflowStepDefinition(
                step_id="step",
                executor_name="executor",
            ),
        ),
    )

    with pytest.raises(
        WorkflowInputError,
        match="exactly match",
    ):
        await _engine(executor).run(
            definition,
            inputs=(
                WorkflowInput(
                    name="unexpected",
                    value="value",
                ),
            ),
        )

    assert executor.calls == []


@pytest.mark.anyio
async def test_runtime_input_type_mismatch_fails_before_execution() -> None:
    executor = StateRecordingExecutor("executor")

    definition = WorkflowDefinition(
        workflow_id="typed-input",
        version="1",
        inputs=(
            WorkflowInputDefinition(
                name="count",
                input_type=WorkflowInputType.INTEGER,
            ),
        ),
        steps=(
            WorkflowStepDefinition(
                step_id="step",
                executor_name="executor",
            ),
        ),
    )

    with pytest.raises(
        WorkflowInputError,
        match="declared type",
    ):
        await _engine(executor).run(
            definition,
            inputs=(
                WorkflowInput(
                    name="count",
                    value=True,
                ),
            ),
        )

    assert executor.calls == []


@pytest.mark.anyio
async def test_false_condition_skips_executor_and_records_state() -> None:
    executor = StateRecordingExecutor("executor")

    definition = WorkflowDefinition(
        workflow_id="conditional-skip",
        version="1",
        inputs=(
            WorkflowInputDefinition(
                name="enabled",
                input_type=WorkflowInputType.BOOLEAN,
            ),
        ),
        steps=(
            WorkflowStepDefinition(
                step_id="conditional",
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

    assert executor.calls == []

    assert result.executions == ()

    assert result.state is not None

    assert result.state.status is WorkflowRunStatus.COMPLETED

    assert result.state.skipped_step_ids == ("conditional",)

    assert result.state.steps[0].status is WorkflowStepStatus.SKIPPED


@pytest.mark.anyio
async def test_true_condition_executes_and_inputs_reach_context() -> None:
    executor = StateRecordingExecutor("executor")

    definition = WorkflowDefinition(
        workflow_id="conditional-run",
        version="1",
        inputs=(
            WorkflowInputDefinition(
                name="region",
                input_type=WorkflowInputType.STRING,
            ),
        ),
        steps=(
            WorkflowStepDefinition(
                step_id="conditional",
                executor_name="executor",
                condition=WorkflowCondition(
                    input_name="region",
                    operator=WorkflowConditionOperator.NOT_EQUALS,
                    expected_value="blocked",
                ),
            ),
        ),
    )

    result = await _engine(executor).run(
        definition,
        inputs=(
            WorkflowInput(
                name="region",
                value="allowed",
            ),
        ),
    )

    assert len(executor.calls) == 1

    assert executor.calls[0].input_value("region") == "allowed"

    assert result.state is not None

    assert result.state.skipped_step_ids == ()

    assert result.state.steps[0].status is WorkflowStepStatus.COMPLETED


@pytest.mark.anyio
async def test_join_observes_skipped_dependency_explicitly() -> None:
    branch = StateRecordingExecutor("branch")

    join = StateRecordingExecutor("join")

    definition = WorkflowDefinition(
        workflow_id="branch-join",
        version="1",
        inputs=(
            WorkflowInputDefinition(
                name="route",
                input_type=WorkflowInputType.STRING,
            ),
        ),
        steps=(
            WorkflowStepDefinition(
                step_id="branch-a",
                executor_name="branch",
                condition=WorkflowCondition(
                    input_name="route",
                    operator=WorkflowConditionOperator.EQUALS,
                    expected_value="a",
                ),
            ),
            WorkflowStepDefinition(
                step_id="branch-b",
                executor_name="branch",
                condition=WorkflowCondition(
                    input_name="route",
                    operator=WorkflowConditionOperator.EQUALS,
                    expected_value="b",
                ),
            ),
            WorkflowStepDefinition(
                step_id="join",
                executor_name="join",
                depends_on=(
                    "branch-a",
                    "branch-b",
                ),
            ),
        ),
    )

    result = await _engine(
        branch,
        join,
    ).run(
        definition,
        inputs=(
            WorkflowInput(
                name="route",
                value="a",
            ),
        ),
    )

    assert len(branch.calls) == 1

    assert branch.calls[0].step_id == ("branch-a")

    assert len(join.calls) == 1

    join_context = join.calls[0]

    assert tuple(
        execution.step_id for execution in join_context.dependency_results
    ) == ("branch-a",)

    assert join_context.skipped_dependency_ids == ("branch-b",)

    assert join_context.was_dependency_skipped("branch-b") is True

    assert result.state is not None

    assert result.state.skipped_step_ids == ("branch-b",)


@pytest.mark.anyio
async def test_failure_exposes_terminal_failed_run_snapshot() -> None:
    first = StateRecordingExecutor("first")

    failing = StateFailingExecutor("failing")

    definition = WorkflowDefinition(
        workflow_id="failure-state",
        version="1",
        steps=(
            WorkflowStepDefinition(
                step_id="first-step",
                executor_name="first",
            ),
            WorkflowStepDefinition(
                step_id="failing-step",
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

    assert error.state is not None

    assert error.state.status is WorkflowRunStatus.FAILED

    assert tuple(step.status for step in error.state.steps) == (
        WorkflowStepStatus.COMPLETED,
        WorkflowStepStatus.FAILED,
    )

    assert tuple(execution.step_id for execution in error.state.executions) == (
        "first-step",
    )


@pytest.mark.anyio
async def test_run_inputs_are_canonicalized_to_definition_order() -> None:
    executor = StateRecordingExecutor("executor")

    definition = WorkflowDefinition(
        workflow_id="canonical-inputs",
        version="1",
        inputs=(
            WorkflowInputDefinition(
                name="alpha",
                input_type=WorkflowInputType.STRING,
            ),
            WorkflowInputDefinition(
                name="beta",
                input_type=WorkflowInputType.INTEGER,
            ),
        ),
        steps=(
            WorkflowStepDefinition(
                step_id="step",
                executor_name="executor",
            ),
        ),
    )

    result = await _engine(executor).run(
        definition,
        inputs=(
            WorkflowInput(
                name="beta",
                value=2,
            ),
            WorkflowInput(
                name="alpha",
                value="one",
            ),
        ),
    )

    assert result.state is not None

    assert tuple(item.name for item in result.state.inputs) == (
        "alpha",
        "beta",
    )

    assert tuple(item.value for item in result.state.inputs) == (
        "one",
        2,
    )

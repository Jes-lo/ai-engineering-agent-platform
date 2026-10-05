"""Tests for the project-owned deterministic workflow engine."""

from dataclasses import dataclass, field

import pytest

from ai_engineering_agent_platform.domain.workflow import (
    WorkflowDefinition,
    WorkflowRunResult,
    WorkflowStepDefinition,
)
from ai_engineering_agent_platform.services.workflow import (
    WorkflowEngine,
    WorkflowExecutionLimitError,
    WorkflowExecutorRegistry,
    WorkflowExecutorRegistryError,
    WorkflowStepContext,
    WorkflowStepExecutionError,
)


@dataclass
class RecordingExecutor:
    """Synthetic trusted executor used for deterministic workflow tests."""

    name: str
    calls: list[
        tuple[
            str,
            tuple[str, ...],
        ]
    ] = field(default_factory=list)

    async def execute(
        self,
        *,
        step: WorkflowStepDefinition,
        context: WorkflowStepContext,
    ) -> object:
        dependencies = tuple(result.step_id for result in context.dependency_results)

        self.calls.append(
            (
                step.step_id,
                dependencies,
            )
        )

        return f"output:{step.step_id}"


@dataclass
class FailingExecutor:
    """Synthetic executor that records one call and then fails."""

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

        raise RuntimeError("synthetic executor failure")


def _definition() -> WorkflowDefinition:
    return WorkflowDefinition(
        workflow_id="workflow-test",
        version="1",
        steps=(
            WorkflowStepDefinition(
                step_id="start",
                executor_name="alpha",
            ),
            WorkflowStepDefinition(
                step_id="left",
                executor_name="beta",
                depends_on=("start",),
            ),
            WorkflowStepDefinition(
                step_id="right",
                executor_name="beta",
                depends_on=("start",),
            ),
            WorkflowStepDefinition(
                step_id="join",
                executor_name="alpha",
                depends_on=(
                    "left",
                    "right",
                ),
            ),
        ),
    )


@pytest.mark.anyio
async def test_engine_executes_static_dag_in_declaration_order() -> None:
    alpha = RecordingExecutor("alpha")

    beta = RecordingExecutor("beta")

    engine = WorkflowEngine(
        WorkflowExecutorRegistry(
            (
                alpha,
                beta,
            )
        ),
        max_steps=8,
        run_id_factory=lambda: "workflow-run:test",
    )

    result = await engine.run(_definition())
    assert isinstance(result, WorkflowRunResult)

    assert result.run_id == ("workflow-run:test")

    assert tuple(execution.step_id for execution in result.executions) == (
        "start",
        "left",
        "right",
        "join",
    )

    assert alpha.calls == [
        (
            "start",
            (),
        ),
        (
            "join",
            (
                "left",
                "right",
            ),
        ),
    ]

    assert beta.calls == [
        (
            "left",
            ("start",),
        ),
        (
            "right",
            ("start",),
        ),
    ]


@pytest.mark.anyio
async def test_step_receives_only_declared_dependencies() -> None:
    executor = RecordingExecutor("executor")

    definition = WorkflowDefinition(
        workflow_id="dependency-isolation",
        version="1",
        steps=(
            WorkflowStepDefinition(
                step_id="one",
                executor_name="executor",
            ),
            WorkflowStepDefinition(
                step_id="two",
                executor_name="executor",
            ),
            WorkflowStepDefinition(
                step_id="three",
                executor_name="executor",
                depends_on=("two",),
            ),
        ),
    )

    engine = WorkflowEngine(
        WorkflowExecutorRegistry((executor,)),
        run_id_factory=lambda: "workflow-run:dependency",
    )

    await engine.run(definition)

    assert executor.calls[-1] == (
        "three",
        ("two",),
    )


@pytest.mark.anyio
async def test_unknown_executor_fails_before_any_step_runs() -> None:
    known = RecordingExecutor("known")

    definition = WorkflowDefinition(
        workflow_id="preflight",
        version="1",
        steps=(
            WorkflowStepDefinition(
                step_id="first",
                executor_name="known",
            ),
            WorkflowStepDefinition(
                step_id="second",
                executor_name="missing",
                depends_on=("first",),
            ),
        ),
    )

    engine = WorkflowEngine(
        WorkflowExecutorRegistry((known,)),
        run_id_factory=lambda: "workflow-run:preflight",
    )

    with pytest.raises(
        WorkflowExecutorRegistryError,
        match="unregistered executor",
    ):
        await engine.run(definition)

    assert known.calls == []


@pytest.mark.anyio
async def test_step_budget_fails_before_any_step_runs() -> None:
    executor = RecordingExecutor("executor")

    definition = WorkflowDefinition(
        workflow_id="budget",
        version="1",
        steps=(
            WorkflowStepDefinition(
                step_id="one",
                executor_name="executor",
            ),
            WorkflowStepDefinition(
                step_id="two",
                executor_name="executor",
            ),
        ),
    )

    engine = WorkflowEngine(
        WorkflowExecutorRegistry((executor,)),
        max_steps=1,
        run_id_factory=lambda: "workflow-run:budget",
    )

    with pytest.raises(
        WorkflowExecutionLimitError,
        match="step budget",
    ):
        await engine.run(definition)

    assert executor.calls == []


def test_duplicate_executor_names_are_rejected() -> None:
    with pytest.raises(
        WorkflowExecutorRegistryError,
        match="must be unique",
    ):
        WorkflowExecutorRegistry(
            (
                RecordingExecutor("same"),
                RecordingExecutor("same"),
            )
        )


@pytest.mark.anyio
async def test_step_failure_is_not_retried() -> None:
    successful = RecordingExecutor("successful")

    failing = FailingExecutor("failing")

    definition = WorkflowDefinition(
        workflow_id="failure",
        version="1",
        steps=(
            WorkflowStepDefinition(
                step_id="first",
                executor_name="successful",
            ),
            WorkflowStepDefinition(
                step_id="second",
                executor_name="failing",
                depends_on=("first",),
            ),
        ),
    )

    engine = WorkflowEngine(
        WorkflowExecutorRegistry(
            (
                successful,
                failing,
            )
        ),
        run_id_factory=lambda: "workflow-run:failure",
    )

    with pytest.raises(
        WorkflowStepExecutionError,
    ) as captured:
        await engine.run(definition)

    assert captured.value.step_id == ("second")

    assert captured.value.executor_name == ("failing")

    assert successful.calls == [
        (
            "first",
            (),
        ),
    ]

    assert failing.calls == 1


@pytest.mark.anyio
async def test_invalid_run_id_fails_before_executor_side_effect() -> None:
    executor = RecordingExecutor("executor")

    definition = WorkflowDefinition(
        workflow_id="run-id",
        version="1",
        steps=(
            WorkflowStepDefinition(
                step_id="only",
                executor_name="executor",
            ),
        ),
    )

    engine = WorkflowEngine(
        WorkflowExecutorRegistry((executor,)),
        run_id_factory=lambda: "",
    )

    with pytest.raises(
        Exception,
        match="run_id_factory",
    ):
        await engine.run(definition)

    assert executor.calls == []

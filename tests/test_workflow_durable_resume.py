"""Tests for WorkflowEngine durable checkpointing and explicit resume."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from ai_engineering_agent_platform.domain.workflow import (
    WorkflowCondition,
    WorkflowConditionOperator,
    WorkflowDefinition,
    WorkflowEventType,
    WorkflowExecutionEvent,
    WorkflowInput,
    WorkflowInputDefinition,
    WorkflowInputType,
    WorkflowRunResult,
    WorkflowRunState,
    WorkflowRunStatus,
    WorkflowStepDefinition,
    WorkflowStepExecution,
    WorkflowStepState,
    WorkflowStepStatus,
)
from ai_engineering_agent_platform.services.workflow import (
    WorkflowEngine,
    WorkflowExecutorRegistry,
    WorkflowResumeError,
    WorkflowStepContext,
    WorkflowStepExecutionError,
)
from ai_engineering_agent_platform.services.workflow_persistence import (
    WorkflowCheckpoint,
    WorkflowPersistenceError,
)


@dataclass
class MemoryStateStore:
    """Small structural WorkflowStateStore used by engine tests."""

    checkpoint: WorkflowCheckpoint | None = None
    saved: list[WorkflowCheckpoint] = field(default_factory=list)
    loads: list[str] = field(default_factory=list)
    fail_on_save: int | None = None

    async def save(
        self,
        checkpoint: WorkflowCheckpoint,
    ) -> None:
        save_number = len(self.saved) + 1

        if self.fail_on_save == save_number:
            raise WorkflowPersistenceError("synthetic persistence failure")

        self.saved.append(checkpoint)
        self.checkpoint = checkpoint

    async def load(
        self,
        run_id: str,
    ) -> WorkflowCheckpoint | None:
        self.loads.append(run_id)

        if self.checkpoint is None or self.checkpoint.state.run_id != run_id:
            return None

        return self.checkpoint


@dataclass
class RecordingExecutor:
    """Executor recording exact contexts without external side effects."""

    name: str
    calls: list[WorkflowStepContext] = field(default_factory=list)

    async def execute(
        self,
        *,
        step: WorkflowStepDefinition,
        context: WorkflowStepContext,
    ) -> object:
        self.calls.append(context)
        return f"output:{step.step_id}"


@dataclass
class CheckpointAwareExecutor:
    """Executor proving checkpoints exist before each executor call."""

    name: str
    store: MemoryStateStore
    saved_counts: list[int] = field(default_factory=list)

    async def execute(
        self,
        *,
        step: WorkflowStepDefinition,
        context: WorkflowStepContext,
    ) -> object:
        del context

        self.saved_counts.append(len(self.store.saved))

        return f"output:{step.step_id}"


@dataclass
class FailingExecutor:
    """Executor recording one invocation before deterministic failure."""

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


def _event(
    sequence: int,
    event_type: WorkflowEventType,
    *,
    step_id: str | None = None,
    executor_name: str | None = None,
) -> WorkflowExecutionEvent:
    return WorkflowExecutionEvent(
        sequence=sequence,
        run_id="workflow-run:durable-engine",
        workflow_id="durable-engine",
        workflow_version="1",
        event_type=event_type,
        step_id=step_id,
        executor_name=executor_name,
    )


def _linear_definition() -> WorkflowDefinition:
    return WorkflowDefinition(
        workflow_id="durable-engine",
        version="1",
        steps=(
            WorkflowStepDefinition(
                step_id="first",
                executor_name="executor",
            ),
            WorkflowStepDefinition(
                step_id="second",
                executor_name="executor",
                depends_on=("first",),
            ),
        ),
    )


def _resume_definition() -> WorkflowDefinition:
    return WorkflowDefinition(
        workflow_id="durable-engine",
        version="1",
        inputs=(
            WorkflowInputDefinition(
                name="enabled",
                input_type=WorkflowInputType.BOOLEAN,
            ),
        ),
        steps=(
            WorkflowStepDefinition(
                step_id="first",
                executor_name="alpha",
            ),
            WorkflowStepDefinition(
                step_id="conditional",
                executor_name="beta",
                depends_on=("first",),
                condition=WorkflowCondition(
                    input_name="enabled",
                    operator=WorkflowConditionOperator.EQUALS,
                    expected_value=True,
                ),
            ),
            WorkflowStepDefinition(
                step_id="last",
                executor_name="alpha",
                depends_on=(
                    "first",
                    "conditional",
                ),
            ),
        ),
    )


def _resume_inputs() -> tuple[WorkflowInput, ...]:
    return (
        WorkflowInput(
            name="enabled",
            value=False,
        ),
    )


def _resume_checkpoint() -> WorkflowCheckpoint:
    first_execution = WorkflowStepExecution(
        step_id="first",
        executor_name="alpha",
        output="saved:first",
    )

    state = WorkflowRunState(
        run_id="workflow-run:durable-engine",
        workflow_id="durable-engine",
        workflow_version="1",
        status=WorkflowRunStatus.RUNNING,
        inputs=_resume_inputs(),
        steps=(
            WorkflowStepState(
                step_id="first",
                executor_name="alpha",
                status=WorkflowStepStatus.COMPLETED,
                execution=first_execution,
            ),
            WorkflowStepState(
                step_id="conditional",
                executor_name="beta",
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
                step_id="first",
                executor_name="alpha",
            ),
            _event(
                3,
                WorkflowEventType.STEP_SKIPPED,
                step_id="conditional",
                executor_name="beta",
            ),
        ),
    )


@pytest.mark.anyio
async def test_run_checkpoints_before_executor_side_effects() -> None:
    store = MemoryStateStore()

    executor = CheckpointAwareExecutor(
        name="executor",
        store=store,
    )

    engine = WorkflowEngine(
        WorkflowExecutorRegistry((executor,)),
        run_id_factory=lambda: "workflow-run:durable-engine",
        state_store=store,
    )

    result = await engine.run(_linear_definition())

    assert executor.saved_counts == [
        1,
        2,
    ]

    assert tuple(checkpoint.state.status for checkpoint in store.saved) == (
        WorkflowRunStatus.RUNNING,
        WorkflowRunStatus.RUNNING,
        WorkflowRunStatus.RUNNING,
        WorkflowRunStatus.COMPLETED,
    )

    assert tuple(len(checkpoint.events) for checkpoint in store.saved) == (
        1,
        2,
        3,
        4,
    )

    assert result.state is not None
    assert result.state.status is WorkflowRunStatus.COMPLETED


@pytest.mark.anyio
async def test_failed_run_is_checkpointed_before_error_escapes() -> None:
    store = MemoryStateStore()
    failing = FailingExecutor("failing")

    definition = WorkflowDefinition(
        workflow_id="durable-engine",
        version="1",
        steps=(
            WorkflowStepDefinition(
                step_id="only",
                executor_name="failing",
            ),
        ),
    )

    engine = WorkflowEngine(
        WorkflowExecutorRegistry((failing,)),
        run_id_factory=lambda: "workflow-run:durable-engine",
        state_store=store,
    )

    with pytest.raises(WorkflowStepExecutionError):
        await engine.run(definition)

    assert failing.calls == 1
    assert store.checkpoint is not None
    assert store.checkpoint.state.status is WorkflowRunStatus.FAILED
    assert store.checkpoint.state.steps[-1].status is WorkflowStepStatus.FAILED
    assert store.checkpoint.events[-1].event_type is WorkflowEventType.RUN_FAILED


@pytest.mark.anyio
async def test_initial_checkpoint_failure_blocks_executor() -> None:
    store = MemoryStateStore(fail_on_save=1)

    executor = RecordingExecutor("executor")

    engine = WorkflowEngine(
        WorkflowExecutorRegistry((executor,)),
        run_id_factory=lambda: "workflow-run:durable-engine",
        state_store=store,
    )

    with pytest.raises(
        WorkflowPersistenceError,
        match="synthetic persistence failure",
    ):
        await engine.run(_linear_definition())

    assert executor.calls == []


@pytest.mark.anyio
async def test_resume_preserves_completed_and_skipped_progress() -> None:
    store = MemoryStateStore(checkpoint=_resume_checkpoint())

    alpha = RecordingExecutor("alpha")
    beta = RecordingExecutor("beta")

    engine = WorkflowEngine(
        WorkflowExecutorRegistry(
            (
                alpha,
                beta,
            )
        ),
        state_store=store,
    )

    result = await engine.resume(
        _resume_definition(),
        run_id="workflow-run:durable-engine",
        inputs=_resume_inputs(),
    )
    assert isinstance(result, WorkflowRunResult)

    assert beta.calls == []

    assert len(alpha.calls) == 1
    assert alpha.calls[0].step_id == "last"

    assert tuple(item.step_id for item in alpha.calls[0].dependency_results) == (
        "first",
    )

    assert alpha.calls[0].skipped_dependency_ids == ("conditional",)

    assert tuple(execution.step_id for execution in result.executions) == (
        "first",
        "last",
    )

    assert result.state is not None

    assert tuple(step.status for step in result.state.steps) == (
        WorkflowStepStatus.COMPLETED,
        WorkflowStepStatus.SKIPPED,
        WorkflowStepStatus.COMPLETED,
    )

    assert (
        result.trace is not None
        and result.trace.events[-1].event_type is WorkflowEventType.RUN_COMPLETED
    )


@pytest.mark.anyio
async def test_resume_after_last_step_only_finalizes_run() -> None:
    checkpoint = _resume_checkpoint()

    last_execution = WorkflowStepExecution(
        step_id="last",
        executor_name="alpha",
        output="saved:last",
    )

    running_state = WorkflowRunState(
        run_id=checkpoint.state.run_id,
        workflow_id=checkpoint.state.workflow_id,
        workflow_version=checkpoint.state.workflow_version,
        status=WorkflowRunStatus.RUNNING,
        inputs=checkpoint.state.inputs,
        steps=(
            *checkpoint.state.steps,
            WorkflowStepState(
                step_id="last",
                executor_name="alpha",
                status=WorkflowStepStatus.COMPLETED,
                execution=last_execution,
            ),
        ),
    )

    store = MemoryStateStore(
        checkpoint=WorkflowCheckpoint(
            state=running_state,
            events=(
                *checkpoint.events,
                _event(
                    4,
                    WorkflowEventType.STEP_COMPLETED,
                    step_id="last",
                    executor_name="alpha",
                ),
            ),
        )
    )

    alpha = RecordingExecutor("alpha")
    beta = RecordingExecutor("beta")

    engine = WorkflowEngine(
        WorkflowExecutorRegistry(
            (
                alpha,
                beta,
            )
        ),
        state_store=store,
    )

    result = await engine.resume(
        _resume_definition(),
        run_id="workflow-run:durable-engine",
        inputs=_resume_inputs(),
    )

    assert alpha.calls == []
    assert beta.calls == []

    assert len(store.saved) == 1

    assert store.saved[0].state.status is WorkflowRunStatus.COMPLETED

    assert result.state is not None
    assert result.state.status is WorkflowRunStatus.COMPLETED


@pytest.mark.anyio
async def test_unknown_checkpoint_fails_before_executor() -> None:
    store = MemoryStateStore()

    executor = RecordingExecutor("executor")

    engine = WorkflowEngine(
        WorkflowExecutorRegistry((executor,)),
        state_store=store,
    )

    with pytest.raises(
        WorkflowResumeError,
        match="checkpoint not found",
    ):
        await engine.resume(
            _linear_definition(),
            run_id="workflow-run:missing",
        )

    assert executor.calls == []


@pytest.mark.anyio
async def test_terminal_checkpoint_cannot_be_resumed() -> None:
    store = MemoryStateStore()

    executor = RecordingExecutor("executor")

    engine = WorkflowEngine(
        WorkflowExecutorRegistry((executor,)),
        run_id_factory=lambda: "workflow-run:durable-engine",
        state_store=store,
    )

    await engine.run(_linear_definition())

    previous_calls = len(executor.calls)

    with pytest.raises(
        WorkflowResumeError,
        match="not resumable",
    ):
        await engine.resume(
            _linear_definition(),
            run_id="workflow-run:durable-engine",
        )

    assert len(executor.calls) == previous_calls


@pytest.mark.anyio
async def test_input_mismatch_fails_before_resumed_side_effect() -> None:
    store = MemoryStateStore(checkpoint=_resume_checkpoint())

    alpha = RecordingExecutor("alpha")
    beta = RecordingExecutor("beta")

    engine = WorkflowEngine(
        WorkflowExecutorRegistry(
            (
                alpha,
                beta,
            )
        ),
        state_store=store,
    )

    with pytest.raises(
        WorkflowResumeError,
        match="inputs differ",
    ):
        await engine.resume(
            _resume_definition(),
            run_id="workflow-run:durable-engine",
            inputs=(
                WorkflowInput(
                    name="enabled",
                    value=True,
                ),
            ),
        )

    assert alpha.calls == []
    assert beta.calls == []


@pytest.mark.anyio
async def test_event_state_mismatch_fails_before_side_effect() -> None:
    checkpoint = _resume_checkpoint()

    incompatible = WorkflowCheckpoint(
        state=checkpoint.state,
        events=(
            checkpoint.events[0],
            _event(
                2,
                WorkflowEventType.STEP_SKIPPED,
                step_id="first",
                executor_name="alpha",
            ),
            checkpoint.events[2],
        ),
    )

    store = MemoryStateStore(checkpoint=incompatible)

    alpha = RecordingExecutor("alpha")
    beta = RecordingExecutor("beta")

    engine = WorkflowEngine(
        WorkflowExecutorRegistry(
            (
                alpha,
                beta,
            )
        ),
        state_store=store,
    )

    with pytest.raises(
        WorkflowResumeError,
        match="event ledger",
    ):
        await engine.resume(
            _resume_definition(),
            run_id="workflow-run:durable-engine",
            inputs=_resume_inputs(),
        )

    assert alpha.calls == []
    assert beta.calls == []

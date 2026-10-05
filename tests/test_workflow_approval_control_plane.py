"""Workflow control-plane tests for durable authenticated approval pauses."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from ai_engineering_agent_platform.domain.workflow import (
    WorkflowApprovalPause,
    WorkflowDefinition,
    WorkflowEventType,
    WorkflowExecutionEvent,
    WorkflowRunResult,
    WorkflowRunState,
    WorkflowRunStatus,
    WorkflowStepDefinition,
    WorkflowStepExecution,
    WorkflowStepState,
    WorkflowStepStatus,
)
from ai_engineering_agent_platform.services.workflow import (
    WorkflowApprovalPending,
    WorkflowEngine,
    WorkflowExecutorRegistry,
    WorkflowResumeError,
    WorkflowStepContext,
)
from ai_engineering_agent_platform.services.workflow_persistence import (
    WorkflowCheckpoint,
)

RUN_ID = "workflow-run:approval-control"


@dataclass
class MemoryStateStore:
    """Structural durable store for approval control-plane tests."""

    checkpoint: WorkflowCheckpoint | None = None
    saved: list[WorkflowCheckpoint] = field(default_factory=list)

    async def save(
        self,
        checkpoint: WorkflowCheckpoint,
    ) -> None:
        self.saved.append(checkpoint)
        self.checkpoint = checkpoint

    async def load(
        self,
        run_id: str,
    ) -> WorkflowCheckpoint | None:
        if self.checkpoint is None or self.checkpoint.state.run_id != run_id:
            return None

        return self.checkpoint


@dataclass
class ApprovalAwareExecutor:
    """Synthetic executor exposing both legacy and durable HITL surfaces."""

    name: str
    start_outcome: object
    resume_outcomes: list[object] = field(default_factory=list)
    resume_error: Exception | None = None
    legacy_calls: list[WorkflowStepContext] = field(default_factory=list)
    approval_starts: list[WorkflowStepContext] = field(default_factory=list)
    approval_resumes: list[
        tuple[
            WorkflowStepContext,
            WorkflowApprovalPause,
        ]
    ] = field(default_factory=list)

    async def execute(
        self,
        *,
        step: WorkflowStepDefinition,
        context: WorkflowStepContext,
    ) -> object:
        self.legacy_calls.append(context)
        return "legacy-output:" + step.step_id

    async def execute_with_approval_pause(
        self,
        *,
        step: WorkflowStepDefinition,
        context: WorkflowStepContext,
    ) -> object | WorkflowApprovalPause:
        del step

        self.approval_starts.append(context)

        return self.start_outcome

    async def resume_approval(
        self,
        *,
        step: WorkflowStepDefinition,
        context: WorkflowStepContext,
        pause: WorkflowApprovalPause,
    ) -> object | WorkflowApprovalPause:
        del step

        self.approval_resumes.append(
            (
                context,
                pause,
            )
        )

        if self.resume_error is not None:
            raise self.resume_error

        if not self.resume_outcomes:
            raise AssertionError("unexpected approval resume")

        return self.resume_outcomes.pop(0)


@dataclass
class RecordingExecutor:
    """Normal workflow executor used after an approval-aware step."""

    name: str
    calls: list[WorkflowStepContext] = field(default_factory=list)

    async def execute(
        self,
        *,
        step: WorkflowStepDefinition,
        context: WorkflowStepContext,
    ) -> object:
        self.calls.append(context)
        return "output:" + step.step_id


def _pause(
    suffix: str,
) -> WorkflowApprovalPause:
    return WorkflowApprovalPause(
        continuation_identity=(f"call-{suffix}",),
        approval_ids=(f"approval-{suffix}",),
    )


def _definition(
    *,
    include_after: bool = True,
) -> WorkflowDefinition:
    steps = [
        WorkflowStepDefinition(
            step_id="agent-step",
            executor_name="agent",
        )
    ]

    if include_after:
        steps.append(
            WorkflowStepDefinition(
                step_id="after-step",
                executor_name="after",
                depends_on=("agent-step",),
            )
        )

    return WorkflowDefinition(
        workflow_id="approval-control",
        version="1",
        steps=tuple(steps),
    )


def _engine(
    *,
    agent: ApprovalAwareExecutor,
    store: MemoryStateStore | None,
    after: RecordingExecutor | None = None,
) -> WorkflowEngine:
    executors: tuple[
        object,
        ...,
    ]

    executors = (agent,) if after is None else (agent, after)

    return WorkflowEngine(
        WorkflowExecutorRegistry(executors),
        run_id_factory=(lambda: RUN_ID),
        state_store=store,
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
        run_id=RUN_ID,
        workflow_id="approval-control",
        workflow_version="1",
        event_type=event_type,
        step_id=step_id,
        executor_name=executor_name,
    )


@pytest.mark.anyio
async def test_durable_run_pauses_without_run_failure() -> None:
    pause = _pause("initial")

    store = MemoryStateStore()

    agent = ApprovalAwareExecutor(
        name="agent",
        start_outcome=pause,
    )

    after = RecordingExecutor(
        name="after",
    )

    result = await _engine(
        agent=agent,
        store=store,
        after=after,
    ).run(_definition())

    assert isinstance(
        result,
        WorkflowApprovalPending,
    )

    assert result.state.status is WorkflowRunStatus.AWAITING_APPROVAL

    assert result.approval_pause == pause

    assert result.paused_step_id == "agent-step"

    assert tuple(event.event_type for event in result.events) == (
        WorkflowEventType.RUN_STARTED,
        WorkflowEventType.STEP_AWAITING_APPROVAL,
    )

    assert store.checkpoint is not None
    assert store.checkpoint.state.status is WorkflowRunStatus.AWAITING_APPROVAL

    assert agent.legacy_calls == []
    assert len(agent.approval_starts) == 1
    assert after.calls == []


@pytest.mark.anyio
async def test_nondurable_run_preserves_legacy_executor_surface() -> None:
    agent = ApprovalAwareExecutor(
        name="agent",
        start_outcome=_pause("unused"),
    )

    result = await _engine(
        agent=agent,
        store=None,
    ).run(_definition(include_after=False))

    assert isinstance(
        result,
        WorkflowRunResult,
    )

    assert len(agent.legacy_calls) == 1

    assert agent.approval_starts == []


@pytest.mark.anyio
async def test_resume_approval_completes_step_and_continues_dag() -> None:
    store = MemoryStateStore()

    agent = ApprovalAwareExecutor(
        name="agent",
        start_outcome=_pause("first"),
        resume_outcomes=[
            "agent-output",
        ],
    )

    after = RecordingExecutor(
        name="after",
    )

    engine = _engine(
        agent=agent,
        store=store,
        after=after,
    )

    pending = await engine.run(_definition())

    assert isinstance(
        pending,
        WorkflowApprovalPending,
    )

    result = await engine.resume_approval(
        _definition(),
        run_id=RUN_ID,
    )

    assert isinstance(
        result,
        WorkflowRunResult,
    )

    assert (
        result.state is not None and result.state.status is WorkflowRunStatus.COMPLETED
    )

    assert tuple(execution.output for execution in result.executions) == (
        "agent-output",
        "output:after-step",
    )

    assert result.trace is not None

    assert tuple(event.event_type for event in result.trace.events) == (
        WorkflowEventType.RUN_STARTED,
        WorkflowEventType.STEP_AWAITING_APPROVAL,
        WorkflowEventType.STEP_COMPLETED,
        WorkflowEventType.STEP_COMPLETED,
        WorkflowEventType.RUN_COMPLETED,
    )

    assert len(agent.approval_resumes) == 1

    assert len(after.calls) == 1


@pytest.mark.anyio
async def test_resume_approval_can_repause_same_step() -> None:
    first = _pause("first")
    second = _pause("second")

    store = MemoryStateStore()

    agent = ApprovalAwareExecutor(
        name="agent",
        start_outcome=first,
        resume_outcomes=[
            second,
            "completed-output",
        ],
    )

    engine = _engine(
        agent=agent,
        store=store,
    )

    initial = await engine.run(_definition(include_after=False))

    assert isinstance(
        initial,
        WorkflowApprovalPending,
    )

    repaused = await engine.resume_approval(
        _definition(include_after=False),
        run_id=RUN_ID,
    )

    assert isinstance(
        repaused,
        WorkflowApprovalPending,
    )

    assert repaused.approval_pause == second

    assert len(repaused.state.steps) == 1

    assert tuple(event.event_type for event in repaused.events) == (
        WorkflowEventType.RUN_STARTED,
        WorkflowEventType.STEP_AWAITING_APPROVAL,
        WorkflowEventType.STEP_AWAITING_APPROVAL,
    )

    completed = await engine.resume_approval(
        _definition(include_after=False),
        run_id=RUN_ID,
    )

    assert isinstance(
        completed,
        WorkflowRunResult,
    )

    assert completed.trace is not None

    assert tuple(event.event_type for event in completed.trace.events) == (
        WorkflowEventType.RUN_STARTED,
        WorkflowEventType.STEP_AWAITING_APPROVAL,
        WorkflowEventType.STEP_AWAITING_APPROVAL,
        WorkflowEventType.STEP_COMPLETED,
        WorkflowEventType.RUN_COMPLETED,
    )


@pytest.mark.anyio
async def test_generic_resume_rejects_awaiting_approval_checkpoint() -> None:
    store = MemoryStateStore()

    agent = ApprovalAwareExecutor(
        name="agent",
        start_outcome=_pause("generic-reject"),
    )

    engine = _engine(
        agent=agent,
        store=store,
    )

    pending = await engine.run(_definition(include_after=False))

    assert isinstance(
        pending,
        WorkflowApprovalPending,
    )

    with pytest.raises(
        WorkflowResumeError,
        match="not resumable",
    ):
        await engine.resume(
            _definition(include_after=False),
            run_id=RUN_ID,
        )


@pytest.mark.anyio
async def test_running_resume_accepts_historical_pause_ledger() -> None:
    execution = WorkflowStepExecution(
        step_id="agent-step",
        executor_name="agent",
        output="agent-output",
    )

    checkpoint = WorkflowCheckpoint(
        state=WorkflowRunState(
            run_id=RUN_ID,
            workflow_id="approval-control",
            workflow_version="1",
            status=WorkflowRunStatus.RUNNING,
            inputs=(),
            steps=(
                WorkflowStepState(
                    step_id="agent-step",
                    executor_name="agent",
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
                WorkflowEventType.STEP_AWAITING_APPROVAL,
                step_id="agent-step",
                executor_name="agent",
            ),
            _event(
                3,
                WorkflowEventType.STEP_COMPLETED,
                step_id="agent-step",
                executor_name="agent",
            ),
        ),
    )

    store = MemoryStateStore(
        checkpoint=checkpoint,
    )

    agent = ApprovalAwareExecutor(
        name="agent",
        start_outcome="unused",
    )

    after = RecordingExecutor(
        name="after",
    )

    result = await _engine(
        agent=agent,
        store=store,
        after=after,
    ).resume(
        _definition(),
        run_id=RUN_ID,
    )

    assert isinstance(
        result,
        WorkflowRunResult,
    )

    assert tuple(execution.output for execution in result.executions) == (
        "agent-output",
        "output:after-step",
    )

    assert len(after.calls) == 1


@pytest.mark.anyio
async def test_failed_approval_resume_leaves_checkpoint_paused() -> None:
    store = MemoryStateStore()

    agent = ApprovalAwareExecutor(
        name="agent",
        start_outcome=_pause("error"),
    )

    engine = _engine(
        agent=agent,
        store=store,
    )

    pending = await engine.run(_definition(include_after=False))

    assert isinstance(
        pending,
        WorkflowApprovalPending,
    )

    before = store.checkpoint
    saved_count = len(store.saved)

    agent.resume_error = RuntimeError("synthetic approval resume failure")

    with pytest.raises(
        WorkflowResumeError,
        match="approval continuation could not resume",
    ):
        await engine.resume_approval(
            _definition(include_after=False),
            run_id=RUN_ID,
        )

    assert store.checkpoint == before
    assert len(store.saved) == saved_count

    assert store.checkpoint is not None
    assert store.checkpoint.state.status is WorkflowRunStatus.AWAITING_APPROVAL

    assert (
        store.checkpoint.events[-1].event_type
        is WorkflowEventType.STEP_AWAITING_APPROVAL
    )

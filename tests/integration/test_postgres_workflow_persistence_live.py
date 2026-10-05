"""Live PostgreSQL tests for durable workflow checkpoints."""

import asyncio
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from ai_engineering_agent_platform.adapters.postgres.workflow_state_store import (
    PostgreSQLWorkflowStateStore,
)
from ai_engineering_agent_platform.config import (
    Settings,
)
from ai_engineering_agent_platform.domain.workflow import (
    WorkflowDefinition,
    WorkflowEventType,
    WorkflowExecutionEvent,
    WorkflowRunResult,
    WorkflowRunState,
    WorkflowRunStatus,
    WorkflowStepDefinition,
    WorkflowStepStatus,
)
from ai_engineering_agent_platform.runtime.postgres import (
    postgres_pool_runtime,
)
from ai_engineering_agent_platform.services.workflow import (
    WorkflowEngine,
    WorkflowExecutorRegistry,
    WorkflowStepContext,
)
from ai_engineering_agent_platform.services.workflow_persistence import (
    WorkflowCheckpoint,
    WorkflowPersistenceConflictError,
    WorkflowPersistenceIntegrityError,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("AI_PLATFORM_RUN_POSTGRES_INTEGRATION") != "1",
    reason=("live PostgreSQL integration is opt-in"),
)


RUNTIME_KEYS = {
    "AI_PLATFORM_POSTGRES_HOST",
    "AI_PLATFORM_POSTGRES_PORT",
    "AI_PLATFORM_POSTGRES_DATABASE",
    "AI_PLATFORM_POSTGRES_USER",
    "AI_PLATFORM_POSTGRES_PASSWORD",
    "AI_PLATFORM_POSTGRES_SSLMODE",
    "AI_PLATFORM_POSTGRES_CONNECT_TIMEOUT_SECONDS",
    "AI_PLATFORM_POSTGRES_POOL_MIN_SIZE",
    "AI_PLATFORM_POSTGRES_POOL_MAX_SIZE",
    "AI_PLATFORM_POSTGRES_POOL_TIMEOUT_SECONDS",
}


class _SyntheticProcessInterruption(BaseException):
    """Model process loss after a committed durable checkpoint."""


@dataclass
class _RecordingWorkflowExecutor:
    """Record deterministic live workflow executor invocations."""

    name: str
    calls: list[str] = field(default_factory=list)

    async def execute(
        self,
        *,
        step: WorkflowStepDefinition,
        context: WorkflowStepContext,
    ) -> object:
        del context

        self.calls.append(step.step_id)

        return f"live-output:{step.step_id}"


@dataclass
class _InterruptAfterSaveStore:
    """Interrupt only after delegating one selected durable save."""

    delegate: PostgreSQLWorkflowStateStore
    interrupt_after_save: int
    save_count: int = 0

    async def save(
        self,
        checkpoint: WorkflowCheckpoint,
    ) -> None:
        await self.delegate.save(checkpoint)

        self.save_count += 1

        if self.save_count == self.interrupt_after_save:
            raise _SyntheticProcessInterruption

    async def load(
        self,
        run_id: str,
    ) -> WorkflowCheckpoint | None:
        return await self.delegate.load(run_id)


def _engine_definition() -> WorkflowDefinition:
    return WorkflowDefinition(
        workflow_id="workflow-live-engine-resume",
        version="1",
        steps=(
            WorkflowStepDefinition(
                step_id="first",
                executor_name="first",
            ),
            WorkflowStepDefinition(
                step_id="second",
                executor_name="second",
                depends_on=("first",),
            ),
        ),
    )


def _load_env(
    path: Path,
) -> dict[str, str]:
    values: dict[
        str,
        str,
    ] = {}

    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw or raw.startswith("#"):
            continue

        if "=" not in raw:
            raise RuntimeError(f"Malformed environment file: {path}")

        key, value = raw.split(
            "=",
            1,
        )

        if key in values:
            raise RuntimeError(f"Duplicate environment key: {key}")

        values[key] = value

    if set(values) != RUNTIME_KEYS:
        raise RuntimeError(f"Unexpected environment keys: {path}")

    return values


def _event(
    sequence: int,
    event_type: WorkflowEventType,
) -> WorkflowExecutionEvent:
    return WorkflowExecutionEvent(
        sequence=sequence,
        run_id="workflow-run:live-postgres",
        workflow_id="workflow-live-postgres",
        workflow_version="1",
        event_type=event_type,
    )


def _running() -> WorkflowCheckpoint:
    return WorkflowCheckpoint(
        state=WorkflowRunState(
            run_id="workflow-run:live-postgres",
            workflow_id="workflow-live-postgres",
            workflow_version="1",
            status=WorkflowRunStatus.RUNNING,
            inputs=(),
            steps=(),
        ),
        events=(
            _event(
                1,
                WorkflowEventType.RUN_STARTED,
            ),
        ),
    )


def _completed() -> WorkflowCheckpoint:
    running = _running()

    return WorkflowCheckpoint(
        state=replace(
            running.state,
            status=WorkflowRunStatus.COMPLETED,
        ),
        events=(
            *running.events,
            _event(
                2,
                WorkflowEventType.RUN_COMPLETED,
            ),
        ),
    )


async def _exercise() -> None:
    runtime_path = Path(os.environ["AI_PLATFORM_POSTGRES_INTEGRATION_RUNTIME_ENV"])

    runtime = _load_env(runtime_path)

    os.environ.update(runtime)

    settings = Settings()

    if settings.postgres_user != "ai_platform_runtime":
        raise RuntimeError("Integration test must use runtime role")

    async with postgres_pool_runtime(settings) as pool:
        store = PostgreSQLWorkflowStateStore(pool)

        running = _running()

        assert await store.load(running.state.run_id) is None

        await store.save(running)

        assert await store.load(running.state.run_id) == running

        await store.save(running)

        assert await store.load(running.state.run_id) == running

        completed = _completed()

        await store.save(completed)

        assert await store.load(completed.state.run_id) == completed

        with pytest.raises(
            WorkflowPersistenceConflictError,
        ):
            await store.save(running)

        async with pool.connection() as connection:
            await connection.execute(
                """
                UPDATE ai_platform.workflow_checkpoints
                SET checkpoint_payload = '{}'
                WHERE run_id = %s
                """,
                (completed.state.run_id,),
            )

        with pytest.raises(
            WorkflowPersistenceIntegrityError,
            match="checksum mismatch",
        ):
            await store.load(completed.state.run_id)


def test_live_postgres_workflow_checkpoint_store() -> None:
    """Exercise real durable workflow checkpoint persistence."""
    asyncio.run(_exercise())


async def _exercise_engine_restart_and_resume() -> None:
    runtime_path = Path(os.environ["AI_PLATFORM_POSTGRES_INTEGRATION_RUNTIME_ENV"])

    runtime = _load_env(runtime_path)

    os.environ.update(runtime)

    settings = Settings()

    if settings.postgres_user != "ai_platform_runtime":
        raise RuntimeError("Integration test must use runtime role")

    async with postgres_pool_runtime(settings) as pool:
        store = PostgreSQLWorkflowStateStore(pool)

        run_id = "workflow-run:live-engine-resume"

        definition = _engine_definition()

        first_before_restart = _RecordingWorkflowExecutor("first")

        second_before_restart = _RecordingWorkflowExecutor("second")

        interrupting_store = _InterruptAfterSaveStore(
            delegate=store,
            interrupt_after_save=2,
        )

        first_engine = WorkflowEngine(
            WorkflowExecutorRegistry(
                (
                    first_before_restart,
                    second_before_restart,
                )
            ),
            run_id_factory=lambda: run_id,
            state_store=interrupting_store,
        )

        with pytest.raises(_SyntheticProcessInterruption):
            await first_engine.run(definition)

        assert first_before_restart.calls == ["first"]

        assert second_before_restart.calls == []

        persisted = await store.load(run_id)

        assert persisted is not None

        assert persisted.state.status is WorkflowRunStatus.RUNNING

        assert len(persisted.state.steps) == 1

        assert persisted.state.steps[0].step_id == "first"

        assert persisted.state.steps[0].status is WorkflowStepStatus.COMPLETED

        assert tuple(event.event_type for event in persisted.events) == (
            WorkflowEventType.RUN_STARTED,
            WorkflowEventType.STEP_COMPLETED,
        )

        first_after_restart = _RecordingWorkflowExecutor("first")

        second_after_restart = _RecordingWorkflowExecutor("second")

        resumed_engine = WorkflowEngine(
            WorkflowExecutorRegistry(
                (
                    first_after_restart,
                    second_after_restart,
                )
            ),
            state_store=store,
        )

        result = await resumed_engine.resume(
            definition,
            run_id=run_id,
        )
        assert isinstance(result, WorkflowRunResult)

        assert first_after_restart.calls == []

        assert second_after_restart.calls == ["second"]

        assert tuple(execution.step_id for execution in result.executions) == (
            "first",
            "second",
        )

        assert result.state is not None

        assert result.state.status is WorkflowRunStatus.COMPLETED

        assert result.trace is not None

        assert result.trace.events[-1].event_type is WorkflowEventType.RUN_COMPLETED

        terminal = await store.load(run_id)

        assert terminal is not None

        assert terminal.state == result.state

        assert terminal.events == result.trace.events


def test_live_workflow_engine_restart_and_resume() -> None:
    """Resume a persisted RUNNING workflow with a new engine instance."""
    asyncio.run(_exercise_engine_restart_and_resume())

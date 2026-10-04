"""Deterministic project-owned workflow execution engine."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from ai_engineering_agent_platform.domain.workflow import (
    MAX_WORKFLOW_DEFINITION_STEPS,
    WorkflowCondition,
    WorkflowConditionOperator,
    WorkflowDefinition,
    WorkflowEventType,
    WorkflowExecutionEvent,
    WorkflowExecutionTrace,
    WorkflowInput,
    WorkflowRunResult,
    WorkflowRunState,
    WorkflowRunStatus,
    WorkflowStepDefinition,
    WorkflowStepExecution,
    WorkflowStepState,
    WorkflowStepStatus,
)
from ai_engineering_agent_platform.services.workflow_persistence import (
    WorkflowCheckpoint,
    WorkflowStateStore,
)

MAX_WORKFLOW_EXECUTION_STEPS = MAX_WORKFLOW_DEFINITION_STEPS


class WorkflowControlError(Exception):
    """Base error for controlled workflow execution."""


class WorkflowExecutorRegistryError(WorkflowControlError):
    """Raised when workflow executor registration/resolution is invalid."""


class WorkflowExecutionLimitError(WorkflowControlError):
    """Raised before execution when a workflow exceeds the run budget."""


class WorkflowInputError(WorkflowControlError):
    """Raised before execution when typed workflow inputs are invalid."""


class WorkflowResumeError(WorkflowControlError):
    """Raised when durable workflow continuation cannot proceed safely."""


class WorkflowStepExecutionError(WorkflowControlError):
    """Raised when one workflow executor fails.

    Earlier successfully completed steps are not rolled back. The engine does
    not retry automatically. ``state`` captures the process-local terminal
    snapshot of this failed attempt.
    """

    def __init__(
        self,
        *,
        step_id: str,
        executor_name: str,
        state: WorkflowRunState | None = None,
        trace: WorkflowExecutionTrace | None = None,
    ) -> None:
        """Capture the failing step without exposing arbitrary cause text."""
        self.step_id = step_id
        self.executor_name = executor_name
        self.state = state
        self.trace = trace

        super().__init__("workflow step execution failed")


@dataclass(
    frozen=True,
    slots=True,
)
class WorkflowStepContext:
    """Declared inputs/dependency state visible to exactly one step."""

    run_id: str
    workflow_id: str
    workflow_version: str
    step_id: str
    dependency_results: tuple[
        WorkflowStepExecution,
        ...,
    ] = ()
    inputs: tuple[
        WorkflowInput,
        ...,
    ] = ()
    skipped_dependency_ids: tuple[
        str,
        ...,
    ] = ()

    def __post_init__(
        self,
    ) -> None:
        """Validate unique context data and dependency-state separation."""
        if not isinstance(
            self.dependency_results,
            tuple,
        ):
            raise ValueError("dependency_results must be a tuple")

        if not isinstance(
            self.inputs,
            tuple,
        ):
            raise ValueError("inputs must be a tuple")

        if not isinstance(
            self.skipped_dependency_ids,
            tuple,
        ):
            raise ValueError("skipped_dependency_ids must be a tuple")

        result_ids = tuple(result.step_id for result in self.dependency_results)

        if len(result_ids) != len(set(result_ids)):
            raise ValueError("dependency result IDs must be unique")

        skipped_ids = self.skipped_dependency_ids

        if len(skipped_ids) != len(set(skipped_ids)):
            raise ValueError("skipped dependency IDs must be unique")

        if set(result_ids).intersection(skipped_ids):
            raise ValueError("dependency cannot be both completed and skipped")

        input_names = tuple(item.name for item in self.inputs)

        if len(input_names) != len(set(input_names)):
            raise ValueError("workflow context input names must be unique")

    def result_for(
        self,
        step_id: str,
    ) -> WorkflowStepExecution:
        """Return one successfully completed declared dependency."""
        matches = tuple(
            result for result in self.dependency_results if result.step_id == step_id
        )

        if len(matches) != 1:
            raise KeyError(step_id)

        return matches[0]

    def input_value(
        self,
        name: str,
    ) -> object:
        """Return one validated workflow input value."""
        matches = tuple(item for item in self.inputs if item.name == name)

        if len(matches) != 1:
            raise KeyError(name)

        return matches[0].value

    def was_dependency_skipped(
        self,
        step_id: str,
    ) -> bool:
        """Return whether a declared dependency was conditionally skipped."""
        return step_id in self.skipped_dependency_ids


class WorkflowStepExecutor(Protocol):
    """Trusted application executor used by the workflow control plane."""

    @property
    def name(
        self,
    ) -> str:
        """Return the stable registry name."""

    async def execute(
        self,
        *,
        step: WorkflowStepDefinition,
        context: WorkflowStepContext,
    ) -> object:
        """Execute one step through its application-owned boundary."""


class WorkflowExecutorRegistry:
    """Immutable name-to-executor registry."""

    def __init__(
        self,
        executors: tuple[
            WorkflowStepExecutor,
            ...,
        ],
    ) -> None:
        """Validate and index trusted workflow executors."""
        if not isinstance(
            executors,
            tuple,
        ):
            raise WorkflowExecutorRegistryError("executors must be a tuple")

        if not executors:
            raise WorkflowExecutorRegistryError(
                "at least one workflow executor is required"
            )

        by_name: dict[
            str,
            WorkflowStepExecutor,
        ] = {}

        for executor in executors:
            name = executor.name

            if (
                not isinstance(
                    name,
                    str,
                )
                or not name.strip()
            ):
                raise WorkflowExecutorRegistryError(
                    "workflow executor names must be non-empty strings"
                )

            if name in by_name:
                raise WorkflowExecutorRegistryError(
                    "workflow executor names must be unique"
                )

            by_name[name] = executor

        self._executors = by_name

    def executor(
        self,
        name: str,
    ) -> WorkflowStepExecutor:
        """Resolve one exact trusted executor."""
        try:
            return self._executors[name]
        except KeyError as exc:
            raise WorkflowExecutorRegistryError(
                "workflow references an unregistered executor"
            ) from exc

    def names(
        self,
    ) -> tuple[str, ...]:
        """Return deterministic executor names."""
        return tuple(self._executors)


def _default_run_id() -> str:
    """Create one process-local workflow run identity."""
    return "workflow-run:" + uuid4().hex


def _validate_inputs(
    definition: WorkflowDefinition,
    inputs: tuple[
        WorkflowInput,
        ...,
    ],
) -> tuple[
    WorkflowInput,
    ...,
]:
    """Validate exact required input set and canonicalize definition order."""
    if not isinstance(
        inputs,
        tuple,
    ):
        raise WorkflowInputError("workflow inputs must be a tuple")

    if any(
        not isinstance(
            item,
            WorkflowInput,
        )
        for item in inputs
    ):
        raise WorkflowInputError("workflow inputs must contain WorkflowInput values")

    provided_names = tuple(item.name for item in inputs)

    if len(provided_names) != len(set(provided_names)):
        raise WorkflowInputError("workflow input names must be unique")

    provided = {item.name: item for item in inputs}

    expected_names = tuple(
        input_definition.name for input_definition in definition.inputs
    )

    if set(provided) != set(expected_names):
        raise WorkflowInputError(
            "workflow inputs must exactly match declared input schema"
        )

    canonical: list[WorkflowInput] = []

    for input_definition in definition.inputs:
        item = provided[input_definition.name]

        if not input_definition.accepts(item.value):
            raise WorkflowInputError(
                "workflow input value does not match declared type"
            )

        canonical.append(item)

    return tuple(canonical)


def _condition_matches(
    condition: WorkflowCondition,
    inputs: tuple[
        WorkflowInput,
        ...,
    ],
) -> bool:
    """Evaluate one validated condition without expression execution."""
    input_by_name = {item.name: item.value for item in inputs}

    actual = input_by_name[condition.input_name]

    if condition.operator is WorkflowConditionOperator.EQUALS:
        return actual == condition.expected_value

    if condition.operator is WorkflowConditionOperator.NOT_EQUALS:
        return actual != condition.expected_value

    raise WorkflowControlError("unsupported workflow condition operator")


class WorkflowEngine:
    """Execute one validated static DAG sequentially and without retries.

    The engine owns orchestration order, typed input validation, deterministic
    input conditions, and process-local run state. It does not call model,
    tool, retrieval, MCP, database, filesystem, network, or provider
    implementations directly. Capabilities remain behind trusted application
    executors.

    Before the first executor call, the complete workflow is preflighted for
    executor availability, exact input schema, and the hard run-step budget.
    Runtime step failures are fail-stop. Earlier side effects are not rolled
    back and the engine never automatically retries a failed step.
    """

    def __init__(
        self,
        registry: WorkflowExecutorRegistry,
        *,
        max_steps: int = 16,
        run_id_factory: Callable[
            [],
            str,
        ]
        | None = None,
        state_store: WorkflowStateStore | None = None,
    ) -> None:
        """Create one bounded workflow engine."""
        if not isinstance(
            registry,
            WorkflowExecutorRegistry,
        ):
            raise ValueError("registry must be a WorkflowExecutorRegistry")

        if (
            not isinstance(
                max_steps,
                int,
            )
            or isinstance(
                max_steps,
                bool,
            )
            or max_steps <= 0
            or max_steps > MAX_WORKFLOW_EXECUTION_STEPS
        ):
            raise ValueError(
                f"max_steps must be between 1 and {MAX_WORKFLOW_EXECUTION_STEPS}"
            )

        self._registry = registry
        self._max_steps = max_steps
        self._run_id_factory = (
            _default_run_id if run_id_factory is None else run_id_factory
        )
        self._state_store = state_store

    @staticmethod
    def _running_state(
        *,
        run_id: str,
        definition: WorkflowDefinition,
        inputs: tuple[
            WorkflowInput,
            ...,
        ],
        steps: tuple[
            WorkflowStepState,
            ...,
        ],
    ) -> WorkflowRunState:
        """Build one validated non-terminal workflow snapshot."""
        return WorkflowRunState(
            run_id=run_id,
            workflow_id=definition.workflow_id,
            workflow_version=definition.version,
            status=WorkflowRunStatus.RUNNING,
            inputs=inputs,
            steps=steps,
        )

    async def _save_checkpoint(
        self,
        *,
        state: WorkflowRunState,
        events: list[WorkflowExecutionEvent,],
    ) -> None:
        """Persist progress when a durable state store is configured."""
        if self._state_store is None:
            return

        await self._state_store.save(
            WorkflowCheckpoint(
                state=state,
                events=tuple(events),
            )
        )

    @staticmethod
    def _validate_resume_checkpoint(
        *,
        definition: WorkflowDefinition,
        validated_inputs: tuple[
            WorkflowInput,
            ...,
        ],
        checkpoint: WorkflowCheckpoint,
        run_id: str,
    ) -> None:
        """Fail closed unless saved progress exactly matches this definition."""
        state = checkpoint.state

        if (
            state.run_id != run_id
            or state.workflow_id != definition.workflow_id
            or state.workflow_version != definition.version
        ):
            raise WorkflowResumeError("workflow checkpoint identity mismatch")

        if state.status is not WorkflowRunStatus.RUNNING:
            raise WorkflowResumeError("workflow checkpoint is not resumable")

        if state.inputs != validated_inputs:
            raise WorkflowResumeError(
                "workflow checkpoint inputs differ from supplied inputs"
            )

        if len(state.steps) > len(definition.steps):
            raise WorkflowResumeError("workflow checkpoint progress exceeds definition")

        if len(checkpoint.events) != len(state.steps) + 1:
            raise WorkflowResumeError(
                "workflow checkpoint event ledger is incompatible"
            )

        for index, saved_step in enumerate(state.steps):
            definition_step = definition.steps[index]

            if (
                saved_step.step_id != definition_step.step_id
                or saved_step.executor_name != definition_step.executor_name
            ):
                raise WorkflowResumeError(
                    "workflow checkpoint progress is incompatible"
                )

            condition = definition_step.condition

            should_skip = condition is not None and not _condition_matches(
                condition,
                validated_inputs,
            )

            expected_status = (
                WorkflowStepStatus.SKIPPED
                if should_skip
                else WorkflowStepStatus.COMPLETED
            )

            if saved_step.status is not expected_status:
                raise WorkflowResumeError(
                    "workflow checkpoint step status is incompatible"
                )

            expected_event_type = (
                WorkflowEventType.STEP_SKIPPED
                if expected_status is WorkflowStepStatus.SKIPPED
                else WorkflowEventType.STEP_COMPLETED
            )

            event = checkpoint.events[index + 1]

            if (
                event.event_type is not expected_event_type
                or event.step_id != saved_step.step_id
                or event.executor_name != saved_step.executor_name
            ):
                raise WorkflowResumeError(
                    "workflow checkpoint event ledger is incompatible"
                )

    async def run(
        self,
        definition: WorkflowDefinition,
        *,
        inputs: tuple[
            WorkflowInput,
            ...,
        ] = (),
    ) -> WorkflowRunResult:
        """Execute one complete static workflow attempt."""
        if not isinstance(
            definition,
            WorkflowDefinition,
        ):
            raise WorkflowControlError("definition must be a WorkflowDefinition")

        if len(definition.steps) > self._max_steps:
            raise WorkflowExecutionLimitError(
                "workflow exceeds configured execution step budget"
            )

        resolved = tuple(
            (
                step,
                self._registry.executor(step.executor_name),
            )
            for step in definition.steps
        )

        validated_inputs = _validate_inputs(
            definition,
            inputs,
        )

        run_id = self._run_id_factory()

        if (
            not isinstance(
                run_id,
                str,
            )
            or not run_id.strip()
        ):
            raise WorkflowControlError("run_id_factory must return a non-empty string")

        events: list[WorkflowExecutionEvent] = []

        def record_event(
            event_type: WorkflowEventType,
            *,
            step: WorkflowStepDefinition | None = None,
        ) -> None:
            events.append(
                WorkflowExecutionEvent(
                    sequence=len(events) + 1,
                    run_id=run_id,
                    workflow_id=definition.workflow_id,
                    workflow_version=definition.version,
                    event_type=event_type,
                    step_id=(None if step is None else step.step_id),
                    executor_name=(None if step is None else step.executor_name),
                )
            )

        record_event(WorkflowEventType.RUN_STARTED)

        executions: list[WorkflowStepExecution] = []

        step_states: list[WorkflowStepState] = []

        by_step_id: dict[
            str,
            WorkflowStepExecution,
        ] = {}

        skipped_step_ids: set[str] = set()

        await self._save_checkpoint(
            state=self._running_state(
                run_id=run_id,
                definition=definition,
                inputs=validated_inputs,
                steps=(),
            ),
            events=events,
        )

        for (
            step,
            executor,
        ) in resolved:
            condition = step.condition

            if condition is not None and not _condition_matches(
                condition,
                validated_inputs,
            ):
                step_states.append(
                    WorkflowStepState(
                        step_id=step.step_id,
                        executor_name=step.executor_name,
                        status=WorkflowStepStatus.SKIPPED,
                    )
                )

                skipped_step_ids.add(step.step_id)

                record_event(
                    WorkflowEventType.STEP_SKIPPED,
                    step=step,
                )

                await self._save_checkpoint(
                    state=self._running_state(
                        run_id=run_id,
                        definition=definition,
                        inputs=validated_inputs,
                        steps=tuple(step_states),
                    ),
                    events=events,
                )

                continue

            dependency_results = tuple(
                by_step_id[dependency]
                for dependency in step.depends_on
                if dependency in by_step_id
            )

            skipped_dependency_ids = tuple(
                dependency
                for dependency in step.depends_on
                if dependency in skipped_step_ids
            )

            context = WorkflowStepContext(
                run_id=run_id,
                workflow_id=definition.workflow_id,
                workflow_version=definition.version,
                step_id=step.step_id,
                dependency_results=dependency_results,
                inputs=validated_inputs,
                skipped_dependency_ids=skipped_dependency_ids,
            )

            try:
                output = await executor.execute(
                    step=step,
                    context=context,
                )
            except Exception as exc:
                record_event(
                    WorkflowEventType.STEP_FAILED,
                    step=step,
                )

                record_event(WorkflowEventType.RUN_FAILED)

                failed_trace = WorkflowExecutionTrace(
                    run_id=run_id,
                    workflow_id=definition.workflow_id,
                    workflow_version=definition.version,
                    events=tuple(events),
                )

                failed_state = WorkflowRunState(
                    run_id=run_id,
                    workflow_id=definition.workflow_id,
                    workflow_version=definition.version,
                    status=WorkflowRunStatus.FAILED,
                    inputs=validated_inputs,
                    steps=(
                        *step_states,
                        WorkflowStepState(
                            step_id=step.step_id,
                            executor_name=step.executor_name,
                            status=WorkflowStepStatus.FAILED,
                        ),
                    ),
                )

                await self._save_checkpoint(
                    state=failed_state,
                    events=events,
                )

                raise WorkflowStepExecutionError(
                    step_id=step.step_id,
                    executor_name=step.executor_name,
                    state=failed_state,
                    trace=failed_trace,
                ) from exc

            execution = WorkflowStepExecution(
                step_id=step.step_id,
                executor_name=step.executor_name,
                output=output,
            )

            executions.append(execution)

            by_step_id[step.step_id] = execution

            record_event(
                WorkflowEventType.STEP_COMPLETED,
                step=step,
            )

            step_states.append(
                WorkflowStepState(
                    step_id=step.step_id,
                    executor_name=step.executor_name,
                    status=WorkflowStepStatus.COMPLETED,
                    execution=execution,
                )
            )

            await self._save_checkpoint(
                state=self._running_state(
                    run_id=run_id,
                    definition=definition,
                    inputs=validated_inputs,
                    steps=tuple(step_states),
                ),
                events=events,
            )

        record_event(WorkflowEventType.RUN_COMPLETED)

        completed_trace = WorkflowExecutionTrace(
            run_id=run_id,
            workflow_id=definition.workflow_id,
            workflow_version=definition.version,
            events=tuple(events),
        )

        completed_state = WorkflowRunState(
            run_id=run_id,
            workflow_id=definition.workflow_id,
            workflow_version=definition.version,
            status=WorkflowRunStatus.COMPLETED,
            inputs=validated_inputs,
            steps=tuple(step_states),
        )

        await self._save_checkpoint(
            state=completed_state,
            events=events,
        )

        return WorkflowRunResult(
            run_id=run_id,
            workflow_id=definition.workflow_id,
            workflow_version=definition.version,
            executions=tuple(executions),
            state=completed_state,
            trace=completed_trace,
        )

    async def resume(
        self,
        definition: WorkflowDefinition,
        *,
        run_id: str,
        inputs: tuple[
            WorkflowInput,
            ...,
        ] = (),
    ) -> WorkflowRunResult:
        """Explicitly continue one validated non-terminal durable checkpoint."""
        if self._state_store is None:
            raise WorkflowResumeError("workflow resume requires durable state store")

        if not isinstance(
            definition,
            WorkflowDefinition,
        ):
            raise WorkflowControlError("definition must be a WorkflowDefinition")

        if len(definition.steps) > self._max_steps:
            raise WorkflowExecutionLimitError(
                "workflow exceeds configured execution step budget"
            )

        resolved = tuple(
            (
                step,
                self._registry.executor(step.executor_name),
            )
            for step in definition.steps
        )

        validated_inputs = _validate_inputs(
            definition,
            inputs,
        )

        if (
            not isinstance(
                run_id,
                str,
            )
            or not run_id.strip()
        ):
            raise WorkflowResumeError("run_id must be a non-empty string")

        checkpoint = await self._state_store.load(run_id)

        if checkpoint is None:
            raise WorkflowResumeError("workflow checkpoint not found")

        if not isinstance(
            checkpoint,
            WorkflowCheckpoint,
        ):
            raise WorkflowResumeError(
                "workflow state store returned invalid checkpoint"
            )

        self._validate_resume_checkpoint(
            definition=definition,
            validated_inputs=validated_inputs,
            checkpoint=checkpoint,
            run_id=run_id,
        )

        events = list(checkpoint.events)

        step_states = list(checkpoint.state.steps)

        executions = list(checkpoint.state.executions)

        by_step_id = {execution.step_id: execution for execution in executions}

        skipped_step_ids = set(checkpoint.state.skipped_step_ids)

        def record_event(
            event_type: WorkflowEventType,
            *,
            step: WorkflowStepDefinition | None = None,
        ) -> None:
            events.append(
                WorkflowExecutionEvent(
                    sequence=len(events) + 1,
                    run_id=run_id,
                    workflow_id=definition.workflow_id,
                    workflow_version=definition.version,
                    event_type=event_type,
                    step_id=(None if step is None else step.step_id),
                    executor_name=(None if step is None else step.executor_name),
                )
            )

        remaining = resolved[len(step_states) :]

        for (
            step,
            executor,
        ) in remaining:
            condition = step.condition

            if condition is not None and not _condition_matches(
                condition,
                validated_inputs,
            ):
                step_states.append(
                    WorkflowStepState(
                        step_id=step.step_id,
                        executor_name=step.executor_name,
                        status=WorkflowStepStatus.SKIPPED,
                    )
                )

                skipped_step_ids.add(step.step_id)

                record_event(
                    WorkflowEventType.STEP_SKIPPED,
                    step=step,
                )

                await self._save_checkpoint(
                    state=self._running_state(
                        run_id=run_id,
                        definition=definition,
                        inputs=validated_inputs,
                        steps=tuple(step_states),
                    ),
                    events=events,
                )

                continue

            dependency_results = tuple(
                by_step_id[dependency]
                for dependency in step.depends_on
                if dependency in by_step_id
            )

            skipped_dependency_ids = tuple(
                dependency
                for dependency in step.depends_on
                if dependency in skipped_step_ids
            )

            context = WorkflowStepContext(
                run_id=run_id,
                workflow_id=definition.workflow_id,
                workflow_version=definition.version,
                step_id=step.step_id,
                dependency_results=dependency_results,
                inputs=validated_inputs,
                skipped_dependency_ids=skipped_dependency_ids,
            )

            try:
                output = await executor.execute(
                    step=step,
                    context=context,
                )
            except Exception as exc:
                record_event(
                    WorkflowEventType.STEP_FAILED,
                    step=step,
                )

                record_event(WorkflowEventType.RUN_FAILED)

                failed_trace = WorkflowExecutionTrace(
                    run_id=run_id,
                    workflow_id=definition.workflow_id,
                    workflow_version=definition.version,
                    events=tuple(events),
                )

                failed_state = WorkflowRunState(
                    run_id=run_id,
                    workflow_id=definition.workflow_id,
                    workflow_version=definition.version,
                    status=WorkflowRunStatus.FAILED,
                    inputs=validated_inputs,
                    steps=(
                        *step_states,
                        WorkflowStepState(
                            step_id=step.step_id,
                            executor_name=step.executor_name,
                            status=WorkflowStepStatus.FAILED,
                        ),
                    ),
                )

                await self._save_checkpoint(
                    state=failed_state,
                    events=events,
                )

                raise WorkflowStepExecutionError(
                    step_id=step.step_id,
                    executor_name=step.executor_name,
                    state=failed_state,
                    trace=failed_trace,
                ) from exc

            execution = WorkflowStepExecution(
                step_id=step.step_id,
                executor_name=step.executor_name,
                output=output,
            )

            executions.append(execution)

            by_step_id[step.step_id] = execution

            record_event(
                WorkflowEventType.STEP_COMPLETED,
                step=step,
            )

            step_states.append(
                WorkflowStepState(
                    step_id=step.step_id,
                    executor_name=step.executor_name,
                    status=WorkflowStepStatus.COMPLETED,
                    execution=execution,
                )
            )

            await self._save_checkpoint(
                state=self._running_state(
                    run_id=run_id,
                    definition=definition,
                    inputs=validated_inputs,
                    steps=tuple(step_states),
                ),
                events=events,
            )

        record_event(WorkflowEventType.RUN_COMPLETED)

        completed_trace = WorkflowExecutionTrace(
            run_id=run_id,
            workflow_id=definition.workflow_id,
            workflow_version=definition.version,
            events=tuple(events),
        )

        completed_state = WorkflowRunState(
            run_id=run_id,
            workflow_id=definition.workflow_id,
            workflow_version=definition.version,
            status=WorkflowRunStatus.COMPLETED,
            inputs=validated_inputs,
            steps=tuple(step_states),
        )

        await self._save_checkpoint(
            state=completed_state,
            events=events,
        )

        return WorkflowRunResult(
            run_id=run_id,
            workflow_id=definition.workflow_id,
            workflow_version=definition.version,
            executions=tuple(executions),
            state=completed_state,
            trace=completed_trace,
        )

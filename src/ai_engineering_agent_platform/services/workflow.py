"""Deterministic project-owned workflow execution engine."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from uuid import uuid4

from ai_engineering_agent_platform.domain.workflow import (
    MAX_WORKFLOW_DEFINITION_STEPS,
    WorkflowApprovalPause,
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


@dataclass(frozen=True, slots=True)
class WorkflowApprovalPending:
    """Non-terminal workflow control-plane result awaiting human approval."""

    state: WorkflowRunState
    events: tuple[
        WorkflowExecutionEvent,
        ...,
    ]

    def __post_init__(
        self,
    ) -> None:
        """Require one exact durable non-terminal approval pause."""
        if not isinstance(
            self.state,
            WorkflowRunState,
        ):
            raise ValueError("state must be WorkflowRunState")

        if self.state.status is not WorkflowRunStatus.AWAITING_APPROVAL:
            raise ValueError("approval-pending state must be awaiting approval")

        if (
            not isinstance(
                self.events,
                tuple,
            )
            or not self.events
        ):
            raise ValueError("approval-pending events must be a non-empty tuple")

        if any(
            not isinstance(
                event,
                WorkflowExecutionEvent,
            )
            for event in self.events
        ):
            raise ValueError("approval-pending events contain invalid values")

        if tuple(event.sequence for event in self.events) != tuple(
            range(
                1,
                len(self.events) + 1,
            )
        ):
            raise ValueError("approval-pending event sequences must be contiguous")

        if any(
            event.run_id != self.state.run_id
            or event.workflow_id != self.state.workflow_id
            or event.workflow_version != self.state.workflow_version
            for event in self.events
        ):
            raise ValueError("approval-pending event identity mismatch")

        if self.events[0].event_type is not WorkflowEventType.RUN_STARTED:
            raise ValueError("approval-pending ledger must start with RUN_STARTED")

        if any(
            event.event_type
            in {
                WorkflowEventType.RUN_COMPLETED,
                WorkflowEventType.RUN_FAILED,
            }
            for event in self.events
        ):
            raise ValueError("approval-pending ledger must be non-terminal")

        paused_step = self.state.steps[-1]
        final_event = self.events[-1]

        if (
            final_event.event_type is not WorkflowEventType.STEP_AWAITING_APPROVAL
            or final_event.step_id != paused_step.step_id
            or final_event.executor_name != paused_step.executor_name
        ):
            raise ValueError("approval-pending ledger must end with matching pause")

    @property
    def approval_pause(
        self,
    ) -> WorkflowApprovalPause:
        """Return exact durable references for the currently paused step."""
        pause = self.state.steps[-1].approval_pause

        if pause is None:
            raise RuntimeError("awaiting workflow state lost approval pause")

        return pause

    @property
    def paused_step_id(
        self,
    ) -> str:
        """Return the currently paused workflow step identity."""
        return self.state.steps[-1].step_id


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


@runtime_checkable
class WorkflowApprovalAwareExecutor(
    WorkflowStepExecutor,
    Protocol,
):
    """Structural workflow executor supporting durable HITL pause/resume."""

    async def execute_with_approval_pause(
        self,
        *,
        step: WorkflowStepDefinition,
        context: WorkflowStepContext,
    ) -> object | WorkflowApprovalPause:
        """Execute until completion or one durable approval pause."""

    async def resume_approval(
        self,
        *,
        step: WorkflowStepDefinition,
        context: WorkflowStepContext,
        pause: WorkflowApprovalPause,
    ) -> object | WorkflowApprovalPause:
        """Resume one exact durable approval pause."""


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
    @staticmethod
    def _validate_checkpoint_progress(
        *,
        definition: WorkflowDefinition,
        validated_inputs: tuple[
            WorkflowInput,
            ...,
        ],
        checkpoint: WorkflowCheckpoint,
    ) -> None:
        """Validate saved step/event progress including historical HITL pauses."""
        state = checkpoint.state
        events = checkpoint.events

        if state.inputs != validated_inputs:
            raise WorkflowResumeError(
                "workflow checkpoint inputs differ from supplied inputs"
            )

        if len(state.steps) > len(definition.steps):
            raise WorkflowResumeError("workflow checkpoint progress exceeds definition")

        if not events or events[0].event_type is not WorkflowEventType.RUN_STARTED:
            raise WorkflowResumeError(
                "workflow checkpoint event ledger is incompatible"
            )

        event_index = 1

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

            if saved_step.status is WorkflowStepStatus.SKIPPED:
                if not should_skip:
                    raise WorkflowResumeError(
                        "workflow checkpoint step status is incompatible"
                    )

                if event_index >= len(events):
                    raise WorkflowResumeError(
                        "workflow checkpoint event ledger is incompatible"
                    )

                event = events[event_index]

                if (
                    event.event_type is not WorkflowEventType.STEP_SKIPPED
                    or event.step_id != saved_step.step_id
                    or event.executor_name != saved_step.executor_name
                ):
                    raise WorkflowResumeError(
                        "workflow checkpoint event ledger is incompatible"
                    )

                event_index += 1
                continue

            if saved_step.status is WorkflowStepStatus.COMPLETED:
                if should_skip:
                    raise WorkflowResumeError(
                        "workflow checkpoint step status is incompatible"
                    )

                while event_index < len(events):
                    event = events[event_index]

                    if (
                        event.event_type is WorkflowEventType.STEP_AWAITING_APPROVAL
                        and event.step_id == saved_step.step_id
                        and event.executor_name == saved_step.executor_name
                    ):
                        event_index += 1
                        continue

                    break

                if event_index >= len(events):
                    raise WorkflowResumeError(
                        "workflow checkpoint event ledger is incompatible"
                    )

                event = events[event_index]

                if (
                    event.event_type is not WorkflowEventType.STEP_COMPLETED
                    or event.step_id != saved_step.step_id
                    or event.executor_name != saved_step.executor_name
                ):
                    raise WorkflowResumeError(
                        "workflow checkpoint event ledger is incompatible"
                    )

                event_index += 1
                continue

            if saved_step.status is WorkflowStepStatus.AWAITING_APPROVAL:
                if (
                    state.status is not WorkflowRunStatus.AWAITING_APPROVAL
                    or index != len(state.steps) - 1
                    or should_skip
                ):
                    raise WorkflowResumeError(
                        "workflow checkpoint approval pause is incompatible"
                    )

                saw_pause = False

                while event_index < len(events):
                    event = events[event_index]

                    if (
                        event.event_type is not WorkflowEventType.STEP_AWAITING_APPROVAL
                        or event.step_id != saved_step.step_id
                        or event.executor_name != saved_step.executor_name
                    ):
                        raise WorkflowResumeError(
                            "workflow checkpoint approval ledger is incompatible"
                        )

                    saw_pause = True
                    event_index += 1

                if not saw_pause:
                    raise WorkflowResumeError(
                        "workflow checkpoint approval ledger is incomplete"
                    )

                continue

            raise WorkflowResumeError("workflow checkpoint step status is incompatible")

        if event_index != len(events):
            raise WorkflowResumeError(
                "workflow checkpoint event ledger is incompatible"
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
        """Fail closed unless RUNNING progress exactly matches the definition."""
        state = checkpoint.state

        if (
            state.run_id != run_id
            or state.workflow_id != definition.workflow_id
            or state.workflow_version != definition.version
        ):
            raise WorkflowResumeError("workflow checkpoint identity mismatch")

        if state.status is not WorkflowRunStatus.RUNNING:
            raise WorkflowResumeError("workflow checkpoint is not resumable")

        WorkflowEngine._validate_checkpoint_progress(
            definition=definition,
            validated_inputs=validated_inputs,
            checkpoint=checkpoint,
        )

    @staticmethod
    def _validate_approval_resume_checkpoint(
        *,
        definition: WorkflowDefinition,
        validated_inputs: tuple[
            WorkflowInput,
            ...,
        ],
        checkpoint: WorkflowCheckpoint,
        run_id: str,
    ) -> None:
        """Fail closed unless one exact workflow step is durably awaiting approval."""
        state = checkpoint.state

        if (
            state.run_id != run_id
            or state.workflow_id != definition.workflow_id
            or state.workflow_version != definition.version
        ):
            raise WorkflowResumeError("workflow checkpoint identity mismatch")

        if state.status is not WorkflowRunStatus.AWAITING_APPROVAL:
            raise WorkflowResumeError("workflow checkpoint is not awaiting approval")

        WorkflowEngine._validate_checkpoint_progress(
            definition=definition,
            validated_inputs=validated_inputs,
            checkpoint=checkpoint,
        )

    async def _execute_from_progress(
        self,
        *,
        definition: WorkflowDefinition,
        validated_inputs: tuple[
            WorkflowInput,
            ...,
        ],
        resolved: tuple[
            tuple[
                WorkflowStepDefinition,
                WorkflowStepExecutor,
            ],
            ...,
        ],
        run_id: str,
        events: list[WorkflowExecutionEvent,],
        step_states: list[WorkflowStepState,],
        executions: list[WorkflowStepExecution,],
        by_step_id: dict[
            str,
            WorkflowStepExecution,
        ],
        skipped_step_ids: set[str,],
        start_index: int,
    ) -> WorkflowRunResult | WorkflowApprovalPending:
        """Continue deterministic workflow execution from validated progress."""

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

        for (
            step,
            executor,
        ) in resolved[start_index:]:
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

            approval_aware = self._state_store is not None and isinstance(
                executor,
                WorkflowApprovalAwareExecutor,
            )

            try:
                if approval_aware and isinstance(
                    executor, WorkflowApprovalAwareExecutor
                ):
                    output = await executor.execute_with_approval_pause(
                        step=step,
                        context=context,
                    )
                else:
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

            if approval_aware and isinstance(
                output,
                WorkflowApprovalPause,
            ):
                paused_step = WorkflowStepState(
                    step_id=step.step_id,
                    executor_name=step.executor_name,
                    status=WorkflowStepStatus.AWAITING_APPROVAL,
                    approval_pause=output,
                )

                record_event(
                    WorkflowEventType.STEP_AWAITING_APPROVAL,
                    step=step,
                )

                awaiting_state = WorkflowRunState(
                    run_id=run_id,
                    workflow_id=definition.workflow_id,
                    workflow_version=definition.version,
                    status=WorkflowRunStatus.AWAITING_APPROVAL,
                    inputs=validated_inputs,
                    steps=(
                        *step_states,
                        paused_step,
                    ),
                )

                await self._save_checkpoint(
                    state=awaiting_state,
                    events=events,
                )

                return WorkflowApprovalPending(
                    state=awaiting_state,
                    events=tuple(events),
                )

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

    async def run(
        self,
        definition: WorkflowDefinition,
        *,
        inputs: tuple[
            WorkflowInput,
            ...,
        ] = (),
    ) -> WorkflowRunResult | WorkflowApprovalPending:
        """Execute until terminal completion or one durable approval pause."""
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

        events = [
            WorkflowExecutionEvent(
                sequence=1,
                run_id=run_id,
                workflow_id=definition.workflow_id,
                workflow_version=definition.version,
                event_type=WorkflowEventType.RUN_STARTED,
            )
        ]

        await self._save_checkpoint(
            state=self._running_state(
                run_id=run_id,
                definition=definition,
                inputs=validated_inputs,
                steps=(),
            ),
            events=events,
        )

        return await self._execute_from_progress(
            definition=definition,
            validated_inputs=validated_inputs,
            resolved=resolved,
            run_id=run_id,
            events=events,
            step_states=[],
            executions=[],
            by_step_id={},
            skipped_step_ids=set(),
            start_index=0,
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
    ) -> WorkflowRunResult | WorkflowApprovalPending:
        """Continue one validated RUNNING durable checkpoint."""
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

        step_states = list(checkpoint.state.steps)

        executions = list(checkpoint.state.executions)

        by_step_id = {execution.step_id: execution for execution in executions}

        skipped_step_ids = set(checkpoint.state.skipped_step_ids)

        return await self._execute_from_progress(
            definition=definition,
            validated_inputs=validated_inputs,
            resolved=resolved,
            run_id=run_id,
            events=list(checkpoint.events),
            step_states=step_states,
            executions=executions,
            by_step_id=by_step_id,
            skipped_step_ids=skipped_step_ids,
            start_index=len(step_states),
        )

    async def resume_approval(
        self,
        definition: WorkflowDefinition,
        *,
        run_id: str,
        inputs: tuple[
            WorkflowInput,
            ...,
        ] = (),
    ) -> WorkflowRunResult | WorkflowApprovalPending:
        """Resume one exact durable workflow approval pause.

        Human authentication and the approval decision occur outside this method
        through the configured approval boundary. This method resumes only an
        already-decided durable workflow continuation.
        """
        if self._state_store is None:
            raise WorkflowResumeError(
                "workflow approval resume requires durable state store"
            )

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

        self._validate_approval_resume_checkpoint(
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

        paused_index = len(step_states) - 1

        paused_state = step_states[paused_index]

        step, executor = resolved[paused_index]

        if not isinstance(
            executor,
            WorkflowApprovalAwareExecutor,
        ):
            raise WorkflowResumeError(
                "paused workflow executor does not support approval resume"
            )

        pause = paused_state.approval_pause

        if pause is None:
            raise WorkflowResumeError("paused workflow state lost approval references")

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
            output = await executor.resume_approval(
                step=step,
                context=context,
                pause=pause,
            )
        except Exception as exc:
            # Do not manufacture FAILED state or an automatic retry. The durable
            # AWAITING_APPROVAL checkpoint remains the last committed workflow
            # state if approval/continuation resume fails.
            raise WorkflowResumeError(
                "workflow approval continuation could not resume"
            ) from exc

        def record_event(
            event_type: WorkflowEventType,
        ) -> None:
            events.append(
                WorkflowExecutionEvent(
                    sequence=len(events) + 1,
                    run_id=run_id,
                    workflow_id=definition.workflow_id,
                    workflow_version=definition.version,
                    event_type=event_type,
                    step_id=step.step_id,
                    executor_name=step.executor_name,
                )
            )

        if isinstance(
            output,
            WorkflowApprovalPause,
        ):
            step_states[paused_index] = WorkflowStepState(
                step_id=step.step_id,
                executor_name=step.executor_name,
                status=WorkflowStepStatus.AWAITING_APPROVAL,
                approval_pause=output,
            )

            record_event(WorkflowEventType.STEP_AWAITING_APPROVAL)

            awaiting_state = WorkflowRunState(
                run_id=run_id,
                workflow_id=definition.workflow_id,
                workflow_version=definition.version,
                status=WorkflowRunStatus.AWAITING_APPROVAL,
                inputs=validated_inputs,
                steps=tuple(step_states),
            )

            await self._save_checkpoint(
                state=awaiting_state,
                events=events,
            )

            return WorkflowApprovalPending(
                state=awaiting_state,
                events=tuple(events),
            )

        execution = WorkflowStepExecution(
            step_id=step.step_id,
            executor_name=step.executor_name,
            output=output,
        )

        executions.append(execution)

        by_step_id[step.step_id] = execution

        step_states[paused_index] = WorkflowStepState(
            step_id=step.step_id,
            executor_name=step.executor_name,
            status=WorkflowStepStatus.COMPLETED,
            execution=execution,
        )

        record_event(WorkflowEventType.STEP_COMPLETED)

        # Commit the approved step before continuing the DAG. Persistence failure
        # here does not imply exactly-once external side effects; a later replay
        # must fail closed against consumed approval/agent continuation authority.
        await self._save_checkpoint(
            state=self._running_state(
                run_id=run_id,
                definition=definition,
                inputs=validated_inputs,
                steps=tuple(step_states),
            ),
            events=events,
        )

        return await self._execute_from_progress(
            definition=definition,
            validated_inputs=validated_inputs,
            resolved=resolved,
            run_id=run_id,
            events=events,
            step_states=step_states,
            executions=executions,
            by_step_id=by_step_id,
            skipped_step_ids=skipped_step_ids,
            start_index=len(step_states),
        )

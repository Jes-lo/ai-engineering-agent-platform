"""Immutable domain models for project-owned workflow execution."""

import re
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

MAX_WORKFLOW_DEFINITION_STEPS = 32
MAX_WORKFLOW_INPUT_STRING_LENGTH = 4096

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,79}$")


type WorkflowScalar = str | int | float | bool


class WorkflowInputType(StrEnum):
    """Supported deterministic workflow input types."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"


class WorkflowConditionOperator(StrEnum):
    """Small deterministic condition vocabulary."""

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"


class WorkflowStepStatus(StrEnum):
    """Terminal state of one workflow step attempt."""

    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class WorkflowRunStatus(StrEnum):
    """Process-local workflow run state."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowEventType(StrEnum):
    """Privacy-safe structural workflow execution event types."""

    RUN_STARTED = "run_started"
    STEP_COMPLETED = "step_completed"
    STEP_SKIPPED = "step_skipped"
    STEP_FAILED = "step_failed"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"


def _require_identifier(
    value: object,
    *,
    field_name: str,
) -> str:
    """Return one bounded portable identifier."""
    if not isinstance(
        value,
        str,
    ) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a portable identifier")

    return value


def _require_version(
    value: object,
) -> str:
    """Return one bounded non-empty workflow version."""
    if (
        not isinstance(
            value,
            str,
        )
        or not value.strip()
        or len(value) > 80
    ):
        raise ValueError("version must be a non-empty string of at most 80 characters")

    return value


def _validate_scalar(
    value: object,
    *,
    field_name: str,
) -> WorkflowScalar:
    """Validate one deterministic scalar workflow value."""
    if isinstance(
        value,
        bool,
    ):
        return value

    if type(value) is int:
        return value

    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f"{field_name} float must be finite")

        return value

    if isinstance(
        value,
        str,
    ):
        if len(value) > MAX_WORKFLOW_INPUT_STRING_LENGTH:
            raise ValueError(
                f"{field_name} string exceeds "
                f"{MAX_WORKFLOW_INPUT_STRING_LENGTH} characters"
            )

        return value

    raise ValueError(f"{field_name} must be a supported workflow scalar")


def _matches_input_type(
    value: WorkflowScalar,
    input_type: WorkflowInputType,
) -> bool:
    """Return whether one scalar exactly matches one declared type."""
    if input_type is WorkflowInputType.STRING:
        return isinstance(
            value,
            str,
        )

    if input_type is WorkflowInputType.INTEGER:
        return type(value) is int

    if input_type is WorkflowInputType.FLOAT:
        return type(value) is float

    if input_type is WorkflowInputType.BOOLEAN:
        return isinstance(
            value,
            bool,
        )

    return False


@dataclass(
    frozen=True,
    slots=True,
)
class WorkflowInputDefinition:
    """One required typed input accepted by a workflow definition."""

    name: str
    input_type: WorkflowInputType

    def __post_init__(
        self,
    ) -> None:
        """Validate schema identity and type."""
        _require_identifier(
            self.name,
            field_name="workflow input name",
        )

        if not isinstance(
            self.input_type,
            WorkflowInputType,
        ):
            raise ValueError("input_type must be WorkflowInputType")

    def accepts(
        self,
        value: WorkflowScalar,
    ) -> bool:
        """Return whether a scalar exactly matches this input schema."""
        return _matches_input_type(
            value,
            self.input_type,
        )


@dataclass(
    frozen=True,
    slots=True,
)
class WorkflowInput:
    """One concrete typed workflow-run input value."""

    name: str
    value: WorkflowScalar

    def __post_init__(
        self,
    ) -> None:
        """Validate stable input identity and bounded scalar value."""
        _require_identifier(
            self.name,
            field_name="workflow input name",
        )

        _validate_scalar(
            self.value,
            field_name="workflow input",
        )


@dataclass(
    frozen=True,
    slots=True,
)
class WorkflowCondition:
    """One deterministic condition over one declared workflow input."""

    input_name: str
    operator: WorkflowConditionOperator
    expected_value: WorkflowScalar

    def __post_init__(
        self,
    ) -> None:
        """Validate condition shape without evaluating it."""
        _require_identifier(
            self.input_name,
            field_name="condition input name",
        )

        if not isinstance(
            self.operator,
            WorkflowConditionOperator,
        ):
            raise ValueError("operator must be WorkflowConditionOperator")

        _validate_scalar(
            self.expected_value,
            field_name="condition expected value",
        )


@dataclass(
    frozen=True,
    slots=True,
)
class WorkflowStepDefinition:
    """One deterministic application-controlled workflow step.

    ``executor_name`` selects a trusted application executor. It does not
    identify or authorize a model, tool, or provider directly.

    ``depends_on`` is explicit and may reference only earlier steps in the
    containing workflow definition.

    ``condition`` may compare one declared workflow input by exact typed
    equality or inequality. Arbitrary expressions and executable conditions
    are intentionally outside this foundation.
    """

    step_id: str
    executor_name: str
    depends_on: tuple[str, ...] = ()
    condition: WorkflowCondition | None = None

    def __post_init__(
        self,
    ) -> None:
        """Validate portable identity, dependencies, and condition type."""
        _require_identifier(
            self.step_id,
            field_name="step_id",
        )

        _require_identifier(
            self.executor_name,
            field_name="executor_name",
        )

        if not isinstance(
            self.depends_on,
            tuple,
        ):
            raise ValueError("depends_on must be a tuple")

        for dependency in self.depends_on:
            _require_identifier(
                dependency,
                field_name="dependency step ID",
            )

        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("dependency step IDs must be unique")

        if self.step_id in self.depends_on:
            raise ValueError("a workflow step cannot depend on itself")

        if self.condition is not None and not isinstance(
            self.condition,
            WorkflowCondition,
        ):
            raise ValueError("condition must be WorkflowCondition or None")


@dataclass(
    frozen=True,
    slots=True,
)
class WorkflowDefinition:
    """Versioned static workflow DAG and its required typed inputs."""

    workflow_id: str
    version: str
    steps: tuple[
        WorkflowStepDefinition,
        ...,
    ]
    inputs: tuple[
        WorkflowInputDefinition,
        ...,
    ] = ()

    def __post_init__(
        self,
    ) -> None:
        """Validate bounded DAG, input schema, and condition references."""
        _require_identifier(
            self.workflow_id,
            field_name="workflow_id",
        )

        _require_version(self.version)

        if not isinstance(
            self.steps,
            tuple,
        ):
            raise ValueError("steps must be a tuple")

        if not self.steps:
            raise ValueError("workflow must contain at least one step")

        if len(self.steps) > MAX_WORKFLOW_DEFINITION_STEPS:
            raise ValueError(
                "workflow exceeds maximum definition size "
                f"of {MAX_WORKFLOW_DEFINITION_STEPS} steps"
            )

        if any(
            not isinstance(
                step,
                WorkflowStepDefinition,
            )
            for step in self.steps
        ):
            raise ValueError("steps must contain WorkflowStepDefinition values")

        if not isinstance(
            self.inputs,
            tuple,
        ):
            raise ValueError("inputs must be a tuple")

        if any(
            not isinstance(
                input_definition,
                WorkflowInputDefinition,
            )
            for input_definition in self.inputs
        ):
            raise ValueError("inputs must contain WorkflowInputDefinition values")

        input_names = tuple(input_definition.name for input_definition in self.inputs)

        if len(input_names) != len(set(input_names)):
            raise ValueError("workflow input names must be unique")

        input_definitions = {
            input_definition.name: input_definition for input_definition in self.inputs
        }

        step_ids = tuple(step.step_id for step in self.steps)

        if len(step_ids) != len(set(step_ids)):
            raise ValueError("workflow step IDs must be unique")

        earlier_steps: set[str] = set()

        for step in self.steps:
            for dependency in step.depends_on:
                if dependency not in earlier_steps:
                    raise ValueError(
                        "workflow dependencies must reference earlier declared steps"
                    )

            condition = step.condition

            if condition is not None:
                try:
                    input_definition = input_definitions[condition.input_name]
                except KeyError as exc:
                    raise ValueError(
                        "workflow condition must reference a declared input"
                    ) from exc

                if not input_definition.accepts(condition.expected_value):
                    raise ValueError(
                        "workflow condition expected value "
                        "must match declared input type"
                    )

            earlier_steps.add(step.step_id)

    @property
    def ordered_step_ids(
        self,
    ) -> tuple[str, ...]:
        """Return deterministic declaration/execution order."""
        return tuple(step.step_id for step in self.steps)


@dataclass(
    frozen=True,
    slots=True,
)
class WorkflowStepExecution:
    """One successfully completed process-local workflow step."""

    step_id: str
    executor_name: str
    output: object

    def __post_init__(
        self,
    ) -> None:
        """Validate stable execution identity."""
        _require_identifier(
            self.step_id,
            field_name="step_id",
        )

        _require_identifier(
            self.executor_name,
            field_name="executor_name",
        )


@dataclass(
    frozen=True,
    slots=True,
)
class WorkflowStepState:
    """Terminal state of one step in a process-local workflow attempt."""

    step_id: str
    executor_name: str
    status: WorkflowStepStatus
    execution: WorkflowStepExecution | None = None

    def __post_init__(
        self,
    ) -> None:
        """Validate step-state/execution consistency."""
        _require_identifier(
            self.step_id,
            field_name="step_id",
        )

        _require_identifier(
            self.executor_name,
            field_name="executor_name",
        )

        if not isinstance(
            self.status,
            WorkflowStepStatus,
        ):
            raise ValueError("status must be WorkflowStepStatus")

        if self.status is WorkflowStepStatus.COMPLETED:
            if not isinstance(
                self.execution,
                WorkflowStepExecution,
            ):
                raise ValueError("completed step state requires execution")

            if (
                self.execution.step_id != self.step_id
                or self.execution.executor_name != self.executor_name
            ):
                raise ValueError("step state execution identity mismatch")

            return

        if self.execution is not None:
            raise ValueError("skipped/failed step state must not contain execution")


@dataclass(
    frozen=True,
    slots=True,
)
class WorkflowRunState:
    """Immutable process-local snapshot of one workflow run."""

    run_id: str
    workflow_id: str
    workflow_version: str
    status: WorkflowRunStatus
    inputs: tuple[
        WorkflowInput,
        ...,
    ]
    steps: tuple[
        WorkflowStepState,
        ...,
    ]

    def __post_init__(
        self,
    ) -> None:
        """Validate run identity and unique input/step state."""
        _require_identifier(
            self.run_id,
            field_name="run_id",
        )

        _require_identifier(
            self.workflow_id,
            field_name="workflow_id",
        )

        _require_version(self.workflow_version)

        if not isinstance(
            self.status,
            WorkflowRunStatus,
        ):
            raise ValueError("status must be WorkflowRunStatus")

        if not isinstance(
            self.inputs,
            tuple,
        ):
            raise ValueError("inputs must be a tuple")

        if any(
            not isinstance(
                item,
                WorkflowInput,
            )
            for item in self.inputs
        ):
            raise ValueError("inputs must contain WorkflowInput values")

        input_names = tuple(item.name for item in self.inputs)

        if len(input_names) != len(set(input_names)):
            raise ValueError("run input names must be unique")

        if not isinstance(
            self.steps,
            tuple,
        ):
            raise ValueError("steps must be a tuple")

        if any(
            not isinstance(
                step,
                WorkflowStepState,
            )
            for step in self.steps
        ):
            raise ValueError("steps must contain WorkflowStepState values")

        step_ids = tuple(step.step_id for step in self.steps)

        if len(step_ids) != len(set(step_ids)):
            raise ValueError("run step IDs must be unique")

        failed_count = sum(
            step.status is WorkflowStepStatus.FAILED for step in self.steps
        )

        if self.status is WorkflowRunStatus.FAILED and failed_count != 1:
            raise ValueError("failed run state must contain exactly one failed step")

        if self.status is not WorkflowRunStatus.FAILED and failed_count != 0:
            raise ValueError("non-failed run state must not contain failed steps")

    @property
    def executions(
        self,
    ) -> tuple[
        WorkflowStepExecution,
        ...,
    ]:
        """Return successful executions in workflow order."""
        return tuple(
            step.execution
            for step in self.steps
            if (
                step.status is WorkflowStepStatus.COMPLETED
                and step.execution is not None
            )
        )

    @property
    def skipped_step_ids(
        self,
    ) -> tuple[str, ...]:
        """Return skipped step IDs in workflow order."""
        return tuple(
            step.step_id
            for step in self.steps
            if step.status is WorkflowStepStatus.SKIPPED
        )


@dataclass(
    frozen=True,
    slots=True,
)
class WorkflowExecutionEvent:
    """One privacy-safe structural workflow execution event.

    Events intentionally contain no workflow input values, prompts, tool
    arguments, retrieved/generated content, executor outputs, or exception
    messages.
    """

    sequence: int
    run_id: str
    workflow_id: str
    workflow_version: str
    event_type: WorkflowEventType
    step_id: str | None = None
    executor_name: str | None = None

    def __post_init__(
        self,
    ) -> None:
        """Validate sequence, correlation identity, and event shape."""
        if (
            not isinstance(
                self.sequence,
                int,
            )
            or isinstance(
                self.sequence,
                bool,
            )
            or self.sequence <= 0
        ):
            raise ValueError("workflow event sequence must be a positive integer")

        _require_identifier(
            self.run_id,
            field_name="run_id",
        )

        _require_identifier(
            self.workflow_id,
            field_name="workflow_id",
        )

        _require_version(self.workflow_version)

        if not isinstance(
            self.event_type,
            WorkflowEventType,
        ):
            raise ValueError("event_type must be WorkflowEventType")

        step_event_types = {
            WorkflowEventType.STEP_COMPLETED,
            WorkflowEventType.STEP_SKIPPED,
            WorkflowEventType.STEP_FAILED,
        }

        if self.event_type in step_event_types:
            if self.step_id is None or self.executor_name is None:
                raise ValueError("step workflow event requires step identity")

            _require_identifier(
                self.step_id,
                field_name="step_id",
            )

            _require_identifier(
                self.executor_name,
                field_name="executor_name",
            )

            return

        if self.step_id is not None or self.executor_name is not None:
            raise ValueError("run workflow event must not contain step identity")

    @property
    def event_id(
        self,
    ) -> str:
        """Return deterministic per-run event correlation identity."""
        return f"{self.run_id}:event:{self.sequence}"


@dataclass(
    frozen=True,
    slots=True,
)
class WorkflowExecutionTrace:
    """Immutable ordered structural event ledger for one terminal run."""

    run_id: str
    workflow_id: str
    workflow_version: str
    events: tuple[
        WorkflowExecutionEvent,
        ...,
    ]

    def __post_init__(
        self,
    ) -> None:
        """Validate identity, sequence continuity, and terminal event order."""
        _require_identifier(
            self.run_id,
            field_name="run_id",
        )

        _require_identifier(
            self.workflow_id,
            field_name="workflow_id",
        )

        _require_version(self.workflow_version)

        if not isinstance(
            self.events,
            tuple,
        ):
            raise ValueError("workflow trace events must be a tuple")

        if not self.events:
            raise ValueError("workflow trace must contain events")

        if any(
            not isinstance(
                event,
                WorkflowExecutionEvent,
            )
            for event in self.events
        ):
            raise ValueError(
                "workflow trace must contain WorkflowExecutionEvent values"
            )

        expected_sequences = tuple(
            range(
                1,
                len(self.events) + 1,
            )
        )

        actual_sequences = tuple(event.sequence for event in self.events)

        if actual_sequences != expected_sequences:
            raise ValueError("workflow trace event sequences must be contiguous")

        for event in self.events:
            if (
                event.run_id != self.run_id
                or event.workflow_id != self.workflow_id
                or event.workflow_version != self.workflow_version
            ):
                raise ValueError("workflow trace event identity mismatch")

        if self.events[0].event_type is not WorkflowEventType.RUN_STARTED:
            raise ValueError("workflow trace must start with RUN_STARTED")

        terminal_types = {
            WorkflowEventType.RUN_COMPLETED,
            WorkflowEventType.RUN_FAILED,
        }

        if self.events[-1].event_type not in terminal_types:
            raise ValueError("workflow trace must end with one terminal run event")

        for event in self.events[1:-1]:
            if (
                event.event_type is WorkflowEventType.RUN_STARTED
                or event.event_type in terminal_types
            ):
                raise ValueError("workflow trace contains misplaced run event")


@dataclass(
    frozen=True,
    slots=True,
)
class WorkflowRunResult:
    """Immutable successful result for one completed workflow attempt."""

    run_id: str
    workflow_id: str
    workflow_version: str
    executions: tuple[
        WorkflowStepExecution,
        ...,
    ]
    state: WorkflowRunState | None = None
    trace: WorkflowExecutionTrace | None = None

    def __post_init__(
        self,
    ) -> None:
        """Validate one completed process-local run result."""
        _require_identifier(
            self.run_id,
            field_name="run_id",
        )

        _require_identifier(
            self.workflow_id,
            field_name="workflow_id",
        )

        _require_version(self.workflow_version)

        if not isinstance(
            self.executions,
            tuple,
        ):
            raise ValueError("executions must be a tuple")

        if any(
            not isinstance(
                execution,
                WorkflowStepExecution,
            )
            for execution in self.executions
        ):
            raise ValueError("executions must contain WorkflowStepExecution values")

        step_ids = tuple(execution.step_id for execution in self.executions)

        if len(step_ids) != len(set(step_ids)):
            raise ValueError("completed execution step IDs must be unique")

        if self.trace is not None:
            if not isinstance(
                self.trace,
                WorkflowExecutionTrace,
            ):
                raise ValueError("trace must be WorkflowExecutionTrace or None")

            if (
                self.trace.run_id != self.run_id
                or self.trace.workflow_id != self.workflow_id
                or self.trace.workflow_version != self.workflow_version
            ):
                raise ValueError("workflow result trace identity mismatch")

            if self.trace.events[-1].event_type is not WorkflowEventType.RUN_COMPLETED:
                raise ValueError(
                    "successful workflow result trace must end RUN_COMPLETED"
                )

        if self.state is None:
            if not self.executions:
                raise ValueError("completed workflow result must contain executions")

            return

        if not isinstance(
            self.state,
            WorkflowRunState,
        ):
            raise ValueError("state must be WorkflowRunState or None")

        if self.state.status is not WorkflowRunStatus.COMPLETED:
            raise ValueError("workflow result state must be completed")

        if (
            self.state.run_id != self.run_id
            or self.state.workflow_id != self.workflow_id
            or self.state.workflow_version != self.workflow_version
        ):
            raise ValueError("workflow result state identity mismatch")

        if self.state.executions != self.executions:
            raise ValueError("workflow result executions must match run state")

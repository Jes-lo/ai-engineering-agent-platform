"""Fail-closed durable workflow checkpoint contracts and canonical codec."""

import json
import re
from dataclasses import dataclass
from math import isfinite
from typing import Protocol, cast

from ai_engineering_agent_platform.domain.workflow import (
    WorkflowEventType,
    WorkflowExecutionEvent,
    WorkflowInput,
    WorkflowRunState,
    WorkflowRunStatus,
    WorkflowStepExecution,
    WorkflowStepState,
    WorkflowStepStatus,
)

WORKFLOW_CHECKPOINT_FORMAT_VERSION = 1
MAX_WORKFLOW_CHECKPOINT_BYTES = 1_048_576
MAX_DURABLE_VALUE_DEPTH = 12
MAX_DURABLE_COLLECTION_ITEMS = 256
MAX_DURABLE_STRING_LENGTH = 65_536
MAX_DURABLE_INTEGER_DIGITS = 128

_INTEGER_PATTERN = re.compile(r"-?(?:0|[1-9][0-9]*)")


class WorkflowPersistenceError(Exception):
    """Base error for durable workflow state operations."""


class WorkflowSerializationError(WorkflowPersistenceError):
    """Raised when durable state cannot be encoded or validated."""


class WorkflowPersistenceConflictError(WorkflowPersistenceError):
    """Raised when a durable checkpoint would violate monotonic state."""


class WorkflowPersistenceIntegrityError(WorkflowPersistenceError):
    """Raised when stored checkpoint metadata or content is inconsistent."""


@dataclass(
    frozen=True,
    slots=True,
)
class WorkflowCheckpoint:
    """Validated durable snapshot of one workflow execution attempt."""

    state: WorkflowRunState
    events: tuple[
        WorkflowExecutionEvent,
        ...,
    ]

    def __post_init__(
        self,
    ) -> None:
        """Validate event identity, ordering, and run lifecycle shape."""
        if not isinstance(
            self.state,
            WorkflowRunState,
        ):
            raise ValueError("state must be WorkflowRunState")

        if (
            not isinstance(
                self.events,
                tuple,
            )
            or not self.events
        ):
            raise ValueError("events must be a non-empty tuple")

        if any(
            not isinstance(
                event,
                WorkflowExecutionEvent,
            )
            for event in self.events
        ):
            raise ValueError("events must contain WorkflowExecutionEvent values")

        expected_sequences = tuple(
            range(
                1,
                len(self.events) + 1,
            )
        )

        if tuple(event.sequence for event in self.events) != expected_sequences:
            raise ValueError("checkpoint event sequences must be contiguous")

        for event in self.events:
            if (
                event.run_id != self.state.run_id
                or event.workflow_id != self.state.workflow_id
                or event.workflow_version != self.state.workflow_version
            ):
                raise ValueError("checkpoint event identity mismatch")

        if self.events[0].event_type is not WorkflowEventType.RUN_STARTED:
            raise ValueError("checkpoint must start with RUN_STARTED")

        terminal_types = {
            WorkflowEventType.RUN_COMPLETED,
            WorkflowEventType.RUN_FAILED,
        }

        terminal_positions = tuple(
            index
            for index, event in enumerate(self.events)
            if event.event_type in terminal_types
        )

        if self.state.status is WorkflowRunStatus.RUNNING:
            if terminal_positions:
                raise ValueError("running checkpoint must not contain terminal event")

            return

        expected_terminal = (
            WorkflowEventType.RUN_COMPLETED
            if self.state.status is WorkflowRunStatus.COMPLETED
            else WorkflowEventType.RUN_FAILED
        )

        if (
            terminal_positions != (len(self.events) - 1,)
            or self.events[-1].event_type is not expected_terminal
        ):
            raise ValueError("terminal checkpoint event does not match run status")


class WorkflowStateStore(Protocol):
    """Durable storage boundary owned outside WorkflowEngine."""

    async def save(
        self,
        checkpoint: WorkflowCheckpoint,
    ) -> None:
        """Persist one validated checkpoint atomically."""
        ...

    async def load(
        self,
        run_id: str,
    ) -> WorkflowCheckpoint | None:
        """Load one checkpoint or return None when absent."""
        ...


def _require_exact_keys(
    value: object,
    *,
    keys: set[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(
        value,
        dict,
    ):
        raise WorkflowSerializationError(f"{label} must be an object")

    if any(
        not isinstance(
            key,
            str,
        )
        for key in value
    ):
        raise WorkflowSerializationError(f"{label} keys must be strings")

    if set(value) != keys:
        raise WorkflowSerializationError(f"{label} fields differ from schema")

    return value


def _encode_value(
    value: object,
    *,
    depth: int = 0,
) -> object:
    if depth > MAX_DURABLE_VALUE_DEPTH:
        raise WorkflowSerializationError("durable value exceeds maximum depth")

    if value is None:
        return {
            "t": "null",
        }

    if isinstance(
        value,
        bool,
    ):
        return {
            "t": "bool",
            "v": value,
        }

    if type(value) is int:
        encoded = str(value)

        if len(encoded.lstrip("-")) > MAX_DURABLE_INTEGER_DIGITS:
            raise WorkflowSerializationError("durable integer exceeds digit limit")

        return {
            "t": "int",
            "v": encoded,
        }

    if type(value) is float:
        if not isfinite(value):
            raise WorkflowSerializationError("durable float must be finite")

        return {
            "t": "float",
            "v": repr(value),
        }

    if isinstance(
        value,
        str,
    ):
        if len(value) > MAX_DURABLE_STRING_LENGTH:
            raise WorkflowSerializationError("durable string exceeds length limit")

        return {
            "t": "str",
            "v": value,
        }

    if isinstance(
        value,
        tuple,
    ):
        if len(value) > MAX_DURABLE_COLLECTION_ITEMS:
            raise WorkflowSerializationError("durable tuple exceeds item limit")

        return {
            "t": "tuple",
            "v": [
                _encode_value(
                    item,
                    depth=depth + 1,
                )
                for item in value
            ],
        }

    if isinstance(
        value,
        list,
    ):
        if len(value) > MAX_DURABLE_COLLECTION_ITEMS:
            raise WorkflowSerializationError("durable list exceeds item limit")

        return {
            "t": "list",
            "v": [
                _encode_value(
                    item,
                    depth=depth + 1,
                )
                for item in value
            ],
        }

    if isinstance(
        value,
        dict,
    ):
        if len(value) > MAX_DURABLE_COLLECTION_ITEMS:
            raise WorkflowSerializationError("durable mapping exceeds item limit")

        if any(
            not isinstance(
                key,
                str,
            )
            for key in value
        ):
            raise WorkflowSerializationError("durable mapping keys must be strings")

        keys = sorted(value)

        return {
            "t": "dict",
            "v": [
                [
                    key,
                    _encode_value(
                        value[key],
                        depth=depth + 1,
                    ),
                ]
                for key in keys
            ],
        }

    raise WorkflowSerializationError("unsupported durable value type")


def _decode_value(
    raw: object,
    *,
    depth: int = 0,
) -> object:
    if depth > MAX_DURABLE_VALUE_DEPTH:
        raise WorkflowSerializationError("durable value exceeds maximum depth")

    if not isinstance(
        raw,
        dict,
    ):
        raise WorkflowSerializationError("durable value must be an object")

    tag = raw.get("t")

    if not isinstance(
        tag,
        str,
    ):
        raise WorkflowSerializationError("durable value type tag missing")

    if tag == "null":
        _require_exact_keys(
            raw,
            keys={
                "t",
            },
            label="null durable value",
        )
        return None

    data = _require_exact_keys(
        raw,
        keys={
            "t",
            "v",
        },
        label="durable value",
    )

    value = data["v"]

    if tag == "bool":
        if not isinstance(
            value,
            bool,
        ):
            raise WorkflowSerializationError("invalid durable bool")
        return value

    if tag == "int":
        if (
            not isinstance(
                value,
                str,
            )
            or not _INTEGER_PATTERN.fullmatch(value)
            or len(value.lstrip("-")) > MAX_DURABLE_INTEGER_DIGITS
        ):
            raise WorkflowSerializationError("invalid durable integer")
        return int(value)

    if tag == "float":
        if not isinstance(
            value,
            str,
        ):
            raise WorkflowSerializationError("invalid durable float")

        try:
            decoded = float(value)
        except ValueError as exc:
            raise WorkflowSerializationError("invalid durable float") from exc

        if not isfinite(decoded):
            raise WorkflowSerializationError("durable float must be finite")

        return decoded

    if tag == "str":
        if (
            not isinstance(
                value,
                str,
            )
            or len(value) > MAX_DURABLE_STRING_LENGTH
        ):
            raise WorkflowSerializationError("invalid durable string")
        return value

    if tag in {
        "tuple",
        "list",
    }:
        if (
            not isinstance(
                value,
                list,
            )
            or len(value) > MAX_DURABLE_COLLECTION_ITEMS
        ):
            raise WorkflowSerializationError("invalid durable sequence")

        decoded_items = [
            _decode_value(
                item,
                depth=depth + 1,
            )
            for item in value
        ]

        if tag == "tuple":
            return tuple(decoded_items)

        return decoded_items

    if tag == "dict":
        if (
            not isinstance(
                value,
                list,
            )
            or len(value) > MAX_DURABLE_COLLECTION_ITEMS
        ):
            raise WorkflowSerializationError("invalid durable mapping")

        result: dict[
            str,
            object,
        ] = {}

        previous: str | None = None

        for item in value:
            if (
                not isinstance(
                    item,
                    list,
                )
                or len(item) != 2
                or not isinstance(
                    item[0],
                    str,
                )
            ):
                raise WorkflowSerializationError("invalid durable mapping entry")

            key = item[0]

            if previous is not None and key <= previous:
                raise WorkflowSerializationError(
                    "durable mapping keys must be unique and sorted"
                )

            result[key] = _decode_value(
                item[1],
                depth=depth + 1,
            )

            previous = key

        return result

    raise WorkflowSerializationError("unknown durable value type tag")


def _reject_duplicate_keys(
    pairs: list[
        tuple[
            str,
            object,
        ]
    ],
) -> dict[str, object]:
    result: dict[
        str,
        object,
    ] = {}

    for key, value in pairs:
        if key in result:
            raise WorkflowSerializationError("duplicate JSON object key")

        result[key] = value

    return result


class WorkflowCheckpointCodec:
    """Canonical versioned encoding for durable workflow checkpoints."""

    def dumps(
        self,
        checkpoint: WorkflowCheckpoint,
    ) -> str:
        """Serialize one validated checkpoint into canonical JSON."""
        if not isinstance(
            checkpoint,
            WorkflowCheckpoint,
        ):
            raise WorkflowSerializationError("checkpoint must be WorkflowCheckpoint")

        state = checkpoint.state

        payload = {
            "events": [
                {
                    "event_type": event.event_type.value,
                    "executor_name": event.executor_name,
                    "run_id": event.run_id,
                    "sequence": event.sequence,
                    "step_id": event.step_id,
                    "workflow_id": event.workflow_id,
                    "workflow_version": event.workflow_version,
                }
                for event in checkpoint.events
            ],
            "format": WORKFLOW_CHECKPOINT_FORMAT_VERSION,
            "state": {
                "inputs": [
                    {
                        "name": item.name,
                        "value": _encode_value(item.value),
                    }
                    for item in state.inputs
                ],
                "run_id": state.run_id,
                "status": state.status.value,
                "steps": [
                    {
                        "execution": (
                            None
                            if step.execution is None
                            else {
                                "output": _encode_value(step.execution.output),
                            }
                        ),
                        "executor_name": step.executor_name,
                        "status": step.status.value,
                        "step_id": step.step_id,
                    }
                    for step in state.steps
                ],
                "workflow_id": state.workflow_id,
                "workflow_version": state.workflow_version,
            },
        }

        try:
            encoded = json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(
                    ",",
                    ":",
                ),
                sort_keys=True,
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise WorkflowSerializationError("checkpoint JSON encoding failed") from exc

        if len(encoded.encode("utf-8")) > MAX_WORKFLOW_CHECKPOINT_BYTES:
            raise WorkflowSerializationError("checkpoint exceeds maximum encoded size")

        return encoded

    def loads(
        self,
        payload: str,
    ) -> WorkflowCheckpoint:
        """Decode one checkpoint and revalidate all domain invariants."""
        if not isinstance(
            payload,
            str,
        ):
            raise WorkflowSerializationError("checkpoint payload must be a string")

        if len(payload.encode("utf-8")) > MAX_WORKFLOW_CHECKPOINT_BYTES:
            raise WorkflowSerializationError("checkpoint exceeds maximum encoded size")

        try:
            root = json.loads(
                payload,
                object_pairs_hook=(_reject_duplicate_keys),
            )
        except WorkflowSerializationError:
            raise
        except json.JSONDecodeError as exc:
            raise WorkflowSerializationError("checkpoint is not valid JSON") from exc

        root_data = _require_exact_keys(
            root,
            keys={
                "events",
                "format",
                "state",
            },
            label="checkpoint",
        )

        if (
            type(root_data["format"]) is not int
            or root_data["format"] != WORKFLOW_CHECKPOINT_FORMAT_VERSION
        ):
            raise WorkflowSerializationError("unsupported checkpoint format version")

        state_data = _require_exact_keys(
            root_data["state"],
            keys={
                "inputs",
                "run_id",
                "status",
                "steps",
                "workflow_id",
                "workflow_version",
            },
            label="checkpoint state",
        )

        raw_inputs = state_data["inputs"]
        raw_steps = state_data["steps"]
        raw_events = root_data["events"]

        if not isinstance(
            raw_inputs,
            list,
        ):
            raise WorkflowSerializationError("checkpoint inputs must be a list")

        if not isinstance(
            raw_steps,
            list,
        ):
            raise WorkflowSerializationError("checkpoint steps must be a list")

        if not isinstance(
            raw_events,
            list,
        ):
            raise WorkflowSerializationError("checkpoint events must be a list")

        try:
            inputs = tuple(
                WorkflowInput(
                    name=cast(str, input_data["name"]),
                    value=cast(
                        str | int | float | bool,
                        _decode_value(input_data["value"]),
                    ),
                )
                for raw_input in raw_inputs
                for input_data in (
                    _require_exact_keys(
                        raw_input,
                        keys={
                            "name",
                            "value",
                        },
                        label="checkpoint input",
                    ),
                )
            )

            steps: list[WorkflowStepState] = []

            for raw_step in raw_steps:
                step_data = _require_exact_keys(
                    raw_step,
                    keys={
                        "execution",
                        "executor_name",
                        "status",
                        "step_id",
                    },
                    label="checkpoint step",
                )

                status = WorkflowStepStatus(cast(str, step_data["status"]))

                raw_execution = step_data["execution"]

                execution = None

                if raw_execution is not None:
                    execution_data = _require_exact_keys(
                        raw_execution,
                        keys={
                            "output",
                        },
                        label="checkpoint execution",
                    )

                    execution = WorkflowStepExecution(
                        step_id=cast(str, step_data["step_id"]),
                        executor_name=cast(str, step_data["executor_name"]),
                        output=_decode_value(execution_data["output"]),
                    )

                steps.append(
                    WorkflowStepState(
                        step_id=cast(str, step_data["step_id"]),
                        executor_name=cast(str, step_data["executor_name"]),
                        status=status,
                        execution=execution,
                    )
                )

            state = WorkflowRunState(
                run_id=cast(str, state_data["run_id"]),
                workflow_id=cast(str, state_data["workflow_id"]),
                workflow_version=cast(str, state_data["workflow_version"]),
                status=WorkflowRunStatus(cast(str, state_data["status"])),
                inputs=inputs,
                steps=tuple(steps),
            )

            events = tuple(
                WorkflowExecutionEvent(
                    sequence=cast(int, event_data["sequence"]),
                    run_id=cast(str, event_data["run_id"]),
                    workflow_id=cast(str, event_data["workflow_id"]),
                    workflow_version=cast(str, event_data["workflow_version"]),
                    event_type=WorkflowEventType(cast(str, event_data["event_type"])),
                    step_id=cast(str | None, event_data["step_id"]),
                    executor_name=cast(str | None, event_data["executor_name"]),
                )
                for raw_event in raw_events
                for event_data in (
                    _require_exact_keys(
                        raw_event,
                        keys={
                            "event_type",
                            "executor_name",
                            "run_id",
                            "sequence",
                            "step_id",
                            "workflow_id",
                            "workflow_version",
                        },
                        label="checkpoint event",
                    ),
                )
            )

            return WorkflowCheckpoint(
                state=state,
                events=events,
            )
        except WorkflowSerializationError:
            raise
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise WorkflowSerializationError(
                "checkpoint violates workflow domain invariants"
            ) from exc

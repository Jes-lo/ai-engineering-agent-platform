"""Tests for privacy-safe OpenTelemetry workflow trace projection."""

from dataclasses import dataclass, field
from typing import (
    Any,
    cast,
)

from opentelemetry.trace import (
    SpanKind,
    Status,
    StatusCode,
    Tracer,
)

from ai_engineering_agent_platform.domain.workflow import (
    WorkflowEventType,
    WorkflowExecutionEvent,
    WorkflowExecutionTrace,
    WorkflowInput,
    WorkflowRunState,
    WorkflowRunStatus,
    WorkflowStepExecution,
    WorkflowStepState,
    WorkflowStepStatus,
)
from ai_engineering_agent_platform.services.workflow_opentelemetry import (
    WorkflowOpenTelemetryAdapter,
)


@dataclass
class RecordedSpan:
    """Minimal in-memory span shape used to test API projection."""

    name: str
    kind: SpanKind
    attributes: dict[
        str,
        object,
    ]
    events: list[
        tuple[
            str,
            dict[
                str,
                object,
            ],
        ]
    ] = field(default_factory=list)
    status: Status | None = None

    def add_event(
        self,
        name: str,
        attributes: Any = None,
        timestamp: int | None = None,
    ) -> None:
        del timestamp

        self.events.append(
            (
                name,
                dict(attributes or {}),
            )
        )

    def set_status(
        self,
        status: Status,
        description: str | None = None,
    ) -> None:
        del description

        self.status = status


@dataclass
class SpanContext:
    """Context manager returning one synthetic recorded span."""

    span: RecordedSpan

    def __enter__(
        self,
    ) -> RecordedSpan:
        return self.span

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        del exc_type
        del exc_value
        del traceback


@dataclass
class RecordingTracer:
    """Minimal recording tracer implementing the used OpenTelemetry surface."""

    spans: list[RecordedSpan] = field(default_factory=list)
    calls: list[
        dict[
            str,
            object,
        ]
    ] = field(default_factory=list)

    def start_as_current_span(
        self,
        name: str,
        context: object = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Any = None,
        links: object = (),
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> SpanContext:
        del context
        del links
        del start_time
        del end_on_exit

        span = RecordedSpan(
            name=name,
            kind=kind,
            attributes=dict(attributes or {}),
        )

        self.spans.append(span)

        self.calls.append(
            {
                "record_exception": (record_exception),
                "set_status_on_exception": (set_status_on_exception),
            }
        )

        return SpanContext(span)


def _event(
    *,
    sequence: int,
    event_type: WorkflowEventType,
    step_id: str | None = None,
    executor_name: str | None = None,
) -> WorkflowExecutionEvent:
    return WorkflowExecutionEvent(
        sequence=sequence,
        run_id="workflow-run:otel",
        workflow_id="otel-workflow",
        workflow_version="1",
        event_type=event_type,
        step_id=step_id,
        executor_name=executor_name,
    )


def _completed_state_and_trace() -> tuple[
    WorkflowRunState,
    WorkflowExecutionTrace,
]:
    execution = WorkflowStepExecution(
        step_id="step",
        executor_name="executor",
        output="TOP-SECRET-OUTPUT",
    )

    state = WorkflowRunState(
        run_id="workflow-run:otel",
        workflow_id="otel-workflow",
        workflow_version="1",
        status=WorkflowRunStatus.COMPLETED,
        inputs=(
            WorkflowInput(
                name="secret",
                value="TOP-SECRET-INPUT",
            ),
        ),
        steps=(
            WorkflowStepState(
                step_id="step",
                executor_name="executor",
                status=(WorkflowStepStatus.COMPLETED),
                execution=execution,
            ),
        ),
    )

    trace = WorkflowExecutionTrace(
        run_id="workflow-run:otel",
        workflow_id="otel-workflow",
        workflow_version="1",
        events=(
            _event(
                sequence=1,
                event_type=(WorkflowEventType.RUN_STARTED),
            ),
            _event(
                sequence=2,
                event_type=(WorkflowEventType.STEP_COMPLETED),
                step_id="step",
                executor_name="executor",
            ),
            _event(
                sequence=3,
                event_type=(WorkflowEventType.RUN_COMPLETED),
            ),
        ),
    )

    return (
        state,
        trace,
    )


def _failed_state_and_trace() -> tuple[
    WorkflowRunState,
    WorkflowExecutionTrace,
]:
    state = WorkflowRunState(
        run_id="workflow-run:otel",
        workflow_id="otel-workflow",
        workflow_version="1",
        status=WorkflowRunStatus.FAILED,
        inputs=(),
        steps=(
            WorkflowStepState(
                step_id="failed-step",
                executor_name="executor",
                status=(WorkflowStepStatus.FAILED),
            ),
        ),
    )

    trace = WorkflowExecutionTrace(
        run_id="workflow-run:otel",
        workflow_id="otel-workflow",
        workflow_version="1",
        events=(
            _event(
                sequence=1,
                event_type=(WorkflowEventType.RUN_STARTED),
            ),
            _event(
                sequence=2,
                event_type=(WorkflowEventType.STEP_FAILED),
                step_id="failed-step",
                executor_name="executor",
            ),
            _event(
                sequence=3,
                event_type=(WorkflowEventType.RUN_FAILED),
            ),
        ),
    )

    return (
        state,
        trace,
    )


def test_completed_trace_projects_to_one_otel_span() -> None:
    recording = RecordingTracer()

    adapter = WorkflowOpenTelemetryAdapter(
        cast(
            Tracer,
            recording,
        )
    )

    state, trace = _completed_state_and_trace()

    summary = adapter.export_terminal_trace(
        state=state,
        trace=trace,
    )

    assert summary.run_status is WorkflowRunStatus.COMPLETED

    assert len(recording.spans) == 1

    span = recording.spans[0]

    assert span.name == ("workflow.execution")

    assert span.kind is SpanKind.INTERNAL

    assert span.attributes == {
        "workflow.run.id": ("workflow-run:otel"),
        "workflow.id": ("otel-workflow"),
        "workflow.version": "1",
        "workflow.run.status": ("completed"),
        "workflow.steps.completed": 1,
        "workflow.steps.skipped": 0,
        "workflow.steps.failed": 0,
        "workflow.events.count": 3,
        "workflow.observability.mode": ("terminal_trace_projection"),
    }

    assert tuple(event[1]["workflow.event.type"] for event in span.events) == (
        "run_started",
        "step_completed",
        "run_completed",
    )

    assert span.status is not None

    assert span.status.status_code is StatusCode.OK


def test_failed_trace_sets_error_status_without_exception_text() -> None:
    recording = RecordingTracer()

    adapter = WorkflowOpenTelemetryAdapter(
        cast(
            Tracer,
            recording,
        )
    )

    state, trace = _failed_state_and_trace()

    summary = adapter.export_terminal_trace(
        state=state,
        trace=trace,
    )

    assert summary.run_status is WorkflowRunStatus.FAILED

    assert len(recording.spans) == 1

    span = recording.spans[0]

    assert span.status is not None

    assert span.status.status_code is StatusCode.ERROR

    rendered = repr(
        (
            span.attributes,
            span.events,
            span.status,
        )
    )

    assert "TOP-SECRET" not in rendered


def test_adapter_disables_automatic_exception_recording() -> None:
    recording = RecordingTracer()

    adapter = WorkflowOpenTelemetryAdapter(
        cast(
            Tracer,
            recording,
        )
    )

    state, trace = _completed_state_and_trace()

    adapter.export_terminal_trace(
        state=state,
        trace=trace,
    )

    assert recording.calls == [
        {
            "record_exception": False,
            "set_status_on_exception": False,
        }
    ]


def test_projection_contains_no_sensitive_workflow_values() -> None:
    recording = RecordingTracer()

    adapter = WorkflowOpenTelemetryAdapter(
        cast(
            Tracer,
            recording,
        )
    )

    state, trace = _completed_state_and_trace()

    adapter.export_terminal_trace(
        state=state,
        trace=trace,
    )

    rendered = repr(recording.spans)

    assert "TOP-SECRET-INPUT" not in rendered

    assert "TOP-SECRET-OUTPUT" not in rendered


def test_adapter_does_not_claim_execution_timing() -> None:
    recording = RecordingTracer()

    adapter = WorkflowOpenTelemetryAdapter(
        cast(
            Tracer,
            recording,
        )
    )

    state, trace = _completed_state_and_trace()

    adapter.export_terminal_trace(
        state=state,
        trace=trace,
    )

    assert (
        recording.spans[0].attributes["workflow.observability.mode"]
        == "terminal_trace_projection"
    )

"""OpenTelemetry API projection for terminal workflow execution traces."""

from opentelemetry.trace import (
    SpanKind,
    Status,
    StatusCode,
    Tracer,
)

from ai_engineering_agent_platform.domain.workflow import (
    WorkflowExecutionTrace,
    WorkflowRunState,
    WorkflowRunStatus,
)
from ai_engineering_agent_platform.services.workflow_observability import (
    WorkflowObservabilitySummary,
    summarize_workflow_state,
    workflow_event_attributes,
)


class WorkflowOpenTelemetryAdapter:
    """Project privacy-safe terminal trace projection into OpenTelemetry.

    The adapter consumes an already terminal workflow state and structural
    trace. It intentionally does not claim to measure workflow execution
    duration because Feature 17 events do not carry execution timestamps.

    The caller injects an OpenTelemetry API ``Tracer``. This module configures
    no SDK, exporter, collector, endpoint, credentials, or network transport.
    """

    def __init__(
        self,
        tracer: Tracer,
    ) -> None:
        """Store an externally configured OpenTelemetry API tracer."""
        if tracer is None:
            raise ValueError("tracer must not be None")

        self._tracer = tracer

    def export_terminal_trace(
        self,
        *,
        state: WorkflowRunState,
        trace: WorkflowExecutionTrace,
    ) -> WorkflowObservabilitySummary:
        """Project one terminal workflow trace without high-content payloads."""
        summary = summarize_workflow_state(
            state=state,
            trace=trace,
        )

        span_attributes: dict[
            str,
            str | int,
        ] = {
            "workflow.run.id": (summary.run_id),
            "workflow.id": (summary.workflow_id),
            "workflow.version": (summary.workflow_version),
            "workflow.run.status": (summary.run_status.value),
            "workflow.steps.completed": (summary.completed_steps),
            "workflow.steps.skipped": (summary.skipped_steps),
            "workflow.steps.failed": (summary.failed_steps),
            "workflow.events.count": (summary.event_count),
            "workflow.observability.mode": ("terminal_trace_projection"),
        }

        with self._tracer.start_as_current_span(
            "workflow.execution",
            kind=SpanKind.INTERNAL,
            attributes=span_attributes,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            for event in trace.events:
                span.add_event(
                    "workflow.event",
                    attributes=dict(workflow_event_attributes(event)),
                )

            span.set_status(
                Status(
                    StatusCode.OK
                    if (summary.run_status is WorkflowRunStatus.COMPLETED)
                    else StatusCode.ERROR
                )
            )

        return summary

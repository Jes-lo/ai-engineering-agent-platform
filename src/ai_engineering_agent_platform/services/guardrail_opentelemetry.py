"""OpenTelemetry API projection for deterministic guardrail evaluations."""

from opentelemetry.trace import (
    SpanKind,
    Status,
    StatusCode,
    Tracer,
)

from ai_engineering_agent_platform.domain.guardrails import (
    GuardrailEvaluation,
)
from ai_engineering_agent_platform.services.guardrail_observability import (
    GuardrailObservabilitySummary,
    guardrail_evaluation_attributes,
    summarize_guardrail_evaluation,
)


class GuardrailOpenTelemetryAdapter:
    """Project low-content guardrail decisions into OpenTelemetry.

    The adapter configures no SDK, exporter, Collector, endpoint, credentials,
    network transport, automatic exception recording, or telemetry backend.
    """

    def __init__(
        self,
        tracer: Tracer,
    ) -> None:
        """Store an externally configured OpenTelemetry API tracer."""
        if tracer is None:
            raise ValueError("tracer must not be None")

        self._tracer = tracer

    def export_evaluation(
        self,
        evaluation: GuardrailEvaluation,
    ) -> GuardrailObservabilitySummary:
        """Project one completed guardrail decision without content payloads."""
        summary = summarize_guardrail_evaluation(evaluation)

        attributes = dict(guardrail_evaluation_attributes(evaluation))

        with self._tracer.start_as_current_span(
            "guardrail.evaluation",
            kind=SpanKind.INTERNAL,
            attributes=attributes,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            # A BLOCK is a successful policy decision, not an instrumentation
            # failure. The action remains explicit in the span attributes.
            span.set_status(Status(StatusCode.OK))

        return summary

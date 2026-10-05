"""Tests for privacy-safe OpenTelemetry guardrail projection."""

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

from ai_engineering_agent_platform.domain.guardrails import (
    GuardrailAction,
    GuardrailCategory,
    GuardrailEvaluation,
    GuardrailFinding,
    GuardrailSeverity,
    GuardrailStage,
)
from ai_engineering_agent_platform.services.guardrail_opentelemetry import (
    GuardrailOpenTelemetryAdapter,
)


@dataclass
class RecordedSpan:
    """Minimal in-memory span used to inspect projection behavior."""

    name: str
    kind: SpanKind
    attributes: dict[
        str,
        object,
    ]
    status: Status | None = None

    def set_status(
        self,
        status: Status,
        description: str | None = None,
    ) -> None:
        del description
        self.status = status


@dataclass
class SpanContext:
    """Context manager returning one recorded synthetic span."""

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
    """Minimal tracer implementing the OpenTelemetry surface used here."""

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


def _evaluation() -> GuardrailEvaluation:
    """Return a blocking decision with sensitive-looking internals."""
    return GuardrailEvaluation(
        content_id="TOP-SECRET-CONTENT-ID",
        stage=(GuardrailStage.RETRIEVED_CONTEXT),
        action=GuardrailAction.BLOCK,
        findings=(
            GuardrailFinding(
                rule_id="TOP-SECRET-RULE-ID",
                category=(GuardrailCategory.PROMPT_INJECTION_SIGNAL),
                severity=(GuardrailSeverity.CRITICAL),
                message=("TOP-SECRET-FINDING-MESSAGE"),
                start_offset=1,
                end_offset=7,
            ),
        ),
        evaluated_rule_ids=("TOP-SECRET-RULE-ID",),
    )


def test_guardrail_projection_creates_one_low_content_span() -> None:
    """One evaluation should become one structural internal span."""
    recording = RecordingTracer()

    adapter = GuardrailOpenTelemetryAdapter(
        cast(
            Tracer,
            recording,
        )
    )

    summary = adapter.export_evaluation(_evaluation())

    assert summary.action is (GuardrailAction.BLOCK)

    assert len(recording.spans) == 1

    span = recording.spans[0]

    assert span.name == "guardrail.evaluation"
    assert span.kind is SpanKind.INTERNAL

    assert span.attributes["guardrail.stage"] == (
        GuardrailStage.RETRIEVED_CONTEXT.value
    )

    assert span.attributes["guardrail.action"] == GuardrailAction.BLOCK.value

    assert span.attributes["guardrail.findings.max_severity"] == "critical"

    assert span.status is not None
    assert span.status.status_code is (StatusCode.OK)


def test_block_is_policy_success_not_telemetry_error() -> None:
    """A BLOCK decision should remain explicit without error status."""
    recording = RecordingTracer()

    GuardrailOpenTelemetryAdapter(
        cast(
            Tracer,
            recording,
        )
    ).export_evaluation(_evaluation())

    assert recording.spans[0].status is not None
    assert recording.spans[0].status.status_code is StatusCode.OK

    assert recording.spans[0].attributes["guardrail.action"] == "block"


def test_adapter_disables_automatic_exception_recording() -> None:
    """The projection must not ask OTel to capture exception payloads."""
    recording = RecordingTracer()

    GuardrailOpenTelemetryAdapter(
        cast(
            Tracer,
            recording,
        )
    ).export_evaluation(_evaluation())

    assert recording.calls == [
        {
            "record_exception": False,
            "set_status_on_exception": False,
        }
    ]


def test_otel_projection_contains_no_sensitive_guardrail_values() -> None:
    """Configured identifiers and finding text must never be exported."""
    recording = RecordingTracer()

    GuardrailOpenTelemetryAdapter(
        cast(
            Tracer,
            recording,
        )
    ).export_evaluation(_evaluation())

    rendered = repr(recording.spans[0].attributes)

    for secret in (
        "TOP-SECRET-CONTENT-ID",
        "TOP-SECRET-RULE-ID",
        "TOP-SECRET-FINDING-MESSAGE",
    ):
        assert secret not in rendered

    for forbidden_key in (
        "guardrail.content_id",
        "guardrail.rule_id",
        "guardrail.message",
        "guardrail.offset",
        "guardrail.prompt",
        "guardrail.tool_output",
        "guardrail.model_output",
        "guardrail.retrieved_content",
    ):
        assert forbidden_key not in recording.spans[0].attributes

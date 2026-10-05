"""Tests for low-content deterministic guardrail observability."""

import pytest

from ai_engineering_agent_platform.domain.guardrails import (
    GuardrailAction,
    GuardrailCategory,
    GuardrailEvaluation,
    GuardrailFinding,
    GuardrailSeverity,
    GuardrailStage,
)
from ai_engineering_agent_platform.services.guardrail_observability import (
    guardrail_evaluation_attributes,
    summarize_guardrail_evaluation,
)


def _blocked_evaluation() -> GuardrailEvaluation:
    """Return an evaluation containing deliberately sensitive metadata."""
    return GuardrailEvaluation(
        content_id="TOP-SECRET-CONTENT-ID",
        stage=GuardrailStage.TOOL_RESULT,
        action=GuardrailAction.BLOCK,
        findings=(
            GuardrailFinding(
                rule_id="TOP-SECRET-RULE-ID",
                category=(GuardrailCategory.PROMPT_INJECTION_SIGNAL),
                severity=(GuardrailSeverity.HIGH),
                message=("TOP-SECRET-FINDING-MESSAGE"),
                start_offset=3,
                end_offset=9,
            ),
        ),
        evaluated_rule_ids=("TOP-SECRET-RULE-ID",),
    )


def test_guardrail_observability_summary_is_low_content() -> None:
    """Summary should contain only structural policy-decision metadata."""
    evaluation = _blocked_evaluation()

    summary = summarize_guardrail_evaluation(evaluation)

    assert summary.stage is (GuardrailStage.TOOL_RESULT)
    assert summary.action is (GuardrailAction.BLOCK)
    assert summary.evaluated_rule_count == 1
    assert summary.finding_count == 1
    assert summary.finding_categories == (GuardrailCategory.PROMPT_INJECTION_SIGNAL,)
    assert summary.max_severity is (GuardrailSeverity.HIGH)

    rendered = repr(summary)

    for secret in (
        "TOP-SECRET-CONTENT-ID",
        "TOP-SECRET-RULE-ID",
        "TOP-SECRET-FINDING-MESSAGE",
    ):
        assert secret not in rendered


def test_guardrail_attributes_are_strictly_allowlisted() -> None:
    """Telemetry attributes must exclude content and configured identifiers."""
    attributes = dict(guardrail_evaluation_attributes(_blocked_evaluation()))

    assert attributes == {
        "guardrail.stage": (GuardrailStage.TOOL_RESULT.value),
        "guardrail.action": (GuardrailAction.BLOCK.value),
        "guardrail.rules.evaluated.count": 1,
        "guardrail.findings.count": 1,
        "guardrail.observability.mode": ("low_content_projection"),
        "guardrail.findings.categories": (
            GuardrailCategory.PROMPT_INJECTION_SIGNAL.value,
        ),
        "guardrail.findings.max_severity": ("high"),
    }

    rendered = repr(attributes)

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
        assert forbidden_key not in attributes


def test_allow_projection_omits_empty_finding_dimensions() -> None:
    """An allow decision should not fabricate finding metadata."""
    evaluation = GuardrailEvaluation(
        content_id="secret-id",
        stage=GuardrailStage.USER_INPUT,
        action=GuardrailAction.ALLOW,
        findings=(),
        evaluated_rule_ids=("secret-policy-rule",),
    )

    attributes = dict(guardrail_evaluation_attributes(evaluation))

    assert attributes["guardrail.action"] == GuardrailAction.ALLOW.value

    assert attributes["guardrail.findings.count"] == 0

    assert "guardrail.findings.categories" not in attributes

    assert "guardrail.findings.max_severity" not in attributes

    assert "secret-id" not in repr(attributes)
    assert "secret-policy-rule" not in repr(attributes)


def test_observability_rejects_non_evaluation_values() -> None:
    """Projection must reject arbitrary foreign values."""
    with pytest.raises(
        TypeError,
        match="GuardrailEvaluation",
    ):
        summarize_guardrail_evaluation(
            object()  # type: ignore[arg-type]
        )

"""Tests for provider-neutral deterministic guardrail domain values."""

from __future__ import annotations

import pytest

from ai_engineering_agent_platform.domain.guardrails import (
    GuardrailAction,
    GuardrailCategory,
    GuardrailEvaluation,
    GuardrailFinding,
    GuardrailSeverity,
    GuardrailStage,
    GuardrailSubject,
)


def test_subject_preserves_transient_content_and_identity() -> None:
    """Subjects should preserve exact transient runtime content."""
    subject = GuardrailSubject(
        content_id="request-1",
        stage=GuardrailStage.USER_INPUT,
        content="exact user text",
    )

    assert subject.content_id == "request-1"
    assert subject.stage is GuardrailStage.USER_INPUT
    assert subject.content == "exact user text"


def test_subject_rejects_empty_content_identity() -> None:
    """Content identifiers must remain explicit and non-empty."""
    with pytest.raises(
        ValueError,
        match="content_id must not be empty",
    ):
        GuardrailSubject(
            content_id="   ",
            stage=GuardrailStage.USER_INPUT,
            content="text",
        )


def test_subject_allows_empty_content_for_boundary_evaluation() -> None:
    """Empty runtime text may still pass through explicit evaluation."""
    subject = GuardrailSubject(
        content_id="empty-1",
        stage=GuardrailStage.MODEL_OUTPUT,
        content="",
    )

    assert subject.content == ""


def test_finding_requires_complete_offset_pair() -> None:
    """Partial match offsets must fail closed."""
    with pytest.raises(
        ValueError,
        match="must be supplied together",
    ):
        GuardrailFinding(
            rule_id="rule-1",
            category=GuardrailCategory.POLICY_VIOLATION,
            severity=GuardrailSeverity.HIGH,
            message="blocked signal",
            start_offset=0,
        )


def test_finding_rejects_invalid_offset_order() -> None:
    """A finding cannot describe an empty or reversed range."""
    with pytest.raises(
        ValueError,
        match="greater than start_offset",
    ):
        GuardrailFinding(
            rule_id="rule-1",
            category=GuardrailCategory.POLICY_VIOLATION,
            severity=GuardrailSeverity.HIGH,
            message="blocked signal",
            start_offset=3,
            end_offset=3,
        )


def test_evaluation_does_not_store_subject_content() -> None:
    """Evaluation results must not echo raw guarded content."""
    evaluation = GuardrailEvaluation(
        content_id="request-2",
        stage=GuardrailStage.RETRIEVED_CONTEXT,
        action=GuardrailAction.ALLOW,
        findings=(),
        evaluated_rule_ids=("rule-1",),
    )

    assert not hasattr(evaluation, "content")
    assert evaluation.allowed is True


def test_block_evaluation_requires_a_finding() -> None:
    """A blocking decision cannot exist without deterministic evidence."""
    with pytest.raises(
        ValueError,
        match="must contain at least one finding",
    ):
        GuardrailEvaluation(
            content_id="request-3",
            stage=GuardrailStage.TOOL_RESULT,
            action=GuardrailAction.BLOCK,
            findings=(),
            evaluated_rule_ids=("rule-1",),
        )


def test_evaluation_rejects_duplicate_rule_ids() -> None:
    """Audited rule execution identities must remain unique."""
    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        GuardrailEvaluation(
            content_id="request-4",
            stage=GuardrailStage.MODEL_OUTPUT,
            action=GuardrailAction.ALLOW,
            findings=(),
            evaluated_rule_ids=("rule-1", "rule-1"),
        )

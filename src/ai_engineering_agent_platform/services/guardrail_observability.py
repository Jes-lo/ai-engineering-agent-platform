"""Privacy-safe projections for deterministic guardrail evaluations."""

from dataclasses import dataclass

from ai_engineering_agent_platform.domain.guardrails import (
    GuardrailAction,
    GuardrailCategory,
    GuardrailEvaluation,
    GuardrailSeverity,
    GuardrailStage,
)

type GuardrailAttributeValue = (
    str
    | int
    | tuple[
        str,
        ...,
    ]
)


@dataclass(
    frozen=True,
    slots=True,
)
class GuardrailObservabilitySummary:
    """Low-content structural summary suitable for operational telemetry."""

    stage: GuardrailStage
    action: GuardrailAction
    evaluated_rule_count: int
    finding_count: int
    finding_categories: tuple[
        GuardrailCategory,
        ...,
    ]
    max_severity: GuardrailSeverity | None


def summarize_guardrail_evaluation(
    evaluation: GuardrailEvaluation,
) -> GuardrailObservabilitySummary:
    """Summarize one evaluation without content or configured identifiers."""
    if not isinstance(
        evaluation,
        GuardrailEvaluation,
    ):
        raise TypeError("evaluation must be a GuardrailEvaluation")

    categories = tuple(
        sorted(
            {finding.category for finding in evaluation.findings},
            key=lambda item: item.value,
        )
    )

    max_severity = max(
        (finding.severity for finding in evaluation.findings),
        default=None,
    )

    return GuardrailObservabilitySummary(
        stage=evaluation.stage,
        action=evaluation.action,
        evaluated_rule_count=len(evaluation.evaluated_rule_ids),
        finding_count=len(evaluation.findings),
        finding_categories=categories,
        max_severity=max_severity,
    )


def guardrail_evaluation_attributes(
    evaluation: GuardrailEvaluation,
) -> tuple[
    tuple[
        str,
        GuardrailAttributeValue,
    ],
    ...,
]:
    """Return a strict allowlist of low-content telemetry attributes.

    Content identifiers, rule identifiers, finding messages, match offsets,
    prompts, retrieved evidence, tool output, and model output are deliberately
    excluded.
    """
    summary = summarize_guardrail_evaluation(evaluation)

    attributes: list[
        tuple[
            str,
            GuardrailAttributeValue,
        ]
    ] = [
        (
            "guardrail.stage",
            summary.stage.value,
        ),
        (
            "guardrail.action",
            summary.action.value,
        ),
        (
            "guardrail.rules.evaluated.count",
            summary.evaluated_rule_count,
        ),
        (
            "guardrail.findings.count",
            summary.finding_count,
        ),
        (
            "guardrail.observability.mode",
            "low_content_projection",
        ),
    ]

    if summary.finding_categories:
        attributes.append(
            (
                "guardrail.findings.categories",
                tuple(item.value for item in summary.finding_categories),
            )
        )

    if summary.max_severity is not None:
        attributes.append(
            (
                "guardrail.findings.max_severity",
                summary.max_severity.name.lower(),
            )
        )

    return tuple(attributes)

"""Deterministic runtime guardrail evaluation and enforcement primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ai_engineering_agent_platform.domain.guardrails import (
    GuardrailAction,
    GuardrailCategory,
    GuardrailEvaluation,
    GuardrailFinding,
    GuardrailSeverity,
    GuardrailStage,
    GuardrailSubject,
)

_MAX_RULE_ID_LENGTH = 256
_MAX_LITERAL_LENGTH = 4096
_MAX_MESSAGE_LENGTH = 1024


class GuardrailError(RuntimeError):
    """Base error for deterministic runtime guardrail enforcement."""


class GuardrailConfigurationError(GuardrailError):
    """Guardrail configuration is incomplete or contradictory."""


class GuardrailStageError(GuardrailError):
    """A subject reached a stage that policy did not enable."""


class GuardrailInputLimitError(GuardrailError):
    """Guardrail input exceeded the configured deterministic bound."""


class GuardrailRuleExecutionError(GuardrailError):
    """A configured rule failed instead of producing a valid decision."""


class GuardrailRuleContractError(GuardrailError):
    """A rule violated the platform-owned guardrail contract."""


def _require_non_empty_string(
    value: object,
    *,
    field_name: str,
    max_length: int,
) -> str:
    """Return one validated bounded string."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    normalized = value.strip()

    if not normalized:
        raise ValueError(f"{field_name} must not be empty")

    if len(normalized) > max_length:
        raise ValueError(f"{field_name} must not exceed {max_length} characters")

    return normalized


class GuardrailRule(Protocol):
    """Protocol implemented by deterministic project-owned guardrail rules."""

    @property
    def rule_id(self) -> str:
        """Return the stable platform-owned rule identifier."""
        ...

    @property
    def stages(self) -> tuple[GuardrailStage, ...]:
        """Return stages at which the rule is valid."""
        ...

    def evaluate(
        self,
        subject: GuardrailSubject,
    ) -> tuple[GuardrailFinding, ...]:
        """Evaluate one transient subject without performing side effects."""
        ...


@dataclass(frozen=True, slots=True)
class GuardrailPolicy:
    """Platform-owned bounds and blocking threshold for guardrail execution."""

    enabled_stages: tuple[GuardrailStage, ...]
    block_at_or_above: GuardrailSeverity = GuardrailSeverity.HIGH
    max_content_chars: int = 100_000
    max_findings: int = 64

    def __post_init__(self) -> None:
        """Validate deterministic safety-policy bounds."""
        if not isinstance(self.enabled_stages, tuple):
            raise TypeError("enabled_stages must be a tuple")

        if not self.enabled_stages:
            raise ValueError("enabled_stages must not be empty")

        if not all(isinstance(stage, GuardrailStage) for stage in self.enabled_stages):
            raise TypeError("enabled_stages must contain GuardrailStage values")

        if len(self.enabled_stages) != len(set(self.enabled_stages)):
            raise ValueError("enabled_stages must be unique")

        if not isinstance(
            self.block_at_or_above,
            GuardrailSeverity,
        ):
            raise TypeError("block_at_or_above must be a GuardrailSeverity")

        if isinstance(self.max_content_chars, bool) or not isinstance(
            self.max_content_chars, int
        ):
            raise TypeError("max_content_chars must be an integer")

        if self.max_content_chars < 1:
            raise ValueError("max_content_chars must be positive")

        if isinstance(self.max_findings, bool) or not isinstance(
            self.max_findings, int
        ):
            raise TypeError("max_findings must be an integer")

        if self.max_findings < 1:
            raise ValueError("max_findings must be positive")


@dataclass(frozen=True, slots=True)
class GuardrailLiteralPattern:
    """One explicit deterministic literal signal owned by platform policy."""

    literal: str
    category: GuardrailCategory
    severity: GuardrailSeverity
    message: str

    def __post_init__(self) -> None:
        """Validate one bounded literal signal definition."""
        object.__setattr__(
            self,
            "literal",
            _require_non_empty_string(
                self.literal,
                field_name="literal",
                max_length=_MAX_LITERAL_LENGTH,
            ),
        )
        object.__setattr__(
            self,
            "message",
            _require_non_empty_string(
                self.message,
                field_name="message",
                max_length=_MAX_MESSAGE_LENGTH,
            ),
        )

        if not isinstance(self.category, GuardrailCategory):
            raise TypeError("category must be a GuardrailCategory")

        if not isinstance(self.severity, GuardrailSeverity):
            raise TypeError("severity must be a GuardrailSeverity")


@dataclass(frozen=True, slots=True)
class LiteralPatternGuardrailRule:
    """Match explicit literals without claiming semantic threat detection."""

    rule_id: str
    stages: tuple[GuardrailStage, ...]
    patterns: tuple[GuardrailLiteralPattern, ...]
    case_sensitive: bool = False

    def __post_init__(self) -> None:
        """Validate exact rule ownership and deterministic pattern coverage."""
        object.__setattr__(
            self,
            "rule_id",
            _require_non_empty_string(
                self.rule_id,
                field_name="rule_id",
                max_length=_MAX_RULE_ID_LENGTH,
            ),
        )

        if not isinstance(self.stages, tuple):
            raise TypeError("stages must be a tuple")

        if not self.stages:
            raise ValueError("stages must not be empty")

        if not all(isinstance(stage, GuardrailStage) for stage in self.stages):
            raise TypeError("stages must contain GuardrailStage values")

        if len(self.stages) != len(set(self.stages)):
            raise ValueError("stages must be unique")

        if not isinstance(self.patterns, tuple):
            raise TypeError("patterns must be a tuple")

        if not self.patterns:
            raise ValueError("patterns must not be empty")

        if not all(
            isinstance(pattern, GuardrailLiteralPattern) for pattern in self.patterns
        ):
            raise TypeError("patterns must contain GuardrailLiteralPattern values")

        if not isinstance(self.case_sensitive, bool):
            raise TypeError("case_sensitive must be a boolean")

        marker_keys = tuple(
            pattern.literal if self.case_sensitive else pattern.literal.casefold()
            for pattern in self.patterns
        )

        if len(marker_keys) != len(set(marker_keys)):
            raise ValueError("patterns must contain unique literals")

    def evaluate(
        self,
        subject: GuardrailSubject,
    ) -> tuple[GuardrailFinding, ...]:
        """Return deterministic literal signals for one subject."""
        if subject.stage not in self.stages:
            return ()

        findings: list[GuardrailFinding] = []

        for pattern in self.patterns:
            if self.case_sensitive:
                start = subject.content.find(pattern.literal)

                if start < 0:
                    continue

                findings.append(
                    GuardrailFinding(
                        rule_id=self.rule_id,
                        category=pattern.category,
                        severity=pattern.severity,
                        message=pattern.message,
                        start_offset=start,
                        end_offset=start + len(pattern.literal),
                    )
                )
                continue

            if pattern.literal.casefold() not in subject.content.casefold():
                continue

            findings.append(
                GuardrailFinding(
                    rule_id=self.rule_id,
                    category=pattern.category,
                    severity=pattern.severity,
                    message=pattern.message,
                )
            )

        return tuple(findings)


class GuardrailService:
    """Evaluate deterministic guardrails and fail closed on uncertainty."""

    def __init__(
        self,
        *,
        policy: GuardrailPolicy,
        rules: tuple[GuardrailRule, ...],
    ) -> None:
        """Validate complete explicit rule coverage for enabled stages."""
        if not isinstance(policy, GuardrailPolicy):
            raise TypeError("policy must be a GuardrailPolicy")

        if not isinstance(rules, tuple):
            raise TypeError("rules must be a tuple")

        if not rules:
            raise GuardrailConfigurationError("at least one guardrail rule is required")

        rule_ids: list[str] = []

        for rule in rules:
            try:
                rule_id = rule.rule_id
                stages = rule.stages
            except (AttributeError, TypeError) as exc:
                raise GuardrailConfigurationError(
                    "rules must implement GuardrailRule"
                ) from exc

            normalized_rule_id = _require_non_empty_string(
                rule_id,
                field_name="rule_id",
                max_length=_MAX_RULE_ID_LENGTH,
            )

            if not isinstance(stages, tuple):
                raise GuardrailConfigurationError(
                    f"rule {normalized_rule_id!r} stages must be a tuple"
                )

            if not stages:
                raise GuardrailConfigurationError(
                    f"rule {normalized_rule_id!r} has no stages"
                )

            if not all(isinstance(stage, GuardrailStage) for stage in stages):
                raise GuardrailConfigurationError(
                    f"rule {normalized_rule_id!r} has invalid stages"
                )

            if len(stages) != len(set(stages)):
                raise GuardrailConfigurationError(
                    f"rule {normalized_rule_id!r} has duplicate stages"
                )

            unknown_stages = tuple(
                stage for stage in stages if stage not in policy.enabled_stages
            )

            if unknown_stages:
                raise GuardrailConfigurationError(
                    f"rule {normalized_rule_id!r} targets a policy-disabled stage"
                )

            rule_ids.append(normalized_rule_id)

        if len(rule_ids) != len(set(rule_ids)):
            raise GuardrailConfigurationError(
                "guardrail rule identifiers must be unique"
            )

        for stage in policy.enabled_stages:
            if not any(stage in rule.stages for rule in rules):
                raise GuardrailConfigurationError(
                    f"enabled stage {stage.value!r} has no rule coverage"
                )

        self._policy = policy
        self._rules = rules

    @property
    def policy(self) -> GuardrailPolicy:
        """Return immutable platform-owned guardrail policy."""
        return self._policy

    @property
    def rules(self) -> tuple[GuardrailRule, ...]:
        """Return configured deterministic rules."""
        return self._rules

    def evaluate(
        self,
        subject: GuardrailSubject,
    ) -> GuardrailEvaluation:
        """Evaluate one subject and return content-free enforcement state."""
        if not isinstance(subject, GuardrailSubject):
            raise TypeError("subject must be a GuardrailSubject")

        if subject.stage not in self._policy.enabled_stages:
            raise GuardrailStageError(f"stage {subject.stage.value!r} is not enabled")

        if len(subject.content) > self._policy.max_content_chars:
            raise GuardrailInputLimitError(
                "guardrail content exceeds configured character limit"
            )

        findings: list[GuardrailFinding] = []
        evaluated_rule_ids: list[str] = []

        for rule in self._rules:
            if subject.stage not in rule.stages:
                continue

            evaluated_rule_ids.append(rule.rule_id)

            try:
                produced = rule.evaluate(subject)
            except Exception as exc:
                raise GuardrailRuleExecutionError(
                    f"guardrail rule {rule.rule_id!r} failed"
                ) from exc

            if not isinstance(produced, tuple):
                raise GuardrailRuleContractError(
                    f"guardrail rule {rule.rule_id!r} must return a tuple"
                )

            for finding in produced:
                if not isinstance(finding, GuardrailFinding):
                    raise GuardrailRuleContractError(
                        f"guardrail rule {rule.rule_id!r} returned an invalid finding"
                    )

                if finding.rule_id != rule.rule_id:
                    raise GuardrailRuleContractError(
                        f"guardrail rule {rule.rule_id!r} returned a foreign rule_id"
                    )

                if finding.end_offset is not None and finding.end_offset > len(
                    subject.content
                ):
                    raise GuardrailRuleContractError(
                        f"guardrail rule {rule.rule_id!r} "
                        "returned an out-of-range offset"
                    )

                findings.append(finding)

                if len(findings) > self._policy.max_findings:
                    raise GuardrailRuleContractError(
                        "guardrail findings exceed configured limit"
                    )

        if not evaluated_rule_ids:
            raise GuardrailConfigurationError(
                f"stage {subject.stage.value!r} has no executable rule"
            )

        action = (
            GuardrailAction.BLOCK
            if any(
                finding.severity >= self._policy.block_at_or_above
                for finding in findings
            )
            else GuardrailAction.ALLOW
        )

        return GuardrailEvaluation(
            content_id=subject.content_id,
            stage=subject.stage,
            action=action,
            findings=tuple(findings),
            evaluated_rule_ids=tuple(evaluated_rule_ids),
        )

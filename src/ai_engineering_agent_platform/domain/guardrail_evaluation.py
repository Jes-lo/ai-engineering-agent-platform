"""Deterministic adversarial evaluation primitives for runtime guardrails."""

from dataclasses import dataclass
from math import isclose, isfinite

from ai_engineering_agent_platform.domain.guardrails import (
    GuardrailAction,
    GuardrailStage,
    GuardrailSubject,
)


def _require_non_empty_string(
    field_name: str,
    value: object,
) -> None:
    """Require one meaningful string."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _require_non_negative_integer(
    field_name: str,
    value: object,
) -> None:
    """Require one non-negative integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _require_unit_rate(
    field_name: str,
    value: object,
) -> None:
    """Require one finite normalized rate."""
    if isinstance(value, bool) or not isinstance(
        value,
        (
            int,
            float,
        ),
    ):
        raise ValueError(f"{field_name} must be numeric")

    if not isfinite(value):
        raise ValueError(f"{field_name} must be finite")

    if value < 0.0 or value > 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")


@dataclass(
    frozen=True,
    slots=True,
)
class GuardrailAdversarialCase:
    """One explicit runtime-guardrail expectation."""

    case_id: str
    subject: GuardrailSubject
    expected_action: GuardrailAction

    def __post_init__(self) -> None:
        """Validate case identity and expected decision."""
        _require_non_empty_string(
            "case_id",
            self.case_id,
        )

        if not isinstance(
            self.subject,
            GuardrailSubject,
        ):
            raise TypeError("subject must be a GuardrailSubject")

        if not isinstance(
            self.expected_action,
            GuardrailAction,
        ):
            raise TypeError("expected_action must be a GuardrailAction")


@dataclass(
    frozen=True,
    slots=True,
)
class GuardrailAdversarialDataset:
    """Versioned deterministic adversarial cases with provenance."""

    dataset_id: str
    version: str
    provenance_ref: str
    cases: tuple[
        GuardrailAdversarialCase,
        ...,
    ]

    def __post_init__(self) -> None:
        """Validate dataset identity and stable case uniqueness."""
        _require_non_empty_string(
            "dataset_id",
            self.dataset_id,
        )
        _require_non_empty_string(
            "version",
            self.version,
        )
        _require_non_empty_string(
            "provenance_ref",
            self.provenance_ref,
        )

        if not isinstance(
            self.cases,
            tuple,
        ):
            raise TypeError("cases must be a tuple")

        if not self.cases:
            raise ValueError("adversarial dataset must contain at least one case")

        if any(
            not isinstance(
                case,
                GuardrailAdversarialCase,
            )
            for case in self.cases
        ):
            raise TypeError("cases must contain GuardrailAdversarialCase values")

        case_ids = tuple(case.case_id for case in self.cases)

        if len(case_ids) != len(set(case_ids)):
            raise ValueError("adversarial case identifiers must be unique")


@dataclass(
    frozen=True,
    slots=True,
)
class GuardrailAdversarialCaseResult:
    """Low-content deterministic result for one adversarial case."""

    case_id: str
    stage: GuardrailStage
    expected_action: GuardrailAction
    actual_action: GuardrailAction
    action_correct: bool

    def __post_init__(self) -> None:
        """Validate structural result fields."""
        _require_non_empty_string(
            "case_id",
            self.case_id,
        )

        if not isinstance(
            self.stage,
            GuardrailStage,
        ):
            raise TypeError("stage must be a GuardrailStage")

        if not isinstance(
            self.expected_action,
            GuardrailAction,
        ):
            raise TypeError("expected_action must be a GuardrailAction")

        if not isinstance(
            self.actual_action,
            GuardrailAction,
        ):
            raise TypeError("actual_action must be a GuardrailAction")

        if not isinstance(
            self.action_correct,
            bool,
        ):
            raise TypeError("action_correct must be a boolean")

        if self.action_correct is not (self.actual_action is self.expected_action):
            raise ValueError("action_correct does not match expected/actual action")


@dataclass(
    frozen=True,
    slots=True,
)
class GuardrailAdversarialSummary:
    """Transparent aggregate detection metrics for one dataset run."""

    case_count: int
    expected_block_count: int
    expected_allow_count: int
    correctly_blocked_count: int
    correctly_allowed_count: int
    missed_block_count: int
    unexpected_block_count: int
    action_accuracy: float
    expected_block_recall: float
    expected_allow_rate: float

    def __post_init__(self) -> None:
        """Validate count relationships and normalized metrics."""
        if (
            isinstance(
                self.case_count,
                bool,
            )
            or not isinstance(
                self.case_count,
                int,
            )
            or self.case_count <= 0
        ):
            raise ValueError("case_count must be a positive integer")

        for field_name, value in (
            (
                "expected_block_count",
                self.expected_block_count,
            ),
            (
                "expected_allow_count",
                self.expected_allow_count,
            ),
            (
                "correctly_blocked_count",
                self.correctly_blocked_count,
            ),
            (
                "correctly_allowed_count",
                self.correctly_allowed_count,
            ),
            (
                "missed_block_count",
                self.missed_block_count,
            ),
            (
                "unexpected_block_count",
                self.unexpected_block_count,
            ),
        ):
            _require_non_negative_integer(
                field_name,
                value,
            )

        for rate_name, rate_value in (
            (
                "action_accuracy",
                self.action_accuracy,
            ),
            (
                "expected_block_recall",
                self.expected_block_recall,
            ),
            (
                "expected_allow_rate",
                self.expected_allow_rate,
            ),
        ):
            _require_unit_rate(
                rate_name,
                rate_value,
            )

        if self.expected_block_count + self.expected_allow_count != self.case_count:
            raise ValueError("expected action counts must equal case_count")

        if (
            self.correctly_blocked_count + self.missed_block_count
            != self.expected_block_count
        ):
            raise ValueError("block outcome counts are inconsistent")

        if (
            self.correctly_allowed_count + self.unexpected_block_count
            != self.expected_allow_count
        ):
            raise ValueError("allow outcome counts are inconsistent")

        expected_accuracy = (
            self.correctly_blocked_count + self.correctly_allowed_count
        ) / self.case_count

        expected_block_recall = (
            1.0
            if self.expected_block_count == 0
            else (self.correctly_blocked_count / self.expected_block_count)
        )

        expected_allow_rate = (
            1.0
            if self.expected_allow_count == 0
            else (self.correctly_allowed_count / self.expected_allow_count)
        )

        for field_name, actual, expected in (
            (
                "action_accuracy",
                self.action_accuracy,
                expected_accuracy,
            ),
            (
                "expected_block_recall",
                self.expected_block_recall,
                expected_block_recall,
            ),
            (
                "expected_allow_rate",
                self.expected_allow_rate,
                expected_allow_rate,
            ),
        ):
            if not isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{field_name} is inconsistent with counts")

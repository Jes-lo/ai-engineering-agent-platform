"""Provider-neutral domain values for deterministic runtime guardrails."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum

_MAX_IDENTIFIER_LENGTH = 256
_MAX_MESSAGE_LENGTH = 1024


def _require_non_empty_string(
    value: object,
    *,
    field_name: str,
    max_length: int,
) -> str:
    """Return one validated non-empty bounded string."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    normalized = value.strip()

    if not normalized:
        raise ValueError(f"{field_name} must not be empty")

    if len(normalized) > max_length:
        raise ValueError(f"{field_name} must not exceed {max_length} characters")

    return normalized


class GuardrailStage(StrEnum):
    """Runtime boundaries at which deterministic guardrails may execute."""

    USER_INPUT = "user_input"
    RETRIEVED_CONTEXT = "retrieved_context"
    TOOL_RESULT = "tool_result"
    MODEL_OUTPUT = "model_output"


class GuardrailCategory(StrEnum):
    """Policy-owned categories for deterministic safety findings."""

    PROMPT_INJECTION_SIGNAL = "prompt_injection_signal"
    SENSITIVE_DATA_SIGNAL = "sensitive_data_signal"
    UNSAFE_CONTENT_SIGNAL = "unsafe_content_signal"
    POLICY_VIOLATION = "policy_violation"
    OTHER = "other"


class GuardrailSeverity(IntEnum):
    """Ordered severity used by deterministic enforcement policy."""

    INFO = 10
    LOW = 20
    MEDIUM = 30
    HIGH = 40
    CRITICAL = 50


class GuardrailAction(StrEnum):
    """Current enforcement actions supported by the guardrail core."""

    ALLOW = "allow"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class GuardrailSubject:
    """Transient content presented to a deterministic guardrail boundary."""

    content_id: str
    stage: GuardrailStage
    content: str

    def __post_init__(self) -> None:
        """Validate subject identity and runtime value types."""
        object.__setattr__(
            self,
            "content_id",
            _require_non_empty_string(
                self.content_id,
                field_name="content_id",
                max_length=_MAX_IDENTIFIER_LENGTH,
            ),
        )

        if not isinstance(self.stage, GuardrailStage):
            raise TypeError("stage must be a GuardrailStage")

        if not isinstance(self.content, str):
            raise TypeError("content must be a string")


@dataclass(frozen=True, slots=True)
class GuardrailFinding:
    """One deterministic signal returned by a configured guardrail rule."""

    rule_id: str
    category: GuardrailCategory
    severity: GuardrailSeverity
    message: str
    start_offset: int | None = None
    end_offset: int | None = None

    def __post_init__(self) -> None:
        """Validate finding metadata without retaining matched content."""
        object.__setattr__(
            self,
            "rule_id",
            _require_non_empty_string(
                self.rule_id,
                field_name="rule_id",
                max_length=_MAX_IDENTIFIER_LENGTH,
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

        has_start = self.start_offset is not None
        has_end = self.end_offset is not None

        if has_start != has_end:
            raise ValueError("start_offset and end_offset must be supplied together")

        if self.start_offset is None or self.end_offset is None:
            return

        if isinstance(self.start_offset, bool) or not isinstance(
            self.start_offset,
            int,
        ):
            raise TypeError("start_offset must be an integer")

        if isinstance(self.end_offset, bool) or not isinstance(
            self.end_offset,
            int,
        ):
            raise TypeError("end_offset must be an integer")

        if self.start_offset < 0:
            raise ValueError("start_offset must be non-negative")

        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")


@dataclass(frozen=True, slots=True)
class GuardrailEvaluation:
    """Content-free enforcement result suitable for downstream control flow."""

    content_id: str
    stage: GuardrailStage
    action: GuardrailAction
    findings: tuple[GuardrailFinding, ...]
    evaluated_rule_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate immutable evaluation structure."""
        object.__setattr__(
            self,
            "content_id",
            _require_non_empty_string(
                self.content_id,
                field_name="content_id",
                max_length=_MAX_IDENTIFIER_LENGTH,
            ),
        )

        if not isinstance(self.stage, GuardrailStage):
            raise TypeError("stage must be a GuardrailStage")

        if not isinstance(self.action, GuardrailAction):
            raise TypeError("action must be a GuardrailAction")

        if not isinstance(self.findings, tuple):
            raise TypeError("findings must be a tuple")

        if not all(isinstance(finding, GuardrailFinding) for finding in self.findings):
            raise TypeError("findings must contain GuardrailFinding values")

        if not isinstance(self.evaluated_rule_ids, tuple):
            raise TypeError("evaluated_rule_ids must be a tuple")

        normalized_rule_ids = tuple(
            _require_non_empty_string(
                rule_id,
                field_name="evaluated_rule_id",
                max_length=_MAX_IDENTIFIER_LENGTH,
            )
            for rule_id in self.evaluated_rule_ids
        )

        if len(normalized_rule_ids) != len(set(normalized_rule_ids)):
            raise ValueError("evaluated_rule_ids must be unique")

        object.__setattr__(
            self,
            "evaluated_rule_ids",
            normalized_rule_ids,
        )

        if self.action is GuardrailAction.BLOCK and not self.findings:
            raise ValueError("BLOCK evaluations must contain at least one finding")

    @property
    def allowed(self) -> bool:
        """Return whether downstream processing may continue."""
        return self.action is GuardrailAction.ALLOW

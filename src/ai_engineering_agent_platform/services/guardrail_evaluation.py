"""Deterministic adversarial evaluation orchestration for runtime guardrails."""

from dataclasses import dataclass
from typing import Protocol

from ai_engineering_agent_platform.domain.guardrail_evaluation import (
    GuardrailAdversarialCaseResult,
    GuardrailAdversarialDataset,
    GuardrailAdversarialSummary,
)
from ai_engineering_agent_platform.domain.guardrails import (
    GuardrailAction,
    GuardrailEvaluation,
    GuardrailSubject,
)


class GuardrailEvaluator(Protocol):
    """Minimal deterministic guardrail execution contract."""

    def evaluate(
        self,
        subject: GuardrailSubject,
    ) -> GuardrailEvaluation:
        """Evaluate one guardrail subject."""
        ...


class GuardrailAdversarialEvaluationError(RuntimeError):
    """Base guardrail adversarial-evaluation error."""


class GuardrailAdversarialEvaluationContractError(GuardrailAdversarialEvaluationError):
    """An evaluator result violated the evaluation contract."""


@dataclass(
    frozen=True,
    slots=True,
)
class GuardrailAdversarialRunResult:
    """Low-content result for one versioned adversarial dataset."""

    dataset_id: str
    dataset_version: str
    provenance_ref: str
    cases: tuple[
        GuardrailAdversarialCaseResult,
        ...,
    ]
    summary: GuardrailAdversarialSummary

    def __post_init__(self) -> None:
        """Validate result cardinality."""
        if len(self.cases) != self.summary.case_count:
            raise ValueError("case result count must match summary case_count")


def _ratio(
    numerator: int,
    denominator: int,
) -> float:
    """Return one deterministic rate with a vacuous-success zero case."""
    if denominator == 0:
        return 1.0

    return numerator / denominator


def _summarize(
    cases: tuple[
        GuardrailAdversarialCaseResult,
        ...,
    ],
) -> GuardrailAdversarialSummary:
    """Build transparent action-decision metrics."""
    if not cases:
        raise ValueError("cases must not be empty")

    expected_block_count = sum(
        item.expected_action is GuardrailAction.BLOCK for item in cases
    )

    expected_allow_count = sum(
        item.expected_action is GuardrailAction.ALLOW for item in cases
    )

    correctly_blocked_count = sum(
        item.expected_action is GuardrailAction.BLOCK
        and item.actual_action is GuardrailAction.BLOCK
        for item in cases
    )

    correctly_allowed_count = sum(
        item.expected_action is GuardrailAction.ALLOW
        and item.actual_action is GuardrailAction.ALLOW
        for item in cases
    )

    missed_block_count = expected_block_count - correctly_blocked_count

    unexpected_block_count = expected_allow_count - correctly_allowed_count

    case_count = len(cases)

    return GuardrailAdversarialSummary(
        case_count=case_count,
        expected_block_count=expected_block_count,
        expected_allow_count=expected_allow_count,
        correctly_blocked_count=(correctly_blocked_count),
        correctly_allowed_count=(correctly_allowed_count),
        missed_block_count=missed_block_count,
        unexpected_block_count=(unexpected_block_count),
        action_accuracy=(correctly_blocked_count + correctly_allowed_count)
        / case_count,
        expected_block_recall=_ratio(
            correctly_blocked_count,
            expected_block_count,
        ),
        expected_allow_rate=_ratio(
            correctly_allowed_count,
            expected_allow_count,
        ),
    )


class GuardrailAdversarialEvaluationService:
    """Evaluate explicit adversarial expectations without an LLM judge.

    Cases run synchronously in stable dataset order. Evaluator exceptions remain
    visible. The result intentionally retains no subject content, finding
    messages, offsets, or rule identifiers.
    """

    def __init__(
        self,
        evaluator: GuardrailEvaluator,
    ) -> None:
        """Store the injected deterministic evaluation boundary."""
        self._evaluator = evaluator

    def evaluate(
        self,
        dataset: GuardrailAdversarialDataset,
    ) -> GuardrailAdversarialRunResult:
        """Execute one versioned adversarial dataset."""
        if not isinstance(
            dataset,
            GuardrailAdversarialDataset,
        ):
            raise TypeError("dataset must be a GuardrailAdversarialDataset")

        results: list[GuardrailAdversarialCaseResult] = []

        for case in dataset.cases:
            evaluation = self._evaluator.evaluate(case.subject)

            if not isinstance(
                evaluation,
                GuardrailEvaluation,
            ):
                raise GuardrailAdversarialEvaluationContractError(
                    "evaluator must return GuardrailEvaluation"
                )

            if evaluation.content_id != case.subject.content_id:
                raise GuardrailAdversarialEvaluationContractError(
                    "guardrail evaluation content identity mismatch"
                )

            if evaluation.stage is not case.subject.stage:
                raise GuardrailAdversarialEvaluationContractError(
                    "guardrail evaluation stage mismatch"
                )

            results.append(
                GuardrailAdversarialCaseResult(
                    case_id=case.case_id,
                    stage=case.subject.stage,
                    expected_action=(case.expected_action),
                    actual_action=(evaluation.action),
                    action_correct=(evaluation.action is case.expected_action),
                )
            )

        frozen_results = tuple(results)

        return GuardrailAdversarialRunResult(
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.version,
            provenance_ref=(dataset.provenance_ref),
            cases=frozen_results,
            summary=_summarize(frozen_results),
        )

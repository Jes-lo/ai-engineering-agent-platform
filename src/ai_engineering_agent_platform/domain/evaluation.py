"""Infrastructure-independent deterministic RAG evaluation primitives."""

from dataclasses import dataclass
from math import isfinite

from ai_engineering_agent_platform.domain.grounding import (
    GroundedAnswerStatus,
)
from ai_engineering_agent_platform.domain.retrieval import (
    RetrievalRequest,
)


def _require_non_empty_string(
    field_name: str,
    value: object,
) -> None:
    """Require one meaningful string."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")

    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _require_non_negative_integer(
    field_name: str,
    value: object,
) -> None:
    """Require an integer count that is not negative."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _require_unit_score(
    field_name: str,
    value: object,
) -> None:
    """Require a finite normalized score from zero through one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")

    if not isfinite(value):
        raise ValueError(f"{field_name} must be finite")

    if value < 0.0 or value > 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class RAGEvaluationCase:
    """One versionable RAG evaluation expectation."""

    case_id: str
    request: RetrievalRequest
    expected_relevant_chunk_ids: tuple[str, ...]
    expected_status: GroundedAnswerStatus

    def __post_init__(self) -> None:
        """Validate case identity and expected evidence."""
        _require_non_empty_string(
            "case_id",
            self.case_id,
        )

        if not isinstance(
            self.request,
            RetrievalRequest,
        ):
            raise ValueError("request must be a RetrievalRequest")

        if not isinstance(
            self.expected_relevant_chunk_ids,
            tuple,
        ):
            raise ValueError("expected_relevant_chunk_ids must be a tuple")

        for chunk_id in self.expected_relevant_chunk_ids:
            _require_non_empty_string(
                "expected relevant chunk identifier",
                chunk_id,
            )

        if len(self.expected_relevant_chunk_ids) != len(
            set(self.expected_relevant_chunk_ids)
        ):
            raise ValueError("expected relevant chunk identifiers must be unique")

        if not isinstance(
            self.expected_status,
            GroundedAnswerStatus,
        ):
            raise ValueError("expected_status must be a GroundedAnswerStatus")


@dataclass(frozen=True, slots=True)
class RAGEvaluationDataset:
    """Versioned evaluation cases with explicit provenance."""

    dataset_id: str
    version: str
    provenance_ref: str
    cases: tuple[RAGEvaluationCase, ...]

    def __post_init__(self) -> None:
        """Validate dataset identity, provenance, and case uniqueness."""
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

        if not isinstance(self.cases, tuple):
            raise ValueError("cases must be a tuple")

        if not self.cases:
            raise ValueError("evaluation dataset must contain at least one case")

        if any(not isinstance(case, RAGEvaluationCase) for case in self.cases):
            raise ValueError("cases must contain RAGEvaluationCase values")

        case_ids = tuple(case.case_id for case in self.cases)

        if len(case_ids) != len(set(case_ids)):
            raise ValueError("evaluation case identifiers must be unique")


@dataclass(frozen=True, slots=True)
class RAGEvaluationMetrics:
    """Transparent deterministic metrics for one RAG evaluation case."""

    expected_relevant_count: int
    retrieved_count: int
    grounding_count: int
    citation_count: int

    relevant_retrieved_count: int
    relevant_grounding_count: int
    relevant_citation_count: int

    retrieval_precision: float
    retrieval_recall: float

    grounding_precision: float
    grounding_recall: float

    citation_precision: float
    citation_recall: float

    answer_status_correct: bool

    def __post_init__(self) -> None:
        """Validate counts, normalized scores, and status result."""
        count_fields = (
            (
                "expected_relevant_count",
                self.expected_relevant_count,
            ),
            (
                "retrieved_count",
                self.retrieved_count,
            ),
            (
                "grounding_count",
                self.grounding_count,
            ),
            (
                "citation_count",
                self.citation_count,
            ),
            (
                "relevant_retrieved_count",
                self.relevant_retrieved_count,
            ),
            (
                "relevant_grounding_count",
                self.relevant_grounding_count,
            ),
            (
                "relevant_citation_count",
                self.relevant_citation_count,
            ),
        )

        for field_name, count_value in count_fields:
            _require_non_negative_integer(
                field_name,
                count_value,
            )

        if self.relevant_retrieved_count > min(
            self.retrieved_count,
            self.expected_relevant_count,
        ):
            raise ValueError("relevant_retrieved_count exceeds valid bounds")

        if self.relevant_grounding_count > min(
            self.grounding_count,
            self.expected_relevant_count,
        ):
            raise ValueError("relevant_grounding_count exceeds valid bounds")

        if self.relevant_citation_count > min(
            self.citation_count,
            self.expected_relevant_count,
        ):
            raise ValueError("relevant_citation_count exceeds valid bounds")

        score_fields = (
            (
                "retrieval_precision",
                self.retrieval_precision,
            ),
            (
                "retrieval_recall",
                self.retrieval_recall,
            ),
            (
                "grounding_precision",
                self.grounding_precision,
            ),
            (
                "grounding_recall",
                self.grounding_recall,
            ),
            (
                "citation_precision",
                self.citation_precision,
            ),
            (
                "citation_recall",
                self.citation_recall,
            ),
        )

        for field_name, score_value in score_fields:
            _require_unit_score(
                field_name,
                score_value,
            )

        if not isinstance(
            self.answer_status_correct,
            bool,
        ):
            raise ValueError("answer_status_correct must be a boolean")


@dataclass(frozen=True, slots=True)
class RAGEvaluationSummary:
    """Macro-averaged metrics across a non-empty evaluation run."""

    case_count: int

    retrieval_precision: float
    retrieval_recall: float

    grounding_precision: float
    grounding_recall: float

    citation_precision: float
    citation_recall: float

    answer_status_accuracy: float

    def __post_init__(self) -> None:
        """Validate aggregate count and normalized metrics."""
        if (
            isinstance(self.case_count, bool)
            or not isinstance(self.case_count, int)
            or self.case_count <= 0
        ):
            raise ValueError("case_count must be a positive integer")

        for field_name, value in (
            (
                "retrieval_precision",
                self.retrieval_precision,
            ),
            (
                "retrieval_recall",
                self.retrieval_recall,
            ),
            (
                "grounding_precision",
                self.grounding_precision,
            ),
            (
                "grounding_recall",
                self.grounding_recall,
            ),
            (
                "citation_precision",
                self.citation_precision,
            ),
            (
                "citation_recall",
                self.citation_recall,
            ),
            (
                "answer_status_accuracy",
                self.answer_status_accuracy,
            ),
        ):
            _require_unit_score(
                field_name,
                value,
            )

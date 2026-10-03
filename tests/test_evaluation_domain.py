"""Tests for deterministic RAG evaluation domain primitives."""

import pytest

from ai_engineering_agent_platform.domain import (
    GroundedAnswerStatus,
    RAGEvaluationCase,
    RAGEvaluationDataset,
    RAGEvaluationMetrics,
    RAGEvaluationSummary,
    RetrievalRequest,
)


def _case(
    *,
    case_id: str = "case-1",
) -> RAGEvaluationCase:
    """Return one deterministic evaluation case."""
    return RAGEvaluationCase(
        case_id=case_id,
        request=RetrievalRequest(
            query="What synthetic evidence exists?",
            top_k=3,
            namespace="synthetic",
        ),
        expected_relevant_chunk_ids=(
            "chunk-a",
            "chunk-b",
        ),
        expected_status=GroundedAnswerStatus.ANSWERED,
    )


def test_evaluation_case_preserves_expected_contract() -> None:
    """A valid case should preserve request and relevance expectations."""
    case = _case()

    assert case.case_id == "case-1"
    assert case.request.query == ("What synthetic evidence exists?")
    assert case.expected_relevant_chunk_ids == (
        "chunk-a",
        "chunk-b",
    )
    assert case.expected_status is GroundedAnswerStatus.ANSWERED


def test_duplicate_expected_chunk_ids_are_rejected() -> None:
    """Ground-truth evidence identity must not be ambiguous."""
    with pytest.raises(
        ValueError,
        match=("expected relevant chunk identifiers must be unique"),
    ):
        RAGEvaluationCase(
            case_id="duplicate",
            request=RetrievalRequest(
                query="Synthetic?",
                top_k=2,
            ),
            expected_relevant_chunk_ids=(
                "chunk-a",
                "chunk-a",
            ),
            expected_status=GroundedAnswerStatus.ANSWERED,
        )


def test_dataset_requires_versioned_provenance() -> None:
    """Evaluation data should carry explicit version and provenance."""
    dataset = RAGEvaluationDataset(
        dataset_id="synthetic-rag-eval",
        version="1.0.0",
        provenance_ref="synthetic://rag-eval/v1",
        cases=(_case(),),
    )

    assert dataset.dataset_id == "synthetic-rag-eval"
    assert dataset.version == "1.0.0"
    assert dataset.provenance_ref == ("synthetic://rag-eval/v1")
    assert dataset.cases[0].case_id == "case-1"


def test_dataset_rejects_duplicate_case_ids() -> None:
    """Versioned datasets must identify cases uniquely."""
    with pytest.raises(
        ValueError,
        match="evaluation case identifiers must be unique",
    ):
        RAGEvaluationDataset(
            dataset_id="synthetic",
            version="1",
            provenance_ref="synthetic://dataset",
            cases=(
                _case(),
                _case(),
            ),
        )


def test_dataset_rejects_empty_case_collection() -> None:
    """An evaluation run cannot be based on an empty dataset."""
    with pytest.raises(
        ValueError,
        match="must contain at least one case",
    ):
        RAGEvaluationDataset(
            dataset_id="synthetic",
            version="1",
            provenance_ref="synthetic://dataset",
            cases=(),
        )


def test_metrics_preserve_transparent_counts_and_scores() -> None:
    """Metrics expose both denominators/counts and normalized scores."""
    metrics = RAGEvaluationMetrics(
        expected_relevant_count=2,
        retrieved_count=3,
        grounding_count=2,
        citation_count=1,
        relevant_retrieved_count=2,
        relevant_grounding_count=1,
        relevant_citation_count=1,
        retrieval_precision=2 / 3,
        retrieval_recall=1.0,
        grounding_precision=0.5,
        grounding_recall=0.5,
        citation_precision=1.0,
        citation_recall=0.5,
        answer_status_correct=True,
    )

    assert metrics.retrieved_count == 3
    assert metrics.relevant_retrieved_count == 2
    assert metrics.citation_precision == 1.0
    assert metrics.answer_status_correct is True


@pytest.mark.parametrize(
    "score",
    (
        -0.1,
        1.1,
        float("inf"),
        float("nan"),
    ),
)
def test_metrics_reject_invalid_normalized_scores(
    score: float,
) -> None:
    """Evaluation scores must remain finite normalized values."""
    with pytest.raises(ValueError):
        RAGEvaluationMetrics(
            expected_relevant_count=1,
            retrieved_count=1,
            grounding_count=1,
            citation_count=1,
            relevant_retrieved_count=1,
            relevant_grounding_count=1,
            relevant_citation_count=1,
            retrieval_precision=score,
            retrieval_recall=1.0,
            grounding_precision=1.0,
            grounding_recall=1.0,
            citation_precision=1.0,
            citation_recall=1.0,
            answer_status_correct=True,
        )


def test_summary_requires_positive_case_count() -> None:
    """Aggregate evaluation summaries cannot represent zero cases."""
    with pytest.raises(
        ValueError,
        match="case_count must be a positive integer",
    ):
        RAGEvaluationSummary(
            case_count=0,
            retrieval_precision=1.0,
            retrieval_recall=1.0,
            grounding_precision=1.0,
            grounding_recall=1.0,
            citation_precision=1.0,
            citation_recall=1.0,
            answer_status_accuracy=1.0,
        )


def test_evaluation_domain_is_publicly_exported() -> None:
    """Evaluation primitives should be available through the domain."""
    import ai_engineering_agent_platform.domain as domain

    assert {
        "RAGEvaluationCase",
        "RAGEvaluationDataset",
        "RAGEvaluationMetrics",
        "RAGEvaluationSummary",
    } <= set(domain.__all__)

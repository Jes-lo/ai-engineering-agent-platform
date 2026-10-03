"""Tests for deterministic structural RAG evaluation metrics."""

import pytest

from ai_engineering_agent_platform.contracts import (
    FinishReason,
)
from ai_engineering_agent_platform.domain import (
    GroundedAnswer,
    GroundedAnswerStatus,
    GroundedCitation,
    RAGEvaluationCase,
    RAGEvaluationMetrics,
    RerankedEvidence,
    RerankedRetrievalResponse,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedEvidence,
)
from ai_engineering_agent_platform.services import (
    GroundedGenerationResult,
    RAGResult,
    score_rag_result,
    summarize_rag_metrics,
)


def _evidence(
    *,
    chunk_id: str,
    rank: int,
) -> RetrievedEvidence:
    """Return one deterministic retrieved evidence object."""
    text = f"{chunk_id} evidence"

    return RetrievedEvidence(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        text=text,
        score=1.0 - (rank / 10),
        rank=rank,
        start_char=0,
        end_char=len(text),
        source_ref=f"synthetic://{chunk_id}",
        title=f"Document {chunk_id}",
    )


def _case(
    *,
    expected_chunk_ids: tuple[str, ...] = (
        "chunk-a",
        "chunk-b",
    ),
    expected_status: GroundedAnswerStatus = (GroundedAnswerStatus.ANSWERED),
) -> RAGEvaluationCase:
    """Return one deterministic evaluation expectation."""
    return RAGEvaluationCase(
        case_id="case-1",
        request=RetrievalRequest(
            query="What synthetic evidence exists?",
            top_k=3,
            namespace="synthetic",
        ),
        expected_relevant_chunk_ids=expected_chunk_ids,
        expected_status=expected_status,
    )


def _answered_result() -> RAGResult:
    """Return retrieval, reranking, and citation with partial relevance."""
    chunk_a = _evidence(
        chunk_id="chunk-a",
        rank=1,
    )
    chunk_x = _evidence(
        chunk_id="chunk-x",
        rank=2,
    )
    chunk_b = _evidence(
        chunk_id="chunk-b",
        rank=3,
    )

    retrieval = RetrievalResponse(
        query="What synthetic evidence exists?",
        namespace="synthetic",
        results=(
            chunk_a,
            chunk_x,
            chunk_b,
        ),
    )

    grounding = RerankedRetrievalResponse(
        query=retrieval.query,
        namespace=retrieval.namespace,
        model="synthetic-reranker",
        results=(
            RerankedEvidence(
                evidence=chunk_b,
                rerank_score=9.0,
                rank=1,
            ),
            RerankedEvidence(
                evidence=chunk_x,
                rerank_score=8.0,
                rank=2,
            ),
        ),
    )

    citation = GroundedCitation(
        citation_id="C1",
        evidence=chunk_b,
    )

    answer = GroundedAnswer(
        query=retrieval.query,
        namespace=retrieval.namespace,
        status=GroundedAnswerStatus.ANSWERED,
        answer="Supported by beta evidence. [[C1]]",
        citations=(citation,),
    )

    generation = GroundedGenerationResult(
        answer=answer,
        model="generator-model",
        finish_reason=FinishReason.STOP,
        usage=None,
    )

    return RAGResult(
        retrieval=retrieval,
        grounding_input=grounding,
        answer=answer,
        generation=generation,
    )


def _abstained_result() -> RAGResult:
    """Return a valid empty-retrieval abstention."""
    retrieval = RetrievalResponse(
        query="What synthetic evidence exists?",
        namespace="synthetic",
        results=(),
    )

    answer = GroundedAnswer(
        query=retrieval.query,
        namespace=retrieval.namespace,
        status=GroundedAnswerStatus.ABSTAINED,
        answer=None,
        citations=(),
    )

    return RAGResult(
        retrieval=retrieval,
        grounding_input=retrieval,
        answer=answer,
        generation=None,
    )


def test_structural_metrics_cover_retrieval_grounding_and_citations() -> None:
    """Metrics should expose quality loss at each independent RAG stage."""
    metrics = score_rag_result(
        _case(),
        _answered_result(),
    )

    assert metrics.expected_relevant_count == 2

    assert metrics.retrieved_count == 3
    assert metrics.relevant_retrieved_count == 2
    assert metrics.retrieval_precision == pytest.approx(2 / 3)
    assert metrics.retrieval_recall == 1.0

    assert metrics.grounding_count == 2
    assert metrics.relevant_grounding_count == 1
    assert metrics.grounding_precision == 0.5
    assert metrics.grounding_recall == 0.5

    assert metrics.citation_count == 1
    assert metrics.relevant_citation_count == 1
    assert metrics.citation_precision == 1.0
    assert metrics.citation_recall == 0.5

    assert metrics.answer_status_correct is True


def test_correct_unanswerable_abstention_scores_empty_sets_deterministically() -> None:
    """No expected evidence plus abstention should score empty sets exactly."""
    metrics = score_rag_result(
        _case(
            expected_chunk_ids=(),
            expected_status=GroundedAnswerStatus.ABSTAINED,
        ),
        _abstained_result(),
    )

    assert metrics.expected_relevant_count == 0
    assert metrics.retrieved_count == 0
    assert metrics.grounding_count == 0
    assert metrics.citation_count == 0

    assert metrics.retrieval_precision == 1.0
    assert metrics.retrieval_recall == 1.0
    assert metrics.grounding_precision == 1.0
    assert metrics.grounding_recall == 1.0
    assert metrics.citation_precision == 1.0
    assert metrics.citation_recall == 1.0
    assert metrics.answer_status_correct is True


def test_wrong_answer_status_is_scored_not_hidden() -> None:
    """An unexpected abstention should remain visible as status failure."""
    metrics = score_rag_result(
        _case(),
        _abstained_result(),
    )

    assert metrics.retrieval_recall == 0.0
    assert metrics.citation_recall == 0.0
    assert metrics.answer_status_correct is False


def test_query_mismatch_fails_closed() -> None:
    """Evaluation cannot silently score output from another query."""
    case = RAGEvaluationCase(
        case_id="different-query",
        request=RetrievalRequest(
            query="A different query",
            top_k=3,
            namespace="synthetic",
        ),
        expected_relevant_chunk_ids=("chunk-a",),
        expected_status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(
        ValueError,
        match="retrieval query does not match",
    ):
        score_rag_result(
            case,
            _answered_result(),
        )


def test_namespace_mismatch_fails_closed() -> None:
    """Dataset expectations cannot cross retrieval namespaces silently."""
    case = RAGEvaluationCase(
        case_id="different-namespace",
        request=RetrievalRequest(
            query="What synthetic evidence exists?",
            top_k=3,
            namespace="other",
        ),
        expected_relevant_chunk_ids=("chunk-a",),
        expected_status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(
        ValueError,
        match="retrieval namespace does not match",
    ):
        score_rag_result(
            case,
            _answered_result(),
        )


def test_grounding_cannot_introduce_unretrieved_evidence() -> None:
    """Evaluation should detect a forged grounding surface."""
    result = _answered_result()

    forged = _evidence(
        chunk_id="chunk-forged",
        rank=1,
    )

    grounding = RetrievalResponse(
        query=result.retrieval.query,
        namespace=result.retrieval.namespace,
        results=(forged,),
    )

    forged_result = RAGResult(
        retrieval=result.retrieval,
        grounding_input=grounding,
        answer=result.answer,
        generation=result.generation,
    )

    with pytest.raises(
        ValueError,
        match="grounding input contains evidence outside retrieval",
    ):
        score_rag_result(
            _case(),
            forged_result,
        )


def test_citation_cannot_escape_grounding_input() -> None:
    """Evaluation should detect citation evidence excluded by reranking."""
    result = _answered_result()

    retrieval_only = result.retrieval.results[0]

    forged_answer = GroundedAnswer(
        query=result.answer.query,
        namespace=result.answer.namespace,
        status=GroundedAnswerStatus.ANSWERED,
        answer="Forged citation. [[C1]]",
        citations=(
            GroundedCitation(
                citation_id="C1",
                evidence=retrieval_only,
            ),
        ),
    )

    forged_result = RAGResult(
        retrieval=result.retrieval,
        grounding_input=result.grounding_input,
        answer=forged_answer,
        generation=None,
    )

    with pytest.raises(
        ValueError,
        match="citation contains evidence outside grounding input",
    ):
        score_rag_result(
            _case(),
            forged_result,
        )


def _metrics(
    *,
    value: float,
    status_correct: bool,
) -> RAGEvaluationMetrics:
    """Return normalized metrics for summary tests."""
    return RAGEvaluationMetrics(
        expected_relevant_count=1,
        retrieved_count=1,
        grounding_count=1,
        citation_count=1,
        relevant_retrieved_count=1,
        relevant_grounding_count=1,
        relevant_citation_count=1,
        retrieval_precision=value,
        retrieval_recall=value,
        grounding_precision=value,
        grounding_recall=value,
        citation_precision=value,
        citation_recall=value,
        answer_status_correct=status_correct,
    )


def test_summary_uses_macro_averages_without_hidden_overall_score() -> None:
    """Dataset summary should average each signal independently."""
    summary = summarize_rag_metrics(
        (
            _metrics(
                value=1.0,
                status_correct=True,
            ),
            _metrics(
                value=0.0,
                status_correct=False,
            ),
        )
    )

    assert summary.case_count == 2
    assert summary.retrieval_precision == 0.5
    assert summary.retrieval_recall == 0.5
    assert summary.grounding_precision == 0.5
    assert summary.grounding_recall == 0.5
    assert summary.citation_precision == 0.5
    assert summary.citation_recall == 0.5
    assert summary.answer_status_accuracy == 0.5


def test_empty_summary_input_is_rejected() -> None:
    """A run summary requires at least one case metric."""
    with pytest.raises(
        ValueError,
        match="metrics must not be empty",
    ):
        summarize_rag_metrics(())


def test_evaluation_mapping_is_publicly_exported() -> None:
    """Evaluation scoring helpers should be public service utilities."""
    import ai_engineering_agent_platform.services as services

    assert {
        "score_rag_result",
        "summarize_rag_metrics",
    } <= set(services.__all__)

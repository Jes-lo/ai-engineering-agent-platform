"""Pure deterministic scoring for validated RAG execution results."""

from ai_engineering_agent_platform.domain import (
    RAGEvaluationCase,
    RAGEvaluationMetrics,
    RAGEvaluationSummary,
    RerankedRetrievalResponse,
    RetrievalResponse,
    RetrievedEvidence,
)
from ai_engineering_agent_platform.services.rag import (
    RAGResult,
)


def _grounding_evidence(
    result: RAGResult,
) -> tuple[RetrievedEvidence, ...]:
    """Return evidence exactly as supplied to grounded generation."""
    grounding_input = result.grounding_input

    if isinstance(
        grounding_input,
        RetrievalResponse,
    ):
        return grounding_input.results

    if isinstance(
        grounding_input,
        RerankedRetrievalResponse,
    ):
        return tuple(item.evidence for item in grounding_input.results)

    raise ValueError("unsupported RAG grounding input")


def _set_precision(
    predicted: set[str],
    expected: set[str],
) -> float:
    """Return deterministic set precision.

    An empty prediction is perfect only when the expected set is also empty.
    """
    if not predicted:
        return 1.0 if not expected else 0.0

    return len(predicted & expected) / len(predicted)


def _set_recall(
    predicted: set[str],
    expected: set[str],
) -> float:
    """Return deterministic set recall.

    When no relevant evidence is expected, recall is defined as one because
    there is no missing relevant item.
    """
    if not expected:
        return 1.0

    return len(predicted & expected) / len(expected)


def score_rag_result(
    case: RAGEvaluationCase,
    result: RAGResult,
) -> RAGEvaluationMetrics:
    """Score retrieval, grounding, citation, and answer-state behavior.

    These metrics evaluate structural evidence alignment against explicit
    expected chunk identifiers. They do not establish semantic entailment or
    factual correctness of generated claims.
    """
    expected_query = case.request.query
    expected_namespace = case.request.namespace

    if result.retrieval.query != expected_query:
        raise ValueError("RAG retrieval query does not match evaluation case")

    if result.retrieval.namespace != expected_namespace:
        raise ValueError("RAG retrieval namespace does not match evaluation case")

    if result.grounding_input.query != expected_query:
        raise ValueError("RAG grounding query does not match evaluation case")

    if result.grounding_input.namespace != expected_namespace:
        raise ValueError("RAG grounding namespace does not match evaluation case")

    if result.answer.query != expected_query:
        raise ValueError("RAG answer query does not match evaluation case")

    if result.answer.namespace != expected_namespace:
        raise ValueError("RAG answer namespace does not match evaluation case")

    retrieval_evidence = result.retrieval.results

    grounding_evidence = _grounding_evidence(
        result,
    )

    retrieval_by_id = {item.chunk_id: item for item in retrieval_evidence}

    grounding_by_id = {item.chunk_id: item for item in grounding_evidence}

    if not set(grounding_by_id).issubset(retrieval_by_id):
        raise ValueError("RAG grounding input contains evidence outside retrieval")

    for chunk_id, evidence in grounding_by_id.items():
        if evidence is not retrieval_by_id[chunk_id]:
            raise ValueError(
                "RAG grounding evidence is not the retrieved evidence object"
            )

    citation_chunk_ids: list[str] = []

    for citation in result.answer.citations:
        chunk_id = citation.evidence.chunk_id

        if chunk_id not in grounding_by_id:
            raise ValueError("RAG citation contains evidence outside grounding input")

        if citation.evidence is not grounding_by_id[chunk_id]:
            raise ValueError(
                "RAG citation is not bound to the grounding evidence object"
            )

        citation_chunk_ids.append(
            chunk_id,
        )

    if len(citation_chunk_ids) != len(set(citation_chunk_ids)):
        raise ValueError("RAG citations reference duplicate evidence chunks")

    expected = set(case.expected_relevant_chunk_ids)

    retrieved = set(retrieval_by_id)

    grounding = set(grounding_by_id)

    cited = set(citation_chunk_ids)

    relevant_retrieved = retrieved & expected
    relevant_grounding = grounding & expected
    relevant_citations = cited & expected

    return RAGEvaluationMetrics(
        expected_relevant_count=len(expected),
        retrieved_count=len(retrieved),
        grounding_count=len(grounding),
        citation_count=len(cited),
        relevant_retrieved_count=len(relevant_retrieved),
        relevant_grounding_count=len(relevant_grounding),
        relevant_citation_count=len(relevant_citations),
        retrieval_precision=_set_precision(
            retrieved,
            expected,
        ),
        retrieval_recall=_set_recall(
            retrieved,
            expected,
        ),
        grounding_precision=_set_precision(
            grounding,
            expected,
        ),
        grounding_recall=_set_recall(
            grounding,
            expected,
        ),
        citation_precision=_set_precision(
            cited,
            expected,
        ),
        citation_recall=_set_recall(
            cited,
            expected,
        ),
        answer_status_correct=(result.answer.status is case.expected_status),
    )


def summarize_rag_metrics(
    metrics: tuple[
        RAGEvaluationMetrics,
        ...,
    ],
) -> RAGEvaluationSummary:
    """Macro-average deterministic RAG metrics."""
    if not isinstance(metrics, tuple):
        raise ValueError("metrics must be a tuple")

    if not metrics:
        raise ValueError("metrics must not be empty")

    if any(
        not isinstance(
            item,
            RAGEvaluationMetrics,
        )
        for item in metrics
    ):
        raise ValueError("metrics must contain RAGEvaluationMetrics values")

    count = len(metrics)

    return RAGEvaluationSummary(
        case_count=count,
        retrieval_precision=sum(item.retrieval_precision for item in metrics) / count,
        retrieval_recall=sum(item.retrieval_recall for item in metrics) / count,
        grounding_precision=sum(item.grounding_precision for item in metrics) / count,
        grounding_recall=sum(item.grounding_recall for item in metrics) / count,
        citation_precision=sum(item.citation_precision for item in metrics) / count,
        citation_recall=sum(item.citation_recall for item in metrics) / count,
        answer_status_accuracy=sum(
            1.0 if item.answer_status_correct else 0.0 for item in metrics
        )
        / count,
    )

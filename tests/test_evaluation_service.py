"""Tests for versioned automated RAG dataset evaluation."""

import pytest

from ai_engineering_agent_platform.contracts import (
    FinishReason,
)
from ai_engineering_agent_platform.domain import (
    GroundedAnswer,
    GroundedAnswerStatus,
    GroundedCitation,
    ProviderExecutionError,
    RAGEvaluationCase,
    RAGEvaluationDataset,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedEvidence,
)
from ai_engineering_agent_platform.services import (
    GroundedGenerationResult,
    RAGEvaluationService,
    RAGResult,
)


def _request(
    query: str,
) -> RetrievalRequest:
    """Return one deterministic evaluation retrieval request."""
    return RetrievalRequest(
        query=query,
        top_k=2,
        namespace="synthetic",
    )


def _answered_result(
    query: str,
) -> RAGResult:
    """Return one perfectly relevant answered RAG result."""
    text = "relevant evidence"

    evidence = RetrievedEvidence(
        chunk_id=f"chunk-{query}",
        document_id=f"doc-{query}",
        text=text,
        score=0.9,
        rank=1,
        start_char=0,
        end_char=len(text),
        source_ref=f"synthetic://{query}",
        title="Synthetic",
    )

    retrieval = RetrievalResponse(
        query=query,
        namespace="synthetic",
        results=(evidence,),
    )

    answer = GroundedAnswer(
        query=query,
        namespace="synthetic",
        status=GroundedAnswerStatus.ANSWERED,
        answer="Synthetic answer. [[C1]]",
        citations=(
            GroundedCitation(
                citation_id="C1",
                evidence=evidence,
            ),
        ),
    )

    generation = GroundedGenerationResult(
        answer=answer,
        model="synthetic-generator",
        finish_reason=FinishReason.STOP,
        usage=None,
    )

    return RAGResult(
        retrieval=retrieval,
        grounding_input=retrieval,
        answer=answer,
        generation=generation,
    )


def _abstained_result(
    query: str,
) -> RAGResult:
    """Return one empty-retrieval abstention."""
    retrieval = RetrievalResponse(
        query=query,
        namespace="synthetic",
        results=(),
    )

    answer = GroundedAnswer(
        query=query,
        namespace="synthetic",
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


class SyntheticRAGRunner:
    """Deterministic RAG execution boundary for dataset tests."""

    def __init__(
        self,
        results: dict[str, RAGResult],
        *,
        error_query: str | None = None,
    ) -> None:
        """Store deterministic outputs and optional failure."""
        self.results = results
        self.error_query = error_query
        self.requests: list[RetrievalRequest] = []

    async def run(
        self,
        request: RetrievalRequest,
    ) -> RAGResult:
        """Record execution and return configured result."""
        self.requests.append(
            request,
        )

        if request.query == self.error_query:
            raise ProviderExecutionError("synthetic RAG execution failure")

        return self.results[request.query]


def _dataset() -> RAGEvaluationDataset:
    """Return one versioned two-case synthetic evaluation dataset."""
    answered_query = "answerable"
    abstain_query = "unanswerable"

    return RAGEvaluationDataset(
        dataset_id="synthetic-rag-evaluation",
        version="1.0.0",
        provenance_ref="synthetic://rag-evaluation/v1",
        cases=(
            RAGEvaluationCase(
                case_id="answered",
                request=_request(answered_query),
                expected_relevant_chunk_ids=(f"chunk-{answered_query}",),
                expected_status=GroundedAnswerStatus.ANSWERED,
            ),
            RAGEvaluationCase(
                case_id="abstained",
                request=_request(abstain_query),
                expected_relevant_chunk_ids=(),
                expected_status=GroundedAnswerStatus.ABSTAINED,
            ),
        ),
    )


@pytest.mark.anyio
async def test_dataset_evaluation_preserves_order_and_provenance() -> None:
    """Versioned dataset cases should execute sequentially and transparently."""
    runner = SyntheticRAGRunner(
        {
            "answerable": _answered_result("answerable"),
            "unanswerable": _abstained_result("unanswerable"),
        }
    )

    service = RAGEvaluationService(runner)

    result = await service.evaluate(_dataset())

    assert tuple(request.query for request in runner.requests) == (
        "answerable",
        "unanswerable",
    )

    assert result.dataset_id == ("synthetic-rag-evaluation")
    assert result.dataset_version == "1.0.0"
    assert result.provenance_ref == ("synthetic://rag-evaluation/v1")

    assert tuple(item.case.case_id for item in result.cases) == (
        "answered",
        "abstained",
    )

    assert all(item.metrics.answer_status_correct for item in result.cases)

    assert result.summary.case_count == 2

    assert result.summary.retrieval_precision == 1.0
    assert result.summary.retrieval_recall == 1.0

    assert result.summary.grounding_precision == 1.0
    assert result.summary.grounding_recall == 1.0

    assert result.summary.citation_precision == 1.0
    assert result.summary.citation_recall == 1.0

    assert result.summary.answer_status_accuracy == 1.0


@pytest.mark.anyio
async def test_runner_failure_aborts_evaluation_without_hiding_error() -> None:
    """Provider/RAG failures should remain visible to evaluation callers."""
    runner = SyntheticRAGRunner(
        {
            "answerable": _answered_result("answerable"),
            "unanswerable": _abstained_result("unanswerable"),
        },
        error_query="unanswerable",
    )

    service = RAGEvaluationService(runner)

    with pytest.raises(
        ProviderExecutionError,
        match="synthetic RAG execution failure",
    ):
        await service.evaluate(_dataset())

    assert tuple(request.query for request in runner.requests) == (
        "answerable",
        "unanswerable",
    )


def test_evaluation_service_is_publicly_exported() -> None:
    """Automated evaluation orchestration should be a public service."""
    import ai_engineering_agent_platform.services as services

    assert {
        "RAGEvaluationCaseExecution",
        "RAGEvaluationRunResult",
        "RAGEvaluationService",
        "RAGRunner",
    } <= set(services.__all__)

    assert hasattr(
        services,
        "RAGEvaluationService",
    )

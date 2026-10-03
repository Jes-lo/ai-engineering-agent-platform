"""Tests for pure reranking mapping."""

import pytest

from ai_engineering_agent_platform.contracts import (
    RerankResponse,
    RerankResult,
)
from ai_engineering_agent_platform.domain import (
    ProviderExecutionError,
    RetrievalResponse,
    RetrievedEvidence,
)
from ai_engineering_agent_platform.services import (
    build_rerank_request,
    build_reranked_retrieval_response,
)


def _evidence(
    *,
    chunk_id: str,
    text: str,
    score: float,
    rank: int,
    start_char: int,
) -> RetrievedEvidence:
    """Return deterministic synthetic retrieval evidence."""
    return RetrievedEvidence(
        chunk_id=chunk_id,
        document_id="shared-document",
        text=text,
        score=score,
        rank=rank,
        start_char=start_char,
        end_char=(start_char + len(text)),
        source_ref="synthetic://shared-document",
        title="Synthetic Document",
    )


def _retrieval_response() -> RetrievalResponse:
    """Return two chunks from the same source document."""
    return RetrievalResponse(
        query="synthetic query",
        namespace="synthetic",
        results=(
            _evidence(
                chunk_id="chunk-a",
                text="alpha evidence",
                score=0.91,
                rank=1,
                start_char=0,
            ),
            _evidence(
                chunk_id="chunk-b",
                text="beta evidence",
                score=0.72,
                rank=2,
                start_char=20,
            ),
        ),
    )


def test_rerank_request_uses_unique_chunk_identifiers() -> None:
    """Multiple chunks from one document must remain distinct candidates."""
    request = build_rerank_request(
        _retrieval_response(),
        model="synthetic-reranker",
        top_n=2,
    )

    assert request.model == "synthetic-reranker"
    assert request.query == "synthetic query"
    assert request.top_n == 2

    assert tuple(document.document_id for document in request.documents) == (
        "chunk-a",
        "chunk-b",
    )

    assert tuple(document.text for document in request.documents) == (
        "alpha evidence",
        "beta evidence",
    )


def test_rerank_request_rejects_empty_retrieval() -> None:
    """A reranker must not be invoked without retrieval candidates."""
    with pytest.raises(
        ValueError,
        match=("cannot rerank an empty retrieval response"),
    ):
        build_rerank_request(
            RetrievalResponse(
                query="synthetic",
                results=(),
            ),
            model="synthetic-reranker",
        )


def test_top_n_validation_remains_contract_owned() -> None:
    """Existing RerankRequest invariants remain authoritative."""
    with pytest.raises(
        ValueError,
        match="top_n must not exceed document count",
    ):
        build_rerank_request(
            _retrieval_response(),
            model="synthetic-reranker",
            top_n=3,
        )


def test_rerank_response_reorders_without_losing_vector_diagnostics() -> None:
    """Final order should preserve original vector evidence intact."""
    retrieval = _retrieval_response()

    rerank = RerankResponse(
        model="synthetic-reranker",
        results=(
            RerankResult(
                document_id="chunk-b",
                score=9.0,
                rank=1,
            ),
            RerankResult(
                document_id="chunk-a",
                score=7.0,
                rank=2,
            ),
        ),
    )

    response = build_reranked_retrieval_response(
        retrieval,
        rerank,
    )

    assert response.query == retrieval.query
    assert response.namespace == retrieval.namespace
    assert response.model == "synthetic-reranker"

    assert tuple(item.evidence.chunk_id for item in response.results) == (
        "chunk-b",
        "chunk-a",
    )

    assert tuple(item.rerank_score for item in response.results) == (
        9.0,
        7.0,
    )

    assert tuple(item.rank for item in response.results) == (
        1,
        2,
    )

    assert response.results[0].evidence.score == 0.72
    assert response.results[0].evidence.rank == 2

    assert response.results[1].evidence.score == 0.91
    assert response.results[1].evidence.rank == 1


def test_reranker_can_return_top_n_subset() -> None:
    """A valid reranker may return only the requested best candidates."""
    response = build_reranked_retrieval_response(
        _retrieval_response(),
        RerankResponse(
            model="synthetic-reranker",
            results=(
                RerankResult(
                    document_id="chunk-b",
                    score=9.0,
                    rank=1,
                ),
            ),
        ),
    )

    assert len(response.results) == 1

    assert response.results[0].evidence.chunk_id == "chunk-b"


def test_unknown_rerank_candidate_fails_closed() -> None:
    """Providers cannot inject evidence outside the retrieved candidate set."""
    with pytest.raises(
        ProviderExecutionError,
        match=("Reranker returned unknown candidate identifier"),
    ):
        build_reranked_retrieval_response(
            _retrieval_response(),
            RerankResponse(
                model="synthetic-reranker",
                results=(
                    RerankResult(
                        document_id="unknown-chunk",
                        score=9.0,
                        rank=1,
                    ),
                ),
            ),
        )


def test_boolean_rerank_score_fails_closed() -> None:
    """A weak provider contract value must not escape into domain output."""
    response = RerankResponse(
        model="synthetic-reranker",
        results=(
            RerankResult(
                document_id="chunk-a",
                score=True,
                rank=1,
            ),
        ),
    )

    with pytest.raises(
        ProviderExecutionError,
        match=("Reranker returned invalid ranking result"),
    ):
        build_reranked_retrieval_response(
            _retrieval_response(),
            response,
        )


def test_reranking_mapping_is_publicly_exported() -> None:
    """Pure reranking helpers should be public service utilities."""
    import ai_engineering_agent_platform.services as services

    assert {
        "build_rerank_request",
        "build_reranked_retrieval_response",
    } <= set(services.__all__)

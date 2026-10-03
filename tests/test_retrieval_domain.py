"""Tests for infrastructure-independent retrieval domain primitives."""

from dataclasses import FrozenInstanceError
from math import inf, nan

import pytest

from ai_engineering_agent_platform.domain import (
    DocumentChunk,
    KnowledgeDocument,
    RetrievalMetadataItem,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedEvidence,
)


def _metadata() -> tuple[RetrievalMetadataItem, ...]:
    """Return deterministic synthetic metadata."""
    return (
        RetrievalMetadataItem(
            key="category",
            value="synthetic",
        ),
        RetrievalMetadataItem(
            key="version",
            value=1,
        ),
    )


def _evidence(
    *,
    chunk_id: str = "doc-1:chunk-0",
    rank: int = 1,
) -> RetrievedEvidence:
    """Return deterministic synthetic retrieved evidence."""
    text = "synthetic evidence"

    return RetrievedEvidence(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=text,
        score=0.75,
        rank=rank,
        start_char=0,
        end_char=len(text),
        source_ref="synthetic://doc-1",
        title="Synthetic Document",
        metadata=_metadata(),
    )


def test_retrieval_metadata_is_immutable() -> None:
    """Metadata values should remain immutable."""
    item = RetrievalMetadataItem(
        key="category",
        value="synthetic",
    )

    with pytest.raises(
        FrozenInstanceError,
    ):
        item.key = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("key", "value", "match"),
    (
        (
            " ",
            "value",
            "metadata key must not be empty",
        ),
        (
            "key",
            nan,
            "metadata float values must be finite",
        ),
        (
            "key",
            inf,
            "metadata float values must be finite",
        ),
    ),
)
def test_retrieval_metadata_rejects_invalid_values(
    key: str,
    value: object,
    match: str,
) -> None:
    """Invalid metadata must fail closed."""
    with pytest.raises(
        ValueError,
        match=match,
    ):
        RetrievalMetadataItem(
            key=key,
            value=value,  # type: ignore[arg-type]
        )


def test_knowledge_document_preserves_provenance() -> None:
    """Documents should retain caller-supplied source provenance."""
    document = KnowledgeDocument(
        document_id="doc-1",
        text="synthetic document text",
        source_ref="synthetic://doc-1",
        title="Synthetic Document",
        metadata=_metadata(),
    )

    assert document.document_id == "doc-1"
    assert document.source_ref == "synthetic://doc-1"
    assert document.metadata == _metadata()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        (
            {
                "document_id": " ",
                "text": "text",
                "source_ref": "source",
            },
            "document_id must not be empty",
        ),
        (
            {
                "document_id": "doc",
                "text": " ",
                "source_ref": "source",
            },
            "document text must not be empty",
        ),
        (
            {
                "document_id": "doc",
                "text": "text",
                "source_ref": " ",
            },
            "source_ref must not be empty",
        ),
        (
            {
                "document_id": "doc",
                "text": "text",
                "source_ref": "source",
                "title": " ",
            },
            "title must not be empty",
        ),
    ),
)
def test_knowledge_document_rejects_invalid_text_fields(
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Document identity and provenance must be meaningful."""
    with pytest.raises(
        ValueError,
        match=match,
    ):
        KnowledgeDocument(
            **kwargs,  # type: ignore[arg-type]
        )


def test_knowledge_document_rejects_duplicate_metadata_keys() -> None:
    """Domain metadata keys must remain unambiguous."""
    duplicate_metadata = (
        RetrievalMetadataItem(
            key="duplicate",
            value=1,
        ),
        RetrievalMetadataItem(
            key="duplicate",
            value=2,
        ),
    )

    with pytest.raises(
        ValueError,
        match="metadata keys must be unique",
    ):
        KnowledgeDocument(
            document_id="doc",
            text="text",
            source_ref="source",
            metadata=duplicate_metadata,
        )


def test_document_chunk_requires_exact_offsets() -> None:
    """Chunk text must correspond to its declared character range."""
    chunk = DocumentChunk(
        chunk_id="doc-1:chunk-0",
        document_id="doc-1",
        text="hello",
        index=0,
        start_char=5,
        end_char=10,
        source_ref="synthetic://doc-1",
    )

    assert chunk.start_char == 5
    assert chunk.end_char == 10

    with pytest.raises(
        ValueError,
        match="chunk offsets must match chunk text length",
    ):
        DocumentChunk(
            chunk_id="doc-1:chunk-0",
            document_id="doc-1",
            text="hello",
            index=0,
            start_char=5,
            end_char=11,
            source_ref="synthetic://doc-1",
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        (
            "index",
            True,
            "index must be an integer",
        ),
        (
            "index",
            -1,
            "index must be non-negative",
        ),
        (
            "start_char",
            True,
            "start_char must be an integer",
        ),
        (
            "start_char",
            -1,
            "start_char must be non-negative",
        ),
    ),
)
def test_document_chunk_uses_strict_non_negative_integer_fields(
    field: str,
    value: object,
    match: str,
) -> None:
    """Chunk ordering and offsets must reject boolean or negative values."""
    kwargs: dict[str, object] = {
        "chunk_id": "chunk",
        "document_id": "doc",
        "text": "hello",
        "index": 0,
        "start_char": 0,
        "end_char": 5,
        "source_ref": "source",
    }

    kwargs[field] = value

    with pytest.raises(
        ValueError,
        match=match,
    ):
        DocumentChunk(
            **kwargs,  # type: ignore[arg-type]
        )


def test_retrieval_request_validates_semantic_query() -> None:
    """Text queries should be validated before embedding generation."""
    request = RetrievalRequest(
        query="find synthetic evidence",
        top_k=5,
        namespace="test",
    )

    assert request.top_k == 5
    assert request.namespace == "test"

    with pytest.raises(
        ValueError,
        match="query must not be empty",
    ):
        RetrievalRequest(
            query=" ",
            top_k=5,
        )

    with pytest.raises(
        ValueError,
        match="top_k must be an integer",
    ):
        RetrievalRequest(
            query="query",
            top_k=True,
        )

    with pytest.raises(
        ValueError,
        match="top_k must be positive",
    ):
        RetrievalRequest(
            query="query",
            top_k=0,
        )

    with pytest.raises(
        ValueError,
        match="namespace must not be empty",
    ):
        RetrievalRequest(
            query="query",
            top_k=1,
            namespace=" ",
        )


@pytest.mark.parametrize(
    ("score", "match"),
    (
        (
            True,
            "retrieval score must be numeric",
        ),
        (
            nan,
            "retrieval score must be finite",
        ),
        (
            inf,
            "retrieval score must be finite",
        ),
    ),
)
def test_retrieved_evidence_rejects_invalid_scores(
    score: object,
    match: str,
) -> None:
    """Evidence scores must remain finite numeric values."""
    text = "synthetic"

    with pytest.raises(
        ValueError,
        match=match,
    ):
        RetrievedEvidence(
            chunk_id="chunk",
            document_id="doc",
            text=text,
            score=score,  # type: ignore[arg-type]
            rank=1,
            start_char=0,
            end_char=len(text),
            source_ref="source",
        )


def test_retrieved_evidence_requires_exact_offsets() -> None:
    """Citation-ready evidence must retain exact source offsets."""
    text = "synthetic"

    evidence = RetrievedEvidence(
        chunk_id="chunk",
        document_id="doc",
        text=text,
        score=0.5,
        rank=1,
        start_char=10,
        end_char=10 + len(text),
        source_ref="synthetic://doc",
    )

    assert evidence.end_char - evidence.start_char == len(evidence.text)

    with pytest.raises(
        ValueError,
        match="evidence offsets must match evidence text length",
    ):
        RetrievedEvidence(
            chunk_id="chunk",
            document_id="doc",
            text=text,
            score=0.5,
            rank=1,
            start_char=10,
            end_char=99,
            source_ref="synthetic://doc",
        )


def test_retrieval_response_allows_no_matches() -> None:
    """A valid semantic query may legitimately return no evidence."""
    response = RetrievalResponse(
        query="no match",
        results=(),
        namespace="test",
    )

    assert response.results == ()


def test_retrieval_response_requires_unique_contiguous_results() -> None:
    """Evidence ordering should be deterministic and unambiguous."""
    first = _evidence(
        chunk_id="chunk-1",
        rank=1,
    )
    second = _evidence(
        chunk_id="chunk-2",
        rank=2,
    )

    response = RetrievalResponse(
        query="synthetic",
        results=(
            first,
            second,
        ),
    )

    assert response.results == (
        first,
        second,
    )

    with pytest.raises(
        ValueError,
        match="retrieval result chunk identifiers must be unique",
    ):
        RetrievalResponse(
            query="synthetic",
            results=(
                first,
                _evidence(
                    chunk_id="chunk-1",
                    rank=2,
                ),
            ),
        )

    with pytest.raises(
        ValueError,
        match="retrieval results must use contiguous rank order",
    ):
        RetrievalResponse(
            query="synthetic",
            results=(
                first,
                _evidence(
                    chunk_id="chunk-2",
                    rank=3,
                ),
            ),
        )


def test_retrieval_domain_objects_are_publicly_exported() -> None:
    """Retrieval primitives should be available from the domain package."""
    import ai_engineering_agent_platform.domain as domain

    expected = {
        "DocumentChunk",
        "KnowledgeDocument",
        "RetrievalMetadataItem",
        "RetrievalMetadataValue",
        "RetrievalRequest",
        "RetrievalResponse",
        "RetrievedEvidence",
    }

    assert expected <= set(domain.__all__)


def test_reranked_evidence_preserves_original_retrieval_values() -> None:
    """Reranking must not overwrite original vector score or rank."""
    from ai_engineering_agent_platform.domain import (
        RerankedEvidence,
    )

    evidence = _evidence(
        chunk_id="chunk-1",
        rank=1,
    )

    reranked = RerankedEvidence(
        evidence=evidence,
        rerank_score=9.5,
        rank=2,
    )

    assert reranked.evidence is evidence
    assert reranked.evidence.score == 0.75
    assert reranked.evidence.rank == 1
    assert reranked.rerank_score == 9.5
    assert reranked.rank == 2


@pytest.mark.parametrize(
    ("score", "match"),
    (
        (
            True,
            "rerank_score must be numeric",
        ),
        (
            nan,
            "rerank_score must be finite",
        ),
        (
            inf,
            "rerank_score must be finite",
        ),
    ),
)
def test_reranked_evidence_rejects_invalid_scores(
    score: object,
    match: str,
) -> None:
    """Final reranking scores must be finite numeric values."""
    from ai_engineering_agent_platform.domain import (
        RerankedEvidence,
    )

    with pytest.raises(
        ValueError,
        match=match,
    ):
        RerankedEvidence(
            evidence=_evidence(),
            rerank_score=score,  # type: ignore[arg-type]
            rank=1,
        )


def test_reranked_response_requires_contiguous_unique_results() -> None:
    """Final reranked results must have deterministic rank ordering."""
    from ai_engineering_agent_platform.domain import (
        RerankedEvidence,
        RerankedRetrievalResponse,
    )

    first = RerankedEvidence(
        evidence=_evidence(
            chunk_id="chunk-1",
            rank=1,
        ),
        rerank_score=2.0,
        rank=1,
    )
    second = RerankedEvidence(
        evidence=_evidence(
            chunk_id="chunk-2",
            rank=2,
        ),
        rerank_score=1.0,
        rank=2,
    )

    response = RerankedRetrievalResponse(
        query="synthetic",
        model="synthetic-reranker",
        results=(
            first,
            second,
        ),
    )

    assert response.results == (
        first,
        second,
    )

    with pytest.raises(
        ValueError,
        match=("reranked evidence chunk identifiers must be unique"),
    ):
        RerankedRetrievalResponse(
            query="synthetic",
            model="synthetic-reranker",
            results=(
                first,
                RerankedEvidence(
                    evidence=_evidence(
                        chunk_id="chunk-1",
                        rank=2,
                    ),
                    rerank_score=1.0,
                    rank=2,
                ),
            ),
        )

    with pytest.raises(
        ValueError,
        match=("reranked results must use contiguous rank order"),
    ):
        RerankedRetrievalResponse(
            query="synthetic",
            model="synthetic-reranker",
            results=(
                first,
                RerankedEvidence(
                    evidence=_evidence(
                        chunk_id="chunk-2",
                        rank=2,
                    ),
                    rerank_score=1.0,
                    rank=3,
                ),
            ),
        )


def test_reranked_domain_objects_are_publicly_exported() -> None:
    """Reranked retrieval types should be public domain primitives."""
    import ai_engineering_agent_platform.domain as domain

    assert {
        "RerankedEvidence",
        "RerankedRetrievalResponse",
    } <= set(domain.__all__)

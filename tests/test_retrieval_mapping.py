"""Tests for pure semantic-retrieval provider mapping."""

import pytest

from ai_engineering_agent_platform.contracts import (
    EmbeddingResponse,
    EmbeddingVector,
    VectorMetadataItem,
    VectorQueryResponse,
    VectorQueryResult,
)
from ai_engineering_agent_platform.domain import (
    ProviderExecutionError,
    RetrievalRequest,
)
from ai_engineering_agent_platform.services import (
    build_retrieval_embedding_request,
    build_retrieval_response,
    build_vector_query_request,
)


def _request(
    *,
    top_k: int = 2,
) -> RetrievalRequest:
    """Return one deterministic synthetic retrieval request."""
    return RetrievalRequest(
        query="synthetic alpha",
        top_k=top_k,
        namespace="synthetic",
    )


def _embedding_response(
    *,
    count: int = 1,
    dimensions: int = 3,
) -> EmbeddingResponse:
    """Return deterministic synthetic query embeddings."""
    return EmbeddingResponse(
        model="synthetic-model",
        embeddings=tuple(
            EmbeddingVector(
                values=tuple(
                    float(vector_index + dimension + 1) / 10.0
                    for dimension in range(dimensions)
                ),
            )
            for vector_index in range(count)
        ),
    )


def _metadata(
    *,
    document_id: object = "doc-1",
    chunk_index: object = 0,
    start_char: object = 0,
    end_char: object = 15,
    source_ref: object = "synthetic://doc-1",
    title: object = "Synthetic Document",
    extra: tuple[
        VectorMetadataItem,
        ...,
    ] = (),
) -> tuple[VectorMetadataItem, ...]:
    """Return the project-owned persisted retrieval metadata shape."""
    return (
        VectorMetadataItem(
            key="retrieval.document_id",
            value=document_id,  # type: ignore[arg-type]
        ),
        VectorMetadataItem(
            key="retrieval.chunk_index",
            value=chunk_index,  # type: ignore[arg-type]
        ),
        VectorMetadataItem(
            key="retrieval.start_char",
            value=start_char,  # type: ignore[arg-type]
        ),
        VectorMetadataItem(
            key="retrieval.end_char",
            value=end_char,  # type: ignore[arg-type]
        ),
        VectorMetadataItem(
            key="retrieval.source_ref",
            value=source_ref,  # type: ignore[arg-type]
        ),
        VectorMetadataItem(
            key="retrieval.title",
            value=title,  # type: ignore[arg-type]
        ),
        VectorMetadataItem(
            key="source.category",
            value="synthetic",
        ),
        *extra,
    )


def _result(
    *,
    record_id: str = "doc-1:chunk:0:0:15",
    rank: int = 1,
    text: str | None = "synthetic alpha",
    metadata: tuple[
        VectorMetadataItem,
        ...,
    ]
    | None = None,
) -> VectorQueryResult:
    """Return one deterministic vector-query result."""
    return VectorQueryResult(
        record_id=record_id,
        score=0.9,
        rank=rank,
        text=text,
        metadata=(_metadata() if metadata is None else metadata),
    )


def test_retrieval_embedding_request_contains_only_query() -> None:
    """Semantic retrieval should embed one query text."""
    request = build_retrieval_embedding_request(
        _request(),
        model="synthetic-model",
        dimensions=256,
    )

    assert request.model == "synthetic-model"
    assert request.texts == ("synthetic alpha",)
    assert request.dimensions == 256


def test_vector_query_request_preserves_retrieval_constraints() -> None:
    """Query vector, top-k and namespace must map exactly."""
    request = build_vector_query_request(
        _request(),
        _embedding_response(),
    )

    assert request.vector == (
        0.1,
        0.2,
        0.3,
    )
    assert request.top_k == 2
    assert request.namespace == "synthetic"


def test_vector_query_rejects_multiple_query_embeddings() -> None:
    """One semantic query must produce exactly one query vector."""
    with pytest.raises(
        ProviderExecutionError,
        match=("Embedding provider returned unexpected result count"),
    ):
        build_vector_query_request(
            _request(),
            _embedding_response(count=2),
        )


def test_empty_vector_response_maps_to_empty_evidence() -> None:
    """No vector matches is a valid retrieval outcome."""
    response = build_retrieval_response(
        _request(),
        VectorQueryResponse(results=()),
    )

    assert response.query == "synthetic alpha"
    assert response.namespace == "synthetic"
    assert response.results == ()


def test_vector_result_maps_to_citation_ready_evidence() -> None:
    """Persisted provenance must be restored into domain evidence."""
    response = build_retrieval_response(
        _request(),
        VectorQueryResponse(results=(_result(),)),
    )

    assert len(response.results) == 1

    evidence = response.results[0]

    assert evidence.chunk_id == ("doc-1:chunk:0:0:15")
    assert evidence.document_id == "doc-1"
    assert evidence.text == "synthetic alpha"
    assert evidence.score == 0.9
    assert evidence.rank == 1
    assert evidence.start_char == 0
    assert evidence.end_char == 15
    assert evidence.source_ref == "synthetic://doc-1"
    assert evidence.title == "Synthetic Document"

    assert tuple(
        (
            item.key,
            item.value,
        )
        for item in evidence.metadata
    ) == (
        (
            "category",
            "synthetic",
        ),
    )


def test_source_metadata_cannot_override_reserved_provenance() -> None:
    """Source metadata stays separate after retrieval restoration."""
    metadata = _metadata(
        extra=(
            VectorMetadataItem(
                key="source.retrieval.document_id",
                value="attacker-controlled",
            ),
        )
    )

    response = build_retrieval_response(
        _request(),
        VectorQueryResponse(
            results=(
                _result(
                    metadata=metadata,
                ),
            )
        ),
    )

    evidence = response.results[0]

    assert evidence.document_id == "doc-1"

    assert tuple(
        (
            item.key,
            item.value,
        )
        for item in evidence.metadata
    ) == (
        (
            "category",
            "synthetic",
        ),
        (
            "retrieval.document_id",
            "attacker-controlled",
        ),
    )


def test_missing_result_text_fails_closed() -> None:
    """Retrieval evidence cannot be citation-ready without source text."""
    with pytest.raises(
        ProviderExecutionError,
        match="result text is required",
    ):
        build_retrieval_response(
            _request(),
            VectorQueryResponse(
                results=(
                    _result(
                        text=None,
                    ),
                )
            ),
        )


@pytest.mark.parametrize(
    ("missing_key", "match"),
    (
        (
            "retrieval.document_id",
            "missing retrieval.document_id",
        ),
        (
            "retrieval.chunk_index",
            "missing retrieval.chunk_index",
        ),
        (
            "retrieval.start_char",
            "missing retrieval.start_char",
        ),
        (
            "retrieval.end_char",
            "missing retrieval.end_char",
        ),
        (
            "retrieval.source_ref",
            "missing retrieval.source_ref",
        ),
        (
            "retrieval.title",
            "missing retrieval.title",
        ),
    ),
)
def test_missing_required_provenance_fails_closed(
    missing_key: str,
    match: str,
) -> None:
    """Every project-owned provenance field is mandatory."""
    metadata = tuple(item for item in _metadata() if item.key != missing_key)

    with pytest.raises(
        ProviderExecutionError,
        match=match,
    ):
        build_retrieval_response(
            _request(),
            VectorQueryResponse(
                results=(
                    _result(
                        metadata=metadata,
                    ),
                )
            ),
        )


@pytest.mark.parametrize(
    ("metadata", "match"),
    (
        (
            _metadata(
                document_id=123,
            ),
            ("retrieval.document_id must be a string"),
        ),
        (
            _metadata(
                chunk_index=True,
            ),
            ("retrieval.chunk_index must be an integer"),
        ),
        (
            _metadata(
                start_char=-1,
            ),
            ("retrieval.start_char must be non-negative"),
        ),
        (
            _metadata(
                end_char=True,
            ),
            ("retrieval.end_char must be an integer"),
        ),
        (
            _metadata(
                source_ref=123,
            ),
            ("retrieval.source_ref must be a string"),
        ),
        (
            _metadata(
                title=123,
            ),
            ("retrieval.title must be a string or null"),
        ),
    ),
)
def test_invalid_provenance_types_fail_closed(
    metadata: tuple[
        VectorMetadataItem,
        ...,
    ],
    match: str,
) -> None:
    """Typed retrieval provenance should not be guessed or coerced."""
    with pytest.raises(
        ProviderExecutionError,
        match=match,
    ):
        build_retrieval_response(
            _request(),
            VectorQueryResponse(
                results=(
                    _result(
                        metadata=metadata,
                    ),
                )
            ),
        )


def test_offset_text_mismatch_fails_closed() -> None:
    """Corrupt source offsets must not become citation-ready evidence."""
    metadata = _metadata(
        end_char=999,
    )

    with pytest.raises(
        ProviderExecutionError,
        match="evidence invariants failed",
    ):
        build_retrieval_response(
            _request(),
            VectorQueryResponse(
                results=(
                    _result(
                        metadata=metadata,
                    ),
                )
            ),
        )


def test_unknown_structural_metadata_fails_closed() -> None:
    """Unknown vector metadata must not be silently trusted."""
    metadata = _metadata(
        extra=(
            VectorMetadataItem(
                key="unexpected.field",
                value="value",
            ),
        )
    )

    with pytest.raises(
        ProviderExecutionError,
        match="unsupported metadata key",
    ):
        build_retrieval_response(
            _request(),
            VectorQueryResponse(
                results=(
                    _result(
                        metadata=metadata,
                    ),
                )
            ),
        )


def test_vector_store_cannot_return_more_than_top_k() -> None:
    """Provider result count must respect the requested upper bound."""
    request = _request(top_k=1)

    response = VectorQueryResponse(
        results=(
            _result(
                record_id="chunk-1",
                rank=1,
            ),
            _result(
                record_id="chunk-2",
                rank=2,
            ),
        )
    )

    with pytest.raises(
        ProviderExecutionError,
        match=("Vector store returned more results than requested"),
    ):
        build_retrieval_response(
            request,
            response,
        )


def test_retrieval_mapping_is_publicly_exported() -> None:
    """Retrieval mapping helpers should be public service utilities."""
    import ai_engineering_agent_platform.services as services

    assert {
        "build_retrieval_embedding_request",
        "build_retrieval_response",
        "build_vector_query_request",
    } <= set(services.__all__)

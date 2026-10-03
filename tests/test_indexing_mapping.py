"""Tests for pure indexing-provider mapping."""

import pytest

from ai_engineering_agent_platform.contracts import (
    EmbeddingResponse,
    EmbeddingVector,
)
from ai_engineering_agent_platform.domain import (
    DocumentChunk,
    RetrievalMetadataItem,
)
from ai_engineering_agent_platform.services import (
    build_embedding_request,
    build_vector_upsert_request,
)


def _chunks() -> tuple[DocumentChunk, ...]:
    """Return deterministic synthetic chunks."""
    return (
        DocumentChunk(
            chunk_id="doc-1:chunk:0:0:5",
            document_id="doc-1",
            text="alpha",
            index=0,
            start_char=0,
            end_char=5,
            source_ref="synthetic://doc-1",
            title="Synthetic Document",
            metadata=(
                RetrievalMetadataItem(
                    key="category",
                    value="synthetic",
                ),
                RetrievalMetadataItem(
                    key="version",
                    value=1,
                ),
            ),
        ),
        DocumentChunk(
            chunk_id="doc-1:chunk:1:5:9",
            document_id="doc-1",
            text="beta",
            index=1,
            start_char=5,
            end_char=9,
            source_ref="synthetic://doc-1",
            title="Synthetic Document",
            metadata=(
                RetrievalMetadataItem(
                    key="category",
                    value="synthetic",
                ),
                RetrievalMetadataItem(
                    key="version",
                    value=1,
                ),
            ),
        ),
    )


def _embedding_response() -> EmbeddingResponse:
    """Return embeddings aligned with the deterministic chunks."""
    return EmbeddingResponse(
        model="synthetic-embedding-model",
        embeddings=(
            EmbeddingVector(
                values=(
                    0.1,
                    0.2,
                    0.3,
                ),
            ),
            EmbeddingVector(
                values=(
                    0.4,
                    0.5,
                    0.6,
                ),
            ),
        ),
        input_tokens=2,
    )


def test_embedding_request_preserves_chunk_order_and_text() -> None:
    """Chunk ordering must define embedding ordering deterministically."""
    chunks = _chunks()

    request = build_embedding_request(
        chunks,
        model="synthetic-embedding-model",
        dimensions=3,
    )

    assert request.model == "synthetic-embedding-model"
    assert request.dimensions == 3
    assert request.texts == (
        "alpha",
        "beta",
    )


def test_embedding_request_rejects_empty_chunk_batch() -> None:
    """Indexing should never request embeddings for an empty batch."""
    with pytest.raises(
        ValueError,
        match="chunks must not be empty",
    ):
        build_embedding_request(
            (),
            model="synthetic-model",
        )


def test_embedding_request_delegates_model_validation() -> None:
    """Existing embedding-contract validation must remain authoritative."""
    with pytest.raises(
        ValueError,
        match="model must not be empty",
    ):
        build_embedding_request(
            _chunks(),
            model=" ",
        )


def test_vector_upsert_preserves_record_order_and_vectors() -> None:
    """Embedding position must map exactly to the corresponding chunk."""
    chunks = _chunks()

    request = build_vector_upsert_request(
        chunks,
        _embedding_response(),
        namespace="synthetic",
    )

    assert request.namespace == "synthetic"

    assert tuple(record.record_id for record in request.records) == (
        "doc-1:chunk:0:0:5",
        "doc-1:chunk:1:5:9",
    )

    assert request.records[0].vector == (
        0.1,
        0.2,
        0.3,
    )

    assert request.records[1].vector == (
        0.4,
        0.5,
        0.6,
    )

    assert request.records[0].text == "alpha"
    assert request.records[1].text == "beta"


def test_vector_metadata_contains_stable_provenance() -> None:
    """Chunk provenance should survive the vector-store mapping."""
    request = build_vector_upsert_request(
        _chunks(),
        _embedding_response(),
    )

    metadata = {item.key: item.value for item in request.records[0].metadata}

    assert metadata == {
        "retrieval.document_id": "doc-1",
        "retrieval.chunk_index": 0,
        "retrieval.start_char": 0,
        "retrieval.end_char": 5,
        "retrieval.source_ref": "synthetic://doc-1",
        "retrieval.title": "Synthetic Document",
        "source.category": "synthetic",
        "source.version": 1,
    }


def test_source_metadata_cannot_overwrite_reserved_provenance() -> None:
    """Caller metadata should remain separated from retrieval metadata."""
    chunk = DocumentChunk(
        chunk_id="chunk",
        document_id="real-document",
        text="text",
        index=0,
        start_char=0,
        end_char=4,
        source_ref="synthetic://source",
        metadata=(
            RetrievalMetadataItem(
                key="retrieval.document_id",
                value="attacker-controlled",
            ),
        ),
    )

    response = EmbeddingResponse(
        model="model",
        embeddings=(
            EmbeddingVector(
                values=(
                    0.1,
                    0.2,
                ),
            ),
        ),
    )

    request = build_vector_upsert_request(
        (chunk,),
        response,
    )

    metadata = {item.key: item.value for item in request.records[0].metadata}

    assert metadata["retrieval.document_id"] == "real-document"

    assert metadata["source.retrieval.document_id"] == "attacker-controlled"


def test_none_title_has_stable_metadata_shape() -> None:
    """Missing titles should remain explicit and portable."""
    chunk = DocumentChunk(
        chunk_id="chunk",
        document_id="doc",
        text="text",
        index=0,
        start_char=0,
        end_char=4,
        source_ref="synthetic://source",
    )

    response = EmbeddingResponse(
        model="model",
        embeddings=(
            EmbeddingVector(
                values=(
                    0.1,
                    0.2,
                ),
            ),
        ),
    )

    request = build_vector_upsert_request(
        (chunk,),
        response,
    )

    metadata = {item.key: item.value for item in request.records[0].metadata}

    assert "retrieval.title" in metadata
    assert metadata["retrieval.title"] is None


def test_vector_upsert_rejects_empty_chunk_batch() -> None:
    """No vector upsert should be built for an empty indexing batch."""
    with pytest.raises(
        ValueError,
        match="chunks must not be empty",
    ):
        build_vector_upsert_request(
            (),
            _embedding_response(),
        )


def test_vector_upsert_rejects_embedding_count_mismatch() -> None:
    """Provider cardinality mismatches must fail before persistence."""
    chunks = _chunks()

    response = EmbeddingResponse(
        model="model",
        embeddings=(
            EmbeddingVector(
                values=(
                    0.1,
                    0.2,
                ),
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="embedding count must match chunk count",
    ):
        build_vector_upsert_request(
            chunks,
            response,
        )


def test_namespace_validation_remains_vector_contract_owned() -> None:
    """Mapping must preserve the vector-store namespace contract."""
    with pytest.raises(
        ValueError,
        match="namespace must not be empty",
    ):
        build_vector_upsert_request(
            _chunks(),
            _embedding_response(),
            namespace=" ",
        )


def test_indexing_mapping_is_publicly_exported() -> None:
    """Pure indexing helpers should remain exposed through services."""
    import ai_engineering_agent_platform.services as services

    assert {
        "build_embedding_request",
        "build_vector_upsert_request",
    } <= set(services.__all__)

"""Pure mapping between retrieval-domain chunks and provider contracts."""

from ai_engineering_agent_platform.contracts import (
    EmbeddingRequest,
    EmbeddingResponse,
    VectorMetadataItem,
    VectorRecord,
    VectorUpsertRequest,
)
from ai_engineering_agent_platform.domain import (
    DocumentChunk,
)

_RETRIEVAL_METADATA_PREFIX = "retrieval."
_SOURCE_METADATA_PREFIX = "source."


def _chunk_vector_metadata(
    chunk: DocumentChunk,
) -> tuple[VectorMetadataItem, ...]:
    """Map chunk provenance into collision-resistant vector metadata."""
    provenance = (
        VectorMetadataItem(
            key=f"{_RETRIEVAL_METADATA_PREFIX}document_id",
            value=chunk.document_id,
        ),
        VectorMetadataItem(
            key=f"{_RETRIEVAL_METADATA_PREFIX}chunk_index",
            value=chunk.index,
        ),
        VectorMetadataItem(
            key=f"{_RETRIEVAL_METADATA_PREFIX}start_char",
            value=chunk.start_char,
        ),
        VectorMetadataItem(
            key=f"{_RETRIEVAL_METADATA_PREFIX}end_char",
            value=chunk.end_char,
        ),
        VectorMetadataItem(
            key=f"{_RETRIEVAL_METADATA_PREFIX}source_ref",
            value=chunk.source_ref,
        ),
        VectorMetadataItem(
            key=f"{_RETRIEVAL_METADATA_PREFIX}title",
            value=chunk.title,
        ),
    )

    source_metadata = tuple(
        VectorMetadataItem(
            key=(f"{_SOURCE_METADATA_PREFIX}{item.key}"),
            value=item.value,
        )
        for item in chunk.metadata
    )

    return (
        *provenance,
        *source_metadata,
    )


def build_embedding_request(
    chunks: tuple[DocumentChunk, ...],
    *,
    model: str,
    dimensions: int | None = None,
) -> EmbeddingRequest:
    """Build one provider-neutral embedding request from ordered chunks."""
    if not chunks:
        raise ValueError("chunks must not be empty")

    return EmbeddingRequest(
        model=model,
        texts=tuple(chunk.text for chunk in chunks),
        dimensions=dimensions,
    )


def build_vector_upsert_request(
    chunks: tuple[DocumentChunk, ...],
    embedding_response: EmbeddingResponse,
    *,
    namespace: str | None = None,
    space_id: str | None = None,
) -> VectorUpsertRequest:
    """Map ordered chunks and embeddings into vector-store records."""
    if not chunks:
        raise ValueError("chunks must not be empty")

    if len(chunks) != len(embedding_response.embeddings):
        raise ValueError("embedding count must match chunk count")

    records = tuple(
        VectorRecord(
            record_id=chunk.chunk_id,
            vector=embedding.values,
            text=chunk.text,
            metadata=_chunk_vector_metadata(chunk),
        )
        for chunk, embedding in zip(
            chunks,
            embedding_response.embeddings,
            strict=True,
        )
    )

    resolved_space_id = embedding_response.model if space_id is None else space_id

    return VectorUpsertRequest(
        records=records,
        space_id=resolved_space_id,
        namespace=namespace,
    )

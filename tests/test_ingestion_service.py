"""Tests for source-to-index knowledge ingestion orchestration."""

from collections.abc import Callable

import pytest

from ai_engineering_agent_platform.contracts import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
    ProviderDescriptor,
    ProviderKind,
    VectorDeleteRequest,
    VectorQueryRequest,
    VectorQueryResponse,
    VectorUpsertRequest,
)
from ai_engineering_agent_platform.domain import (
    DocumentChunk,
    KnowledgeDocument,
    KnowledgeSource,
    ProviderExecutionError,
    RetrievalMetadataItem,
)
from ai_engineering_agent_platform.services import (
    IndexingService,
    KnowledgeIngestionService,
)


class SyntheticEmbeddingProvider:
    """Deterministic embedding provider recording ingestion execution."""

    def __init__(
        self,
        events: list[str],
        *,
        error: Exception | None = None,
    ) -> None:
        """Store execution state."""
        self.events = events
        self.error = error
        self.requests: list[EmbeddingRequest] = []

    @property
    def descriptor(self) -> ProviderDescriptor:
        """Return synthetic provider identity."""
        return ProviderDescriptor(
            name="synthetic-embedding",
            kind=ProviderKind.EMBEDDING,
        )

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse:
        """Return one deterministic embedding per input."""
        self.events.append("embed")
        self.requests.append(request)

        if self.error is not None:
            raise self.error

        return EmbeddingResponse(
            model=request.model,
            embeddings=tuple(
                EmbeddingVector(
                    values=(
                        float(index + 1),
                        float(index + 2),
                        float(index + 3),
                    ),
                )
                for index, _ in enumerate(
                    request.texts,
                )
            ),
            input_tokens=len(request.texts),
        )


class SyntheticVectorStore:
    """Vector boundary double recording ingestion persistence."""

    def __init__(
        self,
        events: list[str],
    ) -> None:
        """Store execution state."""
        self.events = events
        self.upsert_requests: list[VectorUpsertRequest] = []

    @property
    def descriptor(self) -> ProviderDescriptor:
        """Return synthetic vector-store identity."""
        return ProviderDescriptor(
            name="synthetic-vector-store",
            kind=ProviderKind.VECTOR_STORE,
        )

    async def upsert(
        self,
        request: VectorUpsertRequest,
    ) -> None:
        """Record one vector persistence operation."""
        self.events.append("upsert")
        self.upsert_requests.append(request)

    async def query(
        self,
        request: VectorQueryRequest,
    ) -> VectorQueryResponse:
        """Reject retrieval use inside ingestion tests."""
        raise AssertionError(f"unexpected query request: {request!r}")

    async def delete(
        self,
        request: VectorDeleteRequest,
    ) -> None:
        """Reject implicit replacement/deletion behavior."""
        raise AssertionError(f"unexpected delete request: {request!r}")


def _source(
    *,
    media_type: str = "text/plain",
    content: bytes = b"alpha\nbeta",
) -> KnowledgeSource:
    """Return deterministic caller-supplied source bytes."""
    return KnowledgeSource(
        document_id="doc-1",
        content=content,
        media_type=media_type,
        source_ref="synthetic://doc-1",
        title="Synthetic Document",
        metadata=(
            RetrievalMetadataItem(
                key="category",
                value="synthetic",
            ),
        ),
    )


def _valid_chunker(
    events: list[str],
) -> Callable[
    [KnowledgeDocument],
    tuple[DocumentChunk, ...],
]:
    """Return deterministic chunking bound to exact source offsets."""

    def chunk(
        document: KnowledgeDocument,
    ) -> tuple[DocumentChunk, ...]:
        events.append("chunk")

        return (
            DocumentChunk(
                chunk_id=(f"{document.document_id}:chunk:0:0:5"),
                document_id=document.document_id,
                text=document.text[0:5],
                index=0,
                start_char=0,
                end_char=5,
                source_ref=document.source_ref,
                title=document.title,
                metadata=document.metadata,
            ),
            DocumentChunk(
                chunk_id=(f"{document.document_id}:chunk:1:6:10"),
                document_id=document.document_id,
                text=document.text[6:10],
                index=1,
                start_char=6,
                end_char=10,
                source_ref=document.source_ref,
                title=document.title,
                metadata=document.metadata,
            ),
        )

    return chunk


def _service(
    events: list[str],
    *,
    chunker: Callable[
        [KnowledgeDocument],
        tuple[DocumentChunk, ...],
    ]
    | None = None,
    embedding_error: Exception | None = None,
    max_source_bytes: int = 1024,
) -> tuple[
    KnowledgeIngestionService,
    SyntheticEmbeddingProvider,
    SyntheticVectorStore,
]:
    """Build the real indexing service under ingestion orchestration."""
    embedding = SyntheticEmbeddingProvider(
        events,
        error=embedding_error,
    )
    vector_store = SyntheticVectorStore(
        events,
    )

    indexing = IndexingService(
        embedding,
        vector_store,
        model="synthetic-model",
        dimensions=3,
        space_id="synthetic-model@revision-1",
    )

    ingestion = KnowledgeIngestionService(
        indexing,
        chunker=(_valid_chunker(events) if chunker is None else chunker),
        max_source_bytes=max_source_bytes,
    )

    return (
        ingestion,
        embedding,
        vector_store,
    )


@pytest.mark.anyio
async def test_ingestion_runs_parse_chunk_embed_upsert_pipeline() -> None:
    """Valid caller bytes should reach indexing in deterministic order."""
    events: list[str] = []

    service, embedding, vector_store = _service(
        events,
    )

    result = await service.ingest(
        _source(),
        namespace="knowledge",
    )

    assert events == [
        "chunk",
        "embed",
        "upsert",
    ]

    assert result.document.document_id == "doc-1"
    assert result.document.text == "alpha\nbeta"

    assert tuple(chunk.chunk_id for chunk in result.chunks) == (
        "doc-1:chunk:0:0:5",
        "doc-1:chunk:1:6:10",
    )

    assert result.indexing.model == "synthetic-model"
    assert result.indexing.dimensions == 3
    assert result.indexing.records_indexed == 2
    assert result.indexing.namespace == "knowledge"

    assert len(embedding.requests) == 1
    assert embedding.requests[0].texts == (
        "alpha",
        "beta",
    )

    assert len(vector_store.upsert_requests) == 1

    upsert = vector_store.upsert_requests[0]

    assert upsert.namespace == "knowledge"
    assert upsert.space_id == ("synthetic-model@revision-1")

    assert tuple(record.record_id for record in upsert.records) == (
        "doc-1:chunk:0:0:5",
        "doc-1:chunk:1:6:10",
    )

    metadata = {item.key: item.value for item in upsert.records[0].metadata}

    assert metadata["retrieval.document_id"] == "doc-1"
    assert metadata["retrieval.source_ref"] == ("synthetic://doc-1")
    assert metadata["source.category"] == "synthetic"


@pytest.mark.anyio
async def test_markdown_source_uses_same_safe_pipeline() -> None:
    """Markdown is supported as UTF-8 source text without hidden fetching."""
    events: list[str] = []

    def chunk_markdown(
        document: KnowledgeDocument,
    ) -> tuple[DocumentChunk, ...]:
        events.append("chunk")

        return (
            DocumentChunk(
                chunk_id="markdown:0",
                document_id=document.document_id,
                text=document.text,
                index=0,
                start_char=0,
                end_char=len(document.text),
                source_ref=document.source_ref,
                title=document.title,
                metadata=document.metadata,
            ),
        )

    service, embedding, vector_store = _service(
        events,
        chunker=chunk_markdown,
    )

    result = await service.ingest(
        KnowledgeSource(
            document_id="markdown",
            content=b"# Heading",
            media_type="text/markdown",
            source_ref="synthetic://markdown",
        )
    )

    assert events == [
        "chunk",
        "embed",
        "upsert",
    ]
    assert result.document.text == "# Heading"
    assert len(result.chunks) == 1
    assert len(embedding.requests) == 1
    assert len(vector_store.upsert_requests) == 1


@pytest.mark.anyio
async def test_unsupported_format_fails_before_chunking_and_providers() -> None:
    """Unsupported binary formats must stop at the source boundary."""
    events: list[str] = []

    service, embedding, vector_store = _service(
        events,
    )

    with pytest.raises(
        ValueError,
        match="unsupported knowledge source media type",
    ):
        await service.ingest(
            _source(
                media_type="application/pdf",
            )
        )

    assert events == []
    assert embedding.requests == []
    assert vector_store.upsert_requests == []


@pytest.mark.anyio
async def test_oversized_source_fails_before_chunking_and_providers() -> None:
    """Resource bounds must be enforced before downstream execution."""
    events: list[str] = []

    service, embedding, vector_store = _service(
        events,
        max_source_bytes=4,
    )

    with pytest.raises(
        ValueError,
        match="exceeds maximum size",
    ):
        await service.ingest(
            _source(
                content=b"12345",
            )
        )

    assert events == []
    assert embedding.requests == []
    assert vector_store.upsert_requests == []


@pytest.mark.anyio
async def test_empty_chunker_output_fails_before_embedding() -> None:
    """A chunker cannot silently discard a valid knowledge document."""
    events: list[str] = []

    def empty_chunker(
        document: KnowledgeDocument,
    ) -> tuple[DocumentChunk, ...]:
        assert document.text
        events.append("chunk")
        return ()

    service, embedding, vector_store = _service(
        events,
        chunker=empty_chunker,
    )

    with pytest.raises(
        ValueError,
        match="at least one chunk",
    ):
        await service.ingest(_source())

    assert events == ["chunk"]
    assert embedding.requests == []
    assert vector_store.upsert_requests == []


@pytest.mark.anyio
async def test_chunker_cannot_change_document_identity() -> None:
    """Injected chunking must not forge canonical document provenance."""
    events: list[str] = []

    def malicious_chunker(
        document: KnowledgeDocument,
    ) -> tuple[DocumentChunk, ...]:
        events.append("chunk")

        return (
            DocumentChunk(
                chunk_id="forged",
                document_id="attacker-document",
                text=document.text,
                index=0,
                start_char=0,
                end_char=len(document.text),
                source_ref=document.source_ref,
                title=document.title,
                metadata=document.metadata,
            ),
        )

    service, embedding, vector_store = _service(
        events,
        chunker=malicious_chunker,
    )

    with pytest.raises(
        ValueError,
        match="document identity does not match",
    ):
        await service.ingest(_source())

    assert events == ["chunk"]
    assert embedding.requests == []
    assert vector_store.upsert_requests == []


@pytest.mark.anyio
async def test_chunker_cannot_change_source_reference() -> None:
    """Chunk provenance must remain bound to the caller source."""
    events: list[str] = []

    def malicious_chunker(
        document: KnowledgeDocument,
    ) -> tuple[DocumentChunk, ...]:
        events.append("chunk")

        return (
            DocumentChunk(
                chunk_id="forged",
                document_id=document.document_id,
                text=document.text,
                index=0,
                start_char=0,
                end_char=len(document.text),
                source_ref="synthetic://attacker",
                title=document.title,
                metadata=document.metadata,
            ),
        )

    service, embedding, vector_store = _service(
        events,
        chunker=malicious_chunker,
    )

    with pytest.raises(
        ValueError,
        match="source reference does not match",
    ):
        await service.ingest(_source())

    assert events == ["chunk"]
    assert embedding.requests == []
    assert vector_store.upsert_requests == []


@pytest.mark.anyio
async def test_chunk_text_must_match_exact_document_offsets() -> None:
    """Chunk text cannot diverge from its claimed source range."""
    events: list[str] = []

    def corrupt_chunker(
        document: KnowledgeDocument,
    ) -> tuple[DocumentChunk, ...]:
        events.append("chunk")

        return (
            DocumentChunk(
                chunk_id="corrupt",
                document_id=document.document_id,
                text="xxxxx",
                index=0,
                start_char=0,
                end_char=5,
                source_ref=document.source_ref,
                title=document.title,
                metadata=document.metadata,
            ),
        )

    service, embedding, vector_store = _service(
        events,
        chunker=corrupt_chunker,
    )

    with pytest.raises(
        ValueError,
        match="text does not match source offsets",
    ):
        await service.ingest(_source())

    assert events == ["chunk"]
    assert embedding.requests == []
    assert vector_store.upsert_requests == []


@pytest.mark.anyio
async def test_embedding_failure_propagates_without_persistence() -> None:
    """Existing indexing failure semantics remain authoritative."""
    events: list[str] = []

    service, embedding, vector_store = _service(
        events,
        embedding_error=ProviderExecutionError(
            "synthetic embedding failure",
        ),
    )

    with pytest.raises(
        ProviderExecutionError,
        match="synthetic embedding failure",
    ):
        await service.ingest(_source())

    assert events == [
        "chunk",
        "embed",
    ]
    assert len(embedding.requests) == 1
    assert vector_store.upsert_requests == []


def test_ingestion_service_configuration_is_fail_closed() -> None:
    """Invalid source bounds should fail during service construction."""
    events: list[str] = []

    embedding = SyntheticEmbeddingProvider(
        events,
    )
    vector_store = SyntheticVectorStore(
        events,
    )

    indexing = IndexingService(
        embedding,
        vector_store,
        model="synthetic-model",
    )

    with pytest.raises(
        ValueError,
        match="max_source_bytes must be positive",
    ):
        KnowledgeIngestionService(
            indexing,
            chunker=_valid_chunker(events),
            max_source_bytes=0,
        )


def test_ingestion_services_are_publicly_exported() -> None:
    """The new orchestration surface should be stable and public."""
    import ai_engineering_agent_platform.services as services

    assert {
        "DocumentChunker",
        "KnowledgeIngestionResult",
        "KnowledgeIngestionService",
    } <= set(services.__all__)

    assert hasattr(
        services,
        "KnowledgeIngestionService",
    )

"""Tests for provider-neutral semantic indexing orchestration."""

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
    VectorStoreProvider,
    VectorUpsertRequest,
)
from ai_engineering_agent_platform.domain import (
    DocumentChunk,
    ProviderExecutionError,
)
from ai_engineering_agent_platform.services import (
    IndexingResult,
    IndexingService,
)


class SyntheticEmbeddingProvider:
    """Configurable embedding provider for orchestration tests."""

    def __init__(
        self,
        response: EmbeddingResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        """Initialize deterministic synthetic provider behavior."""
        self.response = response
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
        """Record the request and return configured behavior."""
        self.requests.append(request)

        if self.error is not None:
            raise self.error

        if self.response is None:
            raise AssertionError("synthetic embedding response not configured")

        return self.response


class SyntheticVectorStore:
    """Configurable in-memory vector-store boundary double."""

    def __init__(
        self,
        error: Exception | None = None,
    ) -> None:
        """Initialize deterministic synthetic provider behavior."""
        self.error = error
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
        """Record one vector upsert or raise the configured error."""
        self.upsert_requests.append(request)

        if self.error is not None:
            raise self.error

    async def query(
        self,
        request: VectorQueryRequest,
    ) -> VectorQueryResponse:
        """Querying is outside the indexing-service test surface."""
        raise AssertionError(f"unexpected query request: {request}")

    async def delete(
        self,
        request: VectorDeleteRequest,
    ) -> None:
        """Deletion is outside the indexing-service test surface."""
        raise AssertionError(f"unexpected delete request: {request}")


def _chunks() -> tuple[DocumentChunk, ...]:
    """Return deterministic synthetic indexing chunks."""
    return (
        DocumentChunk(
            chunk_id="doc:chunk:0:0:5",
            document_id="doc",
            text="alpha",
            index=0,
            start_char=0,
            end_char=5,
            source_ref="synthetic://doc",
        ),
        DocumentChunk(
            chunk_id="doc:chunk:1:5:9",
            document_id="doc",
            text="beta",
            index=1,
            start_char=5,
            end_char=9,
            source_ref="synthetic://doc",
        ),
    )


def _embedding_response(
    *,
    dimensions: int = 3,
    count: int = 2,
    model: str = "synthetic-model",
) -> EmbeddingResponse:
    """Create deterministic provider output."""
    embeddings = tuple(
        EmbeddingVector(
            values=tuple(
                float(item + dimension + 1) / 10.0 for dimension in range(dimensions)
            ),
        )
        for item in range(count)
    )

    return EmbeddingResponse(
        model=model,
        embeddings=embeddings,
        input_tokens=count,
    )


@pytest.mark.anyio
async def test_index_chunks_calls_providers_in_order() -> None:
    """Successful indexing should embed first and persist second."""
    embedding = SyntheticEmbeddingProvider(response=_embedding_response())
    vector_store = SyntheticVectorStore()

    service = IndexingService(
        embedding,
        vector_store,
        model="synthetic-model",
        dimensions=3,
    )

    result = await service.index_chunks(
        _chunks(),
        namespace="test",
    )

    assert result == IndexingResult(
        model="synthetic-model",
        dimensions=3,
        records_indexed=2,
        namespace="test",
    )

    assert len(embedding.requests) == 1

    assert embedding.requests[0].texts == (
        "alpha",
        "beta",
    )

    assert embedding.requests[0].dimensions == 3

    assert len(vector_store.upsert_requests) == 1

    upsert = vector_store.upsert_requests[0]

    assert upsert.namespace == "test"
    assert upsert.space_id == "synthetic-model"

    assert tuple(record.record_id for record in upsert.records) == (
        "doc:chunk:0:0:5",
        "doc:chunk:1:5:9",
    )


@pytest.mark.anyio
async def test_empty_chunks_fail_before_provider_calls() -> None:
    """Empty indexing requests must stop before any provider execution."""
    embedding = SyntheticEmbeddingProvider(response=_embedding_response())
    vector_store = SyntheticVectorStore()

    service = IndexingService(
        embedding,
        vector_store,
        model="synthetic-model",
    )

    with pytest.raises(
        ValueError,
        match="chunks must not be empty",
    ):
        await service.index_chunks(())

    assert embedding.requests == []
    assert vector_store.upsert_requests == []


@pytest.mark.anyio
async def test_embedding_provider_failure_prevents_upsert() -> None:
    """Embedding failures must never reach vector persistence."""
    error = ProviderExecutionError("synthetic embedding failure")

    embedding = SyntheticEmbeddingProvider(error=error)
    vector_store = SyntheticVectorStore()

    service = IndexingService(
        embedding,
        vector_store,
        model="synthetic-model",
    )

    with pytest.raises(
        ProviderExecutionError,
        match="synthetic embedding failure",
    ):
        await service.index_chunks(_chunks())

    assert len(embedding.requests) == 1
    assert vector_store.upsert_requests == []


@pytest.mark.anyio
async def test_embedding_count_mismatch_prevents_upsert() -> None:
    """Provider cardinality violations must fail closed."""
    embedding = SyntheticEmbeddingProvider(response=_embedding_response(count=1))
    vector_store = SyntheticVectorStore()

    service = IndexingService(
        embedding,
        vector_store,
        model="synthetic-model",
    )

    with pytest.raises(
        ProviderExecutionError,
        match=("Embedding provider returned unexpected result count"),
    ):
        await service.index_chunks(_chunks())

    assert vector_store.upsert_requests == []


@pytest.mark.anyio
async def test_requested_dimension_mismatch_prevents_upsert() -> None:
    """Providers must not silently ignore requested dimensionality."""
    embedding = SyntheticEmbeddingProvider(response=_embedding_response(dimensions=2))
    vector_store = SyntheticVectorStore()

    service = IndexingService(
        embedding,
        vector_store,
        model="synthetic-model",
        dimensions=3,
    )

    with pytest.raises(
        ProviderExecutionError,
        match=("Embedding provider returned unexpected dimensions"),
    ):
        await service.index_chunks(_chunks())

    assert vector_store.upsert_requests == []


@pytest.mark.anyio
async def test_unspecified_dimensions_accept_provider_shape() -> None:
    """Provider-selected dimensions are valid when not constrained."""
    embedding = SyntheticEmbeddingProvider(response=_embedding_response(dimensions=2))
    vector_store = SyntheticVectorStore()

    service = IndexingService(
        embedding,
        vector_store,
        model="synthetic-model",
        dimensions=None,
    )

    result = await service.index_chunks(_chunks())

    assert result.dimensions == 2
    assert result.records_indexed == 2
    assert len(vector_store.upsert_requests) == 1


@pytest.mark.anyio
async def test_vector_store_failure_is_propagated() -> None:
    """Persistence failures should remain provider-domain failures."""
    embedding = SyntheticEmbeddingProvider(response=_embedding_response())

    vector_store = SyntheticVectorStore(
        error=ProviderExecutionError("synthetic vector-store failure")
    )

    service = IndexingService(
        embedding,
        vector_store,
        model="synthetic-model",
    )

    with pytest.raises(
        ProviderExecutionError,
        match="synthetic vector-store failure",
    ):
        await service.index_chunks(_chunks())

    assert len(embedding.requests) == 1
    assert len(vector_store.upsert_requests) == 1


@pytest.mark.anyio
async def test_invalid_namespace_fails_before_upsert() -> None:
    """Vector namespace validation must run before persistence."""
    embedding = SyntheticEmbeddingProvider(response=_embedding_response())
    vector_store = SyntheticVectorStore()

    service = IndexingService(
        embedding,
        vector_store,
        model="synthetic-model",
    )

    with pytest.raises(
        ValueError,
        match="namespace must not be empty",
    ):
        await service.index_chunks(
            _chunks(),
            namespace=" ",
        )

    assert len(embedding.requests) == 1
    assert vector_store.upsert_requests == []


def test_indexing_service_uses_vector_store_protocol() -> None:
    """Synthetic store should satisfy the existing structural contract."""
    provider = SyntheticVectorStore()

    assert isinstance(
        provider,
        VectorStoreProvider,
    )


def test_indexing_service_is_publicly_exported() -> None:
    """Indexing orchestration should be available through services."""
    import ai_engineering_agent_platform.services as services

    assert {
        "IndexingResult",
        "IndexingService",
    } <= set(services.__all__)


@pytest.mark.anyio
async def test_embedding_model_mismatch_prevents_upsert() -> None:
    """A provider must not silently substitute another embedding model."""
    embedding = SyntheticEmbeddingProvider(
        response=_embedding_response(
            model="unexpected-model",
        )
    )
    vector_store = SyntheticVectorStore()

    service = IndexingService(
        embedding,
        vector_store,
        model="synthetic-model",
        dimensions=3,
    )

    with pytest.raises(
        ProviderExecutionError,
        match="Embedding provider returned unexpected model",
    ):
        await service.index_chunks(_chunks())

    assert len(embedding.requests) == 1
    assert vector_store.upsert_requests == []


@pytest.mark.anyio
async def test_indexing_service_propagates_custom_space_id() -> None:
    """A stricter caller-owned vector-space identity must be preserved."""
    embedding = SyntheticEmbeddingProvider(response=_embedding_response())
    vector_store = SyntheticVectorStore()

    service = IndexingService(
        embedding,
        vector_store,
        model="synthetic-model",
        dimensions=3,
        space_id="synthetic-model@revision-2",
    )

    await service.index_chunks(
        _chunks(),
        namespace="test",
    )

    assert len(vector_store.upsert_requests) == 1
    assert vector_store.upsert_requests[0].space_id == "synthetic-model@revision-2"


def test_indexing_service_rejects_empty_space_id() -> None:
    """Invalid vector-space configuration must fail before provider use."""
    embedding = SyntheticEmbeddingProvider(response=_embedding_response())
    vector_store = SyntheticVectorStore()

    with pytest.raises(
        ValueError,
        match="space_id must not be empty",
    ):
        IndexingService(
            embedding,
            vector_store,
            model="synthetic-model",
            space_id=" ",
        )

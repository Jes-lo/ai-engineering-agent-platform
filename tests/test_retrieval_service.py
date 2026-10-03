"""Tests for provider-neutral semantic retrieval orchestration."""

import pytest

from ai_engineering_agent_platform.contracts import (
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
    ProviderDescriptor,
    ProviderKind,
    VectorDeleteRequest,
    VectorMetadataItem,
    VectorQueryRequest,
    VectorQueryResponse,
    VectorQueryResult,
    VectorStoreProvider,
    VectorUpsertRequest,
)
from ai_engineering_agent_platform.domain import (
    ProviderExecutionError,
    RetrievalRequest,
)
from ai_engineering_agent_platform.services import (
    RetrievalService,
)


class SyntheticEmbeddingProvider:
    """Configurable embedding provider for retrieval tests."""

    def __init__(
        self,
        *,
        response: EmbeddingResponse | None = None,
        error: Exception | None = None,
        events: list[str] | None = None,
    ) -> None:
        """Initialize deterministic synthetic behavior."""
        self.response = response
        self.error = error
        self.events = events if events is not None else []
        self.requests: list[EmbeddingRequest] = []

    @property
    def descriptor(self) -> ProviderDescriptor:
        """Return deterministic provider identity."""
        return ProviderDescriptor(
            name="synthetic-embedding",
            kind=ProviderKind.EMBEDDING,
        )

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse:
        """Record one embedding request."""
        self.events.append("embed")
        self.requests.append(request)

        if self.error is not None:
            raise self.error

        if self.response is None:
            raise AssertionError("synthetic embedding response not configured")

        return self.response


class SyntheticVectorStore:
    """Configurable vector-store provider for retrieval tests."""

    def __init__(
        self,
        *,
        response: VectorQueryResponse | None = None,
        error: Exception | None = None,
        events: list[str] | None = None,
    ) -> None:
        """Initialize deterministic synthetic behavior."""
        self.response = response
        self.error = error
        self.events = events if events is not None else []
        self.query_requests: list[VectorQueryRequest] = []

    @property
    def descriptor(self) -> ProviderDescriptor:
        """Return deterministic vector-store identity."""
        return ProviderDescriptor(
            name="synthetic-vector-store",
            kind=ProviderKind.VECTOR_STORE,
        )

    async def upsert(
        self,
        request: VectorUpsertRequest,
    ) -> None:
        """Upsert is outside this retrieval test surface."""
        raise AssertionError(f"unexpected upsert request: {request}")

    async def query(
        self,
        request: VectorQueryRequest,
    ) -> VectorQueryResponse:
        """Record one vector query and return configured behavior."""
        self.events.append("query")
        self.query_requests.append(request)

        if self.error is not None:
            raise self.error

        if self.response is None:
            raise AssertionError("synthetic vector response not configured")

        return self.response

    async def delete(
        self,
        request: VectorDeleteRequest,
    ) -> None:
        """Delete is outside this retrieval test surface."""
        raise AssertionError(f"unexpected delete request: {request}")


def _request(
    *,
    top_k: int = 2,
) -> RetrievalRequest:
    """Return one deterministic semantic query."""
    return RetrievalRequest(
        query="synthetic alpha",
        top_k=top_k,
        namespace="synthetic",
    )


def _embedding_response(
    *,
    dimensions: int = 3,
    count: int = 1,
    model: str = "synthetic-model",
) -> EmbeddingResponse:
    """Return deterministic synthetic embedding output."""
    return EmbeddingResponse(
        model=model,
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
    start_char: int = 0,
    end_char: int = 15,
) -> tuple[VectorMetadataItem, ...]:
    """Return valid persisted retrieval provenance."""
    return (
        VectorMetadataItem(
            key="retrieval.document_id",
            value="doc-1",
        ),
        VectorMetadataItem(
            key="retrieval.chunk_index",
            value=0,
        ),
        VectorMetadataItem(
            key="retrieval.start_char",
            value=start_char,
        ),
        VectorMetadataItem(
            key="retrieval.end_char",
            value=end_char,
        ),
        VectorMetadataItem(
            key="retrieval.source_ref",
            value="synthetic://doc-1",
        ),
        VectorMetadataItem(
            key="retrieval.title",
            value="Synthetic Document",
        ),
        VectorMetadataItem(
            key="source.category",
            value="synthetic",
        ),
    )


def _vector_response() -> VectorQueryResponse:
    """Return one valid citation-ready vector result."""
    return VectorQueryResponse(
        results=(
            VectorQueryResult(
                record_id="doc-1:chunk:0:0:15",
                score=0.9,
                rank=1,
                text="synthetic alpha",
                metadata=_metadata(),
            ),
        )
    )


@pytest.mark.anyio
async def test_retrieve_executes_embed_then_query() -> None:
    """Successful semantic retrieval must preserve execution order."""
    events: list[str] = []

    embedding = SyntheticEmbeddingProvider(
        response=_embedding_response(),
        events=events,
    )
    vector_store = SyntheticVectorStore(
        response=_vector_response(),
        events=events,
    )

    service = RetrievalService(
        embedding,
        vector_store,
        model="synthetic-model",
        dimensions=3,
    )

    response = await service.retrieve(_request())

    assert events == [
        "embed",
        "query",
    ]

    assert len(embedding.requests) == 1

    assert embedding.requests[0].texts == ("synthetic alpha",)

    assert embedding.requests[0].dimensions == 3

    assert len(vector_store.query_requests) == 1

    vector_request = vector_store.query_requests[0]

    assert vector_request.vector == (
        0.1,
        0.2,
        0.3,
    )
    assert vector_request.top_k == 2
    assert vector_request.namespace == "synthetic"
    assert vector_request.space_id == "synthetic-model"

    assert response.query == "synthetic alpha"
    assert response.namespace == "synthetic"

    assert len(response.results) == 1

    evidence = response.results[0]

    assert evidence.document_id == "doc-1"
    assert evidence.chunk_id == "doc-1:chunk:0:0:15"
    assert evidence.text == "synthetic alpha"
    assert evidence.source_ref == "synthetic://doc-1"


@pytest.mark.anyio
async def test_empty_vector_result_is_valid() -> None:
    """A successful search may legitimately produce no evidence."""
    embedding = SyntheticEmbeddingProvider(response=_embedding_response())
    vector_store = SyntheticVectorStore(response=VectorQueryResponse(results=()))

    service = RetrievalService(
        embedding,
        vector_store,
        model="synthetic-model",
    )

    response = await service.retrieve(_request())

    assert response.results == ()


@pytest.mark.anyio
async def test_embedding_failure_prevents_vector_query() -> None:
    """Embedding failure must stop before vector-store execution."""
    error = ProviderExecutionError("synthetic embedding failure")

    embedding = SyntheticEmbeddingProvider(error=error)
    vector_store = SyntheticVectorStore(response=_vector_response())

    service = RetrievalService(
        embedding,
        vector_store,
        model="synthetic-model",
    )

    with pytest.raises(
        ProviderExecutionError,
        match="synthetic embedding failure",
    ):
        await service.retrieve(_request())

    assert len(embedding.requests) == 1
    assert vector_store.query_requests == []


@pytest.mark.anyio
async def test_multiple_query_embeddings_prevent_vector_query() -> None:
    """One semantic query must never fan out into multiple vectors."""
    embedding = SyntheticEmbeddingProvider(response=_embedding_response(count=2))
    vector_store = SyntheticVectorStore(response=_vector_response())

    service = RetrievalService(
        embedding,
        vector_store,
        model="synthetic-model",
    )

    with pytest.raises(
        ProviderExecutionError,
        match=("Embedding provider returned unexpected result count"),
    ):
        await service.retrieve(_request())

    assert vector_store.query_requests == []


@pytest.mark.anyio
async def test_requested_dimension_mismatch_prevents_query() -> None:
    """Providers must honor explicitly requested query dimensions."""
    embedding = SyntheticEmbeddingProvider(response=_embedding_response(dimensions=2))
    vector_store = SyntheticVectorStore(response=_vector_response())

    service = RetrievalService(
        embedding,
        vector_store,
        model="synthetic-model",
        dimensions=3,
    )

    with pytest.raises(
        ProviderExecutionError,
        match=("Embedding provider returned unexpected dimensions"),
    ):
        await service.retrieve(_request())

    assert vector_store.query_requests == []


@pytest.mark.anyio
async def test_unspecified_dimensions_accept_provider_shape() -> None:
    """Provider-selected query dimensions are valid when unconstrained."""
    embedding = SyntheticEmbeddingProvider(response=_embedding_response(dimensions=2))
    vector_store = SyntheticVectorStore(response=VectorQueryResponse(results=()))

    service = RetrievalService(
        embedding,
        vector_store,
        model="synthetic-model",
        dimensions=None,
    )

    response = await service.retrieve(_request())

    assert response.results == ()

    assert vector_store.query_requests[0].dimensions == 2


@pytest.mark.anyio
async def test_vector_store_failure_is_propagated() -> None:
    """Query-provider failures must remain provider-domain failures."""
    embedding = SyntheticEmbeddingProvider(response=_embedding_response())

    vector_store = SyntheticVectorStore(
        error=ProviderExecutionError("synthetic vector query failure")
    )

    service = RetrievalService(
        embedding,
        vector_store,
        model="synthetic-model",
    )

    with pytest.raises(
        ProviderExecutionError,
        match="synthetic vector query failure",
    ):
        await service.retrieve(_request())

    assert len(vector_store.query_requests) == 1


@pytest.mark.anyio
async def test_malformed_vector_evidence_fails_after_query() -> None:
    """Malformed persisted provenance must not escape as evidence."""
    embedding = SyntheticEmbeddingProvider(response=_embedding_response())

    vector_store = SyntheticVectorStore(
        response=VectorQueryResponse(
            results=(
                VectorQueryResult(
                    record_id="bad-chunk",
                    score=0.5,
                    rank=1,
                    text="synthetic alpha",
                    metadata=(
                        VectorMetadataItem(
                            key="source.category",
                            value="synthetic",
                        ),
                    ),
                ),
            )
        )
    )

    service = RetrievalService(
        embedding,
        vector_store,
        model="synthetic-model",
    )

    with pytest.raises(
        ProviderExecutionError,
        match=r"missing retrieval\.document_id",
    ):
        await service.retrieve(_request())

    assert len(vector_store.query_requests) == 1


@pytest.mark.anyio
async def test_top_k_violation_from_store_fails_closed() -> None:
    """A store must not return more results than requested."""
    embedding = SyntheticEmbeddingProvider(response=_embedding_response())

    vector_store = SyntheticVectorStore(
        response=VectorQueryResponse(
            results=(
                VectorQueryResult(
                    record_id="chunk-1",
                    score=0.9,
                    rank=1,
                    text="synthetic alpha",
                    metadata=_metadata(),
                ),
                VectorQueryResult(
                    record_id="chunk-2",
                    score=0.8,
                    rank=2,
                    text="synthetic alpha",
                    metadata=_metadata(),
                ),
            )
        )
    )

    service = RetrievalService(
        embedding,
        vector_store,
        model="synthetic-model",
    )

    with pytest.raises(
        ProviderExecutionError,
        match=("Vector store returned more results than requested"),
    ):
        await service.retrieve(_request(top_k=1))


def test_retrieval_service_satisfies_provider_boundaries() -> None:
    """Synthetic collaborators should satisfy existing protocols."""
    embedding = SyntheticEmbeddingProvider(response=_embedding_response())
    vector_store = SyntheticVectorStore(response=_vector_response())

    assert isinstance(
        embedding,
        EmbeddingProvider,
    )
    assert isinstance(
        vector_store,
        VectorStoreProvider,
    )


def test_retrieval_service_is_publicly_exported() -> None:
    """Retrieval orchestration should be available through services."""
    import ai_engineering_agent_platform.services as services

    assert "RetrievalService" in services.__all__
    assert hasattr(
        services,
        "RetrievalService",
    )


@pytest.mark.anyio
async def test_embedding_model_mismatch_prevents_vector_query() -> None:
    """A substituted embedding model must never reach vector search."""
    embedding = SyntheticEmbeddingProvider(
        response=_embedding_response(
            model="unexpected-model",
        )
    )
    vector_store = SyntheticVectorStore(response=_vector_response())

    service = RetrievalService(
        embedding,
        vector_store,
        model="synthetic-model",
        dimensions=3,
    )

    with pytest.raises(
        ProviderExecutionError,
        match="Embedding provider returned unexpected model",
    ):
        await service.retrieve(_request())

    assert len(embedding.requests) == 1
    assert vector_store.query_requests == []


@pytest.mark.anyio
async def test_retrieval_service_propagates_custom_space_id() -> None:
    """Retrieval must preserve a stricter caller-owned space identity."""
    embedding = SyntheticEmbeddingProvider(response=_embedding_response())
    vector_store = SyntheticVectorStore(response=VectorQueryResponse(results=()))

    service = RetrievalService(
        embedding,
        vector_store,
        model="synthetic-model",
        dimensions=3,
        space_id="synthetic-model@revision-2",
    )

    await service.retrieve(_request())

    assert len(vector_store.query_requests) == 1
    assert vector_store.query_requests[0].space_id == "synthetic-model@revision-2"


def test_retrieval_service_rejects_empty_space_id() -> None:
    """Invalid vector-space configuration must fail before provider use."""
    embedding = SyntheticEmbeddingProvider(response=_embedding_response())
    vector_store = SyntheticVectorStore(response=_vector_response())

    with pytest.raises(
        ValueError,
        match="space_id must not be empty",
    ):
        RetrievalService(
            embedding,
            vector_store,
            model="synthetic-model",
            space_id=" ",
        )

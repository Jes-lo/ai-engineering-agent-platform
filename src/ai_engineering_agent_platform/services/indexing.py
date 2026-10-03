"""Provider-neutral semantic indexing orchestration."""

from dataclasses import dataclass

from ai_engineering_agent_platform.contracts import (
    EmbeddingProvider,
    VectorStoreProvider,
)
from ai_engineering_agent_platform.domain import (
    DocumentChunk,
    ProviderExecutionError,
)
from ai_engineering_agent_platform.services.indexing_mapping import (
    build_embedding_request,
    build_vector_upsert_request,
)


@dataclass(frozen=True, slots=True)
class IndexingResult:
    """Summary of one completed semantic indexing operation."""

    model: str
    dimensions: int
    records_indexed: int
    namespace: str | None


class IndexingService:
    """Coordinate embedding generation and vector persistence.

    The service depends only on provider-neutral contracts. It does not own
    provider lifecycle, network configuration, retries, or database
    transactions beyond the guarantees supplied by each provider.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store_provider: VectorStoreProvider,
        *,
        model: str,
        dimensions: int | None = None,
        space_id: str | None = None,
    ) -> None:
        """Store injected providers and embedding request configuration."""
        resolved_space_id = model if space_id is None else space_id

        if not isinstance(resolved_space_id, str):
            raise ValueError("space_id must be a string")

        if not resolved_space_id.strip():
            raise ValueError("space_id must not be empty")

        self._embedding_provider = embedding_provider
        self._vector_store_provider = vector_store_provider
        self._model = model
        self._dimensions = dimensions
        self._space_id = resolved_space_id

    async def index_chunks(
        self,
        chunks: tuple[DocumentChunk, ...],
        *,
        namespace: str | None = None,
    ) -> IndexingResult:
        """Embed and persist one ordered chunk batch."""
        embedding_request = build_embedding_request(
            chunks,
            model=self._model,
            dimensions=self._dimensions,
        )

        embedding_response = await self._embedding_provider.embed(embedding_request)

        if embedding_response.model != self._model:
            raise ProviderExecutionError("Embedding provider returned unexpected model")

        if len(embedding_response.embeddings) != len(chunks):
            raise ProviderExecutionError(
                "Embedding provider returned unexpected result count"
            )

        if (
            self._dimensions is not None
            and embedding_response.dimensions != self._dimensions
        ):
            raise ProviderExecutionError(
                "Embedding provider returned unexpected dimensions"
            )

        upsert_request = build_vector_upsert_request(
            chunks,
            embedding_response,
            namespace=namespace,
            space_id=self._space_id,
        )

        await self._vector_store_provider.upsert(upsert_request)

        return IndexingResult(
            model=embedding_response.model,
            dimensions=embedding_response.dimensions,
            records_indexed=len(upsert_request.records),
            namespace=upsert_request.namespace,
        )

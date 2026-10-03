"""Provider-neutral semantic retrieval orchestration."""

from ai_engineering_agent_platform.contracts import (
    EmbeddingProvider,
    VectorStoreProvider,
)
from ai_engineering_agent_platform.domain import (
    ProviderExecutionError,
    RetrievalRequest,
    RetrievalResponse,
)
from ai_engineering_agent_platform.services.retrieval_mapping import (
    build_retrieval_embedding_request,
    build_retrieval_response,
    build_vector_query_request,
)


class RetrievalService:
    """Coordinate query embedding, vector search, and evidence mapping.

    The service depends only on provider-neutral contracts and domain objects.
    It does not own provider lifecycle, networking, persistence configuration,
    reranking, LLM generation, or citation rendering.
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
        """Store injected providers and query-embedding configuration."""
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

    async def retrieve(
        self,
        request: RetrievalRequest,
    ) -> RetrievalResponse:
        """Embed one semantic query and return validated evidence."""
        embedding_request = build_retrieval_embedding_request(
            request,
            model=self._model,
            dimensions=self._dimensions,
        )

        embedding_response = await self._embedding_provider.embed(embedding_request)

        if embedding_response.model != self._model:
            raise ProviderExecutionError("Embedding provider returned unexpected model")

        if (
            self._dimensions is not None
            and embedding_response.dimensions != self._dimensions
        ):
            raise ProviderExecutionError(
                "Embedding provider returned unexpected dimensions"
            )

        vector_request = build_vector_query_request(
            request,
            embedding_response,
            space_id=self._space_id,
        )

        vector_response = await self._vector_store_provider.query(vector_request)

        return build_retrieval_response(
            request,
            vector_response,
        )

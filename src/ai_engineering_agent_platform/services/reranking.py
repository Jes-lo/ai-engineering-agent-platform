"""Provider-neutral optional reranking orchestration."""

from ai_engineering_agent_platform.contracts import (
    RerankerProvider,
)
from ai_engineering_agent_platform.domain import (
    ProviderExecutionError,
    RerankedRetrievalResponse,
    RetrievalResponse,
)
from ai_engineering_agent_platform.services.reranking_mapping import (
    build_rerank_request,
    build_reranked_retrieval_response,
)


class RerankingService:
    """Optionally rerank already validated retrieval evidence.

    Empty retrieval results bypass the reranker entirely and are returned
    unchanged.

    Non-empty results are mapped through the provider-neutral reranker
    contract. Original vector retrieval scores and ranks remain embedded in
    each evidence object; reranking produces an independent second-stage
    score and final rank.
    """

    def __init__(
        self,
        reranker_provider: RerankerProvider,
        *,
        model: str,
        top_n: int | None = None,
    ) -> None:
        """Store the injected reranker and request configuration."""
        self._reranker_provider = reranker_provider
        self._model = model
        self._top_n = top_n

    async def rerank(
        self,
        response: RetrievalResponse,
    ) -> RetrievalResponse | RerankedRetrievalResponse:
        """Return original empty retrieval or reranked non-empty evidence."""
        if not response.results:
            return response

        request = build_rerank_request(
            response,
            model=self._model,
            top_n=self._top_n,
        )

        rerank_response = await self._reranker_provider.rerank(request)

        if rerank_response.model != self._model:
            raise ProviderExecutionError("Reranker provider returned unexpected model")

        max_results = (
            request.top_n if request.top_n is not None else len(request.documents)
        )

        if len(rerank_response.results) > max_results:
            raise ProviderExecutionError(
                "Reranker provider returned more results than requested"
            )

        return build_reranked_retrieval_response(
            response,
            rerank_response,
        )

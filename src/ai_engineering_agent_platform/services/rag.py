"""Provider-neutral end-to-end RAG orchestration."""

from dataclasses import dataclass

from ai_engineering_agent_platform.domain import (
    GroundedAnswer,
    GroundedAnswerStatus,
    RerankedRetrievalResponse,
    RetrievalRequest,
    RetrievalResponse,
)
from ai_engineering_agent_platform.services.grounded_generation import (
    GroundedGenerationResult,
    GroundedGenerationService,
)
from ai_engineering_agent_platform.services.reranking import (
    RerankingService,
)
from ai_engineering_agent_platform.services.retrieval import (
    RetrievalService,
)


@dataclass(frozen=True, slots=True)
class RAGResult:
    """Validated outputs preserved across one end-to-end RAG execution."""

    retrieval: RetrievalResponse
    grounding_input: RetrievalResponse | RerankedRetrievalResponse
    answer: GroundedAnswer
    generation: GroundedGenerationResult | None


class RAGService:
    """Coordinate retrieval, optional reranking, and grounded generation.

    The service composes already provider-neutral application services. It does
    not own provider lifecycle, API exposure, authentication, authorization,
    ingestion, observability backends, agent execution, or workflow execution.

    Empty semantic retrieval is a valid no-evidence outcome. In that case the
    service returns an explicit grounded abstention without invoking a reranker
    or language model.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        grounded_generation_service: GroundedGenerationService,
        *,
        reranking_service: RerankingService | None = None,
    ) -> None:
        """Store injected RAG stage services."""
        self._retrieval_service = retrieval_service
        self._grounded_generation_service = grounded_generation_service
        self._reranking_service = reranking_service

    async def run(
        self,
        request: RetrievalRequest,
    ) -> RAGResult:
        """Execute one deterministic retrieval-to-generation operation."""
        retrieval = await self._retrieval_service.retrieve(request)

        if not retrieval.results:
            answer = GroundedAnswer(
                query=retrieval.query,
                status=GroundedAnswerStatus.ABSTAINED,
                answer=None,
                citations=(),
                namespace=retrieval.namespace,
            )

            return RAGResult(
                retrieval=retrieval,
                grounding_input=retrieval,
                answer=answer,
                generation=None,
            )

        grounding_input: RetrievalResponse | RerankedRetrievalResponse = retrieval

        if self._reranking_service is not None:
            grounding_input = await self._reranking_service.rerank(retrieval)

        generation = await self._grounded_generation_service.generate(
            grounding_input,
        )

        return RAGResult(
            retrieval=retrieval,
            grounding_input=grounding_input,
            answer=generation.answer,
            generation=generation,
        )

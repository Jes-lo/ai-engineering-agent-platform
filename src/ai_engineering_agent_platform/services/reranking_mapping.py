"""Pure mapping between retrieved evidence and reranker contracts."""

from ai_engineering_agent_platform.contracts import (
    RerankDocument,
    RerankRequest,
    RerankResponse,
)
from ai_engineering_agent_platform.domain import (
    ProviderExecutionError,
    RerankedEvidence,
    RerankedRetrievalResponse,
    RetrievalResponse,
)


def build_rerank_request(
    response: RetrievalResponse,
    *,
    model: str,
    top_n: int | None = None,
) -> RerankRequest:
    """Build one reranker request from retrieved evidence.

    Chunk identifiers are used as reranker document identifiers because one
    source document may legitimately contribute multiple retrieved chunks.
    """
    if not response.results:
        raise ValueError("cannot rerank an empty retrieval response")

    return RerankRequest(
        model=model,
        query=response.query,
        documents=tuple(
            RerankDocument(
                document_id=evidence.chunk_id,
                text=evidence.text,
            )
            for evidence in response.results
        ),
        top_n=top_n,
    )


def build_reranked_retrieval_response(
    retrieval_response: RetrievalResponse,
    rerank_response: RerankResponse,
) -> RerankedRetrievalResponse:
    """Combine reranker ordering with immutable retrieval evidence."""
    if not retrieval_response.results:
        raise ValueError("cannot apply reranking to an empty retrieval response")

    evidence_by_chunk_id = {
        evidence.chunk_id: evidence for evidence in retrieval_response.results
    }

    reranked: list[RerankedEvidence] = []

    for result in rerank_response.results:
        evidence = evidence_by_chunk_id.get(result.document_id)

        if evidence is None:
            raise ProviderExecutionError(
                "Reranker returned unknown candidate identifier"
            )

        try:
            reranked.append(
                RerankedEvidence(
                    evidence=evidence,
                    rerank_score=result.score,
                    rank=result.rank,
                )
            )
        except ValueError as exc:
            raise ProviderExecutionError(
                "Reranker returned invalid ranking result"
            ) from exc

    try:
        return RerankedRetrievalResponse(
            query=retrieval_response.query,
            model=rerank_response.model,
            results=tuple(reranked),
            namespace=retrieval_response.namespace,
        )
    except ValueError as exc:
        raise ProviderExecutionError(
            "Reranker returned invalid response ordering"
        ) from exc

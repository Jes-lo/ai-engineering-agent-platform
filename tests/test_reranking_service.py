"""Tests for provider-neutral optional reranking orchestration."""

import pytest

from ai_engineering_agent_platform.contracts import (
    ProviderDescriptor,
    ProviderKind,
    RerankerProvider,
    RerankRequest,
    RerankResponse,
    RerankResult,
)
from ai_engineering_agent_platform.domain import (
    ProviderExecutionError,
    RerankedRetrievalResponse,
    RetrievalResponse,
    RetrievedEvidence,
)
from ai_engineering_agent_platform.services import (
    RerankingService,
)


class SyntheticReranker:
    """Configurable reranker provider for orchestration tests."""

    def __init__(
        self,
        *,
        response: RerankResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        """Initialize deterministic synthetic provider behavior."""
        self.response = response
        self.error = error
        self.requests: list[RerankRequest] = []

    @property
    def descriptor(self) -> ProviderDescriptor:
        """Return deterministic reranker identity."""
        return ProviderDescriptor(
            name="synthetic-reranker",
            kind=ProviderKind.RERANKER,
        )

    async def rerank(
        self,
        request: RerankRequest,
    ) -> RerankResponse:
        """Record request and return configured behavior."""
        self.requests.append(request)

        if self.error is not None:
            raise self.error

        if self.response is None:
            raise AssertionError("synthetic reranker response not configured")

        return self.response


def _evidence(
    *,
    chunk_id: str,
    text: str,
    score: float,
    rank: int,
    start_char: int,
) -> RetrievedEvidence:
    """Return deterministic citation-ready evidence."""
    return RetrievedEvidence(
        chunk_id=chunk_id,
        document_id="shared-document",
        text=text,
        score=score,
        rank=rank,
        start_char=start_char,
        end_char=(start_char + len(text)),
        source_ref="synthetic://shared-document",
        title="Synthetic Document",
    )


def _retrieval_response() -> RetrievalResponse:
    """Return deterministic vector retrieval results."""
    return RetrievalResponse(
        query="beta evidence",
        namespace="synthetic",
        results=(
            _evidence(
                chunk_id="chunk-a",
                text="alpha evidence",
                score=0.91,
                rank=1,
                start_char=0,
            ),
            _evidence(
                chunk_id="chunk-b",
                text="beta evidence",
                score=0.72,
                rank=2,
                start_char=20,
            ),
        ),
    )


def _reordered_response() -> RerankResponse:
    """Return deterministic reranker output reversing vector order."""
    return RerankResponse(
        model="synthetic-reranker-model",
        results=(
            RerankResult(
                document_id="chunk-b",
                score=9.0,
                rank=1,
            ),
            RerankResult(
                document_id="chunk-a",
                score=7.0,
                rank=2,
            ),
        ),
    )


@pytest.mark.anyio
async def test_rerank_executes_provider_and_reorders_evidence() -> None:
    """Non-empty retrieval should execute one provider rerank."""
    provider = SyntheticReranker(response=_reordered_response())

    service = RerankingService(
        provider,
        model="synthetic-reranker-model",
        top_n=2,
    )

    original = _retrieval_response()

    result = await service.rerank(original)

    assert isinstance(
        result,
        RerankedRetrievalResponse,
    )

    assert len(provider.requests) == 1

    request = provider.requests[0]

    assert request.model == "synthetic-reranker-model"
    assert request.query == original.query
    assert request.top_n == 2

    assert tuple(document.document_id for document in request.documents) == (
        "chunk-a",
        "chunk-b",
    )

    assert tuple(item.evidence.chunk_id for item in result.results) == (
        "chunk-b",
        "chunk-a",
    )

    assert result.results[0].rerank_score == 9.0
    assert result.results[0].rank == 1

    assert result.results[0].evidence.score == 0.72
    assert result.results[0].evidence.rank == 2

    assert result.results[1].evidence.score == 0.91
    assert result.results[1].evidence.rank == 1


@pytest.mark.anyio
async def test_empty_retrieval_bypasses_reranker() -> None:
    """Empty retrieval should return unchanged without provider execution."""
    provider = SyntheticReranker()

    service = RerankingService(
        provider,
        model="synthetic-reranker-model",
    )

    original = RetrievalResponse(
        query="nothing",
        results=(),
        namespace="synthetic",
    )

    result = await service.rerank(original)

    assert result is original
    assert provider.requests == []


@pytest.mark.anyio
async def test_provider_failure_is_propagated() -> None:
    """Provider execution errors should retain provider-domain semantics."""
    provider = SyntheticReranker(
        error=ProviderExecutionError("synthetic reranker failure")
    )

    service = RerankingService(
        provider,
        model="synthetic-reranker-model",
    )

    with pytest.raises(
        ProviderExecutionError,
        match="synthetic reranker failure",
    ):
        await service.rerank(_retrieval_response())

    assert len(provider.requests) == 1


@pytest.mark.anyio
async def test_provider_model_mismatch_fails_closed() -> None:
    """Providers must identify the same model requested by the service."""
    provider = SyntheticReranker(
        response=RerankResponse(
            model="unexpected-model",
            results=(
                RerankResult(
                    document_id="chunk-a",
                    score=9.0,
                    rank=1,
                ),
            ),
        )
    )

    service = RerankingService(
        provider,
        model="synthetic-reranker-model",
    )

    with pytest.raises(
        ProviderExecutionError,
        match=("Reranker provider returned unexpected model"),
    ):
        await service.rerank(_retrieval_response())


@pytest.mark.anyio
async def test_provider_cannot_exceed_top_n() -> None:
    """Provider output must not exceed the requested result count."""
    provider = SyntheticReranker(response=_reordered_response())

    service = RerankingService(
        provider,
        model="synthetic-reranker-model",
        top_n=1,
    )

    with pytest.raises(
        ProviderExecutionError,
        match=("Reranker provider returned more results than requested"),
    ):
        await service.rerank(_retrieval_response())

    assert len(provider.requests) == 1
    assert provider.requests[0].top_n == 1


@pytest.mark.anyio
async def test_valid_top_n_subset_is_supported() -> None:
    """A provider may return only the requested highest-ranked candidate."""
    provider = SyntheticReranker(
        response=RerankResponse(
            model="synthetic-reranker-model",
            results=(
                RerankResult(
                    document_id="chunk-b",
                    score=9.0,
                    rank=1,
                ),
            ),
        )
    )

    service = RerankingService(
        provider,
        model="synthetic-reranker-model",
        top_n=1,
    )

    result = await service.rerank(_retrieval_response())

    assert isinstance(
        result,
        RerankedRetrievalResponse,
    )

    assert len(result.results) == 1

    assert result.results[0].evidence.chunk_id == "chunk-b"

    assert result.results[0].evidence.rank == 2
    assert result.results[0].rank == 1


@pytest.mark.anyio
async def test_unknown_candidate_from_provider_fails_closed() -> None:
    """Rerankers cannot inject evidence outside retrieved candidates."""
    provider = SyntheticReranker(
        response=RerankResponse(
            model="synthetic-reranker-model",
            results=(
                RerankResult(
                    document_id="unknown",
                    score=9.0,
                    rank=1,
                ),
            ),
        )
    )

    service = RerankingService(
        provider,
        model="synthetic-reranker-model",
    )

    with pytest.raises(
        ProviderExecutionError,
        match=("Reranker returned unknown candidate identifier"),
    ):
        await service.rerank(_retrieval_response())


@pytest.mark.anyio
async def test_invalid_top_n_fails_before_provider_call() -> None:
    """Existing contract validation must run before provider execution."""
    provider = SyntheticReranker(response=_reordered_response())

    service = RerankingService(
        provider,
        model="synthetic-reranker-model",
        top_n=3,
    )

    with pytest.raises(
        ValueError,
        match="top_n must not exceed document count",
    ):
        await service.rerank(_retrieval_response())

    assert provider.requests == []


def test_synthetic_provider_satisfies_reranker_protocol() -> None:
    """Synthetic provider should satisfy structural provider typing."""
    provider = SyntheticReranker(response=_reordered_response())

    assert isinstance(
        provider,
        RerankerProvider,
    )


def test_reranking_service_is_publicly_exported() -> None:
    """Optional reranking orchestration should be publicly available."""
    import ai_engineering_agent_platform.services as services

    assert "RerankingService" in services.__all__
    assert hasattr(
        services,
        "RerankingService",
    )

"""Tests for provider-neutral reranker contracts."""

import asyncio
from dataclasses import FrozenInstanceError

import pytest

from ai_engineering_agent_platform.contracts import (
    ProviderDescriptor,
    ProviderKind,
    RerankDocument,
    RerankerProvider,
    RerankRequest,
    RerankResponse,
    RerankResult,
)


class ExampleRerankerProvider:
    """Minimal structural implementation of the reranker contract."""

    @property
    def descriptor(self) -> ProviderDescriptor:
        """Return deterministic provider metadata."""
        return ProviderDescriptor(
            name="example-reranker",
            kind=ProviderKind.RERANKER,
        )

    async def rerank(
        self,
        request: RerankRequest,
    ) -> RerankResponse:
        """Return deterministic normalized ranking results."""
        results = tuple(
            RerankResult(
                document_id=document.document_id,
                score=float(len(request.documents) - index),
                rank=index + 1,
            )
            for index, document in enumerate(request.documents)
        )

        if request.top_n is not None:
            results = results[: request.top_n]

        return RerankResponse(
            model=request.model,
            results=results,
        )


def _documents() -> tuple[RerankDocument, ...]:
    """Return deterministic document candidates."""
    return (
        RerankDocument(
            document_id="doc-1",
            text="first document",
        ),
        RerankDocument(
            document_id="doc-2",
            text="second document",
        ),
        RerankDocument(
            document_id="doc-3",
            text="third document",
        ),
    )


def _request() -> RerankRequest:
    """Return a deterministic valid rerank request."""
    return RerankRequest(
        model="example-reranker-model",
        query="example query",
        documents=_documents(),
        top_n=2,
    )


def test_rerank_request_is_immutable() -> None:
    """Rerank requests should not mutate after creation."""
    request = _request()

    with pytest.raises(FrozenInstanceError):
        request.query = "changed"  # type: ignore[misc]


def test_rerank_request_rejects_invalid_model_query_and_documents() -> None:
    """Requests require usable model, query, and candidates."""
    with pytest.raises(
        ValueError,
        match="model must not be empty",
    ):
        RerankRequest(
            model=" ",
            query="query",
            documents=_documents(),
        )

    with pytest.raises(
        ValueError,
        match="query must not be empty",
    ):
        RerankRequest(
            model="model",
            query=" ",
            documents=_documents(),
        )

    with pytest.raises(
        ValueError,
        match="documents must not be empty",
    ):
        RerankRequest(
            model="model",
            query="query",
            documents=(),
        )

    with pytest.raises(
        ValueError,
        match="document_id must not be empty",
    ):
        RerankDocument(
            document_id=" ",
            text="text",
        )

    with pytest.raises(
        ValueError,
        match="document text must not be empty",
    ):
        RerankDocument(
            document_id="doc",
            text=" ",
        )


def test_rerank_request_rejects_duplicate_document_ids() -> None:
    """Candidate document identifiers must be unique."""
    with pytest.raises(
        ValueError,
        match="document identifiers must be unique",
    ):
        RerankRequest(
            model="model",
            query="query",
            documents=(
                RerankDocument(
                    document_id="duplicate",
                    text="first",
                ),
                RerankDocument(
                    document_id="duplicate",
                    text="second",
                ),
            ),
        )


def test_rerank_request_validates_top_n() -> None:
    """Requested result count must fit the candidate collection."""
    with pytest.raises(
        ValueError,
        match="top_n must be positive",
    ):
        RerankRequest(
            model="model",
            query="query",
            documents=_documents(),
            top_n=0,
        )

    with pytest.raises(
        ValueError,
        match="top_n must not exceed document count",
    ):
        RerankRequest(
            model="model",
            query="query",
            documents=_documents(),
            top_n=4,
        )


def test_rerank_result_validates_normalized_fields() -> None:
    """Normalized ranking results require valid identifiers and ranks."""
    with pytest.raises(
        ValueError,
        match="document_id must not be empty",
    ):
        RerankResult(
            document_id=" ",
            score=1.0,
            rank=1,
        )

    with pytest.raises(
        ValueError,
        match="rerank score must be finite",
    ):
        RerankResult(
            document_id="doc",
            score=float("nan"),
            rank=1,
        )

    with pytest.raises(
        ValueError,
        match="rerank score must be finite",
    ):
        RerankResult(
            document_id="doc",
            score=float("inf"),
            rank=1,
        )

    with pytest.raises(
        ValueError,
        match="rank must be positive",
    ):
        RerankResult(
            document_id="doc",
            score=1.0,
            rank=0,
        )


def test_rerank_response_requires_model_and_results() -> None:
    """Responses require provider model identity and ranked output."""
    with pytest.raises(
        ValueError,
        match="model must not be empty",
    ):
        RerankResponse(
            model=" ",
            results=(
                RerankResult(
                    document_id="doc",
                    score=1.0,
                    rank=1,
                ),
            ),
        )

    with pytest.raises(
        ValueError,
        match="rerank results must not be empty",
    ):
        RerankResponse(
            model="model",
            results=(),
        )


def test_rerank_response_rejects_duplicate_documents() -> None:
    """One normalized response cannot rank a document twice."""
    with pytest.raises(
        ValueError,
        match=("rerank result document identifiers must be unique"),
    ):
        RerankResponse(
            model="model",
            results=(
                RerankResult(
                    document_id="doc",
                    score=2.0,
                    rank=1,
                ),
                RerankResult(
                    document_id="doc",
                    score=1.0,
                    rank=2,
                ),
            ),
        )


def test_rerank_response_requires_contiguous_rank_order() -> None:
    """Normalized results should be ordered with contiguous ranks."""
    with pytest.raises(
        ValueError,
        match=("rerank results must use contiguous rank order"),
    ):
        RerankResponse(
            model="model",
            results=(
                RerankResult(
                    document_id="doc-1",
                    score=2.0,
                    rank=1,
                ),
                RerankResult(
                    document_id="doc-2",
                    score=1.0,
                    rank=3,
                ),
            ),
        )


def test_reranker_provider_supports_structural_async_typing() -> None:
    """Concrete adapters should satisfy RerankerProvider structurally."""
    provider: RerankerProvider = ExampleRerankerProvider()

    assert isinstance(provider, RerankerProvider)
    assert provider.descriptor.kind is ProviderKind.RERANKER

    response = asyncio.run(provider.rerank(_request()))

    assert response.model == "example-reranker-model"
    assert len(response.results) == 2
    assert response.results[0].document_id == "doc-1"
    assert response.results[0].rank == 1
    assert response.results[1].rank == 2

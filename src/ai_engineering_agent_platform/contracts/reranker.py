"""Provider-neutral contracts for document reranking."""

from dataclasses import dataclass
from math import isfinite
from typing import Protocol, runtime_checkable

from ai_engineering_agent_platform.contracts.provider import Provider


@dataclass(frozen=True, slots=True)
class RerankDocument:
    """Immutable document candidate supplied to a reranker."""

    document_id: str
    text: str

    def __post_init__(self) -> None:
        """Validate provider-independent document invariants."""
        if not self.document_id.strip():
            raise ValueError("document_id must not be empty")

        if not self.text.strip():
            raise ValueError("document text must not be empty")


@dataclass(frozen=True, slots=True)
class RerankRequest:
    """Immutable request sent through a reranker provider."""

    model: str
    query: str
    documents: tuple[RerankDocument, ...]
    top_n: int | None = None

    def __post_init__(self) -> None:
        """Validate provider-independent reranking invariants."""
        if not self.model.strip():
            raise ValueError("model must not be empty")

        if not self.query.strip():
            raise ValueError("query must not be empty")

        if not self.documents:
            raise ValueError("documents must not be empty")

        document_ids = [document.document_id for document in self.documents]

        if len(document_ids) != len(set(document_ids)):
            raise ValueError("document identifiers must be unique")

        if self.top_n is not None:
            if self.top_n <= 0:
                raise ValueError("top_n must be positive")

            if self.top_n > len(self.documents):
                raise ValueError("top_n must not exceed document count")


@dataclass(frozen=True, slots=True)
class RerankResult:
    """Normalized ranking result for one document."""

    document_id: str
    score: float
    rank: int

    def __post_init__(self) -> None:
        """Validate normalized result invariants."""
        if not self.document_id.strip():
            raise ValueError("document_id must not be empty")

        if not isfinite(self.score):
            raise ValueError("rerank score must be finite")

        if self.rank <= 0:
            raise ValueError("rank must be positive")


@dataclass(frozen=True, slots=True)
class RerankResponse:
    """Normalized response returned by a reranker provider."""

    model: str
    results: tuple[RerankResult, ...]

    def __post_init__(self) -> None:
        """Validate normalized response ordering and uniqueness."""
        if not self.model.strip():
            raise ValueError("model must not be empty")

        if not self.results:
            raise ValueError("rerank results must not be empty")

        document_ids = [result.document_id for result in self.results]

        if len(document_ids) != len(set(document_ids)):
            raise ValueError("rerank result document identifiers must be unique")

        ranks = tuple(result.rank for result in self.results)

        expected_ranks = tuple(range(1, len(self.results) + 1))

        if ranks != expected_ranks:
            raise ValueError("rerank results must use contiguous rank order")


@runtime_checkable
class RerankerProvider(Provider, Protocol):
    """Asynchronous provider contract for document reranking."""

    async def rerank(
        self,
        request: RerankRequest,
    ) -> RerankResponse:
        """Rerank candidate documents against the supplied query."""
        ...

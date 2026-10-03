"""Provider-neutral contracts for embedding generation."""

from dataclasses import dataclass
from math import isfinite
from typing import Protocol, runtime_checkable

from ai_engineering_agent_platform.contracts.provider import Provider


@dataclass(frozen=True, slots=True)
class EmbeddingRequest:
    """Immutable request sent through an embedding provider."""

    model: str
    texts: tuple[str, ...]
    dimensions: int | None = None

    def __post_init__(self) -> None:
        """Validate provider-independent embedding request invariants."""
        if not self.model.strip():
            raise ValueError("model must not be empty")

        if not self.texts:
            raise ValueError("texts must not be empty")

        if any(not text.strip() for text in self.texts):
            raise ValueError("embedding texts must not contain empty values")

        if self.dimensions is not None and self.dimensions <= 0:
            raise ValueError("dimensions must be positive")


@dataclass(frozen=True, slots=True)
class EmbeddingVector:
    """Immutable normalized embedding vector."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate vector shape and numeric values."""
        if not self.values:
            raise ValueError("embedding vector must not be empty")

        if any(not isfinite(value) for value in self.values):
            raise ValueError("embedding vector values must be finite")

    @property
    def dimensions(self) -> int:
        """Return the number of dimensions in the vector."""
        return len(self.values)


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    """Normalized embedding provider response."""

    model: str
    embeddings: tuple[EmbeddingVector, ...]
    input_tokens: int | None = None

    def __post_init__(self) -> None:
        """Validate normalized embedding response invariants."""
        if not self.model.strip():
            raise ValueError("model must not be empty")

        if not self.embeddings:
            raise ValueError("embeddings must not be empty")

        expected_dimensions = self.embeddings[0].dimensions

        if any(
            embedding.dimensions != expected_dimensions for embedding in self.embeddings
        ):
            raise ValueError("all embedding vectors must have equal dimensions")

        if self.input_tokens is not None and self.input_tokens < 0:
            raise ValueError("input_tokens must be non-negative")

    @property
    def dimensions(self) -> int:
        """Return the common embedding dimensionality."""
        return self.embeddings[0].dimensions


@runtime_checkable
class EmbeddingProvider(Provider, Protocol):
    """Asynchronous provider contract for embedding generation."""

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse:
        """Generate normalized embeddings for the supplied texts."""
        ...

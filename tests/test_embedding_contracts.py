"""Tests for provider-neutral embedding contracts."""

import asyncio
from dataclasses import FrozenInstanceError

import pytest

from ai_engineering_agent_platform.contracts import (
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
    ProviderDescriptor,
    ProviderKind,
)


class ExampleEmbeddingProvider:
    """Minimal structural implementation of the embedding contract."""

    @property
    def descriptor(self) -> ProviderDescriptor:
        """Return deterministic provider metadata."""
        return ProviderDescriptor(
            name="example-embedding",
            kind=ProviderKind.EMBEDDING,
        )

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse:
        """Return deterministic normalized embeddings."""
        embeddings = tuple(
            EmbeddingVector(
                values=(0.1, 0.2, 0.3),
            )
            for _ in request.texts
        )

        return EmbeddingResponse(
            model=request.model,
            embeddings=embeddings,
            input_tokens=len(request.texts),
        )


def _request() -> EmbeddingRequest:
    """Return a deterministic valid request."""
    return EmbeddingRequest(
        model="example-embedding-model",
        texts=("first", "second"),
    )


def test_embedding_request_is_immutable() -> None:
    """Embedding requests should not mutate after creation."""
    request = _request()

    with pytest.raises(FrozenInstanceError):
        request.model = "different-model"  # type: ignore[misc]


def test_embedding_request_rejects_invalid_model_and_texts() -> None:
    """Embedding requests require usable model and text values."""
    with pytest.raises(
        ValueError,
        match="model must not be empty",
    ):
        EmbeddingRequest(
            model=" ",
            texts=("text",),
        )

    with pytest.raises(
        ValueError,
        match="texts must not be empty",
    ):
        EmbeddingRequest(
            model="model",
            texts=(),
        )

    with pytest.raises(
        ValueError,
        match="embedding texts must not contain empty values",
    ):
        EmbeddingRequest(
            model="model",
            texts=("valid", " "),
        )


def test_embedding_request_rejects_invalid_dimensions() -> None:
    """Requested dimensionality must be positive when supplied."""
    with pytest.raises(
        ValueError,
        match="dimensions must be positive",
    ):
        EmbeddingRequest(
            model="model",
            texts=("text",),
            dimensions=0,
        )


def test_embedding_vector_validates_values() -> None:
    """Vectors must contain finite numeric values."""
    with pytest.raises(
        ValueError,
        match="embedding vector must not be empty",
    ):
        EmbeddingVector(values=())

    with pytest.raises(
        ValueError,
        match="embedding vector values must be finite",
    ):
        EmbeddingVector(
            values=(0.1, float("nan")),
        )

    with pytest.raises(
        ValueError,
        match="embedding vector values must be finite",
    ):
        EmbeddingVector(
            values=(0.1, float("inf")),
        )


def test_embedding_vector_reports_dimensions() -> None:
    """Vector dimensionality should be deterministic."""
    vector = EmbeddingVector(
        values=(0.1, 0.2, 0.3),
    )

    assert vector.dimensions == 3


def test_embedding_response_requires_consistent_dimensions() -> None:
    """All vectors in one response must share dimensionality."""
    with pytest.raises(
        ValueError,
        match="all embedding vectors must have equal dimensions",
    ):
        EmbeddingResponse(
            model="model",
            embeddings=(
                EmbeddingVector(
                    values=(0.1, 0.2),
                ),
                EmbeddingVector(
                    values=(0.1, 0.2, 0.3),
                ),
            ),
        )


def test_embedding_response_validates_usage() -> None:
    """Optional token accounting cannot be negative."""
    response = EmbeddingResponse(
        model="model",
        embeddings=(
            EmbeddingVector(
                values=(0.1, 0.2),
            ),
        ),
        input_tokens=7,
    )

    assert response.input_tokens == 7
    assert response.dimensions == 2

    with pytest.raises(
        ValueError,
        match="input_tokens must be non-negative",
    ):
        EmbeddingResponse(
            model="model",
            embeddings=(
                EmbeddingVector(
                    values=(0.1, 0.2),
                ),
            ),
            input_tokens=-1,
        )


def test_embedding_provider_supports_structural_async_typing() -> None:
    """Concrete adapters should satisfy EmbeddingProvider structurally."""
    provider: EmbeddingProvider = ExampleEmbeddingProvider()

    assert isinstance(provider, EmbeddingProvider)
    assert provider.descriptor.kind is ProviderKind.EMBEDDING

    response = asyncio.run(provider.embed(_request()))

    assert response.model == "example-embedding-model"
    assert len(response.embeddings) == 2
    assert response.dimensions == 3
    assert response.input_tokens == 2

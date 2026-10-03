"""Tests for embedding runtime composition."""

import asyncio
import json

import httpx2
import pytest

from ai_engineering_agent_platform.adapters.ollama import (
    OllamaEmbeddingProvider,
)
from ai_engineering_agent_platform.config import Settings
from ai_engineering_agent_platform.contracts import (
    EmbeddingRequest,
    ProviderKind,
)
from ai_engineering_agent_platform.domain import ProviderExecutionError
from ai_engineering_agent_platform.runtime import (
    ollama_embedding_runtime,
)


class RecordingEmbeddingTransport(httpx2.AsyncBaseTransport):
    """Record embedding requests and transport lifecycle state."""

    def __init__(
        self,
        *,
        malformed_response: bool = False,
    ) -> None:
        """Initialize deterministic transport state."""
        self.requests: list[httpx2.Request] = []
        self.closed = False
        self.malformed_response = malformed_response

    async def handle_async_request(
        self,
        request: httpx2.Request,
    ) -> httpx2.Response:
        """Record one request and return a deterministic response."""
        self.requests.append(request)

        if self.malformed_response:
            return httpx2.Response(
                200,
                json={
                    "model": "example-embedding-model",
                    "embeddings": [
                        [0.1, 0.2],
                        [0.3, 0.4],
                    ],
                    "prompt_eval_count": 4,
                },
            )

        return httpx2.Response(
            200,
            json={
                "model": "example-embedding-model",
                "embeddings": [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                ],
                "prompt_eval_count": 7,
            },
        )

    async def aclose(self) -> None:
        """Record transport closure."""
        self.closed = True


def _settings() -> Settings:
    """Return deterministic Ollama runtime settings."""
    return Settings(
        ollama_base_url="https://models.example.test",
        ollama_request_timeout_seconds=45.5,
    )


def _request() -> EmbeddingRequest:
    """Return one deterministic embedding request."""
    return EmbeddingRequest(
        model="example-embedding-model",
        texts=(
            "first synthetic text",
            "second synthetic text",
        ),
        dimensions=3,
    )


def test_embedding_runtime_composes_provider_and_closes_client() -> None:
    """Runtime should compose the provider and own client cleanup."""
    transport = RecordingEmbeddingTransport()

    async def exercise() -> None:
        assert transport.closed is False

        async with ollama_embedding_runtime(
            _settings(),
            transport=transport,
        ) as provider:
            assert isinstance(
                provider,
                OllamaEmbeddingProvider,
            )
            assert provider.descriptor.name == "ollama"
            assert provider.descriptor.kind is ProviderKind.EMBEDDING

            assert transport.closed is False

            result = await provider.embed(_request())

            assert result.model == "example-embedding-model"
            assert len(result.embeddings) == 2
            assert result.dimensions == 3
            assert result.input_tokens == 7

            assert len(transport.requests) == 1

            observed = transport.requests[0]

            assert observed.method == "POST"
            assert str(observed.url) == ("https://models.example.test/api/embed")

            body = json.loads(observed.content)

            assert body == {
                "model": "example-embedding-model",
                "input": [
                    "first synthetic text",
                    "second synthetic text",
                ],
                "truncate": False,
                "dimensions": 3,
            }

        assert transport.closed is True

    asyncio.run(exercise())


def test_embedding_runtime_closes_client_after_provider_error() -> None:
    """Runtime cleanup must execute when embedding normalization fails."""
    transport = RecordingEmbeddingTransport(
        malformed_response=True,
    )

    async def exercise() -> None:
        with pytest.raises(
            ProviderExecutionError,
            match="dimensions do not match",
        ):
            async with ollama_embedding_runtime(
                _settings(),
                transport=transport,
            ) as provider:
                await provider.embed(_request())

        assert transport.closed is True

    asyncio.run(exercise())

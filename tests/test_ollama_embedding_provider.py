"""Tests for the Ollama embedding provider adapter."""

import asyncio
import json

import httpx2
import pytest

from ai_engineering_agent_platform.adapters.ollama import (
    OllamaEmbeddingProvider,
)
from ai_engineering_agent_platform.contracts import (
    EmbeddingProvider,
    EmbeddingRequest,
    ProviderKind,
)
from ai_engineering_agent_platform.domain import (
    ProviderExecutionError,
    ProviderUnavailableError,
)


def _request() -> EmbeddingRequest:
    """Return a deterministic embedding request."""
    return EmbeddingRequest(
        model="example-embedding-model",
        texts=(
            "first synthetic text",
            "second synthetic text",
        ),
        dimensions=3,
    )


def _successful_response() -> httpx2.Response:
    """Return one deterministic Ollama embedding response."""
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


def test_embedding_provider_satisfies_contract() -> None:
    """The adapter should satisfy EmbeddingProvider structurally."""
    transport = httpx2.MockTransport(lambda request: _successful_response())

    async def exercise() -> None:
        async with httpx2.AsyncClient(
            base_url="http://127.0.0.1:11434",
            transport=transport,
        ) as client:
            provider = OllamaEmbeddingProvider(client)

            typed_provider: EmbeddingProvider = provider

            assert isinstance(
                typed_provider,
                EmbeddingProvider,
            )
            assert provider.descriptor.name == "ollama"
            assert provider.descriptor.kind is ProviderKind.EMBEDDING

    asyncio.run(exercise())


def test_embedding_provider_posts_exact_mapped_request() -> None:
    """Embedding should POST the exact mapped batch payload."""
    observed_requests: list[httpx2.Request] = []

    def handler(
        request: httpx2.Request,
    ) -> httpx2.Response:
        observed_requests.append(request)

        assert request.method == "POST"
        assert str(request.url) == ("http://127.0.0.1:11434/api/embed")

        content_type = request.headers.get("content-type")

        assert content_type is not None
        assert content_type.startswith("application/json")

        body = json.loads(request.content)

        assert body == {
            "model": "example-embedding-model",
            "input": [
                "first synthetic text",
                "second synthetic text",
            ],
            "truncate": False,
            "dimensions": 3,
        }

        return _successful_response()

    async def exercise() -> None:
        transport = httpx2.MockTransport(handler)

        async with httpx2.AsyncClient(
            base_url="http://127.0.0.1:11434",
            timeout=120.0,
            transport=transport,
        ) as client:
            provider = OllamaEmbeddingProvider(client)

            result = await provider.embed(_request())

            assert result.model == "example-embedding-model"
            assert len(result.embeddings) == 2
            assert result.dimensions == 3
            assert result.input_tokens == 7

    asyncio.run(exercise())

    assert len(observed_requests) == 1


def test_embedding_provider_uses_response_mapper() -> None:
    """Malformed successful responses should fail through mapping."""

    def handler(
        request: httpx2.Request,
    ) -> httpx2.Response:
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

    async def exercise() -> None:
        transport = httpx2.MockTransport(handler)

        async with httpx2.AsyncClient(
            base_url="http://127.0.0.1:11434",
            transport=transport,
        ) as client:
            provider = OllamaEmbeddingProvider(client)

            with pytest.raises(
                ProviderExecutionError,
                match="dimensions do not match",
            ):
                await provider.embed(_request())

    asyncio.run(exercise())


def test_embedding_provider_normalizes_timeout() -> None:
    """Timeout failures should become unavailable errors."""

    def handler(
        request: httpx2.Request,
    ) -> httpx2.Response:
        raise httpx2.ReadTimeout(
            "simulated timeout",
            request=request,
        )

    async def exercise() -> None:
        transport = httpx2.MockTransport(handler)

        async with httpx2.AsyncClient(
            base_url="http://127.0.0.1:11434",
            transport=transport,
        ) as client:
            provider = OllamaEmbeddingProvider(client)

            with pytest.raises(
                ProviderUnavailableError,
                match="embedding request timed out",
            ) as captured:
                await provider.embed(_request())

            assert isinstance(
                captured.value.__cause__,
                httpx2.TimeoutException,
            )

    asyncio.run(exercise())


def test_embedding_provider_normalizes_transport_failure() -> None:
    """Transport failures should become unavailable errors."""

    def handler(
        request: httpx2.Request,
    ) -> httpx2.Response:
        raise httpx2.ConnectError(
            "simulated connection failure",
            request=request,
        )

    async def exercise() -> None:
        transport = httpx2.MockTransport(handler)

        async with httpx2.AsyncClient(
            base_url="http://127.0.0.1:11434",
            transport=transport,
        ) as client:
            provider = OllamaEmbeddingProvider(client)

            with pytest.raises(
                ProviderUnavailableError,
                match="transport layer",
            ) as captured:
                await provider.embed(_request())

            assert isinstance(
                captured.value.__cause__,
                httpx2.ConnectError,
            )

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "status_code",
    [
        408,
        429,
        500,
        503,
    ],
)
def test_embedding_provider_normalizes_retryable_http_status(
    status_code: int,
) -> None:
    """Temporary HTTP failures should become unavailable errors."""
    response_body_marker = "sensitive-response-body"

    def handler(
        request: httpx2.Request,
    ) -> httpx2.Response:
        return httpx2.Response(
            status_code,
            text=response_body_marker,
        )

    async def exercise() -> None:
        transport = httpx2.MockTransport(handler)

        async with httpx2.AsyncClient(
            base_url="http://127.0.0.1:11434",
            transport=transport,
        ) as client:
            provider = OllamaEmbeddingProvider(client)

            with pytest.raises(
                ProviderUnavailableError,
                match=rf"HTTP {status_code}",
            ) as captured:
                await provider.embed(_request())

            assert response_body_marker not in str(captured.value)
            assert isinstance(
                captured.value.__cause__,
                httpx2.HTTPStatusError,
            )

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "status_code",
    [
        400,
        401,
        403,
        404,
        422,
    ],
)
def test_embedding_provider_normalizes_non_retryable_http_status(
    status_code: int,
) -> None:
    """Rejected HTTP requests should become execution errors."""
    response_body_marker = "sensitive-response-body"

    def handler(
        request: httpx2.Request,
    ) -> httpx2.Response:
        return httpx2.Response(
            status_code,
            text=response_body_marker,
        )

    async def exercise() -> None:
        transport = httpx2.MockTransport(handler)

        async with httpx2.AsyncClient(
            base_url="http://127.0.0.1:11434",
            transport=transport,
        ) as client:
            provider = OllamaEmbeddingProvider(client)

            with pytest.raises(
                ProviderExecutionError,
                match=rf"HTTP {status_code}",
            ) as captured:
                await provider.embed(_request())

            assert response_body_marker not in str(captured.value)
            assert isinstance(
                captured.value.__cause__,
                httpx2.HTTPStatusError,
            )

    asyncio.run(exercise())


def test_embedding_provider_normalizes_invalid_json() -> None:
    """Invalid successful JSON should become an execution error."""

    def handler(
        request: httpx2.Request,
    ) -> httpx2.Response:
        return httpx2.Response(
            200,
            content=b"not-json",
            headers={
                "content-type": "application/json",
            },
        )

    async def exercise() -> None:
        transport = httpx2.MockTransport(handler)

        async with httpx2.AsyncClient(
            base_url="http://127.0.0.1:11434",
            transport=transport,
        ) as client:
            provider = OllamaEmbeddingProvider(client)

            with pytest.raises(
                ProviderExecutionError,
                match="response was not valid JSON",
            ) as captured:
                await provider.embed(_request())

            assert isinstance(
                captured.value.__cause__,
                ValueError,
            )

    asyncio.run(exercise())

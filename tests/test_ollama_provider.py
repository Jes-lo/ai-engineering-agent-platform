"""Tests for the Ollama LLM provider adapter."""

import asyncio
import json

import httpx2
import pytest

from ai_engineering_agent_platform.adapters.ollama import (
    OllamaLLMProvider,
)
from ai_engineering_agent_platform.contracts import (
    FinishReason,
    LLMMessage,
    LLMProvider,
    LLMRequest,
    MessageRole,
    ProviderKind,
)
from ai_engineering_agent_platform.domain import (
    ProviderExecutionError,
    ProviderUnavailableError,
)


def _request() -> LLMRequest:
    """Return a deterministic provider request."""
    return LLMRequest(
        model="example-model",
        messages=(
            LLMMessage(
                role=MessageRole.SYSTEM,
                content="Be concise.",
            ),
            LLMMessage(
                role=MessageRole.USER,
                content="Hello.",
            ),
        ),
        temperature=0.2,
        max_output_tokens=128,
    )


def _successful_response() -> httpx2.Response:
    """Return a deterministic successful Ollama response."""
    return httpx2.Response(
        200,
        json={
            "model": "example-model",
            "message": {
                "role": "assistant",
                "content": "Hello from Ollama.",
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 8,
            "eval_count": 4,
        },
    )


def test_ollama_provider_satisfies_llm_contract() -> None:
    """The adapter should satisfy LLMProvider structurally."""
    transport = httpx2.MockTransport(lambda request: _successful_response())

    async def exercise() -> None:
        async with httpx2.AsyncClient(
            base_url="http://127.0.0.1:11434",
            transport=transport,
        ) as client:
            provider = OllamaLLMProvider(client)

            typed_provider: LLMProvider = provider

            assert isinstance(
                typed_provider,
                LLMProvider,
            )

            assert provider.descriptor.kind is ProviderKind.LLM

            assert provider.descriptor.name == "ollama"

    asyncio.run(exercise())


def test_ollama_provider_posts_mapped_chat_request() -> None:
    """Generation should POST the exact non-streaming mapped payload."""
    observed_requests: list[httpx2.Request] = []

    def handler(
        request: httpx2.Request,
    ) -> httpx2.Response:
        observed_requests.append(request)

        assert request.method == "POST"

        assert str(request.url) == "http://127.0.0.1:11434/api/chat"

        content_type = request.headers.get("content-type")

        assert content_type is not None
        assert content_type.startswith("application/json")

        body = json.loads(request.content)

        assert body == {
            "model": "example-model",
            "messages": [
                {
                    "role": "system",
                    "content": "Be concise.",
                },
                {
                    "role": "user",
                    "content": "Hello.",
                },
            ],
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 128,
            },
        }

        return _successful_response()

    async def exercise() -> None:
        transport = httpx2.MockTransport(handler)

        async with httpx2.AsyncClient(
            base_url="http://127.0.0.1:11434",
            timeout=120.0,
            transport=transport,
        ) as client:
            provider = OllamaLLMProvider(client)

            result = await provider.generate(_request())

            assert result.model == "example-model"

            assert result.message.role is MessageRole.ASSISTANT

            assert result.message.content == "Hello from Ollama."

            assert result.finish_reason is FinishReason.STOP

            assert result.usage is not None
            assert result.usage.input_tokens == 8
            assert result.usage.output_tokens == 4

    asyncio.run(exercise())

    assert len(observed_requests) == 1


def test_ollama_provider_uses_response_mapper() -> None:
    """Malformed successful responses should fail through mapping validation."""

    def handler(
        request: httpx2.Request,
    ) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "model": "example-model",
                "message": {
                    "role": "user",
                    "content": "invalid role",
                },
                "done": True,
                "done_reason": "stop",
            },
        )

    async def exercise() -> None:
        transport = httpx2.MockTransport(handler)

        async with httpx2.AsyncClient(
            base_url="http://127.0.0.1:11434",
            transport=transport,
        ) as client:
            provider = OllamaLLMProvider(client)

            with pytest.raises(
                ProviderExecutionError,
                match="must use assistant role",
            ):
                await provider.generate(_request())

    asyncio.run(exercise())


def test_ollama_provider_normalizes_timeout() -> None:
    """Timeout failures should become provider-unavailable errors."""

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
            provider = OllamaLLMProvider(client)

            with pytest.raises(
                ProviderUnavailableError,
                match="request timed out",
            ) as captured:
                await provider.generate(_request())

            assert isinstance(
                captured.value.__cause__,
                httpx2.TimeoutException,
            )

    asyncio.run(exercise())


def test_ollama_provider_normalizes_transport_failure() -> None:
    """Network failures should become provider-unavailable errors."""

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
            provider = OllamaLLMProvider(client)

            with pytest.raises(
                ProviderUnavailableError,
                match="transport layer",
            ) as captured:
                await provider.generate(_request())

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
def test_ollama_provider_normalizes_retryable_http_status(
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
            provider = OllamaLLMProvider(client)

            with pytest.raises(
                ProviderUnavailableError,
                match=rf"HTTP {status_code}",
            ) as captured:
                await provider.generate(_request())

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
def test_ollama_provider_normalizes_non_retryable_http_status(
    status_code: int,
) -> None:
    """Rejected requests should become provider-execution errors."""
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
            provider = OllamaLLMProvider(client)

            with pytest.raises(
                ProviderExecutionError,
                match=rf"HTTP {status_code}",
            ) as captured:
                await provider.generate(_request())

            assert response_body_marker not in str(captured.value)

            assert isinstance(
                captured.value.__cause__,
                httpx2.HTTPStatusError,
            )

    asyncio.run(exercise())


def test_ollama_provider_normalizes_invalid_json() -> None:
    """Invalid successful JSON should become a provider-execution error."""

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
            provider = OllamaLLMProvider(client)

            with pytest.raises(
                ProviderExecutionError,
                match="response was not valid JSON",
            ) as captured:
                await provider.generate(_request())

            assert isinstance(
                captured.value.__cause__,
                ValueError,
            )

    asyncio.run(exercise())

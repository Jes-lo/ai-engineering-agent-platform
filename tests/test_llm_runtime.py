"""Tests for LLM runtime composition."""

import asyncio

import httpx2
import pytest

from ai_engineering_agent_platform.adapters.ollama import (
    OllamaLLMProvider,
)
from ai_engineering_agent_platform.config import Settings
from ai_engineering_agent_platform.contracts import (
    FinishReason,
    LLMMessage,
    LLMRequest,
    MessageRole,
)
from ai_engineering_agent_platform.runtime import (
    create_ollama_http_client,
    ollama_llm_runtime,
)


class RecordingTransport(httpx2.AsyncBaseTransport):
    """Record requests and expose transport lifecycle state."""

    def __init__(self) -> None:
        """Initialize deterministic transport state."""
        self.requests: list[httpx2.Request] = []
        self.closed = False

    async def handle_async_request(
        self,
        request: httpx2.Request,
    ) -> httpx2.Response:
        """Return one deterministic Ollama-compatible response."""
        self.requests.append(request)

        return httpx2.Response(
            200,
            json={
                "model": "example-model",
                "message": {
                    "role": "assistant",
                    "content": "Hello from runtime.",
                },
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 5,
                "eval_count": 3,
            },
        )

    async def aclose(self) -> None:
        """Record transport closure."""
        self.closed = True


def _settings() -> Settings:
    """Return isolated deterministic runtime settings."""
    return Settings(
        ollama_base_url="https://models.example.test",
        ollama_request_timeout_seconds=45.5,
    )


def _request() -> LLMRequest:
    """Return a deterministic LLM request."""
    return LLMRequest(
        model="example-model",
        messages=(
            LLMMessage(
                role=MessageRole.USER,
                content="Hello.",
            ),
        ),
    )


def test_create_ollama_http_client_uses_settings() -> None:
    """Client factory should apply configured URL and timeout."""

    async def exercise() -> None:
        client = create_ollama_http_client(_settings())

        try:
            assert str(client.base_url) == "https://models.example.test"

            assert client.timeout.connect == 45.5
            assert client.timeout.read == 45.5
            assert client.timeout.write == 45.5
            assert client.timeout.pool == 45.5

            assert client.is_closed is False
        finally:
            await client.aclose()

        assert client.is_closed is True

    asyncio.run(exercise())


def test_ollama_runtime_composes_provider_and_client() -> None:
    """Runtime should wire settings, HTTP client, and provider together."""
    transport = RecordingTransport()

    async def exercise() -> None:
        async with ollama_llm_runtime(
            _settings(),
            transport=transport,
        ) as provider:
            assert isinstance(
                provider,
                OllamaLLMProvider,
            )

            result = await provider.generate(_request())

            assert result.model == "example-model"
            assert result.message.role is MessageRole.ASSISTANT
            assert result.message.content == "Hello from runtime."
            assert result.finish_reason is FinishReason.STOP
            assert result.usage is not None
            assert result.usage.total_tokens == 8

            assert transport.closed is False

        assert transport.closed is True

    asyncio.run(exercise())

    assert len(transport.requests) == 1

    request = transport.requests[0]

    assert request.method == "POST"

    assert str(request.url) == "https://models.example.test/api/chat"


def test_ollama_runtime_closes_client_when_consumer_fails() -> None:
    """Runtime must close its client even when caller code raises."""
    transport = RecordingTransport()

    async def exercise() -> None:
        with pytest.raises(
            RuntimeError,
            match="simulated consumer failure",
        ):
            async with ollama_llm_runtime(
                _settings(),
                transport=transport,
            ):
                assert transport.closed is False

                raise RuntimeError("simulated consumer failure")

        assert transport.closed is True

    asyncio.run(exercise())

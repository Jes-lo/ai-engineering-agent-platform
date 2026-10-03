"""Runtime composition for LLM provider implementations."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx2

from ai_engineering_agent_platform.adapters.ollama import (
    OllamaLLMProvider,
)
from ai_engineering_agent_platform.config import Settings


def create_ollama_http_client(
    settings: Settings,
    *,
    transport: httpx2.AsyncBaseTransport | None = None,
) -> httpx2.AsyncClient:
    """Create an Ollama HTTP client from validated runtime settings."""
    return httpx2.AsyncClient(
        base_url=settings.ollama_base_url,
        timeout=settings.ollama_request_timeout_seconds,
        transport=transport,
    )


@asynccontextmanager
async def ollama_llm_runtime(
    settings: Settings,
    *,
    transport: httpx2.AsyncBaseTransport | None = None,
) -> AsyncIterator[OllamaLLMProvider]:
    """Yield an Ollama provider and own its HTTP-client lifecycle."""
    client = create_ollama_http_client(
        settings,
        transport=transport,
    )

    try:
        yield OllamaLLMProvider(client)
    finally:
        await client.aclose()

"""Runtime composition for embedding provider implementations."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx2

from ai_engineering_agent_platform.adapters.ollama import (
    OllamaEmbeddingProvider,
)
from ai_engineering_agent_platform.config import Settings
from ai_engineering_agent_platform.runtime.ollama import (
    create_ollama_http_client,
)


@asynccontextmanager
async def ollama_embedding_runtime(
    settings: Settings,
    *,
    transport: httpx2.AsyncBaseTransport | None = None,
) -> AsyncIterator[OllamaEmbeddingProvider]:
    """Yield an Ollama embedding provider and own its client lifecycle."""
    client = create_ollama_http_client(
        settings,
        transport=transport,
    )

    try:
        yield OllamaEmbeddingProvider(client)
    finally:
        await client.aclose()

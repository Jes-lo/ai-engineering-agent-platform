"""Shared runtime composition primitives for Ollama providers."""

import httpx2

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

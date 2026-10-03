"""Shared HTTP error normalization for Ollama adapters."""

from typing import Never

import httpx2

from ai_engineering_agent_platform.domain import (
    ProviderExecutionError,
    ProviderUnavailableError,
)


def raise_ollama_http_status_error(
    error: httpx2.HTTPStatusError,
) -> Never:
    """Raise one normalized platform error for an Ollama HTTP failure."""
    status_code = error.response.status_code

    if status_code in {408, 429} or status_code >= 500:
        raise ProviderUnavailableError(
            f"Ollama service unavailable (HTTP {status_code})"
        ) from error

    raise ProviderExecutionError(
        f"Ollama request rejected (HTTP {status_code})"
    ) from error

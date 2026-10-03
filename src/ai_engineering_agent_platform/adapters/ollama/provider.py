"""Ollama implementation of the platform LLM provider contract."""

import httpx2

from ai_engineering_agent_platform.adapters.ollama.mapping import (
    build_ollama_chat_payload,
    parse_ollama_chat_response,
)
from ai_engineering_agent_platform.contracts import (
    LLMRequest,
    LLMResponse,
    ProviderDescriptor,
    ProviderKind,
)
from ai_engineering_agent_platform.domain import (
    ProviderExecutionError,
    ProviderUnavailableError,
)


class OllamaLLMProvider:
    """Generate LLM responses through an injected Ollama HTTP client."""

    _descriptor = ProviderDescriptor(
        name="ollama",
        kind=ProviderKind.LLM,
    )

    def __init__(
        self,
        client: httpx2.AsyncClient,
    ) -> None:
        """Initialize without taking ownership of the client lifecycle."""
        self._client = client

    @property
    def descriptor(self) -> ProviderDescriptor:
        """Return stable provider identity metadata."""
        return self._descriptor

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """Generate one normalized non-streaming Ollama chat response."""
        try:
            response = await self._client.post(
                "/api/chat",
                json=build_ollama_chat_payload(request),
            )
        except httpx2.TimeoutException as exc:
            raise ProviderUnavailableError("Ollama request timed out") from exc
        except httpx2.RequestError as exc:
            raise ProviderUnavailableError(
                "Ollama request failed at the transport layer"
            ) from exc

        try:
            response.raise_for_status()
        except httpx2.HTTPStatusError as exc:
            self._raise_normalized_status_error(exc)

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderExecutionError("Ollama response was not valid JSON") from exc

        return parse_ollama_chat_response(payload)

    @staticmethod
    def _raise_normalized_status_error(
        error: httpx2.HTTPStatusError,
    ) -> None:
        """Translate HTTP status failures without exposing response bodies."""
        status_code = error.response.status_code

        if status_code in {408, 429} or status_code >= 500:
            raise ProviderUnavailableError(
                f"Ollama service unavailable (HTTP {status_code})"
            ) from error

        raise ProviderExecutionError(
            f"Ollama request rejected (HTTP {status_code})"
        ) from error

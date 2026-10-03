"""Ollama implementation of the platform LLM provider contract."""

import httpx2

from ai_engineering_agent_platform.adapters.ollama.http_errors import (
    raise_ollama_http_status_error,
)
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
            raise_ollama_http_status_error(exc)

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderExecutionError("Ollama response was not valid JSON") from exc

        return parse_ollama_chat_response(
            payload,
            requested_tools=request.tools,
        )

"""Tests for provider-neutral LLM contracts."""

import asyncio
from dataclasses import FrozenInstanceError

import pytest

from ai_engineering_agent_platform.contracts import (
    FinishReason,
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    MessageRole,
    ProviderDescriptor,
    ProviderKind,
    TokenUsage,
)


class ExampleLLMProvider:
    """Minimal structural implementation of the LLM contract."""

    @property
    def descriptor(self) -> ProviderDescriptor:
        """Return deterministic provider metadata."""
        return ProviderDescriptor(
            name="example-llm",
            kind=ProviderKind.LLM,
        )

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """Return a deterministic normalized response."""
        return LLMResponse(
            model=request.model,
            message=LLMMessage(
                role=MessageRole.ASSISTANT,
                content="generated response",
            ),
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(
                input_tokens=3,
                output_tokens=2,
            ),
        )


def _request() -> LLMRequest:
    """Return a deterministic valid request fixture."""
    return LLMRequest(
        model="example-model",
        messages=(
            LLMMessage(
                role=MessageRole.USER,
                content="hello",
            ),
        ),
    )


def test_llm_wire_values_are_stable() -> None:
    """Portable enums should expose predictable serialized values."""
    assert MessageRole.SYSTEM.value == "system"
    assert MessageRole.USER.value == "user"
    assert MessageRole.ASSISTANT.value == "assistant"

    assert FinishReason.STOP.value == "stop"
    assert FinishReason.LENGTH.value == "length"
    assert FinishReason.CONTENT_FILTER.value == "content_filter"
    assert FinishReason.OTHER.value == "other"


def test_llm_request_is_immutable() -> None:
    """LLM requests should not mutate after construction."""
    request = _request()

    with pytest.raises(FrozenInstanceError):
        request.model = "different-model"  # type: ignore[misc]


def test_llm_request_rejects_empty_model() -> None:
    """Requests require an explicit model identifier."""
    with pytest.raises(
        ValueError,
        match="model must not be empty",
    ):
        LLMRequest(
            model=" ",
            messages=(
                LLMMessage(
                    role=MessageRole.USER,
                    content="hello",
                ),
            ),
        )


def test_llm_request_rejects_empty_messages() -> None:
    """Generation requires at least one conversational message."""
    with pytest.raises(
        ValueError,
        match="messages must not be empty",
    ):
        LLMRequest(
            model="example-model",
            messages=(),
        )


def test_llm_request_rejects_invalid_numeric_controls() -> None:
    """Portable numeric controls must reject invalid values."""
    messages = (
        LLMMessage(
            role=MessageRole.USER,
            content="hello",
        ),
    )

    with pytest.raises(
        ValueError,
        match="temperature must be non-negative",
    ):
        LLMRequest(
            model="example-model",
            messages=messages,
            temperature=-0.1,
        )

    with pytest.raises(
        ValueError,
        match="max_output_tokens must be positive",
    ):
        LLMRequest(
            model="example-model",
            messages=messages,
            max_output_tokens=0,
        )


def test_token_usage_validates_and_totals_tokens() -> None:
    """Token accounting must be non-negative and deterministic."""
    usage = TokenUsage(
        input_tokens=11,
        output_tokens=7,
    )

    assert usage.total_tokens == 18

    with pytest.raises(
        ValueError,
        match="input_tokens must be non-negative",
    ):
        TokenUsage(
            input_tokens=-1,
            output_tokens=0,
        )

    with pytest.raises(
        ValueError,
        match="output_tokens must be non-negative",
    ):
        TokenUsage(
            input_tokens=0,
            output_tokens=-1,
        )


def test_llm_response_requires_assistant_role() -> None:
    """Normalized model output must be an assistant message."""
    with pytest.raises(
        ValueError,
        match="LLM response message must use assistant role",
    ):
        LLMResponse(
            model="example-model",
            message=LLMMessage(
                role=MessageRole.USER,
                content="invalid response",
            ),
            finish_reason=FinishReason.STOP,
        )


def test_llm_provider_supports_structural_async_typing() -> None:
    """Concrete adapters should satisfy LLMProvider structurally."""
    provider: LLMProvider = ExampleLLMProvider()

    assert isinstance(provider, LLMProvider)
    assert provider.descriptor.kind is ProviderKind.LLM

    response = asyncio.run(provider.generate(_request()))

    assert response.model == "example-model"
    assert response.message.content == "generated response"
    assert response.finish_reason is FinishReason.STOP
    assert response.usage is not None
    assert response.usage.total_tokens == 5

"""Provider-neutral contracts for large language models."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from ai_engineering_agent_platform.contracts.provider import Provider


class MessageRole(StrEnum):
    """Portable conversational roles understood by the platform."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class FinishReason(StrEnum):
    """Normalized reasons for an LLM generation to finish."""

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """Provider-neutral conversational message."""

    role: MessageRole
    content: str


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """Immutable request sent through an LLM provider."""

    model: str
    messages: tuple[LLMMessage, ...]
    temperature: float | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        """Validate provider-independent request invariants."""
        if not self.model.strip():
            raise ValueError("model must not be empty")

        if not self.messages:
            raise ValueError("messages must not be empty")

        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature must be non-negative")

        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Normalized token accounting returned by a provider."""

    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        """Reject impossible token accounting values."""
        if self.input_tokens < 0:
            raise ValueError("input_tokens must be non-negative")

        if self.output_tokens < 0:
            raise ValueError("output_tokens must be non-negative")

    @property
    def total_tokens(self) -> int:
        """Return total input and output token consumption."""
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Normalized result returned by an LLM provider."""

    model: str
    message: LLMMessage
    finish_reason: FinishReason
    usage: TokenUsage | None = None

    def __post_init__(self) -> None:
        """Validate provider-independent response invariants."""
        if not self.model.strip():
            raise ValueError("model must not be empty")

        if self.message.role is not MessageRole.ASSISTANT:
            raise ValueError("LLM response message must use assistant role")


@runtime_checkable
class LLMProvider(Provider, Protocol):
    """Asynchronous provider contract for LLM generation."""

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """Generate one normalized LLM response."""
        ...

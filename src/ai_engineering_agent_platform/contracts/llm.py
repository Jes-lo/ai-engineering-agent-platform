"""Provider-neutral contracts for large language models."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from ai_engineering_agent_platform.contracts.provider import Provider
from ai_engineering_agent_platform.contracts.tool import (
    ToolArgument,
    ToolDefinition,
)


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
    TOOL_CALLS = "tool_calls"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """Provider-neutral conversational message."""

    role: MessageRole
    content: str


@dataclass(frozen=True, slots=True)
class LLMToolCall:
    """Untrusted provider-neutral tool-call proposal from an LLM.

    A proposal is not an executable ToolInvocation. Execution identity,
    authorization, approval, and policy enforcement belong to application
    orchestration and ToolExecutionService.
    """

    tool_name: str
    arguments: tuple[ToolArgument, ...] = ()
    provider_call_id: str | None = None

    def __post_init__(self) -> None:
        """Validate proposal identity and portable scalar arguments."""
        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise ValueError("tool_name must not be empty")

        if not isinstance(self.arguments, tuple):
            raise ValueError("arguments must be a tuple")

        if any(not isinstance(argument, ToolArgument) for argument in self.arguments):
            raise ValueError("arguments must contain ToolArgument values")

        argument_names = tuple(argument.name for argument in self.arguments)

        if len(argument_names) != len(set(argument_names)):
            raise ValueError("tool-call argument names must be unique")

        if self.provider_call_id is not None and (
            not isinstance(self.provider_call_id, str)
            or not self.provider_call_id.strip()
        ):
            raise ValueError("provider_call_id must not be empty when present")


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """Immutable request sent through an LLM provider."""

    model: str
    messages: tuple[LLMMessage, ...]
    temperature: float | None = None
    max_output_tokens: int | None = None
    tools: tuple[ToolDefinition, ...] = ()

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

        if not isinstance(self.tools, tuple):
            raise ValueError("tools must be a tuple")

        if any(not isinstance(tool, ToolDefinition) for tool in self.tools):
            raise ValueError("tools must contain ToolDefinition values")

        tool_names = tuple(tool.name for tool in self.tools)

        if len(tool_names) != len(set(tool_names)):
            raise ValueError("LLM request tool names must be unique")


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
    tool_calls: tuple[LLMToolCall, ...] = ()

    def __post_init__(self) -> None:
        """Validate provider-independent response invariants."""
        if not self.model.strip():
            raise ValueError("model must not be empty")

        if self.message.role is not MessageRole.ASSISTANT:
            raise ValueError("LLM response message must use assistant role")

        if not isinstance(self.tool_calls, tuple):
            raise ValueError("tool_calls must be a tuple")

        if any(not isinstance(tool_call, LLMToolCall) for tool_call in self.tool_calls):
            raise ValueError("tool_calls must contain LLMToolCall values")

        provider_call_ids = tuple(
            tool_call.provider_call_id
            for tool_call in self.tool_calls
            if tool_call.provider_call_id is not None
        )

        if len(provider_call_ids) != len(set(provider_call_ids)):
            raise ValueError("provider tool-call identifiers must be unique")

        if self.tool_calls:
            if self.finish_reason is not FinishReason.TOOL_CALLS:
                raise ValueError(
                    "responses with tool calls must use TOOL_CALLS finish reason"
                )
        elif self.finish_reason is FinishReason.TOOL_CALLS:
            raise ValueError("TOOL_CALLS finish reason requires at least one tool call")


@runtime_checkable
class LLMProvider(Provider, Protocol):
    """Asynchronous provider contract for LLM generation."""

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """Generate one normalized LLM response."""
        ...

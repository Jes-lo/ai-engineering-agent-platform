"""Provider-neutral contracts for large language models."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from ai_engineering_agent_platform.contracts.provider import Provider
from ai_engineering_agent_platform.contracts.tool import (
    ToolArgument,
    ToolDefinition,
    ToolResult,
)


class MessageRole(StrEnum):
    """Portable conversational roles understood by the platform."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(StrEnum):
    """Normalized reasons for an LLM generation to finish."""

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """Provider-neutral plain conversational message."""

    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        """Validate plain-message invariants."""
        if not isinstance(
            self.content,
            str,
        ):
            raise ValueError("message content must be a string")

        if self.role is MessageRole.TOOL:
            raise ValueError("tool role requires LLMToolResultMessage")


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
        if (
            not isinstance(
                self.tool_name,
                str,
            )
            or not self.tool_name.strip()
        ):
            raise ValueError("tool_name must not be empty")

        if not isinstance(
            self.arguments,
            tuple,
        ):
            raise ValueError("arguments must be a tuple")

        if any(
            not isinstance(
                argument,
                ToolArgument,
            )
            for argument in self.arguments
        ):
            raise ValueError("arguments must contain ToolArgument values")

        argument_names = tuple(argument.name for argument in self.arguments)

        if len(argument_names) != len(set(argument_names)):
            raise ValueError("tool-call argument names must be unique")

        if self.provider_call_id is not None and (
            not isinstance(
                self.provider_call_id,
                str,
            )
            or not self.provider_call_id.strip()
        ):
            raise ValueError("provider_call_id must not be empty when present")


@dataclass(frozen=True, slots=True)
class LLMAssistantToolCallMessage:
    """Assistant transcript entry containing inert tool-call proposals."""

    content: str
    tool_calls: tuple[LLMToolCall, ...]

    def __post_init__(self) -> None:
        """Validate assistant tool-call transcript invariants."""
        if not isinstance(
            self.content,
            str,
        ):
            raise ValueError("assistant tool-call content must be a string")

        if not isinstance(
            self.tool_calls,
            tuple,
        ):
            raise ValueError("assistant tool_calls must be a tuple")

        if not self.tool_calls:
            raise ValueError(
                "assistant tool-call message requires at least one tool call"
            )

        if any(
            not isinstance(
                tool_call,
                LLMToolCall,
            )
            for tool_call in self.tool_calls
        ):
            raise ValueError("assistant tool_calls must contain LLMToolCall values")

        provider_call_ids = tuple(
            tool_call.provider_call_id
            for tool_call in self.tool_calls
            if tool_call.provider_call_id is not None
        )

        if len(provider_call_ids) != len(set(provider_call_ids)):
            raise ValueError("assistant provider tool-call identifiers must be unique")

    @property
    def role(
        self,
    ) -> MessageRole:
        """Return the fixed assistant role."""
        return MessageRole.ASSISTANT


@dataclass(frozen=True, slots=True)
class LLMToolResultMessage:
    """Validated tool result represented as provider-neutral LLM context.

    The platform-controlled ToolResult.call_id is retained for internal
    transcript/audit correlation. A provider_call_id, when present, remains
    provider metadata and is not execution authority.
    """

    result: ToolResult
    provider_call_id: str | None = None

    def __post_init__(self) -> None:
        """Validate tool-result transcript metadata."""
        if not isinstance(
            self.result,
            ToolResult,
        ):
            raise ValueError("result must be a ToolResult")

        if self.provider_call_id is not None and (
            not isinstance(
                self.provider_call_id,
                str,
            )
            or not self.provider_call_id.strip()
        ):
            raise ValueError("provider_call_id must not be empty when present")

    @property
    def role(
        self,
    ) -> MessageRole:
        """Return the fixed tool role."""
        return MessageRole.TOOL

    @property
    def content(
        self,
    ) -> str:
        """Return normalized tool-result content."""
        return self.result.content

    @property
    def tool_name(
        self,
    ) -> str:
        """Return the normalized tool name."""
        return self.result.tool_name


type LLMConversationMessage = (
    LLMMessage | LLMAssistantToolCallMessage | LLMToolResultMessage
)


def _validate_conversation_messages(
    messages: tuple[
        LLMConversationMessage,
        ...,
    ],
) -> None:
    """Reject orphaned, incomplete, or mismatched tool-result transcripts."""
    pending_tool_calls: (
        tuple[
            LLMToolCall,
            ...,
        ]
        | None
    ) = None

    pending_index = 0

    for message in messages:
        if isinstance(
            message,
            LLMAssistantToolCallMessage,
        ):
            if pending_tool_calls is not None:
                raise ValueError(
                    "assistant tool calls require all prior tool results first"
                )

            pending_tool_calls = message.tool_calls
            pending_index = 0
            continue

        if isinstance(
            message,
            LLMToolResultMessage,
        ):
            if pending_tool_calls is None:
                raise ValueError(
                    "tool result message has no preceding assistant tool call"
                )

            expected = pending_tool_calls[pending_index]

            if message.result.tool_name != expected.tool_name:
                raise ValueError("tool result name does not match preceding tool call")

            if message.provider_call_id != expected.provider_call_id:
                raise ValueError(
                    "tool result provider_call_id does not match preceding tool call"
                )

            pending_index += 1

            if pending_index == len(pending_tool_calls):
                pending_tool_calls = None
                pending_index = 0

            continue

        if pending_tool_calls is not None:
            raise ValueError(
                "assistant tool calls must be followed by their tool results"
            )

    if pending_tool_calls is not None:
        raise ValueError("conversation ends before all tool results are present")


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """Immutable request sent through an LLM provider."""

    model: str
    messages: tuple[
        LLMConversationMessage,
        ...,
    ]
    temperature: float | None = None
    max_output_tokens: int | None = None
    tools: tuple[ToolDefinition, ...] = ()

    def __post_init__(self) -> None:
        """Validate provider-independent request invariants."""
        if not self.model.strip():
            raise ValueError("model must not be empty")

        if not self.messages:
            raise ValueError("messages must not be empty")

        if any(
            not isinstance(
                message,
                (
                    LLMMessage,
                    LLMAssistantToolCallMessage,
                    LLMToolResultMessage,
                ),
            )
            for message in self.messages
        ):
            raise ValueError("messages contain an unsupported conversation value")

        _validate_conversation_messages(self.messages)

        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature must be non-negative")

        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")

        if not isinstance(
            self.tools,
            tuple,
        ):
            raise ValueError("tools must be a tuple")

        if any(
            not isinstance(
                tool,
                ToolDefinition,
            )
            for tool in self.tools
        ):
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
    def total_tokens(
        self,
    ) -> int:
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

        if not isinstance(
            self.tool_calls,
            tuple,
        ):
            raise ValueError("tool_calls must be a tuple")

        if any(
            not isinstance(
                tool_call,
                LLMToolCall,
            )
            for tool_call in self.tool_calls
        ):
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

"""Tests for provider-neutral tool-result conversation messages."""

import pytest

from ai_engineering_agent_platform.adapters.ollama.mapping import (
    build_ollama_chat_payload,
)
from ai_engineering_agent_platform.contracts import (
    LLMAssistantToolCallMessage,
    LLMMessage,
    LLMRequest,
    LLMToolCall,
    LLMToolResultMessage,
    MessageRole,
    ToolArgument,
    ToolDefinition,
    ToolExecutionStatus,
    ToolParameter,
    ToolParameterType,
    ToolResult,
)


def _proposal(
    *,
    tool_name: str = "lookup",
    provider_call_id: str | None = "provider-1",
) -> LLMToolCall:
    """Return one inert provider-neutral proposal."""
    return LLMToolCall(
        tool_name=tool_name,
        arguments=(
            ToolArgument(
                name="query",
                value="synthetic",
            ),
        ),
        provider_call_id=provider_call_id,
    )


def _result(
    *,
    tool_name: str = "lookup",
    call_id: str = "platform-call-1",
    content: str = "synthetic-result",
) -> ToolResult:
    """Return one validated synthetic tool result."""
    return ToolResult(
        call_id=call_id,
        tool_name=tool_name,
        status=ToolExecutionStatus.SUCCESS,
        content=content,
    )


def _definition() -> ToolDefinition:
    """Return the matching synthetic tool definition."""
    return ToolDefinition(
        name="lookup",
        description="Read synthetic information.",
        parameters=(
            ToolParameter(
                name="query",
                parameter_type=(ToolParameterType.STRING),
                required=True,
            ),
        ),
    )


def test_message_role_exposes_tool_role() -> None:
    """Tool results have an explicit provider-neutral role."""
    assert MessageRole.TOOL.value == "tool"


def test_plain_message_rejects_tool_role() -> None:
    """Tool messages require the richer typed result contract."""
    with pytest.raises(
        ValueError,
        match="requires LLMToolResultMessage",
    ):
        LLMMessage(
            role=MessageRole.TOOL,
            content="unsafe-untyped-result",
        )


def test_assistant_tool_call_message_requires_proposal() -> None:
    """An assistant tool-call transcript cannot be empty."""
    with pytest.raises(
        ValueError,
        match="requires at least one tool call",
    ):
        LLMAssistantToolCallMessage(
            content="",
            tool_calls=(),
        )


def test_valid_tool_exchange_is_accepted() -> None:
    """A matched assistant-call/result pair forms valid model context."""
    proposal = _proposal()

    request = LLMRequest(
        model="synthetic-model",
        messages=(
            LLMMessage(
                role=MessageRole.USER,
                content="Look it up.",
            ),
            LLMAssistantToolCallMessage(
                content="",
                tool_calls=(proposal,),
            ),
            LLMToolResultMessage(
                result=_result(),
                provider_call_id=(proposal.provider_call_id),
            ),
        ),
    )

    assert len(request.messages) == 3

    assert request.messages[1].role is MessageRole.ASSISTANT

    assert request.messages[2].role is MessageRole.TOOL


def test_orphan_tool_result_is_rejected() -> None:
    """A tool result cannot appear without a preceding model proposal."""
    with pytest.raises(
        ValueError,
        match="no preceding assistant tool call",
    ):
        LLMRequest(
            model="synthetic-model",
            messages=(
                LLMMessage(
                    role=MessageRole.USER,
                    content="Synthetic.",
                ),
                LLMToolResultMessage(
                    result=_result(),
                    provider_call_id="provider-1",
                ),
            ),
        )


def test_incomplete_tool_exchange_is_rejected() -> None:
    """Every assistant tool call must receive a result before generation."""
    proposal = _proposal()

    with pytest.raises(
        ValueError,
        match="ends before all tool results",
    ):
        LLMRequest(
            model="synthetic-model",
            messages=(
                LLMMessage(
                    role=MessageRole.USER,
                    content="Synthetic.",
                ),
                LLMAssistantToolCallMessage(
                    content="",
                    tool_calls=(proposal,),
                ),
            ),
        )


def test_tool_result_name_must_match_proposal() -> None:
    """A result cannot be attached to a different proposed tool."""
    proposal = _proposal()

    with pytest.raises(
        ValueError,
        match="name does not match",
    ):
        LLMRequest(
            model="synthetic-model",
            messages=(
                LLMMessage(
                    role=MessageRole.USER,
                    content="Synthetic.",
                ),
                LLMAssistantToolCallMessage(
                    content="",
                    tool_calls=(proposal,),
                ),
                LLMToolResultMessage(
                    result=_result(tool_name="different-tool"),
                    provider_call_id="provider-1",
                ),
            ),
        )


def test_provider_call_identity_must_match_proposal() -> None:
    """Provider-side transcript correlation is exact when an ID exists."""
    proposal = _proposal(provider_call_id="provider-1")

    with pytest.raises(
        ValueError,
        match="provider_call_id does not match",
    ):
        LLMRequest(
            model="synthetic-model",
            messages=(
                LLMMessage(
                    role=MessageRole.USER,
                    content="Synthetic.",
                ),
                LLMAssistantToolCallMessage(
                    content="",
                    tool_calls=(proposal,),
                ),
                LLMToolResultMessage(
                    result=_result(),
                    provider_call_id="provider-2",
                ),
            ),
        )


def test_ollama_maps_validated_tool_exchange() -> None:
    """Ollama receives assistant calls and tool results in its wire shape."""
    proposal = _proposal(provider_call_id="provider-7")

    request = LLMRequest(
        model="synthetic-model",
        messages=(
            LLMMessage(
                role=MessageRole.USER,
                content="Look it up.",
            ),
            LLMAssistantToolCallMessage(
                content="",
                tool_calls=(proposal,),
            ),
            LLMToolResultMessage(
                result=_result(
                    call_id="platform-secret-call-77",
                    content="synthetic-ok",
                ),
                provider_call_id="provider-7",
            ),
        ),
        tools=(_definition(),),
    )

    payload = build_ollama_chat_payload(request)

    assert payload["messages"] == [
        {
            "role": "user",
            "content": "Look it up.",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "lookup",
                        "arguments": {
                            "query": "synthetic",
                        },
                    },
                    "id": "provider-7",
                }
            ],
        },
        {
            "role": "tool",
            "content": "synthetic-ok",
            "tool_name": "lookup",
        },
    ]

    assert "platform-secret-call-77" not in repr(payload)


def test_basic_ollama_message_mapping_remains_compatible() -> None:
    """Ordinary messages keep the original role/content wire shape."""
    payload = build_ollama_chat_payload(
        LLMRequest(
            model="synthetic-model",
            messages=(
                LLMMessage(
                    role=MessageRole.USER,
                    content="hello",
                ),
            ),
        )
    )

    assert payload["messages"] == [
        {
            "role": "user",
            "content": "hello",
        }
    ]

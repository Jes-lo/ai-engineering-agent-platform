"""Tests for provider-neutral LLM tool-call proposal semantics."""

import pytest

from ai_engineering_agent_platform.adapters.ollama.mapping import (
    build_ollama_chat_payload,
    parse_ollama_chat_response,
)
from ai_engineering_agent_platform.contracts import (
    FinishReason,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMToolCall,
    MessageRole,
    ToolArgument,
    ToolDefinition,
    ToolParameter,
    ToolParameterType,
)
from ai_engineering_agent_platform.domain import (
    ProviderExecutionError,
)


def _tools() -> tuple[ToolDefinition, ...]:
    """Return deterministic portable tool definitions."""
    return (
        ToolDefinition(
            name="lookup",
            description="Read synthetic information.",
            parameters=(
                ToolParameter(
                    name="query",
                    parameter_type=ToolParameterType.STRING,
                    required=True,
                    description="Synthetic lookup query.",
                ),
                ToolParameter(
                    name="limit",
                    parameter_type=ToolParameterType.INTEGER,
                    required=False,
                ),
            ),
        ),
        ToolDefinition(
            name="change",
            description="Propose a synthetic change.",
            parameters=(
                ToolParameter(
                    name="value",
                    parameter_type=ToolParameterType.NUMBER,
                    required=True,
                ),
            ),
        ),
    )


def _request(
    *,
    tools: tuple[ToolDefinition, ...] | None = None,
) -> LLMRequest:
    """Return one deterministic LLM request."""
    return LLMRequest(
        model="synthetic-model",
        messages=(
            LLMMessage(
                role=MessageRole.USER,
                content="Use a synthetic tool if useful.",
            ),
        ),
        tools=(_tools() if tools is None else tools),
    )


def _tool_response(
    *,
    tool_name: str = "lookup",
    arguments: object | None = None,
    provider_call_id: object | None = None,
) -> dict[str, object]:
    """Return a deterministic completed Ollama tool-call payload."""
    function: dict[str, object] = {
        "name": tool_name,
        "arguments": (
            {
                "query": "synthetic",
                "limit": 2,
            }
            if arguments is None
            else arguments
        ),
    }

    tool_call: dict[str, object] = {
        "function": function,
    }

    if provider_call_id is not None:
        tool_call["id"] = provider_call_id

    return {
        "model": "synthetic-model",
        "done": True,
        "done_reason": "stop",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                tool_call,
            ],
        },
    }


def test_tool_call_finish_reason_wire_value_is_stable() -> None:
    """Tool proposals have one explicit normalized finish reason."""
    assert FinishReason.TOOL_CALLS.value == "tool_calls"


def test_llm_request_accepts_unique_portable_tool_definitions() -> None:
    """A request can expose an explicit provider-neutral tool set."""
    request = _request()

    assert tuple(tool.name for tool in request.tools) == (
        "lookup",
        "change",
    )


def test_llm_request_rejects_duplicate_tool_names() -> None:
    """A model request cannot expose ambiguous duplicate tool names."""
    duplicate = _tools()[0]

    with pytest.raises(
        ValueError,
        match="LLM request tool names must be unique",
    ):
        _request(
            tools=(
                duplicate,
                duplicate,
            )
        )


def test_llm_tool_call_is_a_non_executable_proposal_contract() -> None:
    """A normalized proposal preserves model-selected tool arguments."""
    proposal = LLMToolCall(
        tool_name="lookup",
        arguments=(
            ToolArgument(
                name="query",
                value="synthetic",
            ),
        ),
        provider_call_id="provider-call-1",
    )

    assert proposal.tool_name == "lookup"
    assert proposal.arguments[0].name == "query"
    assert proposal.provider_call_id == "provider-call-1"


def test_llm_tool_call_rejects_duplicate_argument_names() -> None:
    """Proposal arguments remain unambiguous before execution validation."""
    with pytest.raises(
        ValueError,
        match="tool-call argument names must be unique",
    ):
        LLMToolCall(
            tool_name="lookup",
            arguments=(
                ToolArgument(
                    name="query",
                    value="one",
                ),
                ToolArgument(
                    name="query",
                    value="two",
                ),
            ),
        )


def test_llm_response_requires_tool_calls_finish_reason() -> None:
    """Tool proposals cannot masquerade as a normal STOP response."""
    proposal = LLMToolCall(
        tool_name="lookup",
    )

    with pytest.raises(
        ValueError,
        match=("responses with tool calls must use TOOL_CALLS finish reason"),
    ):
        LLMResponse(
            model="synthetic-model",
            message=LLMMessage(
                role=MessageRole.ASSISTANT,
                content="",
            ),
            finish_reason=FinishReason.STOP,
            tool_calls=(proposal,),
        )


def test_llm_response_rejects_empty_tool_calls_finish_reason() -> None:
    """TOOL_CALLS cannot be emitted without at least one proposal."""
    with pytest.raises(
        ValueError,
        match=("TOOL_CALLS finish reason requires at least one tool call"),
    ):
        LLMResponse(
            model="synthetic-model",
            message=LLMMessage(
                role=MessageRole.ASSISTANT,
                content="",
            ),
            finish_reason=FinishReason.TOOL_CALLS,
        )


def test_llm_response_rejects_duplicate_provider_call_ids() -> None:
    """Provider identifiers must remain unambiguous when present."""
    first = LLMToolCall(
        tool_name="lookup",
        provider_call_id="provider-call-1",
    )

    second = LLMToolCall(
        tool_name="change",
        provider_call_id="provider-call-1",
    )

    with pytest.raises(
        ValueError,
        match="provider tool-call identifiers must be unique",
    ):
        LLMResponse(
            model="synthetic-model",
            message=LLMMessage(
                role=MessageRole.ASSISTANT,
                content="",
            ),
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                first,
                second,
            ),
        )


def test_ollama_payload_maps_tool_definitions_without_policy() -> None:
    """Only portable definitions are exposed to the model."""
    payload = build_ollama_chat_payload(_request())

    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Read synthetic information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Synthetic lookup query.",
                        },
                        "limit": {
                            "type": "integer",
                        },
                    },
                    "required": [
                        "query",
                    ],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "change",
                "description": "Propose a synthetic change.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "number",
                        },
                    },
                    "required": [
                        "value",
                    ],
                },
            },
        },
    ]


def test_ollama_payload_omits_tools_when_request_has_none() -> None:
    """Legacy non-tool generation payloads remain unchanged."""
    payload = build_ollama_chat_payload(
        _request(
            tools=(),
        )
    )

    assert "tools" not in payload


def test_ollama_parser_normalizes_requested_tool_proposal() -> None:
    """A requested model proposal becomes an inert LLMToolCall."""
    response = parse_ollama_chat_response(
        _tool_response(),
        requested_tools=_tools(),
    )

    assert response.finish_reason is (FinishReason.TOOL_CALLS)

    assert response.message.content == ""

    assert len(response.tool_calls) == 1

    proposal = response.tool_calls[0]

    assert proposal.tool_name == "lookup"

    assert tuple(
        (
            argument.name,
            argument.value,
        )
        for argument in proposal.arguments
    ) == (
        (
            "query",
            "synthetic",
        ),
        (
            "limit",
            2,
        ),
    )

    assert proposal.provider_call_id is None


def test_ollama_parser_preserves_optional_provider_call_id() -> None:
    """Provider-originating identifiers remain metadata, not execution IDs."""
    response = parse_ollama_chat_response(
        _tool_response(
            provider_call_id="ollama-call-7",
        ),
        requested_tools=_tools(),
    )

    assert response.tool_calls[0].provider_call_id == ("ollama-call-7")


def test_ollama_parser_still_rejects_calls_without_requested_tools() -> None:
    """Tool proposals remain fail-closed when caller exposed no tools."""
    with pytest.raises(
        ProviderExecutionError,
        match="contains unsupported tool calls",
    ):
        parse_ollama_chat_response(
            _tool_response(),
        )


def test_ollama_parser_rejects_unrequested_tool_name() -> None:
    """A model cannot expand its own tool authority."""
    with pytest.raises(
        ProviderExecutionError,
        match="tool that was not requested: ghost",
    ):
        parse_ollama_chat_response(
            _tool_response(
                tool_name="ghost",
            ),
            requested_tools=_tools(),
        )


@pytest.mark.parametrize(
    "arguments",
    (
        [
            "not",
            "an",
            "object",
        ],
        "not-an-object",
        123,
    ),
)
def test_ollama_parser_rejects_non_object_arguments(
    arguments: object,
) -> None:
    """Provider tool arguments must arrive as one JSON object."""
    with pytest.raises(
        ProviderExecutionError,
        match=r"function\.arguments must be an object",
    ):
        parse_ollama_chat_response(
            _tool_response(
                arguments=arguments,
            ),
            requested_tools=_tools(),
        )


@pytest.mark.parametrize(
    "value",
    (
        {
            "nested": "object",
        },
        [
            "array",
        ],
    ),
)
def test_ollama_parser_rejects_non_scalar_argument_values(
    value: object,
) -> None:
    """Current portable tool contracts intentionally remain scalar-only."""
    with pytest.raises(
        ProviderExecutionError,
        match="must be a portable scalar",
    ):
        parse_ollama_chat_response(
            _tool_response(
                arguments={
                    "query": value,
                },
            ),
            requested_tools=_tools(),
        )


def test_ollama_parser_rejects_invalid_provider_call_id() -> None:
    """Optional provider IDs must be meaningful strings when present."""
    with pytest.raises(
        ProviderExecutionError,
        match="id must be a non-empty string",
    ):
        parse_ollama_chat_response(
            _tool_response(
                provider_call_id=123,
            ),
            requested_tools=_tools(),
        )


def test_tool_calling_contract_is_publicly_exported() -> None:
    """Tool-call proposal semantics should be public provider contracts."""
    import ai_engineering_agent_platform.contracts as contracts

    assert "LLMToolCall" in contracts.__all__
    assert hasattr(
        contracts,
        "LLMToolCall",
    )

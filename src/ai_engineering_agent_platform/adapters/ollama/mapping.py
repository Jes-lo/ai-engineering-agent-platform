"""Translate between platform LLM contracts and Ollama chat payloads."""

from typing import cast

from ai_engineering_agent_platform.adapters.ollama.tool_mapping import (
    build_ollama_tools,
    parse_ollama_tool_calls,
)
from ai_engineering_agent_platform.contracts import (
    FinishReason,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    MessageRole,
    TokenUsage,
    ToolDefinition,
)
from ai_engineering_agent_platform.domain import ProviderExecutionError

type JsonObject = dict[str, object]


def build_ollama_chat_payload(
    request: LLMRequest,
) -> JsonObject:
    """Build a non-streaming Ollama chat request from a platform request."""
    messages: list[JsonObject] = [
        {
            "role": message.role.value,
            "content": message.content,
        }
        for message in request.messages
    ]

    payload: JsonObject = {
        "model": request.model,
        "messages": messages,
        "stream": False,
    }

    if request.tools:
        payload["tools"] = build_ollama_tools(request.tools)

    options: JsonObject = {}

    if request.temperature is not None:
        options["temperature"] = request.temperature

    if request.max_output_tokens is not None:
        options["num_predict"] = request.max_output_tokens

    if options:
        payload["options"] = options

    return payload


def parse_ollama_chat_response(
    payload: object,
    *,
    requested_tools: tuple[ToolDefinition, ...] = (),
) -> LLMResponse:
    """Parse one completed non-streaming Ollama chat response."""
    response = _require_object(
        payload,
        "Ollama response",
    )

    model = _require_non_empty_string(
        response.get("model"),
        "model",
    )

    if response.get("done") is not True:
        raise ProviderExecutionError("Ollama response is not a completed generation")

    message_payload = _require_object(
        response.get("message"),
        "message",
    )

    role = _require_non_empty_string(
        message_payload.get("role"),
        "message.role",
    )

    if role != MessageRole.ASSISTANT.value:
        raise ProviderExecutionError("Ollama response message must use assistant role")

    content = message_payload.get("content")

    if not isinstance(content, str):
        raise ProviderExecutionError("Ollama response message.content must be a string")

    tool_calls = parse_ollama_tool_calls(
        message_payload,
        requested_tools=requested_tools,
    )

    finish_reason = (
        FinishReason.TOOL_CALLS
        if tool_calls
        else _parse_finish_reason(response.get("done_reason"))
    )

    usage = _parse_usage(response)

    return LLMResponse(
        model=model,
        message=LLMMessage(
            role=MessageRole.ASSISTANT,
            content=content,
        ),
        finish_reason=finish_reason,
        usage=usage,
        tool_calls=tool_calls,
    )


def _require_object(
    value: object,
    field_name: str,
) -> JsonObject:
    """Return a JSON object or raise a normalized provider error."""
    if not isinstance(value, dict):
        raise ProviderExecutionError(f"Ollama {field_name} must be a JSON object")

    return cast(JsonObject, value)


def _require_non_empty_string(
    value: object,
    field_name: str,
) -> str:
    """Return a non-empty string field from an Ollama response."""
    if not isinstance(value, str) or not value.strip():
        raise ProviderExecutionError(
            f"Ollama response {field_name} must be a non-empty string"
        )

    return value


def _parse_finish_reason(
    value: object,
) -> FinishReason:
    """Normalize Ollama completion reasons into the platform contract."""
    if not isinstance(value, str):
        return FinishReason.OTHER

    normalized = value.strip().lower()

    mapping = {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
        "content_filter": FinishReason.CONTENT_FILTER,
    }

    return mapping.get(
        normalized,
        FinishReason.OTHER,
    )


def _parse_usage(
    response: JsonObject,
) -> TokenUsage | None:
    """Normalize optional Ollama token accounting."""
    input_tokens = response.get("prompt_eval_count")
    output_tokens = response.get("eval_count")

    if input_tokens is None and output_tokens is None:
        return None

    if input_tokens is None or output_tokens is None:
        raise ProviderExecutionError(
            "Ollama response token usage must include both "
            "prompt_eval_count and eval_count"
        )

    validated_input = _require_non_negative_integer(
        input_tokens,
        "prompt_eval_count",
    )

    validated_output = _require_non_negative_integer(
        output_tokens,
        "eval_count",
    )

    return TokenUsage(
        input_tokens=validated_input,
        output_tokens=validated_output,
    )


def _require_non_negative_integer(
    value: object,
    field_name: str,
) -> int:
    """Validate one Ollama token-count field."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProviderExecutionError(f"Ollama response {field_name} must be an integer")

    if value < 0:
        raise ProviderExecutionError(
            f"Ollama response {field_name} must be non-negative"
        )

    return value

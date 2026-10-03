"""Map controlled platform tool contracts to and from Ollama tool payloads."""

from math import isfinite

from ai_engineering_agent_platform.contracts import (
    LLMToolCall,
    ToolArgument,
    ToolDefinition,
)
from ai_engineering_agent_platform.contracts.tool import ToolArgumentValue
from ai_engineering_agent_platform.domain import ProviderExecutionError

type JsonObject = dict[str, object]


def build_ollama_tools(
    tools: tuple[ToolDefinition, ...],
) -> list[JsonObject]:
    """Build deterministic Ollama function-tool definitions."""
    result: list[JsonObject] = []

    for tool in tools:
        properties: JsonObject = {}
        required: list[str] = []

        for parameter in tool.parameters:
            property_schema: JsonObject = {
                "type": parameter.parameter_type.value,
            }

            if parameter.description is not None:
                property_schema["description"] = parameter.description

            properties[parameter.name] = property_schema

            if parameter.required:
                required.append(parameter.name)

        parameters: JsonObject = {
            "type": "object",
            "properties": properties,
        }

        if required:
            parameters["required"] = required

        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters,
                },
            }
        )

    return result


def parse_ollama_tool_calls(
    message: JsonObject,
    *,
    requested_tools: tuple[ToolDefinition, ...],
) -> tuple[LLMToolCall, ...]:
    """Parse untrusted Ollama tool-call proposals without executing them."""
    raw_tool_calls = message.get("tool_calls")

    if raw_tool_calls is None:
        return ()

    if not isinstance(raw_tool_calls, list):
        raise ProviderExecutionError(
            "Ollama response message.tool_calls must be a list"
        )

    if not raw_tool_calls:
        return ()

    if not requested_tools:
        raise ProviderExecutionError("Ollama response contains unsupported tool calls")

    requested_tool_names = tuple(tool.name for tool in requested_tools)

    if len(requested_tool_names) != len(set(requested_tool_names)):
        raise ProviderExecutionError("requested Ollama tool names must be unique")

    requested_tool_name_set = set(requested_tool_names)

    parsed: list[LLMToolCall] = []
    provider_call_ids: list[str] = []

    for index, raw_tool_call in enumerate(
        raw_tool_calls,
        start=1,
    ):
        tool_call = _require_object(
            raw_tool_call,
            f"message.tool_calls[{index - 1}]",
        )

        function = _require_object(
            tool_call.get("function"),
            f"message.tool_calls[{index - 1}].function",
        )

        tool_name = _require_non_empty_string(
            function.get("name"),
            f"message.tool_calls[{index - 1}].function.name",
        )

        if tool_name not in requested_tool_name_set:
            raise ProviderExecutionError(
                "Ollama response proposed a tool that was not requested: " + tool_name
            )

        arguments_payload = _require_object(
            function.get("arguments"),
            f"message.tool_calls[{index - 1}].function.arguments",
        )

        arguments: list[ToolArgument] = []

        for argument_name, raw_value in arguments_payload.items():
            if not isinstance(argument_name, str) or not argument_name.strip():
                raise ProviderExecutionError(
                    "Ollama tool argument names must be non-empty strings"
                )

            value = _parse_scalar_argument_value(
                raw_value,
                tool_name=tool_name,
                argument_name=argument_name,
            )

            try:
                arguments.append(
                    ToolArgument(
                        name=argument_name,
                        value=value,
                    )
                )
            except ValueError as exc:
                raise ProviderExecutionError(
                    "Ollama tool argument is invalid: " + argument_name
                ) from exc

        provider_call_id_value = tool_call.get("id")
        provider_call_id: str | None = None

        if provider_call_id_value is not None:
            provider_call_id = _require_non_empty_string(
                provider_call_id_value,
                f"message.tool_calls[{index - 1}].id",
            )

            provider_call_ids.append(provider_call_id)

        try:
            parsed.append(
                LLMToolCall(
                    tool_name=tool_name,
                    arguments=tuple(arguments),
                    provider_call_id=provider_call_id,
                )
            )
        except ValueError as exc:
            raise ProviderExecutionError(
                f"Ollama tool call {index} is invalid"
            ) from exc

    if len(provider_call_ids) != len(set(provider_call_ids)):
        raise ProviderExecutionError(
            "Ollama provider tool-call identifiers must be unique"
        )

    return tuple(parsed)


def _parse_scalar_argument_value(
    value: object,
    *,
    tool_name: str,
    argument_name: str,
) -> ToolArgumentValue:
    """Accept only portable scalar values understood by platform tools."""
    if value is None:
        return None

    if isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not isfinite(value):
            raise ProviderExecutionError(
                "Ollama tool argument float must be finite: "
                + tool_name
                + "."
                + argument_name
            )

        return value

    raise ProviderExecutionError(
        "Ollama tool argument must be a portable scalar: "
        + tool_name
        + "."
        + argument_name
    )


def _require_object(
    value: object,
    field_name: str,
) -> JsonObject:
    """Require one JSON object."""
    if not isinstance(value, dict):
        raise ProviderExecutionError(f"Ollama response {field_name} must be an object")

    if any(not isinstance(key, str) for key in value):
        raise ProviderExecutionError(
            f"Ollama response {field_name} keys must be strings"
        )

    return value


def _require_non_empty_string(
    value: object,
    field_name: str,
) -> str:
    """Require one non-empty JSON string."""
    if not isinstance(value, str) or not value.strip():
        raise ProviderExecutionError(
            f"Ollama response {field_name} must be a non-empty string"
        )

    return value

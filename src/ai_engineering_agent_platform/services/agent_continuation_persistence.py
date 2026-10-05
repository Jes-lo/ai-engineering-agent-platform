"""Canonical persistence codec for approval-required agent continuations."""

import json
import math

from ai_engineering_agent_platform.contracts.llm import (
    FinishReason,
    LLMAssistantToolCallMessage,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMToolCall,
    LLMToolResultMessage,
    MessageRole,
    TokenUsage,
)
from ai_engineering_agent_platform.contracts.tool import (
    ToolArgument,
    ToolExecutionStatus,
    ToolInvocation,
    ToolResult,
)
from ai_engineering_agent_platform.services.agent import (
    AgentPlannedToolCall,
    AgentTurnResult,
    AgentTurnStatus,
)
from ai_engineering_agent_platform.services.agent_continuation import (
    AgentContinuationIntegrityError,
    AgentContinuationKind,
    validate_agent_continuation_identity,
)
from ai_engineering_agent_platform.services.agent_loop import (
    AgentLoopRequest,
    AgentLoopResult,
    AgentLoopStatus,
)

AGENT_CONTINUATION_FORMAT_VERSION = 1
MAX_AGENT_CONTINUATION_PAYLOAD_BYTES = 1_048_576
MAX_AGENT_CONTINUATION_STRING_LENGTH = 262_144
MAX_AGENT_CONTINUATION_COLLECTION_ITEMS = 512
MAX_AGENT_CONTINUATION_INTEGER_DIGITS = 128

type AgentContinuationSnapshot = AgentTurnResult | AgentLoopResult


class AgentContinuationSerializationError(AgentContinuationIntegrityError):
    """Raised when continuation JSON is invalid or non-canonical."""


def _reject_json_constant(
    value: str,
) -> object:
    raise AgentContinuationSerializationError(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _exact_object(
    raw: object,
    *,
    keys: set[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(
        raw,
        dict,
    ):
        raise AgentContinuationSerializationError(f"{label} must be an object")

    if set(raw) != keys:
        raise AgentContinuationSerializationError(
            f"{label} contains unexpected or missing keys"
        )

    if any(
        not isinstance(
            key,
            str,
        )
        for key in raw
    ):
        raise AgentContinuationSerializationError(f"{label} keys must be strings")

    return raw


def _string(
    raw: object,
    *,
    label: str,
) -> str:
    if not isinstance(
        raw,
        str,
    ):
        raise AgentContinuationSerializationError(f"{label} must be a string")

    if len(raw) > MAX_AGENT_CONTINUATION_STRING_LENGTH:
        raise AgentContinuationSerializationError(f"{label} exceeds maximum length")

    return raw


def _optional_string(
    raw: object,
    *,
    label: str,
) -> str | None:
    if raw is None:
        return None

    return _string(
        raw,
        label=label,
    )


def _integer(
    raw: object,
    *,
    label: str,
) -> int:
    if not isinstance(
        raw,
        int,
    ) or isinstance(
        raw,
        bool,
    ):
        raise AgentContinuationSerializationError(f"{label} must be an integer")

    return raw


def _optional_float(
    raw: object,
    *,
    label: str,
) -> float | None:
    if raw is None:
        return None

    if not isinstance(
        raw,
        (int, float),
    ) or isinstance(
        raw,
        bool,
    ):
        raise AgentContinuationSerializationError(f"{label} must be numeric or null")

    value = float(raw)

    if not math.isfinite(value):
        raise AgentContinuationSerializationError(f"{label} must be finite")

    return value


def _list(
    raw: object,
    *,
    label: str,
) -> list[object]:
    if not isinstance(
        raw,
        list,
    ):
        raise AgentContinuationSerializationError(f"{label} must be a list")

    if len(raw) > MAX_AGENT_CONTINUATION_COLLECTION_ITEMS:
        raise AgentContinuationSerializationError(f"{label} exceeds maximum items")

    return raw


def _encode_scalar(
    value: str | int | float | bool | None,
) -> dict[str, object]:
    if value is None:
        return {
            "type": "null",
            "value": None,
        }

    if type(value) is bool:
        return {
            "type": "bool",
            "value": value,
        }

    if type(value) is int:
        if len(str(abs(value))) > MAX_AGENT_CONTINUATION_INTEGER_DIGITS:
            raise AgentContinuationSerializationError(
                "tool argument integer exceeds maximum digits"
            )

        return {
            "type": "int",
            "value": str(value),
        }

    if type(value) is float:
        if not math.isfinite(value):
            raise AgentContinuationSerializationError(
                "tool argument float must be finite"
            )

        return {
            "type": "float",
            "value": repr(value),
        }

    if isinstance(
        value,
        str,
    ):
        _string(
            value,
            label="tool argument string",
        )

        return {
            "type": "str",
            "value": value,
        }

    raise AgentContinuationSerializationError("unsupported tool argument scalar")


def _decode_scalar(
    raw: object,
) -> str | int | float | bool | None:
    data = _exact_object(
        raw,
        keys={
            "type",
            "value",
        },
        label="tool argument value",
    )

    kind = _string(
        data["type"],
        label="tool argument value type",
    )

    value = data["value"]

    if kind == "null":
        if value is not None:
            raise AgentContinuationSerializationError("null scalar payload is invalid")

        return None

    if kind == "bool":
        if type(value) is not bool:
            raise AgentContinuationSerializationError("bool scalar payload is invalid")

        return value

    if kind == "int":
        encoded = _string(
            value,
            label="integer scalar",
        )

        digits = encoded[1:] if encoded.startswith("-") else encoded

        if (
            not digits
            or not digits.isdigit()
            or len(digits) > MAX_AGENT_CONTINUATION_INTEGER_DIGITS
            or (len(digits) > 1 and digits.startswith("0"))
            or encoded == "-0"
        ):
            raise AgentContinuationSerializationError(
                "integer scalar is invalid or non-canonical"
            )

        return int(encoded)

    if kind == "float":
        encoded = _string(
            value,
            label="float scalar",
        )

        try:
            parsed = float(encoded)
        except ValueError as exc:
            raise AgentContinuationSerializationError(
                "float scalar is invalid"
            ) from exc

        if not math.isfinite(parsed) or repr(parsed) != encoded:
            raise AgentContinuationSerializationError(
                "float scalar is invalid or non-canonical"
            )

        return parsed

    if kind == "str":
        return _string(
            value,
            label="string scalar",
        )

    raise AgentContinuationSerializationError(
        "tool argument scalar type is unsupported"
    )


def _encode_argument(
    argument: ToolArgument,
) -> dict[str, object]:
    return {
        "name": argument.name,
        "value": _encode_scalar(argument.value),
    }


def _decode_argument(
    raw: object,
) -> ToolArgument:
    data = _exact_object(
        raw,
        keys={
            "name",
            "value",
        },
        label="tool argument",
    )

    return ToolArgument(
        name=_string(
            data["name"],
            label="tool argument name",
        ),
        value=_decode_scalar(data["value"]),
    )


def _encode_tool_call(
    call: LLMToolCall,
) -> dict[str, object]:
    return {
        "tool_name": call.tool_name,
        "arguments": [_encode_argument(argument) for argument in call.arguments],
        "provider_call_id": call.provider_call_id,
    }


def _decode_tool_call(
    raw: object,
) -> LLMToolCall:
    data = _exact_object(
        raw,
        keys={
            "tool_name",
            "arguments",
            "provider_call_id",
        },
        label="LLM tool call",
    )

    return LLMToolCall(
        tool_name=_string(
            data["tool_name"],
            label="LLM tool name",
        ),
        arguments=tuple(
            _decode_argument(item)
            for item in _list(
                data["arguments"],
                label="LLM tool arguments",
            )
        ),
        provider_call_id=_optional_string(
            data["provider_call_id"],
            label="provider_call_id",
        ),
    )


def _encode_invocation(
    invocation: ToolInvocation,
) -> dict[str, object]:
    return {
        "call_id": invocation.call_id,
        "tool_name": invocation.tool_name,
        "arguments": [_encode_argument(argument) for argument in invocation.arguments],
    }


def _decode_invocation(
    raw: object,
) -> ToolInvocation:
    data = _exact_object(
        raw,
        keys={
            "call_id",
            "tool_name",
            "arguments",
        },
        label="tool invocation",
    )

    return ToolInvocation(
        call_id=_string(
            data["call_id"],
            label="tool call_id",
        ),
        tool_name=_string(
            data["tool_name"],
            label="tool invocation name",
        ),
        arguments=tuple(
            _decode_argument(item)
            for item in _list(
                data["arguments"],
                label="tool invocation arguments",
            )
        ),
    )


def _encode_tool_result(
    result: ToolResult,
) -> dict[str, object]:
    return {
        "call_id": result.call_id,
        "tool_name": result.tool_name,
        "status": result.status.value,
        "content": result.content,
    }


def _decode_tool_result(
    raw: object,
) -> ToolResult:
    data = _exact_object(
        raw,
        keys={
            "call_id",
            "tool_name",
            "status",
            "content",
        },
        label="tool result",
    )

    try:
        status = ToolExecutionStatus(
            _string(
                data["status"],
                label="tool result status",
            )
        )
    except ValueError as exc:
        raise AgentContinuationSerializationError(
            "tool result status is invalid"
        ) from exc

    return ToolResult(
        call_id=_string(
            data["call_id"],
            label="tool result call_id",
        ),
        tool_name=_string(
            data["tool_name"],
            label="tool result name",
        ),
        status=status,
        content=_string(
            data["content"],
            label="tool result content",
        ),
    )


def _encode_message(
    message: (LLMMessage | LLMAssistantToolCallMessage | LLMToolResultMessage),
) -> dict[str, object]:
    if isinstance(
        message,
        LLMAssistantToolCallMessage,
    ):
        return {
            "type": "assistant_tool_calls",
            "content": message.content,
            "tool_calls": [_encode_tool_call(call) for call in message.tool_calls],
        }

    if isinstance(
        message,
        LLMToolResultMessage,
    ):
        return {
            "type": "tool_result",
            "result": _encode_tool_result(message.result),
            "provider_call_id": message.provider_call_id,
        }

    if isinstance(
        message,
        LLMMessage,
    ):
        return {
            "type": "message",
            "role": message.role.value,
            "content": message.content,
        }

    raise AgentContinuationSerializationError("unsupported conversation message")


def _decode_message(
    raw: object,
) -> LLMMessage | LLMAssistantToolCallMessage | LLMToolResultMessage:
    if not isinstance(
        raw,
        dict,
    ):
        raise AgentContinuationSerializationError(
            "conversation message must be an object"
        )

    message_type = _string(
        raw.get("type"),
        label="conversation message type",
    )

    if message_type == "message":
        data = _exact_object(
            raw,
            keys={
                "type",
                "role",
                "content",
            },
            label="plain conversation message",
        )

        try:
            role = MessageRole(
                _string(
                    data["role"],
                    label="message role",
                )
            )
        except ValueError as exc:
            raise AgentContinuationSerializationError(
                "message role is invalid"
            ) from exc

        return LLMMessage(
            role=role,
            content=_string(
                data["content"],
                label="message content",
            ),
        )

    if message_type == "assistant_tool_calls":
        data = _exact_object(
            raw,
            keys={
                "type",
                "content",
                "tool_calls",
            },
            label="assistant tool-call message",
        )

        return LLMAssistantToolCallMessage(
            content=_string(
                data["content"],
                label="assistant tool-call content",
            ),
            tool_calls=tuple(
                _decode_tool_call(item)
                for item in _list(
                    data["tool_calls"],
                    label="assistant tool calls",
                )
            ),
        )

    if message_type == "tool_result":
        data = _exact_object(
            raw,
            keys={
                "type",
                "result",
                "provider_call_id",
            },
            label="tool-result message",
        )

        return LLMToolResultMessage(
            result=_decode_tool_result(data["result"]),
            provider_call_id=_optional_string(
                data["provider_call_id"],
                label="tool-result provider_call_id",
            ),
        )

    raise AgentContinuationSerializationError(
        "conversation message type is unsupported"
    )


def _encode_usage(
    usage: TokenUsage | None,
) -> dict[str, int] | None:
    if usage is None:
        return None

    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }


def _decode_usage(
    raw: object,
) -> TokenUsage | None:
    if raw is None:
        return None

    data = _exact_object(
        raw,
        keys={
            "input_tokens",
            "output_tokens",
        },
        label="token usage",
    )

    return TokenUsage(
        input_tokens=_integer(
            data["input_tokens"],
            label="input_tokens",
        ),
        output_tokens=_integer(
            data["output_tokens"],
            label="output_tokens",
        ),
    )


def _encode_response(
    response: LLMResponse,
) -> dict[str, object]:
    return {
        "model": response.model,
        "message": _encode_message(response.message),
        "finish_reason": response.finish_reason.value,
        "usage": _encode_usage(response.usage),
        "tool_calls": [_encode_tool_call(call) for call in response.tool_calls],
    }


def _decode_response(
    raw: object,
) -> LLMResponse:
    data = _exact_object(
        raw,
        keys={
            "model",
            "message",
            "finish_reason",
            "usage",
            "tool_calls",
        },
        label="LLM response",
    )

    message = _decode_message(data["message"])

    if not isinstance(
        message,
        LLMMessage,
    ):
        raise AgentContinuationSerializationError(
            "LLM response message must be a plain assistant message"
        )

    try:
        finish_reason = FinishReason(
            _string(
                data["finish_reason"],
                label="finish_reason",
            )
        )
    except ValueError as exc:
        raise AgentContinuationSerializationError("finish_reason is invalid") from exc

    return LLMResponse(
        model=_string(
            data["model"],
            label="response model",
        ),
        message=message,
        finish_reason=finish_reason,
        usage=_decode_usage(data["usage"]),
        tool_calls=tuple(
            _decode_tool_call(item)
            for item in _list(
                data["tool_calls"],
                label="response tool_calls",
            )
        ),
    )


def _encode_request(
    request: LLMRequest,
) -> dict[str, object]:
    if request.tools:
        raise AgentContinuationSerializationError(
            "durable loop request must not contain caller-supplied tools"
        )

    return {
        "model": request.model,
        "messages": [_encode_message(message) for message in request.messages],
        "temperature": request.temperature,
        "max_output_tokens": request.max_output_tokens,
    }


def _decode_request(
    raw: object,
) -> LLMRequest:
    data = _exact_object(
        raw,
        keys={
            "model",
            "messages",
            "temperature",
            "max_output_tokens",
        },
        label="LLM request",
    )

    max_output_tokens_raw = data["max_output_tokens"]

    max_output_tokens = (
        None
        if max_output_tokens_raw is None
        else _integer(
            max_output_tokens_raw,
            label="max_output_tokens",
        )
    )

    return LLMRequest(
        model=_string(
            data["model"],
            label="request model",
        ),
        messages=tuple(
            _decode_message(item)
            for item in _list(
                data["messages"],
                label="request messages",
            )
        ),
        temperature=_optional_float(
            data["temperature"],
            label="temperature",
        ),
        max_output_tokens=max_output_tokens,
    )


def _encode_planned_step(
    step: AgentPlannedToolCall,
) -> dict[str, object]:
    return {
        "step_number": step.step_number,
        "proposal": _encode_tool_call(step.proposal),
        "invocation": _encode_invocation(step.invocation),
    }


def _decode_planned_step(
    raw: object,
) -> AgentPlannedToolCall:
    data = _exact_object(
        raw,
        keys={
            "step_number",
            "proposal",
            "invocation",
        },
        label="planned tool call",
    )

    return AgentPlannedToolCall(
        step_number=_integer(
            data["step_number"],
            label="step_number",
        ),
        proposal=_decode_tool_call(data["proposal"]),
        invocation=_decode_invocation(data["invocation"]),
    )


def _validate_turn_snapshot(
    turn: AgentTurnResult,
    identity: tuple[str, ...],
) -> None:
    if turn.status is not AgentTurnStatus.APPROVAL_REQUIRED:
        raise AgentContinuationSerializationError(
            "only approval-required turn snapshots are durable"
        )

    if turn.executions:
        raise AgentContinuationSerializationError(
            "approval-required durable turn must not contain executions"
        )

    expected_identity = tuple(step.invocation.call_id for step in turn.planned_steps)

    if expected_identity != identity:
        raise AgentContinuationSerializationError(
            "turn continuation identity does not match planned calls"
        )


def _encode_turn(
    turn: AgentTurnResult,
) -> dict[str, object]:
    return {
        "run_id": turn.run_id,
        "status": turn.status.value,
        "response": _encode_response(turn.response),
        "planned_steps": [_encode_planned_step(step) for step in turn.planned_steps],
        "executions": [],
        "pending_approval_call_ids": list(turn.pending_approval_call_ids),
    }


def _decode_turn(
    raw: object,
) -> AgentTurnResult:
    data = _exact_object(
        raw,
        keys={
            "run_id",
            "status",
            "response",
            "planned_steps",
            "executions",
            "pending_approval_call_ids",
        },
        label="agent turn snapshot",
    )

    executions = _list(
        data["executions"],
        label="turn executions",
    )

    if executions:
        raise AgentContinuationSerializationError(
            "approval-required durable turn cannot contain executions"
        )

    try:
        status = AgentTurnStatus(
            _string(
                data["status"],
                label="turn status",
            )
        )
    except ValueError as exc:
        raise AgentContinuationSerializationError("turn status is invalid") from exc

    return AgentTurnResult(
        run_id=_string(
            data["run_id"],
            label="turn run_id",
        ),
        status=status,
        response=_decode_response(data["response"]),
        planned_steps=tuple(
            _decode_planned_step(item)
            for item in _list(
                data["planned_steps"],
                label="planned steps",
            )
        ),
        executions=(),
        pending_approval_call_ids=tuple(
            _string(
                item,
                label="pending approval call_id",
            )
            for item in _list(
                data["pending_approval_call_ids"],
                label="pending approval call IDs",
            )
        ),
    )


def _encode_loop_request(
    request: AgentLoopRequest,
) -> dict[str, object]:
    return {
        "run_id": request.run_id,
        "llm_request": _encode_request(request.llm_request),
        "exposed_tool_names": list(request.exposed_tool_names),
        "max_model_turns": request.max_model_turns,
        "max_tool_calls": request.max_tool_calls,
    }


def _decode_loop_request(
    raw: object,
) -> AgentLoopRequest:
    data = _exact_object(
        raw,
        keys={
            "run_id",
            "llm_request",
            "exposed_tool_names",
            "max_model_turns",
            "max_tool_calls",
        },
        label="agent loop request",
    )

    return AgentLoopRequest(
        run_id=_string(
            data["run_id"],
            label="loop run_id",
        ),
        llm_request=_decode_request(data["llm_request"]),
        exposed_tool_names=tuple(
            _string(
                item,
                label="exposed tool name",
            )
            for item in _list(
                data["exposed_tool_names"],
                label="exposed tool names",
            )
        ),
        max_model_turns=_integer(
            data["max_model_turns"],
            label="max_model_turns",
        ),
        max_tool_calls=_integer(
            data["max_tool_calls"],
            label="max_tool_calls",
        ),
    )


def _validate_loop_snapshot(
    loop: AgentLoopResult,
    identity: tuple[str, ...],
) -> None:
    if loop.status is not AgentLoopStatus.APPROVAL_REQUIRED:
        raise AgentContinuationSerializationError(
            "only approval-required loop snapshots are durable"
        )

    if loop.final_response is not None:
        raise AgentContinuationSerializationError(
            "approval-required durable loop cannot have final_response"
        )

    if loop.pending_turn is None:
        raise AgentContinuationSerializationError(
            "approval-required durable loop requires pending_turn"
        )

    expected_identity = tuple(
        step.invocation.call_id for step in loop.pending_turn.planned_steps
    )

    if expected_identity != identity:
        raise AgentContinuationSerializationError(
            "loop continuation identity does not match planned calls"
        )


def _encode_loop(
    loop: AgentLoopResult,
) -> dict[str, object]:
    if loop.pending_turn is None:
        raise AgentContinuationSerializationError(
            "approval-required durable loop requires pending_turn"
        )

    return {
        "request": _encode_loop_request(loop.request),
        "status": loop.status.value,
        "messages": [_encode_message(message) for message in loop.messages],
        "model_turns_used": loop.model_turns_used,
        "tool_calls_used": loop.tool_calls_used,
        "final_response": None,
        "pending_turn": _encode_turn(loop.pending_turn),
    }


def _decode_loop(
    raw: object,
) -> AgentLoopResult:
    data = _exact_object(
        raw,
        keys={
            "request",
            "status",
            "messages",
            "model_turns_used",
            "tool_calls_used",
            "final_response",
            "pending_turn",
        },
        label="agent loop snapshot",
    )

    if data["final_response"] is not None:
        raise AgentContinuationSerializationError(
            "approval-required durable loop final_response must be null"
        )

    try:
        status = AgentLoopStatus(
            _string(
                data["status"],
                label="loop status",
            )
        )
    except ValueError as exc:
        raise AgentContinuationSerializationError("loop status is invalid") from exc

    pending_turn = _decode_turn(data["pending_turn"])

    return AgentLoopResult(
        request=_decode_loop_request(data["request"]),
        status=status,
        messages=tuple(
            _decode_message(item)
            for item in _list(
                data["messages"],
                label="loop messages",
            )
        ),
        model_turns_used=_integer(
            data["model_turns_used"],
            label="model_turns_used",
        ),
        tool_calls_used=_integer(
            data["tool_calls_used"],
            label="tool_calls_used",
        ),
        final_response=None,
        pending_turn=pending_turn,
    )


class AgentContinuationCodec:
    """Versioned canonical JSON codec for active approval continuations."""

    def dumps(
        self,
        *,
        kind: AgentContinuationKind,
        identity: tuple[str, ...],
        snapshot: AgentContinuationSnapshot,
    ) -> str:
        try:
            validated_identity = validate_agent_continuation_identity(identity)
        except ValueError as exc:
            raise AgentContinuationSerializationError(
                "continuation identity is invalid"
            ) from exc

        if kind is AgentContinuationKind.TURN:
            if not isinstance(
                snapshot,
                AgentTurnResult,
            ):
                raise AgentContinuationSerializationError(
                    "turn continuation requires AgentTurnResult"
                )

            _validate_turn_snapshot(
                snapshot,
                validated_identity,
            )

            encoded_snapshot = _encode_turn(snapshot)

        elif kind is AgentContinuationKind.LOOP:
            if not isinstance(
                snapshot,
                AgentLoopResult,
            ):
                raise AgentContinuationSerializationError(
                    "loop continuation requires AgentLoopResult"
                )

            _validate_loop_snapshot(
                snapshot,
                validated_identity,
            )

            encoded_snapshot = _encode_loop(snapshot)

        else:
            raise AgentContinuationSerializationError("unsupported continuation kind")

        payload = json.dumps(
            {
                "format": AGENT_CONTINUATION_FORMAT_VERSION,
                "kind": kind.value,
                "identity": list(validated_identity),
                "snapshot": encoded_snapshot,
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )

        if len(payload.encode("utf-8")) > MAX_AGENT_CONTINUATION_PAYLOAD_BYTES:
            raise AgentContinuationSerializationError(
                "continuation payload exceeds maximum bytes"
            )

        return payload

    def loads(
        self,
        payload: str,
    ) -> tuple[
        AgentContinuationKind,
        tuple[str, ...],
        AgentContinuationSnapshot,
    ]:
        if not isinstance(
            payload,
            str,
        ):
            raise AgentContinuationSerializationError(
                "continuation payload must be a string"
            )

        if len(payload.encode("utf-8")) > MAX_AGENT_CONTINUATION_PAYLOAD_BYTES:
            raise AgentContinuationSerializationError(
                "continuation payload exceeds maximum bytes"
            )

        try:
            raw = json.loads(
                payload,
                parse_constant=_reject_json_constant,
            )
        except (
            json.JSONDecodeError,
            UnicodeError,
        ) as exc:
            raise AgentContinuationSerializationError(
                "continuation payload is not valid JSON"
            ) from exc

        data = _exact_object(
            raw,
            keys={
                "format",
                "kind",
                "identity",
                "snapshot",
            },
            label="continuation envelope",
        )

        if (
            _integer(
                data["format"],
                label="continuation format",
            )
            != AGENT_CONTINUATION_FORMAT_VERSION
        ):
            raise AgentContinuationSerializationError("unsupported continuation format")

        try:
            kind = AgentContinuationKind(
                _string(
                    data["kind"],
                    label="continuation kind",
                )
            )
        except ValueError as exc:
            raise AgentContinuationSerializationError(
                "continuation kind is invalid"
            ) from exc

        try:
            identity = validate_agent_continuation_identity(
                tuple(
                    _string(
                        item,
                        label="continuation identity value",
                    )
                    for item in _list(
                        data["identity"],
                        label="continuation identity",
                    )
                )
            )
        except ValueError as exc:
            raise AgentContinuationSerializationError(
                "continuation identity is invalid"
            ) from exc

        snapshot: AgentContinuationSnapshot

        try:
            if kind is AgentContinuationKind.TURN:
                turn_snapshot = _decode_turn(data["snapshot"])

                _validate_turn_snapshot(
                    turn_snapshot,
                    identity,
                )

                snapshot = turn_snapshot

            else:
                loop_snapshot = _decode_loop(data["snapshot"])

                _validate_loop_snapshot(
                    loop_snapshot,
                    identity,
                )

                snapshot = loop_snapshot

        except AgentContinuationSerializationError:
            raise
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise AgentContinuationSerializationError(
                "continuation domain state is invalid"
            ) from exc

        canonical = self.dumps(
            kind=kind,
            identity=identity,
            snapshot=snapshot,
        )

        if canonical != payload:
            raise AgentContinuationSerializationError(
                "continuation payload is not canonical"
            )

        return (
            kind,
            identity,
            snapshot,
        )

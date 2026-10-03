"""Tests for Ollama chat request and response mapping."""

import pytest

from ai_engineering_agent_platform.adapters.ollama.mapping import (
    build_ollama_chat_payload,
    parse_ollama_chat_response,
)
from ai_engineering_agent_platform.contracts import (
    FinishReason,
    LLMMessage,
    LLMRequest,
    MessageRole,
)
from ai_engineering_agent_platform.domain import ProviderExecutionError


def _request(
    *,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> LLMRequest:
    """Return a deterministic platform LLM request."""
    return LLMRequest(
        model="gemma4:example",
        messages=(
            LLMMessage(
                role=MessageRole.SYSTEM,
                content="Answer concisely.",
            ),
            LLMMessage(
                role=MessageRole.USER,
                content="Hello.",
            ),
        ),
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def _response(
    **overrides: object,
) -> dict[str, object]:
    """Return a deterministic completed Ollama response."""
    response: dict[str, object] = {
        "model": "gemma4:example",
        "message": {
            "role": "assistant",
            "content": "Hello.",
        },
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 11,
        "eval_count": 4,
    }

    response.update(overrides)

    return response


def test_build_ollama_chat_payload_minimal() -> None:
    """Mapping should preserve model/messages and disable streaming."""
    payload = build_ollama_chat_payload(_request())

    assert payload == {
        "model": "gemma4:example",
        "messages": [
            {
                "role": "system",
                "content": "Answer concisely.",
            },
            {
                "role": "user",
                "content": "Hello.",
            },
        ],
        "stream": False,
    }


def test_build_ollama_chat_payload_maps_generation_controls() -> None:
    """Portable controls should map to Ollama runtime options."""
    payload = build_ollama_chat_payload(
        _request(
            temperature=0.25,
            max_output_tokens=256,
        )
    )

    assert payload["options"] == {
        "temperature": 0.25,
        "num_predict": 256,
    }


def test_parse_ollama_chat_response() -> None:
    """A completed Ollama response should normalize to LLMResponse."""
    result = parse_ollama_chat_response(_response())

    assert result.model == "gemma4:example"
    assert result.message.role is MessageRole.ASSISTANT
    assert result.message.content == "Hello."
    assert result.finish_reason is FinishReason.STOP

    assert result.usage is not None
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 4
    assert result.usage.total_tokens == 15


@pytest.mark.parametrize(
    ("ollama_reason", "expected"),
    [
        ("stop", FinishReason.STOP),
        ("length", FinishReason.LENGTH),
        ("content_filter", FinishReason.CONTENT_FILTER),
        ("future_reason", FinishReason.OTHER),
    ],
)
def test_parse_ollama_finish_reasons(
    ollama_reason: str,
    expected: FinishReason,
) -> None:
    """Known reasons should normalize and unknown reasons stay portable."""
    result = parse_ollama_chat_response(
        _response(
            done_reason=ollama_reason,
        )
    )

    assert result.finish_reason is expected


@pytest.mark.parametrize(
    "reason",
    [
        None,
        123,
        "",
    ],
)
def test_missing_or_nonstandard_finish_reason_maps_to_other(
    reason: object,
) -> None:
    """Missing or unusable finish reasons should remain forward compatible."""
    result = parse_ollama_chat_response(
        _response(
            done_reason=reason,
        )
    )

    assert result.finish_reason is FinishReason.OTHER


def test_parse_ollama_response_allows_missing_usage() -> None:
    """Usage remains optional when both Ollama token counters are absent."""
    payload = _response()

    del payload["prompt_eval_count"]
    del payload["eval_count"]

    result = parse_ollama_chat_response(payload)

    assert result.usage is None


@pytest.mark.parametrize(
    "done",
    [
        False,
        None,
    ],
)
def test_parse_ollama_response_requires_completed_generation(
    done: object,
) -> None:
    """Non-streaming mapping must reject incomplete generation state."""
    with pytest.raises(
        ProviderExecutionError,
        match="not a completed generation",
    ):
        parse_ollama_chat_response(_response(done=done))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "model",
            "",
            "model must be a non-empty string",
        ),
        (
            "model",
            123,
            "model must be a non-empty string",
        ),
        (
            "message",
            "invalid",
            "message must be a JSON object",
        ),
        (
            "message",
            {
                "role": "user",
                "content": "invalid",
            },
            "must use assistant role",
        ),
        (
            "message",
            {
                "role": "assistant",
                "content": 123,
            },
            "message.content must be a string",
        ),
    ],
)
def test_parse_ollama_response_rejects_malformed_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    """Malformed provider responses must fail before reaching the domain."""
    with pytest.raises(
        ProviderExecutionError,
        match=message,
    ):
        parse_ollama_chat_response(
            _response(
                **{field: value},
            )
        )


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "response",
    ],
)
def test_parse_ollama_response_requires_json_object(
    payload: object,
) -> None:
    """Provider responses must have a JSON-object top level."""
    with pytest.raises(
        ProviderExecutionError,
        match="must be a JSON object",
    ):
        parse_ollama_chat_response(payload)


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens", "message"),
    [
        (
            None,
            4,
            "must include both",
        ),
        (
            11,
            None,
            "must include both",
        ),
        (
            -1,
            4,
            "prompt_eval_count must be non-negative",
        ),
        (
            11,
            -1,
            "eval_count must be non-negative",
        ),
        (
            True,
            4,
            "prompt_eval_count must be an integer",
        ),
        (
            11,
            1.5,
            "eval_count must be an integer",
        ),
    ],
)
def test_parse_ollama_response_validates_usage(
    input_tokens: object,
    output_tokens: object,
    message: str,
) -> None:
    """Partial, negative, or non-integer usage must be rejected."""
    with pytest.raises(
        ProviderExecutionError,
        match=message,
    ):
        parse_ollama_chat_response(
            _response(
                prompt_eval_count=input_tokens,
                eval_count=output_tokens,
            )
        )


def test_parse_ollama_response_rejects_tool_calls() -> None:
    """Tool calls cannot be silently discarded before contract support exists."""
    with pytest.raises(
        ProviderExecutionError,
        match="contains unsupported tool calls",
    ):
        parse_ollama_chat_response(
            _response(
                message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "unsafe_future_tool",
                                "arguments": {},
                            }
                        }
                    ],
                },
            )
        )


def test_parse_ollama_response_allows_empty_tool_call_list() -> None:
    """An explicitly empty tool-call array is equivalent to no tool call."""
    result = parse_ollama_chat_response(
        _response(
            message={
                "role": "assistant",
                "content": "Hello.",
                "tool_calls": [],
            },
        )
    )

    assert result.message.content == "Hello."


def test_parse_ollama_response_rejects_invalid_tool_call_shape() -> None:
    """Malformed tool-call metadata should not be silently ignored."""
    with pytest.raises(
        ProviderExecutionError,
        match=r"message\.tool_calls must be a list",
    ):
        parse_ollama_chat_response(
            _response(
                message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": {},
                },
            )
        )

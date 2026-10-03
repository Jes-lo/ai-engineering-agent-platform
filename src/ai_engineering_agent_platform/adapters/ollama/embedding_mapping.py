"""Translate between platform embedding contracts and Ollama payloads."""

from math import isfinite
from typing import cast

from ai_engineering_agent_platform.contracts import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
)
from ai_engineering_agent_platform.domain import ProviderExecutionError

type JsonObject = dict[str, object]


def build_ollama_embedding_payload(
    request: EmbeddingRequest,
) -> JsonObject:
    """Build one Ollama embedding request from a platform request."""
    payload: JsonObject = {
        "model": request.model,
        "input": list(request.texts),
        "truncate": False,
    }

    if request.dimensions is not None:
        payload["dimensions"] = request.dimensions

    return payload


def parse_ollama_embedding_response(
    payload: object,
    request: EmbeddingRequest,
) -> EmbeddingResponse:
    """Normalize one Ollama embedding response against its request."""
    response = _require_object(
        payload,
        "Ollama embedding response",
    )

    model = _require_non_empty_string(
        response.get("model"),
        "model",
    )

    embeddings_payload = response.get("embeddings")

    if not isinstance(embeddings_payload, list):
        raise ProviderExecutionError(
            "Ollama embedding response embeddings must be a list"
        )

    if len(embeddings_payload) != len(request.texts):
        raise ProviderExecutionError(
            "Ollama embedding response count must match request text count"
        )

    embeddings = tuple(
        _parse_embedding_vector(
            vector,
            index=index,
            expected_dimensions=request.dimensions,
        )
        for index, vector in enumerate(embeddings_payload)
    )

    input_tokens = _parse_input_tokens(response.get("prompt_eval_count"))

    try:
        return EmbeddingResponse(
            model=model,
            embeddings=embeddings,
            input_tokens=input_tokens,
        )
    except ValueError as exc:
        raise ProviderExecutionError(
            "Ollama embedding response violated platform invariants"
        ) from exc


def _parse_embedding_vector(
    value: object,
    *,
    index: int,
    expected_dimensions: int | None,
) -> EmbeddingVector:
    """Validate and normalize one provider embedding vector."""
    if not isinstance(value, list):
        raise ProviderExecutionError(
            f"Ollama embedding response vector {index} must be a list"
        )

    if not value:
        raise ProviderExecutionError(
            f"Ollama embedding response vector {index} must not be empty"
        )

    normalized_values: list[float] = []

    for position, component in enumerate(value):
        if isinstance(component, bool) or not isinstance(
            component,
            (int, float),
        ):
            raise ProviderExecutionError(
                "Ollama embedding response vector "
                f"{index} component {position} must be numeric"
            )

        normalized = float(component)

        if not isfinite(normalized):
            raise ProviderExecutionError(
                "Ollama embedding response vector "
                f"{index} component {position} must be finite"
            )

        normalized_values.append(normalized)

    if (
        expected_dimensions is not None
        and len(normalized_values) != expected_dimensions
    ):
        raise ProviderExecutionError(
            "Ollama embedding response dimensions do not match the requested dimensions"
        )

    try:
        return EmbeddingVector(
            values=tuple(normalized_values),
        )
    except ValueError as exc:
        raise ProviderExecutionError(
            f"Ollama embedding response vector {index} is invalid"
        ) from exc


def _parse_input_tokens(
    value: object,
) -> int | None:
    """Normalize optional Ollama embedding token accounting."""
    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(value, int):
        raise ProviderExecutionError(
            "Ollama embedding response prompt_eval_count must be an integer"
        )

    if value < 0:
        raise ProviderExecutionError(
            "Ollama embedding response prompt_eval_count must be non-negative"
        )

    return value


def _require_object(
    value: object,
    field_name: str,
) -> JsonObject:
    """Return a JSON object or raise a normalized provider error."""
    if not isinstance(value, dict):
        raise ProviderExecutionError(f"{field_name} must be a JSON object")

    return cast(JsonObject, value)


def _require_non_empty_string(
    value: object,
    field_name: str,
) -> str:
    """Return a required non-empty string response field."""
    if not isinstance(value, str) or not value.strip():
        raise ProviderExecutionError(
            f"Ollama embedding response {field_name} must be a non-empty string"
        )

    return value

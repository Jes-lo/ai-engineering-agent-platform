"""Tests for Ollama embedding request and response mapping."""

from copy import deepcopy

import pytest

from ai_engineering_agent_platform.adapters.ollama.embedding_mapping import (
    build_ollama_embedding_payload,
    parse_ollama_embedding_response,
)
from ai_engineering_agent_platform.contracts import (
    EmbeddingRequest,
)
from ai_engineering_agent_platform.domain import ProviderExecutionError


def _request(
    *,
    dimensions: int | None = None,
) -> EmbeddingRequest:
    """Return a deterministic platform embedding request."""
    return EmbeddingRequest(
        model="example-embedding-model",
        texts=(
            "first synthetic text",
            "second synthetic text",
        ),
        dimensions=dimensions,
    )


def _response() -> dict[str, object]:
    """Return one deterministic Ollama-compatible embedding response."""
    return {
        "model": "example-embedding-model",
        "embeddings": [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        "total_duration": 100,
        "load_duration": 20,
        "prompt_eval_count": 7,
    }


def test_build_ollama_embedding_payload() -> None:
    """Batch inputs should map without provider details leaking outward."""
    payload = build_ollama_embedding_payload(_request())

    assert payload == {
        "model": "example-embedding-model",
        "input": [
            "first synthetic text",
            "second synthetic text",
        ],
        "truncate": False,
    }


def test_build_ollama_embedding_payload_maps_dimensions() -> None:
    """Requested dimensionality should map to Ollama directly."""
    payload = build_ollama_embedding_payload(
        _request(
            dimensions=3,
        )
    )

    assert payload["dimensions"] == 3
    assert payload["truncate"] is False


def test_parse_ollama_embedding_response() -> None:
    """A valid batch response should normalize to platform contracts."""
    result = parse_ollama_embedding_response(
        _response(),
        _request(),
    )

    assert result.model == "example-embedding-model"
    assert len(result.embeddings) == 2
    assert result.embeddings[0].values == (
        0.1,
        0.2,
        0.3,
    )
    assert result.embeddings[1].values == (
        0.4,
        0.5,
        0.6,
    )
    assert result.dimensions == 3
    assert result.input_tokens == 7


def test_parse_embedding_response_allows_missing_usage() -> None:
    """Token accounting remains optional when Ollama omits it."""
    payload = _response()
    del payload["prompt_eval_count"]

    result = parse_ollama_embedding_response(
        payload,
        _request(),
    )

    assert result.input_tokens is None


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "response",
    ],
)
def test_parse_embedding_response_requires_object(
    payload: object,
) -> None:
    """Top-level provider responses must be JSON objects."""
    with pytest.raises(
        ProviderExecutionError,
        match="must be a JSON object",
    ):
        parse_ollama_embedding_response(
            payload,
            _request(),
        )


@pytest.mark.parametrize(
    "model",
    [
        None,
        "",
        "   ",
        123,
    ],
)
def test_parse_embedding_response_requires_model(
    model: object,
) -> None:
    """Provider model identity must be a usable string."""
    payload = _response()
    payload["model"] = model

    with pytest.raises(
        ProviderExecutionError,
        match="model must be a non-empty string",
    ):
        parse_ollama_embedding_response(
            payload,
            _request(),
        )


@pytest.mark.parametrize(
    "embeddings",
    [
        None,
        {},
        "vectors",
    ],
)
def test_parse_embedding_response_requires_embedding_list(
    embeddings: object,
) -> None:
    """The provider embeddings field must be a JSON array."""
    payload = _response()
    payload["embeddings"] = embeddings

    with pytest.raises(
        ProviderExecutionError,
        match="embeddings must be a list",
    ):
        parse_ollama_embedding_response(
            payload,
            _request(),
        )


@pytest.mark.parametrize(
    "embeddings",
    [
        [],
        [[0.1, 0.2, 0.3]],
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ],
    ],
)
def test_parse_embedding_response_requires_matching_count(
    embeddings: list[object],
) -> None:
    """There must be exactly one output vector per requested text."""
    payload = _response()
    payload["embeddings"] = embeddings

    with pytest.raises(
        ProviderExecutionError,
        match="count must match request text count",
    ):
        parse_ollama_embedding_response(
            payload,
            _request(),
        )


@pytest.mark.parametrize(
    ("vector", "message"),
    [
        (
            "invalid",
            "vector 0 must be a list",
        ),
        (
            [],
            "vector 0 must not be empty",
        ),
        (
            [0.1, "invalid", 0.3],
            "component 1 must be numeric",
        ),
        (
            [0.1, True, 0.3],
            "component 1 must be numeric",
        ),
        (
            [0.1, float("nan"), 0.3],
            "component 1 must be finite",
        ),
        (
            [0.1, float("inf"), 0.3],
            "component 1 must be finite",
        ),
        (
            [0.1, float("-inf"), 0.3],
            "component 1 must be finite",
        ),
    ],
)
def test_parse_embedding_response_rejects_invalid_vectors(
    vector: object,
    message: str,
) -> None:
    """Malformed vector components must never reach domain models."""
    payload = _response()

    embeddings = cast_embedding_list(payload["embeddings"])

    embeddings[0] = vector

    with pytest.raises(
        ProviderExecutionError,
        match=message,
    ):
        parse_ollama_embedding_response(
            payload,
            _request(),
        )


def test_parse_embedding_response_accepts_integer_components() -> None:
    """JSON numeric integers should normalize safely to floats."""
    payload = _response()

    embeddings = cast_embedding_list(payload["embeddings"])

    embeddings[0] = [1, 2, 3]

    result = parse_ollama_embedding_response(
        payload,
        _request(),
    )

    assert result.embeddings[0].values == (
        1.0,
        2.0,
        3.0,
    )


def test_parse_embedding_response_rejects_requested_dimension_mismatch() -> None:
    """Returned vectors must honor explicit requested dimensionality."""
    with pytest.raises(
        ProviderExecutionError,
        match="dimensions do not match",
    ):
        parse_ollama_embedding_response(
            _response(),
            _request(
                dimensions=2,
            ),
        )


def test_parse_embedding_response_rejects_inconsistent_dimensions() -> None:
    """Provider vectors in one response must share dimensionality."""
    payload = _response()

    embeddings = cast_embedding_list(payload["embeddings"])

    embeddings[1] = [0.4, 0.5]

    with pytest.raises(
        ProviderExecutionError,
        match="violated platform invariants",
    ):
        parse_ollama_embedding_response(
            payload,
            _request(),
        )


@pytest.mark.parametrize(
    "usage",
    [
        -1,
        True,
        1.5,
        "7",
    ],
)
def test_parse_embedding_response_validates_usage(
    usage: object,
) -> None:
    """Token usage must be a non-negative JSON integer."""
    payload = _response()
    payload["prompt_eval_count"] = usage

    with pytest.raises(
        ProviderExecutionError,
        match="prompt_eval_count",
    ):
        parse_ollama_embedding_response(
            payload,
            _request(),
        )


def test_parser_does_not_mutate_provider_payload() -> None:
    """Normalization must not mutate caller-owned provider data."""
    payload = _response()
    original = deepcopy(payload)

    parse_ollama_embedding_response(
        payload,
        _request(),
    )

    assert payload == original


def cast_embedding_list(
    value: object,
) -> list[object]:
    """Return the synthetic embeddings list used by these tests."""
    assert isinstance(value, list)
    return value

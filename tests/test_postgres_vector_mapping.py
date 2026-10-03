"""Tests for pure PostgreSQL vector-store mappings."""

import json

import pytest

from ai_engineering_agent_platform.adapters.postgres import (
    build_metadata_payload,
    build_vector_literal,
    distance_to_score,
    parse_metadata_payload,
)
from ai_engineering_agent_platform.contracts import (
    VectorMetadataItem,
)


def test_build_vector_literal_is_deterministic() -> None:
    """Vector values should map to valid compact pgvector input syntax."""
    literal = build_vector_literal((1.0, -2.5, 0.0))

    assert literal == "[1.0,-2.5,0.0]"

    decoded = json.loads(literal)

    assert decoded == [1.0, -2.5, 0.0]


@pytest.mark.parametrize(
    "vector",
    [
        (),
        (float("nan"),),
        (float("inf"),),
        (float("-inf"),),
    ],
)
def test_build_vector_literal_rejects_invalid_values(
    vector: tuple[float, ...],
) -> None:
    """PostgreSQL mapping must fail closed on invalid vectors."""
    with pytest.raises(ValueError):
        build_vector_literal(vector)


def test_metadata_payload_preserves_order_and_types() -> None:
    """Metadata should round-trip in the contract's ordered representation."""
    metadata = (
        VectorMetadataItem(
            key="string",
            value="value",
        ),
        VectorMetadataItem(
            key="integer",
            value=7,
        ),
        VectorMetadataItem(
            key="float",
            value=1.25,
        ),
        VectorMetadataItem(
            key="boolean",
            value=True,
        ),
        VectorMetadataItem(
            key="nullable",
            value=None,
        ),
    )

    payload = build_metadata_payload(metadata)

    assert payload == [
        {
            "key": "string",
            "value": "value",
        },
        {
            "key": "integer",
            "value": 7,
        },
        {
            "key": "float",
            "value": 1.25,
        },
        {
            "key": "boolean",
            "value": True,
        },
        {
            "key": "nullable",
            "value": None,
        },
    ]

    assert parse_metadata_payload(payload) == metadata


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [1],
        [{"key": "only-key"}],
        [
            {
                "key": "x",
                "value": "y",
                "extra": "z",
            }
        ],
        [
            {
                "key": 1,
                "value": "value",
            }
        ],
        [
            {
                "key": "x",
                "value": ["unsupported"],
            }
        ],
    ],
)
def test_parse_metadata_payload_rejects_invalid_shapes(
    payload: object,
) -> None:
    """Unexpected database JSON shapes must never be silently accepted."""
    with pytest.raises(ValueError):
        parse_metadata_payload(payload)


def test_distance_score_is_higher_for_closer_matches() -> None:
    """L2 distance normalization should expose higher-is-better scores."""
    exact = distance_to_score(0.0)
    near = distance_to_score(1.0)
    far = distance_to_score(3.0)

    assert exact == 1.0
    assert exact > near > far > 0.0
    assert near == pytest.approx(0.5)
    assert far == pytest.approx(0.25)


@pytest.mark.parametrize(
    "distance",
    [
        -1.0,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_distance_score_rejects_invalid_distance(
    distance: float,
) -> None:
    """Distance normalization must fail closed for invalid DB values."""
    with pytest.raises(ValueError):
        distance_to_score(distance)

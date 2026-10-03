"""Pure mappings between vector-store contracts and PostgreSQL values."""

import json
from math import isfinite

from ai_engineering_agent_platform.contracts import (
    VectorMetadataItem,
)

type JsonScalar = str | int | float | bool | None
type JsonObject = dict[str, JsonScalar]

type MetadataPayload = list[JsonObject]


def build_vector_literal(
    vector: tuple[float, ...],
) -> str:
    """Serialize a validated provider-neutral vector for pgvector input."""
    if not vector:
        raise ValueError("vector must not be empty")

    normalized: list[float] = []

    for value in vector:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise ValueError("vector values must be numeric and not boolean")

        try:
            normalized_value = float(value)
        except OverflowError as exc:
            raise ValueError("vector values must be finite") from exc

        if not isfinite(normalized_value):
            raise ValueError("vector values must be finite")

        normalized.append(normalized_value)

    return json.dumps(
        normalized,
        allow_nan=False,
        separators=(",", ":"),
    )


def build_metadata_payload(
    metadata: tuple[VectorMetadataItem, ...],
) -> MetadataPayload:
    """Serialize ordered contract metadata into the JSONB array shape."""
    return [
        {
            "key": item.key,
            "value": item.value,
        }
        for item in metadata
    ]


def parse_metadata_payload(
    payload: object,
) -> tuple[VectorMetadataItem, ...]:
    """Parse PostgreSQL JSONB metadata without weakening contract validation."""
    if not isinstance(payload, list):
        raise ValueError("PostgreSQL vector metadata must be a JSON array")

    items: list[VectorMetadataItem] = []

    for raw_item in payload:
        if not isinstance(raw_item, dict):
            raise ValueError("PostgreSQL vector metadata items must be JSON objects")

        if set(raw_item) != {"key", "value"}:
            raise ValueError(
                "PostgreSQL vector metadata items must contain exactly key and value"
            )

        key = raw_item["key"]
        value = raw_item["value"]

        if not isinstance(key, str):
            raise ValueError("PostgreSQL vector metadata key must be a string")

        if value is not None and not isinstance(
            value,
            (str, int, float, bool),
        ):
            raise ValueError("PostgreSQL vector metadata value has unsupported type")

        items.append(
            VectorMetadataItem(
                key=key,
                value=value,
            )
        )

    return tuple(items)


def distance_to_score(
    distance: float,
) -> float:
    """Normalize non-negative L2 distance to a higher-is-better score."""
    if isinstance(distance, bool) or not isinstance(
        distance,
        (int, float),
    ):
        raise ValueError("vector distance must be numeric and not boolean")

    try:
        normalized_distance = float(distance)
    except OverflowError as exc:
        raise ValueError("vector distance must be finite") from exc

    if not isfinite(normalized_distance) or normalized_distance < 0:
        raise ValueError("vector distance must be finite and non-negative")

    return 1.0 / (1.0 + normalized_distance)

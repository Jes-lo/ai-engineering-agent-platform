"""Provider-neutral contracts for vector storage and retrieval."""

from dataclasses import dataclass
from math import isfinite
from typing import Protocol, runtime_checkable

from ai_engineering_agent_platform.contracts.provider import Provider

type MetadataValue = str | int | float | bool | None


def _validate_vector(values: tuple[float, ...]) -> None:
    """Validate a provider-neutral vector."""
    if not values:
        raise ValueError("vector must not be empty")

    if any(not isfinite(value) for value in values):
        raise ValueError("vector values must be finite")


def _validate_namespace(namespace: str | None) -> None:
    """Validate an optional vector-store namespace."""
    if namespace is not None and not namespace.strip():
        raise ValueError("namespace must not be empty")


@dataclass(frozen=True, slots=True)
class VectorMetadataItem:
    """Immutable metadata key/value pair."""

    key: str
    value: MetadataValue

    def __post_init__(self) -> None:
        """Validate portable metadata values."""
        if not self.key.strip():
            raise ValueError("metadata key must not be empty")

        if isinstance(self.value, float) and not isfinite(self.value):
            raise ValueError("metadata float values must be finite")


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """Immutable record persisted by a vector store."""

    record_id: str
    vector: tuple[float, ...]
    text: str | None = None
    metadata: tuple[VectorMetadataItem, ...] = ()

    def __post_init__(self) -> None:
        """Validate vector record invariants."""
        if not self.record_id.strip():
            raise ValueError("record_id must not be empty")

        _validate_vector(self.vector)

        if self.text is not None and not self.text.strip():
            raise ValueError("record text must not be empty")

        metadata_keys = [item.key for item in self.metadata]

        if len(metadata_keys) != len(set(metadata_keys)):
            raise ValueError("metadata keys must be unique")

    @property
    def dimensions(self) -> int:
        """Return vector dimensionality."""
        return len(self.vector)


@dataclass(frozen=True, slots=True)
class VectorUpsertRequest:
    """Immutable request for inserting or replacing vector records."""

    records: tuple[VectorRecord, ...]
    namespace: str | None = None

    def __post_init__(self) -> None:
        """Validate upsert collection invariants."""
        if not self.records:
            raise ValueError("records must not be empty")

        _validate_namespace(self.namespace)

        record_ids = [record.record_id for record in self.records]

        if len(record_ids) != len(set(record_ids)):
            raise ValueError("record identifiers must be unique")

        expected_dimensions = self.records[0].dimensions

        if any(record.dimensions != expected_dimensions for record in self.records):
            raise ValueError("all vectors must have equal dimensions")

    @property
    def dimensions(self) -> int:
        """Return dimensionality shared by all records."""
        return self.records[0].dimensions


@dataclass(frozen=True, slots=True)
class VectorQueryRequest:
    """Immutable nearest-neighbor query."""

    vector: tuple[float, ...]
    top_k: int
    namespace: str | None = None

    def __post_init__(self) -> None:
        """Validate vector query invariants."""
        _validate_vector(self.vector)
        _validate_namespace(self.namespace)

        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

    @property
    def dimensions(self) -> int:
        """Return query vector dimensionality."""
        return len(self.vector)


@dataclass(frozen=True, slots=True)
class VectorQueryResult:
    """Normalized nearest-neighbor result."""

    record_id: str
    score: float
    rank: int
    text: str | None = None
    metadata: tuple[VectorMetadataItem, ...] = ()

    def __post_init__(self) -> None:
        """Validate normalized query result invariants."""
        if not self.record_id.strip():
            raise ValueError("record_id must not be empty")

        if not isfinite(self.score):
            raise ValueError("vector query score must be finite")

        if self.rank <= 0:
            raise ValueError("rank must be positive")

        if self.text is not None and not self.text.strip():
            raise ValueError("result text must not be empty")

        metadata_keys = [item.key for item in self.metadata]

        if len(metadata_keys) != len(set(metadata_keys)):
            raise ValueError("metadata keys must be unique")


@dataclass(frozen=True, slots=True)
class VectorQueryResponse:
    """Normalized response returned by a vector store."""

    results: tuple[VectorQueryResult, ...]

    def __post_init__(self) -> None:
        """Validate result uniqueness and rank ordering."""
        record_ids = [result.record_id for result in self.results]

        if len(record_ids) != len(set(record_ids)):
            raise ValueError("query result record identifiers must be unique")

        ranks = tuple(result.rank for result in self.results)

        expected_ranks = tuple(range(1, len(self.results) + 1))

        if ranks != expected_ranks:
            raise ValueError("query results must use contiguous rank order")


@dataclass(frozen=True, slots=True)
class VectorDeleteRequest:
    """Immutable request for deleting vector records."""

    record_ids: tuple[str, ...]
    namespace: str | None = None

    def __post_init__(self) -> None:
        """Validate deletion request invariants."""
        if not self.record_ids:
            raise ValueError("record_ids must not be empty")

        _validate_namespace(self.namespace)

        if any(not record_id.strip() for record_id in self.record_ids):
            raise ValueError("record_ids must not contain empty values")

        if len(self.record_ids) != len(set(self.record_ids)):
            raise ValueError("record identifiers must be unique")


@runtime_checkable
class VectorStoreProvider(Provider, Protocol):
    """Asynchronous contract for vector persistence and retrieval."""

    async def upsert(
        self,
        request: VectorUpsertRequest,
    ) -> None:
        """Insert or replace vector records."""
        ...

    async def query(
        self,
        request: VectorQueryRequest,
    ) -> VectorQueryResponse:
        """Return normalized nearest-neighbor results."""
        ...

    async def delete(
        self,
        request: VectorDeleteRequest,
    ) -> None:
        """Delete vector records by identifier."""
        ...

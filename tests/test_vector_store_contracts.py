"""Tests for provider-neutral vector store contracts."""

import asyncio
from dataclasses import FrozenInstanceError

import pytest

from ai_engineering_agent_platform.contracts import (
    ProviderDescriptor,
    ProviderKind,
    VectorDeleteRequest,
    VectorMetadataItem,
    VectorQueryRequest,
    VectorQueryResponse,
    VectorQueryResult,
    VectorRecord,
    VectorStoreProvider,
    VectorUpsertRequest,
)


class ExampleVectorStoreProvider:
    """Minimal structural implementation of the vector-store contract."""

    def __init__(self) -> None:
        """Initialize deterministic in-memory test state."""
        self._records: dict[str, VectorRecord] = {}

    @property
    def descriptor(self) -> ProviderDescriptor:
        """Return deterministic provider metadata."""
        return ProviderDescriptor(
            name="example-vector-store",
            kind=ProviderKind.VECTOR_STORE,
        )

    async def upsert(
        self,
        request: VectorUpsertRequest,
    ) -> None:
        """Store records deterministically for contract testing."""
        for record in request.records:
            self._records[record.record_id] = record

    async def query(
        self,
        request: VectorQueryRequest,
    ) -> VectorQueryResponse:
        """Return deterministic ordered query results."""
        selected = tuple(self._records.values())[: request.top_k]

        return VectorQueryResponse(
            results=tuple(
                VectorQueryResult(
                    record_id=record.record_id,
                    score=float(len(selected) - index),
                    rank=index + 1,
                    text=record.text,
                    metadata=record.metadata,
                )
                for index, record in enumerate(selected)
            ),
        )

    async def delete(
        self,
        request: VectorDeleteRequest,
    ) -> None:
        """Delete deterministic test records."""
        for record_id in request.record_ids:
            self._records.pop(record_id, None)


def _records() -> tuple[VectorRecord, ...]:
    """Return deterministic vector records."""
    return (
        VectorRecord(
            record_id="record-1",
            vector=(0.1, 0.2, 0.3),
            text="first record",
            metadata=(
                VectorMetadataItem(
                    key="source",
                    value="test",
                ),
            ),
        ),
        VectorRecord(
            record_id="record-2",
            vector=(0.4, 0.5, 0.6),
            text="second record",
        ),
    )


def test_vector_metadata_validates_key_and_float_value() -> None:
    """Metadata requires usable keys and finite float values."""
    with pytest.raises(
        ValueError,
        match="metadata key must not be empty",
    ):
        VectorMetadataItem(
            key=" ",
            value="value",
        )

    with pytest.raises(
        ValueError,
        match="metadata float values must be finite",
    ):
        VectorMetadataItem(
            key="score",
            value=float("nan"),
        )


def test_vector_record_is_immutable() -> None:
    """Vector records should not mutate after creation."""
    record = _records()[0]

    with pytest.raises(FrozenInstanceError):
        record.record_id = "changed"  # type: ignore[misc]


def test_vector_record_validates_fields() -> None:
    """Records require valid identifiers, vectors, and payloads."""
    with pytest.raises(
        ValueError,
        match="record_id must not be empty",
    ):
        VectorRecord(
            record_id=" ",
            vector=(0.1,),
        )

    with pytest.raises(
        ValueError,
        match="vector must not be empty",
    ):
        VectorRecord(
            record_id="record",
            vector=(),
        )

    with pytest.raises(
        ValueError,
        match="vector values must be finite",
    ):
        VectorRecord(
            record_id="record",
            vector=(float("inf"),),
        )

    with pytest.raises(
        ValueError,
        match="record text must not be empty",
    ):
        VectorRecord(
            record_id="record",
            vector=(0.1,),
            text=" ",
        )

    with pytest.raises(
        ValueError,
        match="metadata keys must be unique",
    ):
        VectorRecord(
            record_id="record",
            vector=(0.1,),
            metadata=(
                VectorMetadataItem(
                    key="source",
                    value="a",
                ),
                VectorMetadataItem(
                    key="source",
                    value="b",
                ),
            ),
        )


def test_vector_upsert_validates_collection() -> None:
    """Upserts require unique records with equal dimensions."""
    request = VectorUpsertRequest(
        space_id="test-space",
        records=_records(),
        namespace="tenant-a",
    )

    assert request.dimensions == 3

    with pytest.raises(
        ValueError,
        match="records must not be empty",
    ):
        VectorUpsertRequest(space_id="test-space", records=())

    with pytest.raises(
        ValueError,
        match="record identifiers must be unique",
    ):
        VectorUpsertRequest(
            space_id="test-space",
            records=(
                VectorRecord(
                    record_id="duplicate",
                    vector=(0.1,),
                ),
                VectorRecord(
                    record_id="duplicate",
                    vector=(0.2,),
                ),
            ),
        )

    with pytest.raises(
        ValueError,
        match="all vectors must have equal dimensions",
    ):
        VectorUpsertRequest(
            space_id="test-space",
            records=(
                VectorRecord(
                    record_id="one",
                    vector=(0.1,),
                ),
                VectorRecord(
                    record_id="two",
                    vector=(0.1, 0.2),
                ),
            ),
        )


def test_vector_query_validates_controls() -> None:
    """Vector queries require valid vectors, limits, and namespaces."""
    request = VectorQueryRequest(
        space_id="test-space",
        vector=(0.1, 0.2, 0.3),
        top_k=5,
        namespace="tenant-a",
    )

    assert request.dimensions == 3

    with pytest.raises(
        ValueError,
        match="vector must not be empty",
    ):
        VectorQueryRequest(
            space_id="test-space",
            vector=(),
            top_k=1,
        )

    with pytest.raises(
        ValueError,
        match="top_k must be positive",
    ):
        VectorQueryRequest(
            space_id="test-space",
            vector=(0.1,),
            top_k=0,
        )

    with pytest.raises(
        ValueError,
        match="namespace must not be empty",
    ):
        VectorQueryRequest(
            space_id="test-space",
            vector=(0.1,),
            top_k=1,
            namespace=" ",
        )


def test_vector_delete_validates_identifiers_and_namespace() -> None:
    """Deletion requests require unique usable identifiers."""
    with pytest.raises(
        ValueError,
        match="record_ids must not be empty",
    ):
        VectorDeleteRequest(
            space_id="test-space",
            record_ids=(),
        )

    with pytest.raises(
        ValueError,
        match="record_ids must not contain empty values",
    ):
        VectorDeleteRequest(
            space_id="test-space",
            record_ids=("record", " "),
        )

    with pytest.raises(
        ValueError,
        match="record identifiers must be unique",
    ):
        VectorDeleteRequest(
            space_id="test-space",
            record_ids=("record", "record"),
        )

    with pytest.raises(
        ValueError,
        match="namespace must not be empty",
    ):
        VectorDeleteRequest(
            space_id="test-space",
            record_ids=("record",),
            namespace=" ",
        )


def test_vector_query_result_validates_fields() -> None:
    """Query results require valid identifiers, scores, and ranks."""
    with pytest.raises(
        ValueError,
        match="record_id must not be empty",
    ):
        VectorQueryResult(
            record_id=" ",
            score=1.0,
            rank=1,
        )

    with pytest.raises(
        ValueError,
        match="vector query score must be finite",
    ):
        VectorQueryResult(
            record_id="record",
            score=float("nan"),
            rank=1,
        )

    with pytest.raises(
        ValueError,
        match="rank must be positive",
    ):
        VectorQueryResult(
            record_id="record",
            score=1.0,
            rank=0,
        )


def test_vector_query_response_allows_empty_results() -> None:
    """An empty vector store may validly return no matches."""
    response = VectorQueryResponse(
        results=(),
    )

    assert response.results == ()


def test_vector_query_response_validates_order_and_uniqueness() -> None:
    """Query responses require unique records and contiguous ranks."""
    with pytest.raises(
        ValueError,
        match="query result record identifiers must be unique",
    ):
        VectorQueryResponse(
            results=(
                VectorQueryResult(
                    record_id="record",
                    score=2.0,
                    rank=1,
                ),
                VectorQueryResult(
                    record_id="record",
                    score=1.0,
                    rank=2,
                ),
            ),
        )

    with pytest.raises(
        ValueError,
        match="query results must use contiguous rank order",
    ):
        VectorQueryResponse(
            results=(
                VectorQueryResult(
                    record_id="record-1",
                    score=2.0,
                    rank=1,
                ),
                VectorQueryResult(
                    record_id="record-2",
                    score=1.0,
                    rank=3,
                ),
            ),
        )


def test_vector_store_provider_supports_structural_async_typing() -> None:
    """Concrete adapters should satisfy VectorStoreProvider structurally."""
    provider: VectorStoreProvider = ExampleVectorStoreProvider()

    assert isinstance(provider, VectorStoreProvider)
    assert provider.descriptor.kind is ProviderKind.VECTOR_STORE

    asyncio.run(
        provider.upsert(
            VectorUpsertRequest(
                space_id="test-space",
                records=_records(),
                namespace="tenant-a",
            )
        )
    )

    response = asyncio.run(
        provider.query(
            VectorQueryRequest(
                space_id="test-space",
                vector=(0.1, 0.2, 0.3),
                top_k=2,
                namespace="tenant-a",
            )
        )
    )

    assert len(response.results) == 2
    assert response.results[0].record_id == "record-1"
    assert response.results[0].rank == 1
    assert response.results[1].rank == 2

    asyncio.run(
        provider.delete(
            VectorDeleteRequest(
                space_id="test-space",
                record_ids=("record-1",),
                namespace="tenant-a",
            )
        )
    )


@pytest.mark.parametrize(
    "value",
    [
        True,
        "0.1",
        None,
    ],
)
def test_vector_contract_rejects_non_numeric_components(
    value: object,
) -> None:
    """Vector components must be real numeric values, never bool/coerced data."""
    with pytest.raises(
        ValueError,
        match="vector values must be numeric",
    ):
        VectorRecord(
            record_id="record",
            vector=(value,),  # type: ignore[arg-type]
        )


def test_vector_contract_normalizes_numeric_overflow_failure() -> None:
    """Extremely large numeric values must fail as contract validation."""
    with pytest.raises(
        ValueError,
        match="vector values must be finite",
    ):
        VectorRecord(
            record_id="record",
            vector=(10**10000,),
        )


@pytest.mark.parametrize(
    "top_k",
    [
        True,
        1.5,
        "5",
    ],
)
def test_vector_query_requires_integer_top_k(
    top_k: object,
) -> None:
    """Nearest-neighbor limits must be real integers."""
    with pytest.raises(
        ValueError,
        match="top_k must be an integer",
    ):
        VectorQueryRequest(
            space_id="test-space",
            vector=(0.1, 0.2),
            top_k=top_k,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "score",
    [
        True,
        "1.0",
        None,
    ],
)
def test_vector_query_result_requires_numeric_score(
    score: object,
) -> None:
    """Normalized query scores must be real numeric values."""
    with pytest.raises(
        ValueError,
        match="vector query score must be numeric",
    ):
        VectorQueryResult(
            record_id="record",
            score=score,  # type: ignore[arg-type]
            rank=1,
        )


@pytest.mark.parametrize(
    "rank",
    [
        True,
        1.5,
        "1",
    ],
)
def test_vector_query_result_requires_integer_rank(
    rank: object,
) -> None:
    """Normalized query ranks must be real integers."""
    with pytest.raises(
        ValueError,
        match="rank must be an integer",
    ):
        VectorQueryResult(
            record_id="record",
            score=1.0,
            rank=rank,  # type: ignore[arg-type]
        )


def test_vector_requests_preserve_explicit_space_id() -> None:
    """Low-level vector operations must preserve caller-owned space identity."""
    record = VectorRecord(
        record_id="record-1",
        vector=(0.1, 0.2),
    )

    upsert = VectorUpsertRequest(
        records=(record,),
        space_id="explicit-space",
    )
    query = VectorQueryRequest(
        vector=(0.1, 0.2),
        top_k=1,
        space_id="explicit-space",
    )
    delete = VectorDeleteRequest(
        record_ids=("record-1",),
        space_id="explicit-space",
    )

    assert upsert.space_id == "explicit-space"
    assert query.space_id == "explicit-space"
    assert delete.space_id == "explicit-space"


def test_vector_requests_reject_empty_space_id() -> None:
    """Every vector operation must target one non-empty vector space."""
    record = VectorRecord(
        record_id="record-1",
        vector=(0.1, 0.2),
    )

    with pytest.raises(
        ValueError,
        match="space_id must not be empty",
    ):
        VectorUpsertRequest(
            records=(record,),
            space_id=" ",
        )

    with pytest.raises(
        ValueError,
        match="space_id must not be empty",
    ):
        VectorQueryRequest(
            vector=(0.1, 0.2),
            top_k=1,
            space_id=" ",
        )

    with pytest.raises(
        ValueError,
        match="space_id must not be empty",
    ):
        VectorDeleteRequest(
            record_ids=("record-1",),
            space_id=" ",
        )

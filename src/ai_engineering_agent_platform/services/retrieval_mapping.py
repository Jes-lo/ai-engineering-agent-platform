"""Pure mapping between retrieval-domain objects and provider contracts."""

from collections.abc import Mapping

from ai_engineering_agent_platform.contracts import (
    EmbeddingRequest,
    EmbeddingResponse,
    VectorMetadataItem,
    VectorQueryRequest,
    VectorQueryResponse,
    VectorQueryResult,
)
from ai_engineering_agent_platform.domain import (
    ProviderExecutionError,
    RetrievalMetadataItem,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedEvidence,
)

_RETRIEVAL_PREFIX = "retrieval."
_SOURCE_PREFIX = "source."

_DOCUMENT_ID_KEY = "retrieval.document_id"
_CHUNK_INDEX_KEY = "retrieval.chunk_index"
_START_CHAR_KEY = "retrieval.start_char"
_END_CHAR_KEY = "retrieval.end_char"
_SOURCE_REF_KEY = "retrieval.source_ref"
_TITLE_KEY = "retrieval.title"

_REQUIRED_RETRIEVAL_KEYS = frozenset(
    {
        _DOCUMENT_ID_KEY,
        _CHUNK_INDEX_KEY,
        _START_CHAR_KEY,
        _END_CHAR_KEY,
        _SOURCE_REF_KEY,
        _TITLE_KEY,
    }
)


def _provider_mapping_error(
    message: str,
) -> ProviderExecutionError:
    """Create one normalized malformed-provider-response error."""
    return ProviderExecutionError(
        f"Vector result contains invalid retrieval provenance: {message}"
    )


def _required_string(
    metadata: Mapping[str, object],
    key: str,
) -> str:
    """Extract one required non-empty string provenance value."""
    if key not in metadata:
        raise _provider_mapping_error(f"missing {key}")

    value = metadata[key]

    if not isinstance(
        value,
        str,
    ):
        raise _provider_mapping_error(f"{key} must be a string")

    if not value.strip():
        raise _provider_mapping_error(f"{key} must not be empty")

    return value


def _required_non_negative_integer(
    metadata: Mapping[str, object],
    key: str,
) -> int:
    """Extract one strict non-negative integer provenance value."""
    if key not in metadata:
        raise _provider_mapping_error(f"missing {key}")

    value = metadata[key]

    if isinstance(
        value,
        bool,
    ) or not isinstance(
        value,
        int,
    ):
        raise _provider_mapping_error(f"{key} must be an integer")

    if value < 0:
        raise _provider_mapping_error(f"{key} must be non-negative")

    return value


def _optional_title(
    metadata: Mapping[str, object],
) -> str | None:
    """Extract the required metadata slot for an optional title."""
    if _TITLE_KEY not in metadata:
        raise _provider_mapping_error(f"missing {_TITLE_KEY}")

    value = metadata[_TITLE_KEY]

    if value is None:
        return None

    if not isinstance(
        value,
        str,
    ):
        raise _provider_mapping_error(f"{_TITLE_KEY} must be a string or null")

    if not value.strip():
        raise _provider_mapping_error(f"{_TITLE_KEY} must not be empty")

    return value


def _validate_metadata_schema(
    metadata: tuple[VectorMetadataItem, ...],
) -> None:
    """Reject records outside the project-owned retrieval metadata schema."""
    for item in metadata:
        key = item.key

        if key in _REQUIRED_RETRIEVAL_KEYS:
            continue

        if key.startswith(_SOURCE_PREFIX):
            source_key = key[len(_SOURCE_PREFIX) :]

            if not source_key.strip():
                raise _provider_mapping_error("source metadata key must not be empty")

            continue

        raise _provider_mapping_error(f"unsupported metadata key {key!r}")


def _source_metadata(
    metadata: tuple[VectorMetadataItem, ...],
) -> tuple[RetrievalMetadataItem, ...]:
    """Restore caller/source metadata without mixing reserved provenance."""
    restored: list[RetrievalMetadataItem] = []

    for item in metadata:
        if not item.key.startswith(_SOURCE_PREFIX):
            continue

        source_key = item.key[len(_SOURCE_PREFIX) :]

        try:
            restored.append(
                RetrievalMetadataItem(
                    key=source_key,
                    value=item.value,
                )
            )
        except ValueError as exc:
            raise _provider_mapping_error("invalid source metadata") from exc

    return tuple(restored)


def build_retrieval_embedding_request(
    request: RetrievalRequest,
    *,
    model: str,
    dimensions: int | None = None,
) -> EmbeddingRequest:
    """Build one embedding request for a semantic retrieval query."""
    return EmbeddingRequest(
        model=model,
        texts=(request.query,),
        dimensions=dimensions,
    )


def build_vector_query_request(
    request: RetrievalRequest,
    embedding_response: EmbeddingResponse,
    *,
    space_id: str | None = None,
) -> VectorQueryRequest:
    """Map one query embedding into the vector-store query contract."""
    if len(embedding_response.embeddings) != 1:
        raise ProviderExecutionError(
            "Embedding provider returned unexpected result count"
        )

    resolved_space_id = embedding_response.model if space_id is None else space_id

    return VectorQueryRequest(
        vector=(embedding_response.embeddings[0].values),
        top_k=request.top_k,
        space_id=resolved_space_id,
        namespace=request.namespace,
    )


def _map_vector_result(
    result: VectorQueryResult,
) -> RetrievedEvidence:
    """Map one vector result into citation-ready retrieval evidence."""
    if result.text is None:
        raise _provider_mapping_error("result text is required")

    _validate_metadata_schema(result.metadata)

    metadata_values = {item.key: item.value for item in result.metadata}

    document_id = _required_string(
        metadata_values,
        _DOCUMENT_ID_KEY,
    )
    chunk_index = _required_non_negative_integer(
        metadata_values,
        _CHUNK_INDEX_KEY,
    )
    start_char = _required_non_negative_integer(
        metadata_values,
        _START_CHAR_KEY,
    )
    end_char = _required_non_negative_integer(
        metadata_values,
        _END_CHAR_KEY,
    )
    source_ref = _required_string(
        metadata_values,
        _SOURCE_REF_KEY,
    )
    title = _optional_title(metadata_values)

    if chunk_index < 0:
        raise _provider_mapping_error(f"{_CHUNK_INDEX_KEY} must be non-negative")

    try:
        return RetrievedEvidence(
            chunk_id=result.record_id,
            document_id=document_id,
            text=result.text,
            score=result.score,
            rank=result.rank,
            start_char=start_char,
            end_char=end_char,
            source_ref=source_ref,
            title=title,
            metadata=_source_metadata(result.metadata),
        )
    except ValueError as exc:
        raise _provider_mapping_error("evidence invariants failed") from exc


def build_retrieval_response(
    request: RetrievalRequest,
    vector_response: VectorQueryResponse,
) -> RetrievalResponse:
    """Map provider-neutral vector results into retrieval evidence."""
    if len(vector_response.results) > request.top_k:
        raise ProviderExecutionError(
            "Vector store returned more results than requested"
        )

    results = tuple(_map_vector_result(result) for result in vector_response.results)

    return RetrievalResponse(
        query=request.query,
        results=results,
        namespace=request.namespace,
    )

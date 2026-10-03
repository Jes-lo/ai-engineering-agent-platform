"""Infrastructure-independent retrieval domain primitives."""

from dataclasses import dataclass
from math import isfinite

type RetrievalMetadataValue = str | int | float | bool | None


def _validate_required_text(
    name: str,
    value: str,
) -> None:
    """Reject empty required textual values."""
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _validate_optional_text(
    name: str,
    value: str | None,
) -> None:
    """Reject empty optional textual values when supplied."""
    if value is not None and not value.strip():
        raise ValueError(f"{name} must not be empty")


def _validate_non_negative_integer(
    name: str,
    value: int,
) -> None:
    """Reject booleans, non-integers, and negative values."""
    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise ValueError(f"{name} must be an integer")

    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_positive_integer(
    name: str,
    value: int,
) -> None:
    """Reject booleans, non-integers, and non-positive values."""
    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise ValueError(f"{name} must be an integer")

    if value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class RetrievalMetadataItem:
    """Immutable domain metadata carried through retrieval."""

    key: str
    value: RetrievalMetadataValue

    def __post_init__(self) -> None:
        """Validate provider-independent metadata."""
        _validate_required_text(
            "metadata key",
            self.key,
        )

        if self.value is not None and not isinstance(
            self.value,
            (str, int, float, bool),
        ):
            raise ValueError("metadata value has unsupported type")

        if isinstance(self.value, float) and not isfinite(self.value):
            raise ValueError("metadata float values must be finite")


def _validate_metadata(
    metadata: tuple[RetrievalMetadataItem, ...],
) -> None:
    """Require unique metadata keys."""
    keys = tuple(item.key for item in metadata)

    if len(keys) != len(set(keys)):
        raise ValueError("metadata keys must be unique")


@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
    """Canonical text document accepted by the retrieval subsystem."""

    document_id: str
    text: str
    source_ref: str
    title: str | None = None
    metadata: tuple[RetrievalMetadataItem, ...] = ()

    def __post_init__(self) -> None:
        """Validate canonical document identity and source information."""
        _validate_required_text(
            "document_id",
            self.document_id,
        )
        _validate_required_text(
            "document text",
            self.text,
        )
        _validate_required_text(
            "source_ref",
            self.source_ref,
        )
        _validate_optional_text(
            "title",
            self.title,
        )
        _validate_metadata(
            self.metadata,
        )


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """Exact character range derived from one canonical document."""

    chunk_id: str
    document_id: str
    text: str
    index: int
    start_char: int
    end_char: int
    source_ref: str
    title: str | None = None
    metadata: tuple[RetrievalMetadataItem, ...] = ()

    def __post_init__(self) -> None:
        """Validate chunk provenance and exact character offsets."""
        _validate_required_text(
            "chunk_id",
            self.chunk_id,
        )
        _validate_required_text(
            "document_id",
            self.document_id,
        )
        _validate_required_text(
            "chunk text",
            self.text,
        )
        _validate_required_text(
            "source_ref",
            self.source_ref,
        )
        _validate_optional_text(
            "title",
            self.title,
        )

        _validate_non_negative_integer(
            "index",
            self.index,
        )
        _validate_non_negative_integer(
            "start_char",
            self.start_char,
        )
        _validate_non_negative_integer(
            "end_char",
            self.end_char,
        )

        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")

        if self.end_char - self.start_char != len(self.text):
            raise ValueError("chunk offsets must match chunk text length")

        _validate_metadata(
            self.metadata,
        )


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """Semantic retrieval request before embedding generation."""

    query: str
    top_k: int
    namespace: str | None = None

    def __post_init__(self) -> None:
        """Validate retrieval request invariants."""
        _validate_required_text(
            "query",
            self.query,
        )
        _validate_positive_integer(
            "top_k",
            self.top_k,
        )
        _validate_optional_text(
            "namespace",
            self.namespace,
        )


@dataclass(frozen=True, slots=True)
class RetrievedEvidence:
    """One citation-ready chunk returned by semantic retrieval.

    ``source_ref`` is opaque provenance data. Its presence does not imply
    that the platform trusts, opens, fetches, or executes the referenced
    source.
    """

    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int
    start_char: int
    end_char: int
    source_ref: str
    title: str | None = None
    metadata: tuple[RetrievalMetadataItem, ...] = ()

    def __post_init__(self) -> None:
        """Validate retrieved evidence identity, ordering, and provenance."""
        _validate_required_text(
            "chunk_id",
            self.chunk_id,
        )
        _validate_required_text(
            "document_id",
            self.document_id,
        )
        _validate_required_text(
            "evidence text",
            self.text,
        )
        _validate_required_text(
            "source_ref",
            self.source_ref,
        )
        _validate_optional_text(
            "title",
            self.title,
        )

        if isinstance(self.score, bool) or not isinstance(
            self.score,
            (int, float),
        ):
            raise ValueError("retrieval score must be numeric")

        try:
            finite_score = isfinite(self.score)
        except OverflowError as exc:
            raise ValueError("retrieval score must be finite") from exc

        if not finite_score:
            raise ValueError("retrieval score must be finite")

        _validate_positive_integer(
            "rank",
            self.rank,
        )
        _validate_non_negative_integer(
            "start_char",
            self.start_char,
        )
        _validate_non_negative_integer(
            "end_char",
            self.end_char,
        )

        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")

        if self.end_char - self.start_char != len(self.text):
            raise ValueError("evidence offsets must match evidence text length")

        _validate_metadata(
            self.metadata,
        )


@dataclass(frozen=True, slots=True)
class RetrievalResponse:
    """Ordered evidence returned for one semantic query."""

    query: str
    results: tuple[RetrievedEvidence, ...]
    namespace: str | None = None

    def __post_init__(self) -> None:
        """Validate result identity and rank ordering."""
        _validate_required_text(
            "query",
            self.query,
        )
        _validate_optional_text(
            "namespace",
            self.namespace,
        )

        chunk_ids = tuple(result.chunk_id for result in self.results)

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("retrieval result chunk identifiers must be unique")

        ranks = tuple(result.rank for result in self.results)

        expected_ranks = tuple(
            range(
                1,
                len(self.results) + 1,
            )
        )

        if ranks != expected_ranks:
            raise ValueError("retrieval results must use contiguous rank order")


@dataclass(frozen=True, slots=True)
class RerankedEvidence:
    """Retrieved evidence plus an independent reranker result.

    ``evidence.score`` and ``evidence.rank`` remain the original semantic
    vector-retrieval values. ``rerank_score`` and ``rank`` describe the
    optional second-stage ranking. This prevents reranking from destroying
    retrieval diagnostics or citation provenance.
    """

    evidence: RetrievedEvidence
    rerank_score: float
    rank: int

    def __post_init__(self) -> None:
        """Validate reranker score and final rank independently."""
        if isinstance(
            self.rerank_score,
            bool,
        ) or not isinstance(
            self.rerank_score,
            (int, float),
        ):
            raise ValueError("rerank_score must be numeric")

        try:
            finite_score = isfinite(self.rerank_score)
        except OverflowError as exc:
            raise ValueError("rerank_score must be finite") from exc

        if not finite_score:
            raise ValueError("rerank_score must be finite")

        _validate_positive_integer(
            "rank",
            self.rank,
        )


@dataclass(frozen=True, slots=True)
class RerankedRetrievalResponse:
    """Evidence reordered by one explicit reranker model."""

    query: str
    model: str
    results: tuple[RerankedEvidence, ...]
    namespace: str | None = None

    def __post_init__(self) -> None:
        """Validate final reranked ordering and identity."""
        _validate_required_text(
            "query",
            self.query,
        )
        _validate_required_text(
            "model",
            self.model,
        )
        _validate_optional_text(
            "namespace",
            self.namespace,
        )

        if not self.results:
            raise ValueError("reranked results must not be empty")

        chunk_ids = tuple(item.evidence.chunk_id for item in self.results)

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("reranked evidence chunk identifiers must be unique")

        ranks = tuple(item.rank for item in self.results)

        expected_ranks = tuple(
            range(
                1,
                len(self.results) + 1,
            )
        )

        if ranks != expected_ranks:
            raise ValueError("reranked results must use contiguous rank order")

"""Domain models for bounded knowledge-source ingestion."""

from dataclasses import dataclass

from ai_engineering_agent_platform.domain.retrieval import (
    RetrievalMetadataItem,
)


def _require_non_empty_string(
    field_name: str,
    value: object,
) -> None:
    """Require one meaningful string value."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")

    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


@dataclass(frozen=True, slots=True)
class KnowledgeSource:
    """Immutable caller-supplied source payload for knowledge ingestion.

    ``source_ref`` is opaque provenance metadata. Constructing this object does
    not authorize filesystem access or network retrieval.
    """

    document_id: str
    content: bytes
    media_type: str
    source_ref: str
    title: str | None = None
    metadata: tuple[RetrievalMetadataItem, ...] = ()

    def __post_init__(self) -> None:
        """Validate source identity, payload shape, and metadata invariants."""
        _require_non_empty_string(
            "document_id",
            self.document_id,
        )
        _require_non_empty_string(
            "media_type",
            self.media_type,
        )
        _require_non_empty_string(
            "source_ref",
            self.source_ref,
        )

        if not isinstance(self.content, bytes):
            raise ValueError("content must be bytes")

        if not self.content:
            raise ValueError("content must not be empty")

        if self.title is not None:
            _require_non_empty_string(
                "title",
                self.title,
            )

        if not isinstance(self.metadata, tuple):
            raise ValueError("metadata must be a tuple")

        if any(not isinstance(item, RetrievalMetadataItem) for item in self.metadata):
            raise ValueError("metadata must contain RetrievalMetadataItem values")

        metadata_keys = tuple(item.key for item in self.metadata)

        if len(metadata_keys) != len(set(metadata_keys)):
            raise ValueError("metadata keys must be unique")

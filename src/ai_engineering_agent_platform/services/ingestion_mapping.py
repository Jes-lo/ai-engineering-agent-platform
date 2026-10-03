"""Pure mapping from bounded source bytes into knowledge documents."""

from ai_engineering_agent_platform.domain import (
    KnowledgeDocument,
)
from ai_engineering_agent_platform.domain.ingestion import (
    KnowledgeSource,
)

SUPPORTED_TEXT_MEDIA_TYPES = frozenset(
    {
        "text/markdown",
        "text/plain",
    }
)


def normalize_knowledge_media_type(
    media_type: str,
) -> str:
    """Return one canonical supported media type."""
    normalized = media_type.strip().lower()

    if normalized not in SUPPORTED_TEXT_MEDIA_TYPES:
        raise ValueError("unsupported knowledge source media type")

    return normalized


def parse_knowledge_source(
    source: KnowledgeSource,
    *,
    max_source_bytes: int,
) -> KnowledgeDocument:
    """Decode one bounded caller-supplied text source.

    This function performs no filesystem access and no network access.
    ``source_ref`` remains opaque provenance metadata.
    """
    if isinstance(max_source_bytes, bool) or not isinstance(max_source_bytes, int):
        raise ValueError("max_source_bytes must be an integer")

    if max_source_bytes <= 0:
        raise ValueError("max_source_bytes must be positive")

    if len(source.content) > max_source_bytes:
        raise ValueError("knowledge source exceeds maximum size")

    normalize_knowledge_media_type(
        source.media_type,
    )

    try:
        text = source.content.decode(
            "utf-8",
            errors="strict",
        )
    except UnicodeDecodeError as exc:
        raise ValueError("knowledge source content must be valid UTF-8") from exc

    if "\x00" in text:
        raise ValueError("knowledge source content must not contain NUL")

    return KnowledgeDocument(
        document_id=source.document_id,
        text=text,
        source_ref=source.source_ref,
        title=source.title,
        metadata=source.metadata,
    )

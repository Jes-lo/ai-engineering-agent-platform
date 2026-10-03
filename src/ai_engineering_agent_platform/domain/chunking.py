"""Deterministic infrastructure-independent text chunking."""

from dataclasses import dataclass

from ai_engineering_agent_platform.domain.retrieval import (
    DocumentChunk,
    KnowledgeDocument,
)


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


@dataclass(frozen=True, slots=True)
class TextChunkingConfig:
    """Immutable deterministic character-window chunking configuration.

    ``max_chars`` defines the raw source window size.

    ``overlap_chars`` defines the raw character overlap between consecutive
    windows. Leading and trailing whitespace may be excluded from emitted
    chunks, so the visible chunk overlap may be smaller than this raw-window
    overlap.
    """

    max_chars: int
    overlap_chars: int = 0

    def __post_init__(self) -> None:
        """Validate deterministic chunk-window invariants."""
        _validate_positive_integer(
            "max_chars",
            self.max_chars,
        )
        _validate_non_negative_integer(
            "overlap_chars",
            self.overlap_chars,
        )

        if self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")


def _content_bounds(
    text: str,
    start: int,
    end: int,
) -> tuple[int, int] | None:
    """Return non-whitespace bounds within an exact source window."""
    content_start = start
    content_end = end

    while content_start < content_end and text[content_start].isspace():
        content_start += 1

    while content_end > content_start and text[content_end - 1].isspace():
        content_end -= 1

    if content_start == content_end:
        return None

    return (
        content_start,
        content_end,
    )


def _chunk_id(
    *,
    document_id: str,
    index: int,
    start_char: int,
    end_char: int,
) -> str:
    """Build one deterministic chunk identifier."""
    return f"{document_id}:chunk:{index}:{start_char}:{end_char}"


@dataclass(frozen=True, slots=True)
class DeterministicTextChunker:
    """Split canonical text into reproducible exact-source chunks.

    The chunker performs no text normalization, tokenization, network access,
    embedding generation, persistence, or model invocation.

    Every emitted ``DocumentChunk.text`` is an exact Python character slice
    of the source document at ``start_char:end_char``.
    """

    config: TextChunkingConfig

    def chunk(
        self,
        document: KnowledgeDocument,
    ) -> tuple[DocumentChunk, ...]:
        """Return deterministic chunks for one canonical document."""
        text = document.text
        text_length = len(text)

        raw_start = 0
        chunk_index = 0
        chunks: list[DocumentChunk] = []

        while raw_start < text_length:
            raw_end = min(
                raw_start + self.config.max_chars,
                text_length,
            )

            bounds = _content_bounds(
                text,
                raw_start,
                raw_end,
            )

            if bounds is not None:
                (
                    start_char,
                    end_char,
                ) = bounds

                chunk_text = text[start_char:end_char]

                chunks.append(
                    DocumentChunk(
                        chunk_id=_chunk_id(
                            document_id=document.document_id,
                            index=chunk_index,
                            start_char=start_char,
                            end_char=end_char,
                        ),
                        document_id=document.document_id,
                        text=chunk_text,
                        index=chunk_index,
                        start_char=start_char,
                        end_char=end_char,
                        source_ref=document.source_ref,
                        title=document.title,
                        metadata=document.metadata,
                    )
                )

                chunk_index += 1

            if raw_end == text_length:
                break

            raw_start = raw_end - self.config.overlap_chars

        return tuple(chunks)

"""Tests for deterministic infrastructure-independent text chunking."""

from dataclasses import FrozenInstanceError

import pytest

from ai_engineering_agent_platform.domain import (
    DeterministicTextChunker,
    KnowledgeDocument,
    RetrievalMetadataItem,
    TextChunkingConfig,
)


def _document(
    text: str,
) -> KnowledgeDocument:
    """Return one deterministic synthetic document."""
    return KnowledgeDocument(
        document_id="doc-1",
        text=text,
        source_ref="synthetic://doc-1",
        title="Synthetic Document",
        metadata=(
            RetrievalMetadataItem(
                key="category",
                value="synthetic",
            ),
        ),
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        (
            {
                "max_chars": True,
            },
            "max_chars must be an integer",
        ),
        (
            {
                "max_chars": 0,
            },
            "max_chars must be positive",
        ),
        (
            {
                "max_chars": -1,
            },
            "max_chars must be positive",
        ),
        (
            {
                "max_chars": 10,
                "overlap_chars": True,
            },
            "overlap_chars must be an integer",
        ),
        (
            {
                "max_chars": 10,
                "overlap_chars": -1,
            },
            "overlap_chars must be non-negative",
        ),
        (
            {
                "max_chars": 10,
                "overlap_chars": 10,
            },
            "overlap_chars must be smaller than max_chars",
        ),
        (
            {
                "max_chars": 10,
                "overlap_chars": 11,
            },
            "overlap_chars must be smaller than max_chars",
        ),
    ),
)
def test_chunking_config_rejects_invalid_values(
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Chunk-window configuration must fail closed."""
    with pytest.raises(
        ValueError,
        match=match,
    ):
        TextChunkingConfig(
            **kwargs,  # type: ignore[arg-type]
        )


def test_chunking_config_is_immutable() -> None:
    """Chunking configuration should not mutate after construction."""
    config = TextChunkingConfig(
        max_chars=100,
        overlap_chars=10,
    )

    with pytest.raises(
        FrozenInstanceError,
    ):
        config.max_chars = 200  # type: ignore[misc]


def test_fixed_windows_are_deterministic() -> None:
    """Repeated chunking must return exactly equal chunks."""
    document = _document("abcdefghij")
    chunker = DeterministicTextChunker(
        TextChunkingConfig(
            max_chars=6,
            overlap_chars=2,
        )
    )

    first = chunker.chunk(document)
    second = chunker.chunk(document)

    assert first == second

    assert tuple(
        (
            chunk.index,
            chunk.start_char,
            chunk.end_char,
            chunk.text,
        )
        for chunk in first
    ) == (
        (
            0,
            0,
            6,
            "abcdef",
        ),
        (
            1,
            4,
            10,
            "efghij",
        ),
    )


def test_chunks_are_exact_source_slices() -> None:
    """Every emitted chunk must point to its exact original characters."""
    document = _document("abcdefghijklmno")
    chunker = DeterministicTextChunker(
        TextChunkingConfig(
            max_chars=5,
            overlap_chars=2,
        )
    )

    chunks = chunker.chunk(document)

    assert chunks

    for chunk in chunks:
        assert document.text[chunk.start_char : chunk.end_char] == chunk.text

        assert chunk.end_char - chunk.start_char == len(chunk.text)


def test_chunk_ids_are_deterministic_and_range_specific() -> None:
    """Chunk identifiers should encode stable source ranges."""
    chunks = DeterministicTextChunker(
        TextChunkingConfig(
            max_chars=6,
            overlap_chars=2,
        )
    ).chunk(_document("abcdefghij"))

    assert tuple(chunk.chunk_id for chunk in chunks) == (
        "doc-1:chunk:0:0:6",
        "doc-1:chunk:1:4:10",
    )


def test_boundary_whitespace_is_trimmed_with_offsets_adjusted() -> None:
    """Whitespace outside visible content must not corrupt provenance."""
    document = _document("  abc   def  ")
    chunks = DeterministicTextChunker(
        TextChunkingConfig(
            max_chars=8,
        )
    ).chunk(document)

    assert tuple(
        (
            chunk.start_char,
            chunk.end_char,
            chunk.text,
        )
        for chunk in chunks
    ) == (
        (
            2,
            5,
            "abc",
        ),
        (
            8,
            11,
            "def",
        ),
    )

    for chunk in chunks:
        assert document.text[chunk.start_char : chunk.end_char] == chunk.text


def test_whitespace_only_windows_are_not_emitted() -> None:
    """Raw windows containing no content should not become empty chunks."""
    document = _document("abc      def")
    chunks = DeterministicTextChunker(
        TextChunkingConfig(
            max_chars=3,
        )
    ).chunk(document)

    assert tuple(
        (
            chunk.index,
            chunk.start_char,
            chunk.end_char,
            chunk.text,
        )
        for chunk in chunks
    ) == (
        (
            0,
            0,
            3,
            "abc",
        ),
        (
            1,
            9,
            12,
            "def",
        ),
    )


def test_provenance_title_and_metadata_are_propagated() -> None:
    """Chunking must not lose document provenance or metadata."""
    document = _document("abcdefgh")

    chunks = DeterministicTextChunker(
        TextChunkingConfig(
            max_chars=4,
        )
    ).chunk(document)

    assert chunks

    for chunk in chunks:
        assert chunk.document_id == document.document_id
        assert chunk.source_ref == document.source_ref
        assert chunk.title == document.title
        assert chunk.metadata == document.metadata


def test_unicode_offsets_use_python_character_positions() -> None:
    """Offsets should remain exact for Unicode source text."""
    document = _document("áβ🙂XYZ")

    chunks = DeterministicTextChunker(
        TextChunkingConfig(
            max_chars=3,
        )
    ).chunk(document)

    assert tuple(
        (
            chunk.start_char,
            chunk.end_char,
            chunk.text,
        )
        for chunk in chunks
    ) == (
        (
            0,
            3,
            "áβ🙂",
        ),
        (
            3,
            6,
            "XYZ",
        ),
    )


def test_source_ref_remains_opaque_data() -> None:
    """Chunking must propagate source references without interpreting them."""
    document = KnowledgeDocument(
        document_id="doc-opaque",
        text="synthetic",
        source_ref=("https://example.invalid/path?not-fetched=true#opaque"),
    )

    chunks = DeterministicTextChunker(
        TextChunkingConfig(
            max_chars=20,
        )
    ).chunk(document)

    assert len(chunks) == 1
    assert chunks[0].source_ref == document.source_ref


def test_chunker_is_publicly_exported() -> None:
    """Chunking primitives should be available from the domain package."""
    import ai_engineering_agent_platform.domain as domain

    assert {
        "DeterministicTextChunker",
        "TextChunkingConfig",
    } <= set(domain.__all__)

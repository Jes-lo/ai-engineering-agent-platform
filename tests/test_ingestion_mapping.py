"""Tests for bounded text knowledge-source parsing."""

import pytest

from ai_engineering_agent_platform.domain import (
    KnowledgeSource,
    RetrievalMetadataItem,
)
from ai_engineering_agent_platform.services import (
    normalize_knowledge_media_type,
    parse_knowledge_source,
)


def _source(
    *,
    content: bytes = b"alpha\nbeta",
    media_type: str = "text/plain",
) -> KnowledgeSource:
    """Return one deterministic synthetic source."""
    return KnowledgeSource(
        document_id="doc-1",
        content=content,
        media_type=media_type,
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
    ("value", "expected"),
    (
        ("text/plain", "text/plain"),
        (" TEXT/PLAIN ", "text/plain"),
        ("text/markdown", "text/markdown"),
        ("TEXT/MARKDOWN", "text/markdown"),
    ),
)
def test_supported_media_types_are_canonicalized(
    value: str,
    expected: str,
) -> None:
    """Supported text media types should have deterministic identity."""
    assert normalize_knowledge_media_type(value) == expected


def test_plain_text_source_maps_to_existing_knowledge_document() -> None:
    """UTF-8 text should preserve identity, text, and provenance."""
    source = _source()

    document = parse_knowledge_source(
        source,
        max_source_bytes=1024,
    )

    assert document.document_id == source.document_id
    assert document.text == "alpha\nbeta"
    assert document.source_ref == source.source_ref
    assert document.title == source.title
    assert document.metadata == source.metadata


def test_markdown_is_preserved_as_source_text() -> None:
    """Markdown support should decode content without hidden rewriting."""
    source = _source(
        content=b"# Heading\n\nBody **bold**.",
        media_type="text/markdown",
    )

    document = parse_knowledge_source(
        source,
        max_source_bytes=1024,
    )

    assert document.text == "# Heading\n\nBody **bold**."


def test_unsupported_media_type_fails_closed() -> None:
    """Binary or unsupported formats must not be guessed."""
    with pytest.raises(
        ValueError,
        match="unsupported knowledge source media type",
    ):
        parse_knowledge_source(
            _source(
                media_type="application/pdf",
            ),
            max_source_bytes=1024,
        )


def test_media_type_parameters_are_not_silently_guessed() -> None:
    """The initial ingestion contract intentionally uses exact media types."""
    with pytest.raises(
        ValueError,
        match="unsupported knowledge source media type",
    ):
        parse_knowledge_source(
            _source(
                media_type="text/plain; charset=utf-8",
            ),
            max_source_bytes=1024,
        )


def test_invalid_utf8_fails_closed() -> None:
    """Malformed text bytes must fail before chunking or indexing."""
    with pytest.raises(
        ValueError,
        match="must be valid UTF-8",
    ):
        parse_knowledge_source(
            _source(
                content=b"\xff\xfe",
            ),
            max_source_bytes=1024,
        )


def test_source_size_limit_is_enforced_before_decoding() -> None:
    """Oversized caller payloads should be rejected deterministically."""
    with pytest.raises(
        ValueError,
        match="exceeds maximum size",
    ):
        parse_knowledge_source(
            _source(
                content=b"12345",
            ),
            max_source_bytes=4,
        )


def test_exact_size_limit_is_allowed() -> None:
    """The configured maximum is inclusive."""
    document = parse_knowledge_source(
        _source(
            content=b"12345",
        ),
        max_source_bytes=5,
    )

    assert document.text == "12345"


def test_nul_character_fails_closed() -> None:
    """NUL-bearing text should not enter downstream document processing."""
    with pytest.raises(
        ValueError,
        match="must not contain NUL",
    ):
        parse_knowledge_source(
            _source(
                content=b"alpha\x00beta",
            ),
            max_source_bytes=1024,
        )


@pytest.mark.parametrize(
    "invalid",
    (
        True,
        0,
        -1,
    ),
)
def test_invalid_source_size_configuration_is_rejected(
    invalid: object,
) -> None:
    """Resource bounds must be explicit positive integers."""
    match = "must be an integer" if isinstance(invalid, bool) else "must be positive"

    with pytest.raises(
        ValueError,
        match=match,
    ):
        parse_knowledge_source(
            _source(),
            max_source_bytes=invalid,  # type: ignore[arg-type]
        )


def test_source_ref_remains_opaque_and_is_never_fetched() -> None:
    """Parsing must preserve URL-like references purely as provenance."""
    source = KnowledgeSource(
        document_id="opaque",
        content=b"local caller bytes",
        media_type="text/plain",
        source_ref=("https://example.invalid/must-not-be-fetched"),
    )

    document = parse_knowledge_source(
        source,
        max_source_bytes=1024,
    )

    assert document.source_ref == source.source_ref
    assert document.text == "local caller bytes"


def test_ingestion_mapping_is_publicly_exported() -> None:
    """Source parsing helpers should be public through services."""
    import ai_engineering_agent_platform.services as services

    assert {
        "normalize_knowledge_media_type",
        "parse_knowledge_source",
    } <= set(services.__all__)

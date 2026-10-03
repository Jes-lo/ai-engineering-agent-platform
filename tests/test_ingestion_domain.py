"""Tests for bounded knowledge-source domain contracts."""

import pytest

from ai_engineering_agent_platform.domain import (
    KnowledgeSource,
    RetrievalMetadataItem,
)


def test_knowledge_source_preserves_valid_payload() -> None:
    """Valid source identity, bytes, and provenance should be immutable."""
    metadata = (
        RetrievalMetadataItem(
            key="category",
            value="synthetic",
        ),
    )

    source = KnowledgeSource(
        document_id="doc-1",
        content=b"alpha",
        media_type="text/plain",
        source_ref="synthetic://doc-1",
        title="Synthetic",
        metadata=metadata,
    )

    assert source.document_id == "doc-1"
    assert source.content == b"alpha"
    assert source.media_type == "text/plain"
    assert source.source_ref == "synthetic://doc-1"
    assert source.title == "Synthetic"
    assert source.metadata is metadata


@pytest.mark.parametrize(
    ("field_name", "kwargs", "match"),
    (
        (
            "document_id",
            {"document_id": " "},
            "document_id must not be empty",
        ),
        (
            "media_type",
            {"media_type": " "},
            "media_type must not be empty",
        ),
        (
            "source_ref",
            {"source_ref": " "},
            "source_ref must not be empty",
        ),
        (
            "content",
            {"content": b""},
            "content must not be empty",
        ),
        (
            "title",
            {"title": " "},
            "title must not be empty",
        ),
    ),
)
def test_knowledge_source_rejects_empty_required_values(
    field_name: str,
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Identity and source payload fields must remain meaningful."""
    values: dict[str, object] = {
        "document_id": "doc",
        "content": b"text",
        "media_type": "text/plain",
        "source_ref": "synthetic://doc",
        "title": None,
    }

    values.update(kwargs)

    with pytest.raises(
        ValueError,
        match=match,
    ):
        KnowledgeSource(
            **values,  # type: ignore[arg-type]
        )

    assert field_name


def test_knowledge_source_requires_bytes() -> None:
    """Text must cross the ingestion boundary as explicit bytes."""
    with pytest.raises(
        ValueError,
        match="content must be bytes",
    ):
        KnowledgeSource(
            document_id="doc",
            content="text",  # type: ignore[arg-type]
            media_type="text/plain",
            source_ref="synthetic://doc",
        )


def test_knowledge_source_rejects_duplicate_metadata_keys() -> None:
    """Source metadata keys must remain unique before parsing."""
    metadata = (
        RetrievalMetadataItem(
            key="duplicate",
            value=1,
        ),
        RetrievalMetadataItem(
            key="duplicate",
            value=2,
        ),
    )

    with pytest.raises(
        ValueError,
        match="metadata keys must be unique",
    ):
        KnowledgeSource(
            document_id="doc",
            content=b"text",
            media_type="text/plain",
            source_ref="synthetic://doc",
            metadata=metadata,
        )


def test_knowledge_source_is_publicly_exported() -> None:
    """The source boundary should be public through the domain package."""
    import ai_engineering_agent_platform.domain as domain

    assert "KnowledgeSource" in domain.__all__
    assert hasattr(
        domain,
        "KnowledgeSource",
    )

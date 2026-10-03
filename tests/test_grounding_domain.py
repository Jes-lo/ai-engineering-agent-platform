"""Tests for infrastructure-independent grounding domain primitives."""

import pytest

from ai_engineering_agent_platform.domain.grounding import (
    GroundedAnswer,
    GroundedAnswerStatus,
    GroundedCitation,
)
from ai_engineering_agent_platform.domain.retrieval import RetrievedEvidence


def _evidence(
    *,
    chunk_id: str = "chunk-1",
) -> RetrievedEvidence:
    """Return deterministic synthetic retrieved evidence."""
    text = "synthetic evidence"

    return RetrievedEvidence(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=text,
        score=0.9,
        rank=1,
        start_char=0,
        end_char=len(text),
        source_ref="synthetic://doc-1",
        title="Synthetic Document",
    )


def test_answered_grounding_preserves_retrieved_evidence() -> None:
    """Citations should retain the validated retrieval object unchanged."""
    evidence = _evidence()

    citation = GroundedCitation(
        citation_id="C1",
        evidence=evidence,
    )

    answer = GroundedAnswer(
        query="What is synthetic?",
        status=GroundedAnswerStatus.ANSWERED,
        answer="It is synthetic. [[C1]]",
        citations=(citation,),
        namespace="synthetic",
    )

    assert answer.citations[0].evidence is evidence
    assert answer.citations[0].citation_id == "C1"


@pytest.mark.parametrize(
    "citation_id",
    [
        "",
        "1",
        "C",
        "C0",
        "C-1",
        "C1.0",
        "C01",
        "C\u0661",
    ],
)
def test_invalid_citation_identifier_is_rejected(
    citation_id: str,
) -> None:
    """Citation identifiers are platform-owned positive C numbers."""
    with pytest.raises(ValueError):
        GroundedCitation(
            citation_id=citation_id,
            evidence=_evidence(),
        )


def test_answered_result_requires_citations() -> None:
    """An asserted grounded answer cannot exist without provenance."""
    with pytest.raises(
        ValueError,
        match="answered result must contain citations",
    ):
        GroundedAnswer(
            query="Synthetic?",
            status=GroundedAnswerStatus.ANSWERED,
            answer="Synthetic.",
            citations=(),
        )


def test_abstention_contains_no_answer_or_citations() -> None:
    """Explicit abstention should carry no fabricated answer provenance."""
    result = GroundedAnswer(
        query="Unknown?",
        status=GroundedAnswerStatus.ABSTAINED,
        answer=None,
        citations=(),
    )

    assert result.answer is None
    assert result.citations == ()


def test_abstention_rejects_answer_text() -> None:
    """An abstained result cannot simultaneously claim an answer."""
    with pytest.raises(
        ValueError,
        match="abstained result must not contain answer text",
    ):
        GroundedAnswer(
            query="Unknown?",
            status=GroundedAnswerStatus.ABSTAINED,
            answer="Maybe.",
            citations=(),
        )


def test_duplicate_grounded_citations_are_rejected() -> None:
    """Resolved citation collections should contain unique identifiers."""
    citation = GroundedCitation(
        citation_id="C1",
        evidence=_evidence(),
    )

    with pytest.raises(
        ValueError,
        match="grounded citations must use unique identifiers",
    ):
        GroundedAnswer(
            query="Synthetic?",
            status=GroundedAnswerStatus.ANSWERED,
            answer="Synthetic. [[C1]]",
            citations=(
                citation,
                citation,
            ),
        )


def test_grounding_domain_is_publicly_exported() -> None:
    """Grounding domain primitives should be available from the domain package."""
    import ai_engineering_agent_platform.domain as domain

    assert {
        "GroundedAnswer",
        "GroundedAnswerStatus",
        "GroundedCitation",
    } <= set(domain.__all__)

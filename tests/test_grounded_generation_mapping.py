"""Tests for grounded context assembly and citation resolution."""

import json

import pytest

from ai_engineering_agent_platform.contracts import (
    FinishReason,
    LLMMessage,
    LLMResponse,
    MessageRole,
)
from ai_engineering_agent_platform.domain import (
    ProviderExecutionError,
    RerankedEvidence,
    RerankedRetrievalResponse,
    RetrievalResponse,
    RetrievedEvidence,
)
from ai_engineering_agent_platform.domain.grounding import (
    GroundedAnswerStatus,
)
from ai_engineering_agent_platform.services.grounded_generation_mapping import (
    INSUFFICIENT_EVIDENCE_SENTINEL,
    build_grounded_answer,
    build_grounded_llm_request,
    ordered_grounding_evidence,
)


def _evidence(
    *,
    chunk_id: str,
    text: str,
    rank: int,
    start_char: int,
) -> RetrievedEvidence:
    """Return deterministic citation-ready evidence."""
    return RetrievedEvidence(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=text,
        score=1.0 - (rank / 10),
        rank=rank,
        start_char=start_char,
        end_char=start_char + len(text),
        source_ref="synthetic://doc-1",
        title="Synthetic Document",
    )


def _retrieval() -> RetrievalResponse:
    """Return two ordered retrieval candidates."""
    return RetrievalResponse(
        query="Which evidence is available?",
        namespace="synthetic",
        results=(
            _evidence(
                chunk_id="chunk-a",
                text="alpha evidence",
                rank=1,
                start_char=0,
            ),
            _evidence(
                chunk_id="chunk-b",
                text="beta evidence",
                rank=2,
                start_char=20,
            ),
        ),
    )


def _reranked() -> RerankedRetrievalResponse:
    """Return the same evidence in reversed reranker order."""
    retrieval = _retrieval()

    return RerankedRetrievalResponse(
        query=retrieval.query,
        model="synthetic-reranker",
        namespace=retrieval.namespace,
        results=(
            RerankedEvidence(
                evidence=retrieval.results[1],
                rerank_score=9.0,
                rank=1,
            ),
            RerankedEvidence(
                evidence=retrieval.results[0],
                rerank_score=7.0,
                rank=2,
            ),
        ),
    )


def _llm_response(
    content: str,
) -> LLMResponse:
    """Return one normalized synthetic model response."""
    return LLMResponse(
        model="generator-model",
        message=LLMMessage(
            role=MessageRole.ASSISTANT,
            content=content,
        ),
        finish_reason=FinishReason.STOP,
    )


def test_context_assembly_uses_only_selected_evidence_payload() -> None:
    """Prompt payload should exclude provider-owned source resolution data."""
    request = build_grounded_llm_request(
        _retrieval(),
        model="generator-model",
        temperature=0.0,
        max_output_tokens=256,
    )

    assert request.model == "generator-model"
    assert request.temperature == 0.0
    assert request.max_output_tokens == 256
    assert len(request.messages) == 2
    assert request.messages[0].role is MessageRole.SYSTEM
    assert request.messages[1].role is MessageRole.USER

    payload = json.loads(request.messages[1].content)

    assert payload["query"] == "Which evidence is available?"

    assert payload["evidence"] == [
        {
            "citation_id": "C1",
            "text": "alpha evidence",
            "title": "Synthetic Document",
        },
        {
            "citation_id": "C2",
            "text": "beta evidence",
            "title": "Synthetic Document",
        },
    ]

    assert "source_ref" not in request.messages[1].content
    assert "document_id" not in request.messages[1].content


def test_reranked_order_becomes_grounding_citation_order() -> None:
    """Grounding should honor optional reranker order without losing evidence."""
    evidence = ordered_grounding_evidence(_reranked())

    assert tuple(item.chunk_id for item in evidence) == (
        "chunk-b",
        "chunk-a",
    )

    request = build_grounded_llm_request(
        _reranked(),
        model="generator-model",
    )

    payload = json.loads(request.messages[1].content)

    assert payload["evidence"][0]["citation_id"] == "C1"
    assert payload["evidence"][0]["text"] == "beta evidence"


def test_empty_retrieval_cannot_build_generation_request() -> None:
    """Grounded generation must not run without retrieved evidence."""
    retrieval = RetrievalResponse(
        query="Unknown?",
        results=(),
    )

    with pytest.raises(
        ValueError,
        match="without evidence",
    ):
        build_grounded_llm_request(
            retrieval,
            model="generator-model",
        )


def test_valid_citations_resolve_only_to_retrieved_evidence() -> None:
    """Model markers should resolve to project-owned provenance objects."""
    retrieval = _retrieval()

    result = build_grounded_answer(
        retrieval,
        _llm_response(
            "Beta is available [[C2]], then alpha [[C1]], "
            "and beta remains available [[C2]]."
        ),
    )

    assert result.status is GroundedAnswerStatus.ANSWERED

    assert tuple(citation.citation_id for citation in result.citations) == (
        "C2",
        "C1",
    )

    assert tuple(citation.evidence.chunk_id for citation in result.citations) == (
        "chunk-b",
        "chunk-a",
    )

    assert result.citations[0].evidence is retrieval.results[1]
    assert result.citations[1].evidence is retrieval.results[0]


def test_unknown_citation_fails_closed() -> None:
    """The model cannot cite evidence outside the retrieved set."""
    with pytest.raises(
        ProviderExecutionError,
        match="cited unknown evidence identifier",
    ):
        build_grounded_answer(
            _retrieval(),
            _llm_response("Unsupported citation [[C3]]."),
        )


def test_malformed_citation_fails_closed() -> None:
    """Citation-like syntax must use the exact platform marker grammar."""
    with pytest.raises(
        ProviderExecutionError,
        match="malformed citation marker",
    ):
        build_grounded_answer(
            _retrieval(),
            _llm_response("Malformed citation [[C01]]."),
        )


@pytest.mark.parametrize(
    "content",
    [
        "Valid [[C1]] plus foreign [[X1]].",
        "Valid [[C1]] plus lowercase [[c1]].",
        "Valid [[C1]] plus zero identifier [[C0]].",
    ],
)
def test_non_platform_double_bracket_marker_fails_closed(
    content: str,
) -> None:
    """Only exact platform-owned citation markers may survive parsing."""
    with pytest.raises(
        ProviderExecutionError,
        match="malformed citation marker",
    ):
        build_grounded_answer(
            _retrieval(),
            _llm_response(content),
        )


def test_answer_without_citation_fails_closed() -> None:
    """A normal answer cannot claim grounding without evidence references."""
    with pytest.raises(
        ProviderExecutionError,
        match="grounded answer without citations",
    ):
        build_grounded_answer(
            _retrieval(),
            _llm_response("An uncited answer."),
        )


def test_exact_insufficient_evidence_token_abstains() -> None:
    """The model may explicitly abstain without fabricating citations."""
    result = build_grounded_answer(
        _retrieval(),
        _llm_response(INSUFFICIENT_EVIDENCE_SENTINEL),
    )

    assert result.status is GroundedAnswerStatus.ABSTAINED
    assert result.answer is None
    assert result.citations == ()


def test_ambiguous_insufficient_evidence_output_fails_closed() -> None:
    """The abstention token cannot be mixed with asserted answer text."""
    with pytest.raises(
        ProviderExecutionError,
        match="ambiguous insufficient-evidence output",
    ):
        build_grounded_answer(
            _retrieval(),
            _llm_response(f"{INSUFFICIENT_EVIDENCE_SENTINEL} but maybe alpha [[C1]]."),
        )


def test_grounded_generation_mapping_is_publicly_exported() -> None:
    """Grounding mapping helpers should be public service utilities."""
    import ai_engineering_agent_platform.services as services

    assert {
        "build_grounded_answer",
        "build_grounded_llm_request",
        "ordered_grounding_evidence",
    } <= set(services.__all__)

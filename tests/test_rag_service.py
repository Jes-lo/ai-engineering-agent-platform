"""Tests for provider-neutral end-to-end RAG orchestration."""

import pytest

from ai_engineering_agent_platform.contracts import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
    FinishReason,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    MessageRole,
    ProviderDescriptor,
    ProviderKind,
    RerankRequest,
    RerankResponse,
    RerankResult,
    TokenUsage,
    VectorDeleteRequest,
    VectorMetadataItem,
    VectorQueryRequest,
    VectorQueryResponse,
    VectorQueryResult,
    VectorUpsertRequest,
)
from ai_engineering_agent_platform.domain import (
    GroundedAnswerStatus,
    ProviderExecutionError,
    RerankedRetrievalResponse,
    RetrievalRequest,
)
from ai_engineering_agent_platform.domain.guardrails import (
    GuardrailCategory,
    GuardrailFinding,
    GuardrailSeverity,
    GuardrailStage,
    GuardrailSubject,
)
from ai_engineering_agent_platform.services import (
    GroundedGenerationService,
    RAGService,
    RerankingService,
    RetrievalService,
)
from ai_engineering_agent_platform.services.guardrails import (
    GuardrailLiteralPattern,
    GuardrailPolicy,
    GuardrailRule,
    GuardrailService,
    LiteralPatternGuardrailRule,
)


class SyntheticEmbeddingProvider:
    """Deterministic embedding provider recording execution order."""

    def __init__(
        self,
        events: list[str],
    ) -> None:
        """Store shared execution events."""
        self.events = events
        self.requests: list[EmbeddingRequest] = []

    @property
    def descriptor(self) -> ProviderDescriptor:
        """Return deterministic provider identity."""
        return ProviderDescriptor(
            name="synthetic-embedding",
            kind=ProviderKind.EMBEDDING,
        )

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse:
        """Return one deterministic query embedding."""
        self.events.append("embed")
        self.requests.append(request)

        return EmbeddingResponse(
            model="embedding-model",
            embeddings=(
                EmbeddingVector(
                    values=(
                        0.1,
                        0.2,
                        0.3,
                    ),
                ),
            ),
            input_tokens=3,
        )


class SyntheticVectorStore:
    """Deterministic vector store recording semantic query execution."""

    def __init__(
        self,
        events: list[str],
        *,
        response: VectorQueryResponse,
        error: Exception | None = None,
    ) -> None:
        """Store deterministic query behavior."""
        self.events = events
        self.response = response
        self.error = error
        self.query_requests: list[VectorQueryRequest] = []

    @property
    def descriptor(self) -> ProviderDescriptor:
        """Return deterministic provider identity."""
        return ProviderDescriptor(
            name="synthetic-vector-store",
            kind=ProviderKind.VECTOR_STORE,
        )

    async def upsert(
        self,
        request: VectorUpsertRequest,
    ) -> None:
        """Reject unexpected write use in retrieval tests."""
        raise AssertionError(f"unexpected vector upsert: {request!r}")

    async def query(
        self,
        request: VectorQueryRequest,
    ) -> VectorQueryResponse:
        """Return configured semantic query results."""
        self.events.append("vector_query")
        self.query_requests.append(request)

        if self.error is not None:
            raise self.error

        return self.response

    async def delete(
        self,
        request: VectorDeleteRequest,
    ) -> None:
        """Reject unexpected delete use in retrieval tests."""
        raise AssertionError(f"unexpected vector delete: {request!r}")


class SyntheticReranker:
    """Deterministic reranker recording execution order."""

    def __init__(
        self,
        events: list[str],
        *,
        response: RerankResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        """Store deterministic reranker behavior."""
        self.events = events
        self.response = response
        self.error = error
        self.requests: list[RerankRequest] = []

    @property
    def descriptor(self) -> ProviderDescriptor:
        """Return deterministic provider identity."""
        return ProviderDescriptor(
            name="synthetic-reranker",
            kind=ProviderKind.RERANKER,
        )

    async def rerank(
        self,
        request: RerankRequest,
    ) -> RerankResponse:
        """Return configured reranker output."""
        self.events.append("rerank")
        self.requests.append(request)

        if self.error is not None:
            raise self.error

        if self.response is None:
            raise AssertionError("synthetic reranker response not configured")

        return self.response


class SyntheticLLMProvider:
    """Deterministic LLM provider recording execution order."""

    def __init__(
        self,
        events: list[str],
        *,
        response: LLMResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        """Store deterministic generation behavior."""
        self.events = events
        self.response = response
        self.error = error
        self.requests: list[LLMRequest] = []

    @property
    def descriptor(self) -> ProviderDescriptor:
        """Return deterministic provider identity."""
        return ProviderDescriptor(
            name="synthetic-llm",
            kind=ProviderKind.LLM,
        )

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """Return configured grounded-generation output."""
        self.events.append("llm")
        self.requests.append(request)

        if self.error is not None:
            raise self.error

        if self.response is None:
            raise AssertionError("synthetic LLM response not configured")

        return self.response


def _metadata(
    *,
    document_id: str,
    chunk_index: int,
    start_char: int,
    end_char: int,
) -> tuple[VectorMetadataItem, ...]:
    """Return valid project-owned retrieval provenance metadata."""
    return (
        VectorMetadataItem(
            key="retrieval.document_id",
            value=document_id,
        ),
        VectorMetadataItem(
            key="retrieval.chunk_index",
            value=chunk_index,
        ),
        VectorMetadataItem(
            key="retrieval.start_char",
            value=start_char,
        ),
        VectorMetadataItem(
            key="retrieval.end_char",
            value=end_char,
        ),
        VectorMetadataItem(
            key="retrieval.source_ref",
            value=f"synthetic://{document_id}",
        ),
        VectorMetadataItem(
            key="retrieval.title",
            value="Synthetic Document",
        ),
    )


def _vector_response() -> VectorQueryResponse:
    """Return two deterministic retrieval candidates."""
    alpha = "alpha evidence"
    beta = "beta evidence"

    return VectorQueryResponse(
        results=(
            VectorQueryResult(
                record_id="chunk-a",
                score=0.91,
                rank=1,
                text=alpha,
                metadata=_metadata(
                    document_id="doc-a",
                    chunk_index=0,
                    start_char=0,
                    end_char=len(alpha),
                ),
            ),
            VectorQueryResult(
                record_id="chunk-b",
                score=0.72,
                rank=2,
                text=beta,
                metadata=_metadata(
                    document_id="doc-b",
                    chunk_index=0,
                    start_char=20,
                    end_char=20 + len(beta),
                ),
            ),
        ),
    )


def _rerank_response() -> RerankResponse:
    """Return deterministic reranking that reverses vector order."""
    return RerankResponse(
        model="reranker-model",
        results=(
            RerankResult(
                document_id="chunk-b",
                score=9.0,
                rank=1,
            ),
            RerankResult(
                document_id="chunk-a",
                score=7.0,
                rank=2,
            ),
        ),
    )


def _llm_response(
    *,
    content: str = "Grounded synthetic answer. [[C1]]",
) -> LLMResponse:
    """Return deterministic successful model output."""
    return LLMResponse(
        model="generator-model",
        message=LLMMessage(
            role=MessageRole.ASSISTANT,
            content=content,
        ),
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(
            input_tokens=20,
            output_tokens=5,
        ),
    )


def _request() -> RetrievalRequest:
    """Return one deterministic RAG request."""
    return RetrievalRequest(
        query="What synthetic evidence exists?",
        top_k=2,
        namespace="synthetic",
    )


def _retrieval_service(
    events: list[str],
    *,
    vector_response: VectorQueryResponse,
    vector_error: Exception | None = None,
) -> RetrievalService:
    """Build real retrieval orchestration over synthetic providers."""
    return RetrievalService(
        SyntheticEmbeddingProvider(events),
        SyntheticVectorStore(
            events,
            response=vector_response,
            error=vector_error,
        ),
        model="embedding-model",
        dimensions=3,
        space_id="embedding-model@revision-1",
    )


class _RecordingGuardrailRule:
    """Test rule recording exact evidence identities in evaluation order."""

    def __init__(
        self,
        seen_content_ids: list[str],
    ) -> None:
        self._seen_content_ids = seen_content_ids

    @property
    def rule_id(self) -> str:
        return "rag-recording-rule"

    @property
    def stages(self) -> tuple[GuardrailStage, ...]:
        return (GuardrailStage.RETRIEVED_CONTEXT,)

    def evaluate(
        self,
        subject: GuardrailSubject,
    ) -> tuple[GuardrailFinding, ...]:
        self._seen_content_ids.append(subject.content_id)
        return ()


def _rag_guardrail_service(
    seen_content_ids: list[str] | None = None,
) -> GuardrailService:
    """Return deterministic retrieved-context policy for RAG tests."""
    rule: GuardrailRule

    if seen_content_ids is not None:
        rule = _RecordingGuardrailRule(seen_content_ids)
    else:
        rule = LiteralPatternGuardrailRule(
            rule_id="rag-test-retrieved-context-rule",
            stages=(GuardrailStage.RETRIEVED_CONTEXT,),
            patterns=(
                GuardrailLiteralPattern(
                    literal="__rag_guardrail_marker_not_present__",
                    category=GuardrailCategory.OTHER,
                    severity=GuardrailSeverity.LOW,
                    message="configured synthetic RAG signal",
                ),
            ),
        )

    return GuardrailService(
        policy=GuardrailPolicy(
            enabled_stages=(GuardrailStage.RETRIEVED_CONTEXT,),
            block_at_or_above=GuardrailSeverity.HIGH,
            max_content_chars=10_000,
            max_findings=16,
        ),
        rules=(rule,),
    )


def _generation_service(
    events: list[str],
    *,
    response: LLMResponse | None = None,
    error: Exception | None = None,
    guardrail_seen_content_ids: list[str] | None = None,
) -> GroundedGenerationService:
    """Build real grounded generation over a synthetic LLM provider."""
    return GroundedGenerationService(
        SyntheticLLMProvider(
            events,
            response=response,
            error=error,
        ),
        guardrail_service=_rag_guardrail_service(guardrail_seen_content_ids),
        model="generator-model",
        temperature=0.0,
        max_output_tokens=128,
    )


def _reranking_service(
    events: list[str],
    *,
    response: RerankResponse | None = None,
    error: Exception | None = None,
) -> RerankingService:
    """Build real reranking orchestration over a synthetic provider."""
    return RerankingService(
        SyntheticReranker(
            events,
            response=response,
            error=error,
        ),
        model="reranker-model",
        top_n=2,
    )


@pytest.mark.anyio
async def test_rag_runs_retrieval_then_grounded_generation() -> None:
    """RAG without reranking should preserve deterministic stage order."""
    events: list[str] = []

    service = RAGService(
        _retrieval_service(
            events,
            vector_response=_vector_response(),
        ),
        _generation_service(
            events,
            response=_llm_response(),
        ),
    )

    result = await service.run(_request())

    assert events == [
        "embed",
        "vector_query",
        "llm",
    ]

    assert result.grounding_input is result.retrieval
    assert result.retrieval.namespace == "synthetic"

    assert result.generation is not None
    assert result.answer is result.generation.answer
    assert result.answer.status is GroundedAnswerStatus.ANSWERED

    assert tuple(citation.citation_id for citation in result.answer.citations) == (
        "C1",
    )

    assert result.answer.citations[0].evidence is result.retrieval.results[0]
    assert result.answer.citations[0].evidence.chunk_id == "chunk-a"


@pytest.mark.anyio
async def test_rag_optionally_reranks_before_generation() -> None:
    """Configured reranking should become the exact grounding order."""
    events: list[str] = []

    service = RAGService(
        _retrieval_service(
            events,
            vector_response=_vector_response(),
        ),
        _generation_service(
            events,
            response=_llm_response(),
        ),
        reranking_service=_reranking_service(
            events,
            response=_rerank_response(),
        ),
    )

    result = await service.run(_request())

    assert events == [
        "embed",
        "vector_query",
        "rerank",
        "llm",
    ]

    assert tuple(item.chunk_id for item in result.retrieval.results) == (
        "chunk-a",
        "chunk-b",
    )

    assert isinstance(
        result.grounding_input,
        RerankedRetrievalResponse,
    )

    assert tuple(item.evidence.chunk_id for item in result.grounding_input.results) == (
        "chunk-b",
        "chunk-a",
    )

    assert result.answer.citations[0].citation_id == "C1"
    assert (
        result.answer.citations[0].evidence
        is result.grounding_input.results[0].evidence
    )
    assert result.answer.citations[0].evidence.chunk_id == "chunk-b"

    assert result.generation is not None
    assert result.answer is result.generation.answer


@pytest.mark.anyio
async def test_empty_retrieval_abstains_without_reranker_or_llm() -> None:
    """No evidence should become controlled abstention without model use."""
    events: list[str] = []

    service = RAGService(
        _retrieval_service(
            events,
            vector_response=VectorQueryResponse(
                results=(),
            ),
        ),
        _generation_service(
            events,
            response=_llm_response(),
        ),
        reranking_service=_reranking_service(
            events,
            response=_rerank_response(),
        ),
    )

    result = await service.run(_request())

    assert events == [
        "embed",
        "vector_query",
    ]

    assert result.retrieval.results == ()
    assert result.grounding_input is result.retrieval
    assert result.generation is None

    assert result.answer.status is GroundedAnswerStatus.ABSTAINED
    assert result.answer.answer is None
    assert result.answer.citations == ()
    assert result.answer.query == _request().query
    assert result.answer.namespace == "synthetic"


@pytest.mark.anyio
async def test_model_insufficient_evidence_abstention_is_preserved() -> None:
    """Model abstention over retrieved evidence should remain explicit."""
    events: list[str] = []

    service = RAGService(
        _retrieval_service(
            events,
            vector_response=_vector_response(),
        ),
        _generation_service(
            events,
            response=_llm_response(
                content="INSUFFICIENT_EVIDENCE",
            ),
        ),
    )

    result = await service.run(_request())

    assert events == [
        "embed",
        "vector_query",
        "llm",
    ]

    assert result.retrieval.results
    assert result.generation is not None

    assert result.answer is result.generation.answer
    assert result.answer.status is GroundedAnswerStatus.ABSTAINED
    assert result.answer.answer is None
    assert result.answer.citations == ()
    assert result.answer.query == _request().query
    assert result.answer.namespace == "synthetic"


@pytest.mark.anyio
async def test_retrieval_failure_prevents_downstream_execution() -> None:
    """Vector retrieval failure must prevent reranking and generation."""
    events: list[str] = []

    service = RAGService(
        _retrieval_service(
            events,
            vector_response=VectorQueryResponse(
                results=(),
            ),
            vector_error=ProviderExecutionError(
                "synthetic vector retrieval failure",
            ),
        ),
        _generation_service(
            events,
            response=_llm_response(),
        ),
        reranking_service=_reranking_service(
            events,
            response=_rerank_response(),
        ),
    )

    with pytest.raises(
        ProviderExecutionError,
        match="synthetic vector retrieval failure",
    ):
        await service.run(_request())

    assert events == [
        "embed",
        "vector_query",
    ]


@pytest.mark.anyio
async def test_reranker_failure_prevents_generation() -> None:
    """Reranking failure must stop before grounded generation."""
    events: list[str] = []

    service = RAGService(
        _retrieval_service(
            events,
            vector_response=_vector_response(),
        ),
        _generation_service(
            events,
            response=_llm_response(),
        ),
        reranking_service=_reranking_service(
            events,
            error=ProviderExecutionError(
                "synthetic reranker failure",
            ),
        ),
    )

    with pytest.raises(
        ProviderExecutionError,
        match="synthetic reranker failure",
    ):
        await service.run(_request())

    assert events == [
        "embed",
        "vector_query",
        "rerank",
    ]


@pytest.mark.anyio
async def test_generation_failure_preserves_existing_error_semantics() -> None:
    """Generation failures should propagate without orchestration wrapping."""
    events: list[str] = []

    service = RAGService(
        _retrieval_service(
            events,
            vector_response=_vector_response(),
        ),
        _generation_service(
            events,
            error=ProviderExecutionError(
                "synthetic generation failure",
            ),
        ),
    )

    with pytest.raises(
        ProviderExecutionError,
        match="synthetic generation failure",
    ):
        await service.run(_request())

    assert events == [
        "embed",
        "vector_query",
        "llm",
    ]


def test_rag_service_and_result_are_publicly_exported() -> None:
    """End-to-end RAG orchestration should be public through services."""
    import ai_engineering_agent_platform.services as services

    assert {
        "RAGResult",
        "RAGService",
    } <= set(services.__all__)

    assert hasattr(
        services,
        "RAGResult",
    )
    assert hasattr(
        services,
        "RAGService",
    )


@pytest.mark.anyio
async def test_rag_guardrail_uses_final_reranked_grounding_order() -> None:
    """Guardrail evaluation must follow the exact final grounding order."""
    events: list[str] = []
    guarded_content_ids: list[str] = []

    service = RAGService(
        _retrieval_service(
            events,
            vector_response=_vector_response(),
        ),
        _generation_service(
            events,
            response=_llm_response(),
            guardrail_seen_content_ids=guarded_content_ids,
        ),
        reranking_service=_reranking_service(
            events,
            response=_rerank_response(),
        ),
    )

    result = await service.run(_request())

    assert events == [
        "embed",
        "vector_query",
        "rerank",
        "llm",
    ]

    assert guarded_content_ids == [
        "chunk-b",
        "chunk-a",
    ]

    assert isinstance(
        result.grounding_input,
        RerankedRetrievalResponse,
    )

    assert tuple(item.evidence.chunk_id for item in result.grounding_input.results) == (
        "chunk-b",
        "chunk-a",
    )

    assert result.answer.citations[0].evidence is (
        result.grounding_input.results[0].evidence
    )

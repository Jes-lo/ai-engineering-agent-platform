"""Tests for provider-neutral grounded generation orchestration."""

import pytest

from ai_engineering_agent_platform.contracts import (
    FinishReason,
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    MessageRole,
    ProviderDescriptor,
    ProviderKind,
    TokenUsage,
)
from ai_engineering_agent_platform.domain import (
    ProviderExecutionError,
    RetrievalResponse,
    RetrievedEvidence,
)
from ai_engineering_agent_platform.domain.grounding import (
    GroundedAnswerStatus,
)
from ai_engineering_agent_platform.services.grounded_generation import (
    GroundedGenerationService,
)


class SyntheticLLMProvider:
    """Configurable deterministic LLM provider for orchestration tests."""

    def __init__(
        self,
        *,
        response: LLMResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        """Initialize synthetic provider behavior."""
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
        """Record one request and return configured behavior."""
        self.requests.append(request)

        if self.error is not None:
            raise self.error

        if self.response is None:
            raise AssertionError("synthetic LLM response not configured")

        return self.response


def _retrieval() -> RetrievalResponse:
    """Return one deterministic retrieval candidate."""
    text = "synthetic grounded evidence"

    return RetrievalResponse(
        query="What evidence exists?",
        namespace="synthetic",
        results=(
            RetrievedEvidence(
                chunk_id="chunk-1",
                document_id="doc-1",
                text=text,
                score=0.95,
                rank=1,
                start_char=0,
                end_char=len(text),
                source_ref="synthetic://doc-1",
                title="Synthetic Document",
            ),
        ),
    )


def _response(
    *,
    model: str = "generator-model",
    content: str = "Synthetic evidence exists. [[C1]]",
    finish_reason: FinishReason = FinishReason.STOP,
) -> LLMResponse:
    """Return one normalized synthetic generation response."""
    return LLMResponse(
        model=model,
        message=LLMMessage(
            role=MessageRole.ASSISTANT,
            content=content,
        ),
        finish_reason=finish_reason,
        usage=TokenUsage(
            input_tokens=20,
            output_tokens=6,
        ),
    )


@pytest.mark.anyio
async def test_service_generates_validated_grounded_answer() -> None:
    """Service should execute one provider call and preserve citations."""
    provider = SyntheticLLMProvider(response=_response())

    typed_provider: LLMProvider = provider

    service = GroundedGenerationService(
        typed_provider,
        model="generator-model",
        temperature=0.0,
        max_output_tokens=128,
    )

    result = await service.generate(_retrieval())

    assert len(provider.requests) == 1

    request = provider.requests[0]

    assert request.model == "generator-model"
    assert request.temperature == 0.0
    assert request.max_output_tokens == 128

    assert result.model == "generator-model"
    assert result.finish_reason is FinishReason.STOP
    assert result.usage is not None
    assert result.usage.total_tokens == 26

    assert result.answer.status is GroundedAnswerStatus.ANSWERED
    assert result.answer.citations[0].citation_id == "C1"
    assert result.answer.citations[0].evidence.chunk_id == "chunk-1"


@pytest.mark.anyio
async def test_provider_model_mismatch_fails_closed() -> None:
    """Grounded generation must bind response identity to configured model."""
    provider = SyntheticLLMProvider(
        response=_response(
            model="unexpected-model",
        )
    )

    service = GroundedGenerationService(
        provider,
        model="generator-model",
    )

    with pytest.raises(
        ProviderExecutionError,
        match="LLM provider returned unexpected model",
    ):
        await service.generate(_retrieval())


@pytest.mark.anyio
async def test_non_stop_finish_reason_fails_closed() -> None:
    """Truncated or otherwise incomplete output cannot claim grounding."""
    provider = SyntheticLLMProvider(
        response=_response(
            finish_reason=FinishReason.LENGTH,
        )
    )

    service = GroundedGenerationService(
        provider,
        model="generator-model",
    )

    with pytest.raises(
        ProviderExecutionError,
        match="did not finish grounded generation with stop",
    ):
        await service.generate(_retrieval())


@pytest.mark.anyio
async def test_provider_failure_is_propagated() -> None:
    """Existing provider-domain failure semantics should remain intact."""
    provider = SyntheticLLMProvider(
        error=ProviderExecutionError("synthetic LLM failure")
    )

    service = GroundedGenerationService(
        provider,
        model="generator-model",
    )

    with pytest.raises(
        ProviderExecutionError,
        match="synthetic LLM failure",
    ):
        await service.generate(_retrieval())

    assert len(provider.requests) == 1


@pytest.mark.anyio
async def test_empty_retrieval_bypasses_provider() -> None:
    """No evidence means no model execution and no hallucinated answer."""
    provider = SyntheticLLMProvider(response=_response())

    service = GroundedGenerationService(
        provider,
        model="generator-model",
    )

    retrieval = RetrievalResponse(
        query="Unknown?",
        results=(),
    )

    with pytest.raises(
        ValueError,
        match="without evidence",
    ):
        await service.generate(retrieval)

    assert provider.requests == []


def test_grounded_generation_service_is_publicly_exported() -> None:
    """Grounded generation orchestration should be publicly exported."""
    import ai_engineering_agent_platform.services as services

    assert {
        "GroundedGenerationResult",
        "GroundedGenerationService",
    } <= set(services.__all__)

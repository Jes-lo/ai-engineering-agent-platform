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
from ai_engineering_agent_platform.domain.guardrails import (
    GuardrailCategory,
    GuardrailSeverity,
    GuardrailStage,
)
from ai_engineering_agent_platform.services.grounded_generation import (
    GroundedGenerationGuardrailBlockedError,
    GroundedGenerationService,
)
from ai_engineering_agent_platform.services.guardrails import (
    GuardrailLiteralPattern,
    GuardrailPolicy,
    GuardrailService,
    GuardrailStageError,
    LiteralPatternGuardrailRule,
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


def _guardrail_service(
    *,
    stages: tuple[GuardrailStage, ...] = (GuardrailStage.RETRIEVED_CONTEXT,),
    literal: str = "__synthetic_guardrail_marker_not_present__",
    severity: GuardrailSeverity = GuardrailSeverity.LOW,
) -> GuardrailService:
    """Return deterministic retrieved-context guardrail policy."""
    return GuardrailService(
        policy=GuardrailPolicy(
            enabled_stages=stages,
            block_at_or_above=GuardrailSeverity.HIGH,
            max_content_chars=10_000,
            max_findings=16,
        ),
        rules=(
            LiteralPatternGuardrailRule(
                rule_id="grounded-generation-test-rule",
                stages=stages,
                patterns=(
                    GuardrailLiteralPattern(
                        literal=literal,
                        category=(GuardrailCategory.PROMPT_INJECTION_SIGNAL),
                        severity=severity,
                        message="configured synthetic guardrail signal",
                    ),
                ),
            ),
        ),
    )


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
        guardrail_service=_guardrail_service(),
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
        guardrail_service=_guardrail_service(),
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
        guardrail_service=_guardrail_service(),
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
        guardrail_service=_guardrail_service(),
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
        guardrail_service=_guardrail_service(),
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


@pytest.mark.anyio
async def test_retrieved_context_guardrail_blocks_before_llm() -> None:
    """Blocking retrieved evidence must never reach the LLM provider."""
    marker = "synthetic grounded evidence"
    provider = SyntheticLLMProvider(
        response=_response(),
    )

    service = GroundedGenerationService(
        provider,
        guardrail_service=_guardrail_service(
            literal=marker,
            severity=GuardrailSeverity.HIGH,
        ),
        model="generator-model",
    )

    with pytest.raises(
        GroundedGenerationGuardrailBlockedError,
        match="blocked by deterministic guardrail policy",
    ) as exc_info:
        await service.generate(_retrieval())

    assert provider.requests == []

    error = exc_info.value

    assert error.content_id == "chunk-1"
    assert error.blocking_rule_ids == ("grounded-generation-test-rule",)

    assert marker not in str(error)
    assert marker not in repr(error)
    assert marker not in error.content_id
    assert all(marker not in rule_id for rule_id in error.blocking_rule_ids)


@pytest.mark.anyio
async def test_retrieved_context_stage_misconfiguration_fails_before_llm() -> None:
    """Missing retrieved-context coverage must fail before LLM execution."""
    provider = SyntheticLLMProvider(
        response=_response(),
    )

    service = GroundedGenerationService(
        provider,
        guardrail_service=_guardrail_service(
            stages=(GuardrailStage.USER_INPUT,),
        ),
        model="generator-model",
    )

    with pytest.raises(
        GuardrailStageError,
        match="retrieved_context",
    ):
        await service.generate(_retrieval())

    assert provider.requests == []


def test_grounded_generation_guardrail_errors_are_publicly_exported() -> None:
    """Callers should be able to catch the public safety boundary errors."""
    import ai_engineering_agent_platform.services as services

    expected = {
        "GroundedGenerationGuardrailBlockedError",
        "GroundedGenerationGuardrailContractError",
        "GroundedGenerationGuardrailError",
    }

    assert expected <= set(services.__all__)

    for name in expected:
        assert hasattr(
            services,
            name,
        )

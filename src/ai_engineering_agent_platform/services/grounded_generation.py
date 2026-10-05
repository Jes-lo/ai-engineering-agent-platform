"""Provider-neutral grounded generation orchestration."""

from dataclasses import dataclass

from ai_engineering_agent_platform.contracts import (
    FinishReason,
    LLMProvider,
    TokenUsage,
)
from ai_engineering_agent_platform.domain import (
    ProviderExecutionError,
    RerankedRetrievalResponse,
    RetrievalResponse,
)
from ai_engineering_agent_platform.domain.grounding import GroundedAnswer
from ai_engineering_agent_platform.domain.guardrails import (
    GuardrailStage,
    GuardrailSubject,
)
from ai_engineering_agent_platform.services.grounded_generation_mapping import (
    build_grounded_answer,
    build_grounded_llm_request,
    ordered_grounding_evidence,
)
from ai_engineering_agent_platform.services.guardrails import GuardrailService


class GroundedGenerationGuardrailError(RuntimeError):
    """Base failure at the retrieved-context guardrail boundary."""


class GroundedGenerationGuardrailContractError(GroundedGenerationGuardrailError):
    """A guardrail result violated the grounded-generation contract."""


class GroundedGenerationGuardrailBlockedError(GroundedGenerationGuardrailError):
    """Retrieved evidence was blocked before LLM provider execution."""

    def __init__(
        self,
        *,
        content_id: str,
        blocking_rule_ids: tuple[str, ...],
    ) -> None:
        """Store only non-content blocking metadata."""
        if not isinstance(content_id, str) or not content_id.strip():
            raise ValueError("content_id must not be empty")

        if not isinstance(blocking_rule_ids, tuple):
            raise TypeError("blocking_rule_ids must be a tuple")

        if not blocking_rule_ids:
            raise ValueError("blocking_rule_ids must not be empty")

        if not all(
            isinstance(rule_id, str) and rule_id.strip()
            for rule_id in blocking_rule_ids
        ):
            raise ValueError("blocking_rule_ids must contain non-empty strings")

        if len(blocking_rule_ids) != len(set(blocking_rule_ids)):
            raise ValueError("blocking_rule_ids must be unique")

        self.content_id = content_id
        self.blocking_rule_ids = blocking_rule_ids

        super().__init__("retrieved context blocked by deterministic guardrail policy")


@dataclass(frozen=True, slots=True)
class GroundedGenerationResult:
    """Grounded answer plus normalized provider-generation metadata."""

    answer: GroundedAnswer
    model: str
    finish_reason: FinishReason
    usage: TokenUsage | None


class GroundedGenerationService:
    """Generate answers only from guarded retrieved evidence."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        *,
        guardrail_service: GuardrailService,
        model: str,
        temperature: float | None = 0.0,
        max_output_tokens: int | None = None,
    ) -> None:
        """Store provider, mandatory guardrail, and generation configuration."""
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must not be empty")

        if not isinstance(guardrail_service, GuardrailService):
            raise TypeError("guardrail_service must be a GuardrailService")

        self._llm_provider = llm_provider
        self._guardrail_service = guardrail_service
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

    def _enforce_retrieved_context_guardrails(
        self,
        retrieval: RetrievalResponse | RerankedRetrievalResponse,
    ) -> None:
        """Fail closed before retrieved evidence can enter an LLM request."""
        evidence = ordered_grounding_evidence(retrieval)

        for item in evidence:
            evaluation = self._guardrail_service.evaluate(
                GuardrailSubject(
                    content_id=item.chunk_id,
                    stage=GuardrailStage.RETRIEVED_CONTEXT,
                    content=item.text,
                )
            )

            if evaluation.content_id != item.chunk_id:
                raise GroundedGenerationGuardrailContractError(
                    "guardrail evaluation content identity mismatch"
                )

            if evaluation.stage is not GuardrailStage.RETRIEVED_CONTEXT:
                raise GroundedGenerationGuardrailContractError(
                    "guardrail evaluation stage mismatch"
                )

            if evaluation.allowed:
                continue

            threshold = self._guardrail_service.policy.block_at_or_above

            blocking_rule_ids = tuple(
                dict.fromkeys(
                    finding.rule_id
                    for finding in evaluation.findings
                    if finding.severity >= threshold
                )
            )

            if not blocking_rule_ids:
                raise GroundedGenerationGuardrailContractError(
                    "blocking evaluation lacks threshold-level findings"
                )

            raise GroundedGenerationGuardrailBlockedError(
                content_id=item.chunk_id,
                blocking_rule_ids=blocking_rule_ids,
            )

    async def generate(
        self,
        retrieval: RetrievalResponse | RerankedRetrievalResponse,
    ) -> GroundedGenerationResult:
        """Guard evidence, then generate and validate one grounded answer."""
        self._enforce_retrieved_context_guardrails(retrieval)

        request = build_grounded_llm_request(
            retrieval,
            model=self._model,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
        )

        response = await self._llm_provider.generate(request)

        if response.model != self._model:
            raise ProviderExecutionError("LLM provider returned unexpected model")

        if response.finish_reason is not FinishReason.STOP:
            raise ProviderExecutionError(
                "LLM provider did not finish grounded generation with stop"
            )

        answer = build_grounded_answer(
            retrieval,
            response,
        )

        return GroundedGenerationResult(
            answer=answer,
            model=response.model,
            finish_reason=response.finish_reason,
            usage=response.usage,
        )

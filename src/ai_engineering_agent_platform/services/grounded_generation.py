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
from ai_engineering_agent_platform.services.grounded_generation_mapping import (
    build_grounded_answer,
    build_grounded_llm_request,
)


@dataclass(frozen=True, slots=True)
class GroundedGenerationResult:
    """Grounded answer plus normalized provider-generation metadata."""

    answer: GroundedAnswer
    model: str
    finish_reason: FinishReason
    usage: TokenUsage | None


class GroundedGenerationService:
    """Generate answers only from already validated retrieved evidence."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        *,
        model: str,
        temperature: float | None = 0.0,
        max_output_tokens: int | None = None,
    ) -> None:
        """Store injected LLM provider and generation configuration."""
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must not be empty")

        self._llm_provider = llm_provider
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

    async def generate(
        self,
        retrieval: RetrievalResponse | RerankedRetrievalResponse,
    ) -> GroundedGenerationResult:
        """Generate and validate one answer over supplied evidence."""
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

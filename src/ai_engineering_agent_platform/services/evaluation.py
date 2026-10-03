"""Automated deterministic evaluation orchestration for RAG datasets."""

from dataclasses import dataclass
from typing import Protocol

from ai_engineering_agent_platform.domain import (
    RAGEvaluationCase,
    RAGEvaluationDataset,
    RAGEvaluationMetrics,
    RAGEvaluationSummary,
    RetrievalRequest,
)
from ai_engineering_agent_platform.services.evaluation_mapping import (
    score_rag_result,
    summarize_rag_metrics,
)
from ai_engineering_agent_platform.services.rag import (
    RAGResult,
)


class RAGRunner(Protocol):
    """Minimal async RAG execution contract required by evaluation."""

    async def run(
        self,
        request: RetrievalRequest,
    ) -> RAGResult:
        """Execute one RAG request."""
        ...


@dataclass(frozen=True, slots=True)
class RAGEvaluationCaseExecution:
    """One executed dataset case plus deterministic evaluation metrics."""

    case: RAGEvaluationCase
    rag_result: RAGResult
    metrics: RAGEvaluationMetrics


@dataclass(frozen=True, slots=True)
class RAGEvaluationRunResult:
    """Complete deterministic result for one versioned evaluation dataset."""

    dataset_id: str
    dataset_version: str
    provenance_ref: str
    cases: tuple[
        RAGEvaluationCaseExecution,
        ...,
    ]
    summary: RAGEvaluationSummary


class RAGEvaluationService:
    """Execute a versioned RAG evaluation dataset sequentially.

    The service intentionally performs no retries, persistence, external
    dataset loading, LLM-as-a-judge scoring, or observability export.
    """

    def __init__(
        self,
        rag_runner: RAGRunner,
    ) -> None:
        """Store the injected RAG execution boundary."""
        self._rag_runner = rag_runner

    async def evaluate(
        self,
        dataset: RAGEvaluationDataset,
    ) -> RAGEvaluationRunResult:
        """Execute cases in stable dataset order and summarize metrics."""
        executions: list[RAGEvaluationCaseExecution] = []

        metrics: list[RAGEvaluationMetrics] = []

        for case in dataset.cases:
            rag_result = await self._rag_runner.run(case.request)

            case_metrics = score_rag_result(
                case,
                rag_result,
            )

            executions.append(
                RAGEvaluationCaseExecution(
                    case=case,
                    rag_result=rag_result,
                    metrics=case_metrics,
                )
            )

            metrics.append(
                case_metrics,
            )

        summary = summarize_rag_metrics(tuple(metrics))

        return RAGEvaluationRunResult(
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.version,
            provenance_ref=dataset.provenance_ref,
            cases=tuple(executions),
            summary=summary,
        )

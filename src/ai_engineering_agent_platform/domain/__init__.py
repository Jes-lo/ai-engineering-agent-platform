"""Infrastructure-independent platform domain primitives."""

from ai_engineering_agent_platform.domain.chunking import (
    DeterministicTextChunker,
    TextChunkingConfig,
)
from ai_engineering_agent_platform.domain.errors import (
    PlatformError,
    ProviderConfigurationError,
    ProviderError,
    ProviderExecutionError,
    ProviderUnavailableError,
)
from ai_engineering_agent_platform.domain.evaluation import (
    RAGEvaluationCase,
    RAGEvaluationDataset,
    RAGEvaluationMetrics,
    RAGEvaluationSummary,
)
from ai_engineering_agent_platform.domain.grounding import (
    GroundedAnswer,
    GroundedAnswerStatus,
    GroundedCitation,
)
from ai_engineering_agent_platform.domain.ingestion import KnowledgeSource
from ai_engineering_agent_platform.domain.retrieval import (
    DocumentChunk,
    KnowledgeDocument,
    RerankedEvidence,
    RerankedRetrievalResponse,
    RetrievalMetadataItem,
    RetrievalMetadataValue,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedEvidence,
)

__all__ = [
    "DeterministicTextChunker",
    "DocumentChunk",
    "GroundedAnswer",
    "GroundedAnswerStatus",
    "GroundedCitation",
    "KnowledgeDocument",
    "KnowledgeSource",
    "PlatformError",
    "ProviderConfigurationError",
    "ProviderError",
    "ProviderExecutionError",
    "ProviderUnavailableError",
    "RAGEvaluationCase",
    "RAGEvaluationDataset",
    "RAGEvaluationMetrics",
    "RAGEvaluationSummary",
    "RerankedEvidence",
    "RerankedRetrievalResponse",
    "RetrievalMetadataItem",
    "RetrievalMetadataValue",
    "RetrievalRequest",
    "RetrievalResponse",
    "RetrievedEvidence",
    "TextChunkingConfig",
]

"""Stable contracts implemented by replaceable platform adapters."""

from ai_engineering_agent_platform.contracts.embedding import (
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
)
from ai_engineering_agent_platform.contracts.llm import (
    FinishReason,
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    MessageRole,
    TokenUsage,
)
from ai_engineering_agent_platform.contracts.provider import (
    Provider,
    ProviderDescriptor,
    ProviderKind,
)
from ai_engineering_agent_platform.contracts.reranker import (
    RerankDocument,
    RerankerProvider,
    RerankRequest,
    RerankResponse,
    RerankResult,
)
from ai_engineering_agent_platform.contracts.tool import (
    ToolArgument,
    ToolDefinition,
    ToolExecutionStatus,
    ToolInvocation,
    ToolParameter,
    ToolParameterType,
    ToolProvider,
    ToolResult,
)
from ai_engineering_agent_platform.contracts.vector_store import (
    VectorDeleteRequest,
    VectorMetadataItem,
    VectorQueryRequest,
    VectorQueryResponse,
    VectorQueryResult,
    VectorRecord,
    VectorStoreProvider,
    VectorUpsertRequest,
)

__all__ = [
    "EmbeddingProvider",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingVector",
    "FinishReason",
    "LLMMessage",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "MessageRole",
    "Provider",
    "ProviderDescriptor",
    "ProviderKind",
    "RerankDocument",
    "RerankRequest",
    "RerankResponse",
    "RerankResult",
    "RerankerProvider",
    "TokenUsage",
    "ToolArgument",
    "ToolDefinition",
    "ToolExecutionStatus",
    "ToolInvocation",
    "ToolParameter",
    "ToolParameterType",
    "ToolProvider",
    "ToolResult",
    "VectorDeleteRequest",
    "VectorMetadataItem",
    "VectorQueryRequest",
    "VectorQueryResponse",
    "VectorQueryResult",
    "VectorRecord",
    "VectorStoreProvider",
    "VectorUpsertRequest",
]

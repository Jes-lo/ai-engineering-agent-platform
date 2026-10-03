"""Ollama-specific adapter implementation."""

from ai_engineering_agent_platform.adapters.ollama.embedding_provider import (
    OllamaEmbeddingProvider,
)
from ai_engineering_agent_platform.adapters.ollama.provider import (
    OllamaLLMProvider,
)

__all__ = [
    "OllamaEmbeddingProvider",
    "OllamaLLMProvider",
]

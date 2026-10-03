"""Runtime composition for provider implementations."""

from ai_engineering_agent_platform.runtime.embedding import (
    ollama_embedding_runtime,
)
from ai_engineering_agent_platform.runtime.llm import (
    ollama_llm_runtime,
)
from ai_engineering_agent_platform.runtime.ollama import (
    create_ollama_http_client,
)

__all__ = [
    "create_ollama_http_client",
    "ollama_embedding_runtime",
    "ollama_llm_runtime",
]

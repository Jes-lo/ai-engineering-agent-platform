"""Application configuration."""

from functools import lru_cache
from math import isfinite
from typing import Literal
from urllib.parse import urlsplit

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="AI_PLATFORM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AI Engineering & Agent Platform"
    environment: Literal["development", "test", "production"] = "development"
    docs_enabled: bool = True

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_request_timeout_seconds: float = 120.0

    def model_post_init(self, context: object, /) -> None:
        """Validate provider runtime configuration."""
        del context

        parsed_url = urlsplit(self.ollama_base_url)

        if parsed_url.scheme not in {"http", "https"} or parsed_url.hostname is None:
            raise ValueError("ollama_base_url must be an absolute HTTP(S) URL")

        try:
            _ = parsed_url.port
        except ValueError as exc:
            raise ValueError("ollama_base_url must contain a valid port") from exc

        if parsed_url.username is not None or parsed_url.password is not None:
            raise ValueError("ollama_base_url must not contain embedded credentials")

        if parsed_url.path not in {"", "/"} or parsed_url.query or parsed_url.fragment:
            raise ValueError("ollama_base_url must contain only scheme and authority")

        if (
            not isfinite(self.ollama_request_timeout_seconds)
            or self.ollama_request_timeout_seconds <= 0
        ):
            raise ValueError(
                "ollama_request_timeout_seconds must be positive and finite"
            )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide application settings."""

    return Settings()

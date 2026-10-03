"""Application configuration."""

from functools import lru_cache
from math import isfinite
from typing import Literal
from urllib.parse import urlsplit

from pydantic import SecretStr, field_validator
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

    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_database: str = "ai_platform"
    postgres_user: str = "ai_platform_runtime"
    postgres_password: SecretStr | None = None
    postgres_sslmode: Literal[
        "disable",
        "allow",
        "prefer",
        "require",
        "verify-ca",
        "verify-full",
    ] = "prefer"
    postgres_connect_timeout_seconds: int = 5
    postgres_pool_min_size: int = 1
    postgres_pool_max_size: int = 5
    postgres_pool_timeout_seconds: float = 10.0

    @field_validator(
        "postgres_port",
        "postgres_connect_timeout_seconds",
        "postgres_pool_min_size",
        "postgres_pool_max_size",
        "postgres_pool_timeout_seconds",
        mode="before",
    )
    @classmethod
    def reject_boolean_postgres_numeric_settings(
        cls,
        value: object,
    ) -> object:
        """Reject booleans before numeric PostgreSQL coercion."""
        del cls

        if isinstance(value, bool):
            raise ValueError("PostgreSQL numeric settings must not be boolean")

        return value

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

        if not self.postgres_host.strip():
            raise ValueError("postgres_host must not be empty")

        if not self.postgres_database.strip():
            raise ValueError("postgres_database must not be empty")

        if not self.postgres_user.strip():
            raise ValueError("postgres_user must not be empty")

        if not 1 <= self.postgres_port <= 65535:
            raise ValueError("postgres_port must be between 1 and 65535")

        if self.postgres_connect_timeout_seconds <= 0:
            raise ValueError("postgres_connect_timeout_seconds must be positive")

        if self.postgres_pool_min_size <= 0:
            raise ValueError("postgres_pool_min_size must be positive")

        if self.postgres_pool_max_size <= 0:
            raise ValueError("postgres_pool_max_size must be positive")

        if self.postgres_pool_min_size > self.postgres_pool_max_size:
            raise ValueError(
                "postgres_pool_min_size must not exceed postgres_pool_max_size"
            )

        if (
            not isfinite(self.postgres_pool_timeout_seconds)
            or self.postgres_pool_timeout_seconds <= 0
        ):
            raise ValueError(
                "postgres_pool_timeout_seconds must be positive and finite"
            )

        if (
            self.postgres_password is not None
            and not self.postgres_password.get_secret_value()
        ):
            raise ValueError("postgres_password must not be empty when configured")


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide application settings."""

    return Settings()

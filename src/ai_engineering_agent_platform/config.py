"""Application configuration."""

from functools import lru_cache
from typing import Literal

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


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide application settings."""

    return Settings()

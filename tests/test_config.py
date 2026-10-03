"""Tests for application configuration."""

import pytest

from ai_engineering_agent_platform.config import Settings


def test_default_settings(
    isolated_settings_environment: None,
) -> None:
    settings = Settings()

    assert settings.app_name == "AI Engineering & Agent Platform"
    assert settings.environment == "development"
    assert settings.docs_enabled is True


def test_environment_variable_override(
    isolated_settings_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_PLATFORM_ENVIRONMENT", "test")
    monkeypatch.setenv("AI_PLATFORM_DOCS_ENABLED", "false")

    settings = Settings()

    assert settings.environment == "test"
    assert settings.docs_enabled is False

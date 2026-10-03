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
    assert settings.ollama_base_url == "http://127.0.0.1:11434"
    assert settings.ollama_request_timeout_seconds == 120.0


def test_environment_variable_override(
    isolated_settings_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_PLATFORM_ENVIRONMENT", "test")
    monkeypatch.setenv("AI_PLATFORM_DOCS_ENABLED", "false")
    monkeypatch.setenv(
        "AI_PLATFORM_OLLAMA_BASE_URL",
        "https://models.example.test",
    )
    monkeypatch.setenv(
        "AI_PLATFORM_OLLAMA_REQUEST_TIMEOUT_SECONDS",
        "45.5",
    )

    settings = Settings()

    assert settings.environment == "test"
    assert settings.docs_enabled is False
    assert settings.ollama_base_url == "https://models.example.test"
    assert settings.ollama_request_timeout_seconds == 45.5


def test_ollama_base_url_rejects_non_http_and_paths(
    isolated_settings_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "AI_PLATFORM_OLLAMA_BASE_URL",
        "localhost:11434",
    )

    with pytest.raises(
        ValueError,
        match=r"absolute HTTP\(S\) URL",
    ):
        Settings()

    monkeypatch.setenv(
        "AI_PLATFORM_OLLAMA_BASE_URL",
        "ftp://localhost:11434",
    )

    with pytest.raises(
        ValueError,
        match=r"absolute HTTP\(S\) URL",
    ):
        Settings()

    monkeypatch.setenv(
        "AI_PLATFORM_OLLAMA_BASE_URL",
        "http://localhost:11434/api",
    )

    with pytest.raises(
        ValueError,
        match="must contain only scheme and authority",
    ):
        Settings()


def test_ollama_base_url_rejects_embedded_credentials(
    isolated_settings_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "AI_PLATFORM_OLLAMA_BASE_URL",
        "http://user:password@localhost:11434",
    )

    with pytest.raises(
        ValueError,
        match="must not contain embedded credentials",
    ):
        Settings()


@pytest.mark.parametrize(
    "timeout",
    [
        "0",
        "-1",
        "nan",
        "inf",
        "-inf",
    ],
)
def test_ollama_timeout_must_be_positive_and_finite(
    timeout: str,
    isolated_settings_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "AI_PLATFORM_OLLAMA_REQUEST_TIMEOUT_SECONDS",
        timeout,
    )

    with pytest.raises(
        ValueError,
        match=("ollama_request_timeout_seconds must be positive and finite"),
    ):
        Settings()

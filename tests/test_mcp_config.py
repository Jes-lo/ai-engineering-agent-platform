"""Tests for MCP runtime configuration."""

import pytest

from ai_engineering_agent_platform.config import Settings


def test_mcp_defaults_are_disabled_and_local(
    isolated_settings_environment: None,
) -> None:
    settings = Settings()

    assert settings.mcp_enabled is False
    assert settings.mcp_host == "127.0.0.1"
    assert settings.mcp_issuer_url is None
    assert settings.mcp_resource_server_url is None
    assert settings.mcp_required_scopes == ("mcp:tools",)
    assert settings.mcp_max_request_body_size == 1024 * 1024


def test_mcp_enabled_requires_issuer_and_resource_urls(
    isolated_settings_environment: None,
) -> None:
    with pytest.raises(
        ValueError,
        match="mcp_issuer_url is required",
    ):
        Settings(
            mcp_enabled=True,
        )

    with pytest.raises(
        ValueError,
        match="mcp_resource_server_url is required",
    ):
        Settings(
            mcp_enabled=True,
            mcp_issuer_url=("https://auth.example.invalid/"),
        )


def test_mcp_resource_url_must_target_mcp_path(
    isolated_settings_environment: None,
) -> None:
    with pytest.raises(
        ValueError,
        match="must target /mcp",
    ):
        Settings(
            mcp_enabled=True,
            mcp_issuer_url=("https://auth.example.invalid/"),
            mcp_resource_server_url=("https://api.example.invalid/not-mcp"),
        )


def test_production_mcp_urls_require_https(
    isolated_settings_environment: None,
) -> None:
    with pytest.raises(
        ValueError,
        match=("production mcp_issuer_url must use HTTPS"),
    ):
        Settings(
            environment="production",
            mcp_enabled=True,
            mcp_issuer_url=("http://auth.example.invalid/"),
            mcp_resource_server_url=("https://api.example.invalid/mcp"),
        )

    with pytest.raises(
        ValueError,
        match=("production mcp_resource_server_url must use HTTPS"),
    ):
        Settings(
            environment="production",
            mcp_enabled=True,
            mcp_issuer_url=("https://auth.example.invalid/"),
            mcp_resource_server_url=("http://api.example.invalid/mcp"),
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        True,
    ],
)
def test_mcp_request_body_size_must_be_positive_integer(
    value: int | bool,
    isolated_settings_environment: None,
) -> None:
    expected = (
        "MCP numeric settings must not be boolean"
        if value is True
        else ("mcp_max_request_body_size must be positive")
    )

    with pytest.raises(
        ValueError,
        match=expected,
    ):
        Settings(
            mcp_max_request_body_size=value,
        )


def test_mcp_scopes_must_be_unique_and_non_empty(
    isolated_settings_environment: None,
) -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        Settings(
            mcp_required_scopes=(),
        )

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        Settings(
            mcp_required_scopes=(
                "mcp:tools",
                "mcp:tools",
            ),
        )


def test_mcp_environment_override(
    isolated_settings_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "AI_PLATFORM_MCP_ENABLED",
        "true",
    )

    monkeypatch.setenv(
        "AI_PLATFORM_MCP_HOST",
        "testserver",
    )

    monkeypatch.setenv(
        "AI_PLATFORM_MCP_ISSUER_URL",
        "https://auth.example.invalid/",
    )

    monkeypatch.setenv(
        "AI_PLATFORM_MCP_RESOURCE_SERVER_URL",
        "http://testserver/mcp",
    )

    monkeypatch.setenv(
        "AI_PLATFORM_MCP_REQUIRED_SCOPES",
        '["mcp:tools","tool:read"]',
    )

    monkeypatch.setenv(
        "AI_PLATFORM_MCP_MAX_REQUEST_BODY_SIZE",
        "524288",
    )

    settings = Settings()

    assert settings.mcp_enabled is True
    assert settings.mcp_host == "testserver"

    assert settings.mcp_required_scopes == (
        "mcp:tools",
        "tool:read",
    )

    assert settings.mcp_max_request_body_size == 524288

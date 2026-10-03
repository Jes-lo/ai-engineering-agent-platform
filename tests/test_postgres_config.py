"""Tests for PostgreSQL runtime configuration."""

import pytest
from pydantic import SecretStr

from ai_engineering_agent_platform.config import Settings


def test_postgres_settings_have_safe_local_defaults() -> None:
    """Database defaults should target local development without a secret."""
    settings = Settings()

    assert settings.postgres_host == "127.0.0.1"
    assert settings.postgres_port == 5432
    assert settings.postgres_database == "ai_platform"
    assert settings.postgres_user == "ai_platform_runtime"
    assert settings.postgres_password is None
    assert settings.postgres_sslmode == "prefer"
    assert settings.postgres_connect_timeout_seconds == 5
    assert settings.postgres_pool_min_size == 1
    assert settings.postgres_pool_max_size == 5
    assert settings.postgres_pool_timeout_seconds == 10.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"postgres_host": " "},
        {"postgres_database": " "},
        {"postgres_user": " "},
    ],
)
def test_postgres_settings_reject_blank_identifiers(
    kwargs: dict[str, str],
) -> None:
    """Required PostgreSQL identifiers must not be blank."""
    with pytest.raises(ValueError):
        Settings.model_validate(kwargs)


@pytest.mark.parametrize(
    "port",
    [
        0,
        65536,
    ],
)
def test_postgres_settings_reject_invalid_ports(
    port: int,
) -> None:
    """PostgreSQL ports must stay in the TCP port range."""
    with pytest.raises(
        ValueError,
        match="postgres_port must be between 1 and 65535",
    ):
        Settings(postgres_port=port)


def test_postgres_settings_reject_boolean_numeric_values() -> None:
    """Boolean values must not silently become PostgreSQL numbers."""
    with pytest.raises(
        ValueError,
        match="PostgreSQL numeric settings must not be boolean",
    ):
        Settings(postgres_port=True)


def test_postgres_settings_reject_inverted_pool_bounds() -> None:
    """Pool minimum size cannot exceed the maximum."""
    with pytest.raises(
        ValueError,
        match="postgres_pool_min_size must not exceed",
    ):
        Settings(
            postgres_pool_min_size=5,
            postgres_pool_max_size=2,
        )


@pytest.mark.parametrize(
    "timeout",
    [
        0.0,
        float("inf"),
    ],
)
def test_postgres_settings_reject_invalid_pool_timeout(
    timeout: float,
) -> None:
    """Pool wait timeout must be positive and finite."""
    with pytest.raises(
        ValueError,
        match="postgres_pool_timeout_seconds",
    ):
        Settings(
            postgres_pool_timeout_seconds=timeout,
        )


def test_postgres_password_is_masked_in_settings_repr() -> None:
    """Configured database secrets must not appear in Settings repr."""
    value = "-".join(("runtime", "credential", "for", "test"))

    settings = Settings(
        postgres_password=SecretStr(value),
    )

    assert value not in repr(settings)
    assert "**********" in repr(settings)

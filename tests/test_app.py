"""Tests for the FastAPI application."""

from fastapi.testclient import TestClient

from ai_engineering_agent_platform import __version__
from ai_engineering_agent_platform.app import create_app
from ai_engineering_agent_platform.config import Settings


def test_application_metadata(
    isolated_settings_environment: None,
) -> None:
    settings = Settings()
    application = create_app(settings)

    assert application.title == "AI Engineering & Agent Platform"
    assert application.version == __version__


def test_health_endpoint(
    isolated_settings_environment: None,
) -> None:
    application = create_app(Settings())
    client = TestClient(application)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_endpoint(
    isolated_settings_environment: None,
) -> None:
    application = create_app(Settings())
    client = TestClient(application)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_documentation_can_be_disabled(
    isolated_settings_environment: None,
) -> None:
    application = create_app(Settings(docs_enabled=False))
    client = TestClient(application)

    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404

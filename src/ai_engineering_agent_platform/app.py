"""FastAPI application factory."""

from fastapi import FastAPI

from ai_engineering_agent_platform import __version__
from ai_engineering_agent_platform.api.routes.health import router as health_router
from ai_engineering_agent_platform.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    resolved_settings = settings or get_settings()

    docs_url = "/docs" if resolved_settings.docs_enabled else None
    redoc_url = "/redoc" if resolved_settings.docs_enabled else None
    openapi_url = "/openapi.json" if resolved_settings.docs_enabled else None

    application = FastAPI(
        title=resolved_settings.app_name,
        version=__version__,
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
    )

    application.state.settings = resolved_settings
    application.include_router(health_router)

    return application


app = create_app()

"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ai_engineering_agent_platform import __version__
from ai_engineering_agent_platform.api.mcp_server import (
    AuthenticatedMCPServerAdapter,
)
from ai_engineering_agent_platform.api.routes.health import (
    router as health_router,
)
from ai_engineering_agent_platform.config import (
    Settings,
    get_settings,
)


def create_app(
    settings: Settings | None = None,
    *,
    mcp_adapter: AuthenticatedMCPServerAdapter | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    resolved_settings = settings or get_settings()

    if resolved_settings.mcp_enabled and mcp_adapter is None:
        raise ValueError("MCP is enabled but no authenticated MCP adapter was provided")

    if not resolved_settings.mcp_enabled and mcp_adapter is not None:
        raise ValueError("MCP adapter must not be mounted while MCP is disabled")

    @asynccontextmanager
    async def lifespan(
        application: FastAPI,
    ) -> AsyncIterator[None]:
        del application

        if mcp_adapter is None:
            yield
            return

        async with mcp_adapter.server.session_manager.run():
            yield

    docs_url = "/docs" if resolved_settings.docs_enabled else None

    redoc_url = "/redoc" if resolved_settings.docs_enabled else None

    openapi_url = "/openapi.json" if resolved_settings.docs_enabled else None

    application = FastAPI(
        title=resolved_settings.app_name,
        version=__version__,
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
        lifespan=lifespan,
    )

    application.state.settings = resolved_settings

    application.include_router(health_router)

    if mcp_adapter is not None:
        application.state.mcp_adapter = mcp_adapter

        application.mount(
            "/",
            mcp_adapter.app,
            name="mcp",
        )

    return application


app = create_app()

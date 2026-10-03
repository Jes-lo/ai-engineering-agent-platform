"""Health and readiness endpoints."""

from fastapi import APIRouter

router = APIRouter(tags=["system"])


@router.get("/health", summary="Liveness probe")
async def health() -> dict[str, str]:
    """Report whether the API process is alive."""

    return {"status": "ok"}


@router.get("/ready", summary="Readiness probe")
async def ready() -> dict[str, str]:
    """Report whether the API is ready to receive traffic."""

    return {"status": "ready"}

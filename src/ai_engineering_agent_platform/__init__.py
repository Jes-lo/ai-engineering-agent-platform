"""AI Engineering & Agent Platform."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ai-engineering-agent-platform")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]

"""Domain-level exception hierarchy."""


class PlatformError(Exception):
    """Base exception for expected platform-domain failures."""


class ProviderError(PlatformError):
    """Base exception for provider-related failures."""


class ProviderConfigurationError(ProviderError):
    """Raised when provider configuration is invalid."""


class ProviderExecutionError(ProviderError):
    """Raised when a provider cannot complete an operation."""


class ProviderUnavailableError(ProviderError):
    """Raised when a provider is temporarily unavailable."""

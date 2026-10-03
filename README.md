# AI Engineering & Agent Platform

A self-hosted AI engineering and agent platform built as a portfolio
project to demonstrate secure, observable, testable, and maintainable AI
platform engineering practices.

## Status

Early development.

The repository currently includes:

- repository governance, security, architecture, and engineering standards;
- a Python/FastAPI application foundation;
- liveness and readiness endpoints;
- provider-neutral contracts for language models, embeddings, rerankers,
  vector stores, and controlled tool execution;
- immutable provider-neutral request and response models;
- a domain-level provider exception hierarchy;
- a concrete Ollama adapter implementing the language-model provider
  contract;
- non-streaming Ollama chat request and response mapping;
- validated Ollama runtime configuration;
- HTTP client and provider runtime composition with explicit lifecycle
  ownership;
- normalized provider errors for timeout, transport, HTTP-status, malformed
  JSON, and malformed provider responses;
- isolated Ollama adapter tests using mocked HTTP transport;
- automated architectural dependency checks;
- CI/CD and software supply-chain validation.

Current model execution is available through the provider runtime boundary
and uses Ollama's non-streaming chat API. The FastAPI application does not
yet expose model-generation endpoints.

Tool calling remains intentionally unsupported by the LLM contract. Non-empty
provider `tool_calls` are rejected rather than executed or silently
discarded.

Persistent databases, RAG pipelines, executable tools, agents, MCP
integrations, workflow execution, and AI observability backends have not yet
been implemented.

## Planned Capabilities

The platform is intended to evolve incrementally toward capabilities
including:

- additional local and remote model adapters;
- streaming and richer model-capability handling;
- embedding-provider adapters;
- PostgreSQL and vector-search persistence;
- retrieval-augmented generation;
- retrieval and reranking pipelines;
- grounded responses and citations;
- agent execution;
- authorized tool calling;
- MCP integrations and a project-owned MCP server;
- workflow automation;
- human-in-the-loop approval;
- guardrails;
- evaluation datasets and automated evaluations;
- AI and application observability;
- OpenTelemetry;
- metrics and dashboards;
- an AI Developer Console.

Capabilities listed here describe project direction and are not considered
implemented until corresponding code, tests, documentation, and validation
are merged.

## Local Development

The project currently targets Python 3.13.15 and uses `uv` for dependency
and environment management.

Synchronize the environment:

    uv sync

Run the API locally:

    uv run uvicorn ai_engineering_agent_platform.app:app --host 127.0.0.1 --port 8000

Current system endpoints:

- `GET /health` - liveness;
- `GET /ready` - readiness.

Run local validation:

    uv run pytest
    uv run ruff format --check src tests
    uv run ruff check src tests
    uv run mypy src tests

### Local Ollama Runtime

Ollama is the first concrete LLM runtime integration. Ollama itself and model
weights are not distributed by this repository.

Runtime configuration uses:

- `AI_PLATFORM_OLLAMA_BASE_URL`, defaulting to
  `http://127.0.0.1:11434`;
- `AI_PLATFORM_OLLAMA_REQUEST_TIMEOUT_SECONDS`, defaulting to `120`.

The configured base URL must be an absolute HTTP or HTTPS origin without
embedded credentials, path components, query parameters, or fragments.

The current adapter supports non-streaming chat generation only. The runtime
owns creation and cleanup of the HTTP client, while `OllamaLLMProvider`
remains independent of HTTP-client lifecycle management.

For an opt-in real-runtime validation procedure, see
[Local Ollama Smoke Test](docs/operations/local-ollama-smoke-test.md).

## Continuous Integration

GitHub Actions validates pull requests and pushes to `main`.

The current CI baseline includes:

- GitHub Actions workflow linting;
- locked Python environment synchronization;
- formatting;
- linting;
- strict static type checking;
- automated tests with warnings treated as errors;
- package build validation;
- Git history and working-tree secret scanning;
- dependency vulnerability auditing;
- dependency-license inventory;
- CycloneDX SBOM generation and validation.

Supply-chain reports are generated during CI and retained as workflow
artifacts for a limited period.

## Architecture

Architecture decisions and supporting documentation are maintained under:

- `docs/adr/`
- `docs/architecture/`
- `docs/security/`
- `docs/ai-safety/`
- `docs/operations/`

## Engineering Standards

Development and feature completion standards are documented in:

- [Development Standards](docs/operations/development-standards.md)
- [Definition of Done](docs/operations/definition-of-done.md)
- [AI Safety Principles](docs/ai-safety/ai-safety-principles.md)

## Security

Security is treated as a first-class design requirement.

See:

- [SECURITY.md](SECURITY.md)
- [Threat Model](docs/security/threat-model.md)

## Third-Party Components

Third-party software, services, models, datasets, specifications, tools,
and trademarks remain subject to their respective licenses and terms.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

Repository-specific material is provided for portfolio, demonstration,
and evaluation purposes.

Copyright (c) 2026 Jesus Lopez.

All Rights Reserved.

See [LICENSE](LICENSE).

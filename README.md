# AI Engineering & Agent Platform

A self-hosted AI engineering and agent platform built as a portfolio
project to demonstrate secure, observable, testable, and maintainable AI
platform engineering practices.

## Status

Early development.

The repository currently contains the initial governance, security, and
architecture foundation.

Runtime services and AI capabilities have not yet been implemented.

## Planned Capabilities

The platform is intended to evolve incrementally toward capabilities
including:

- model-provider abstraction;
- local and remote model adapters;
- embeddings;
- PostgreSQL and vector search;
- retrieval-augmented generation;
- retrieval and reranking;
- grounded responses and citations;
- agent execution;
- tool calling;
- MCP integrations and a project-owned MCP server;
- workflow automation;
- human-in-the-loop approval;
- guardrails;
- evaluation datasets and automated evaluations;
- AI and application observability;
- OpenTelemetry;
- metrics and dashboards;
- CI/CD and software supply-chain controls;
- an AI Developer Console.

Capabilities listed here describe project direction and are not considered
implemented until corresponding code, tests, documentation, and validation
are merged.

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

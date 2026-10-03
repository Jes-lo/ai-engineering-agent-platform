# Third-Party Notices

This document tracks third-party software, services, specifications,
models, datasets, tools, and other external materials used or referenced
by the AI Engineering & Agent Platform.

Third-party components remain subject to their respective licenses,
terms, copyrights, trademarks, and ownership.

No ownership is claimed over third-party materials.

## Components Currently Used

The project currently uses the following direct software dependencies.

| Component | Version policy | Purpose | License |
| --- | --- | --- | --- |
| FastAPI | 0.142.x | HTTP API framework | MIT |
| Uvicorn | 0.54.x | ASGI application server | BSD-3-Clause |
| pydantic-settings | 2.15.x | Runtime configuration | MIT |
| HTTPX2 | 2.13.x | Runtime HTTP client and adapter testing | BSD-3-Clause |
| Psycopg | 3.3.x with binary extra | PostgreSQL runtime driver | LGPL-3.0-only |
| psycopg-pool | 3.3.x | Asynchronous PostgreSQL connection pooling | LGPL-3.0-only |
| Alembic | 1.20.x | PostgreSQL schema migration orchestration | MIT |
| pytest | 9.1.x | Automated testing | MIT |
| Ruff | 0.16.x | Formatting and static linting | MIT |
| mypy | 2.4.x | Static type checking | MIT |

Exact resolved dependency versions, including transitive dependencies,
are recorded in `uv.lock`.

Automated SBOM and complete dependency-license inventory generation are
part of the repository software supply-chain controls.

Alembic resolves SQLAlchemy as a migration dependency. The current resolved
environment contains SQLAlchemy 2.1.x under the MIT license. Transitive
dependency versions and licenses remain tracked by `uv.lock`, the automated
license inventory, and the generated SBOM rather than being duplicated
exhaustively in this document.

## Database and Vector Infrastructure

The local development and CI persistence environment uses:

| Component | Version / reference | Purpose | License |
| --- | --- | --- | --- |
| PostgreSQL | 18.x container base | Relational persistence | PostgreSQL License |
| pgvector | 0.8.6 | PostgreSQL vector type and distance operations | PostgreSQL License |
| `pgvector/pgvector` container image | `0.8.6-pg18-trixie`, digest pinned in `compose.yaml` | Reproducible local and CI PostgreSQL + pgvector runtime | Upstream components retain their respective licenses |

The repository does not vendor PostgreSQL or pgvector source code. The
development container image is referenced by immutable digest and remains a
third-party artifact subject to its upstream licenses, notices, copyrights,
and image-distribution terms.

## External Services and Tooling

The project may reference or interact with third-party development,
infrastructure, AI, observability, automation, or hosting tools.

Such references do not imply ownership, sponsorship, endorsement, or
affiliation.

Their use remains subject to their respective licenses and terms.

Ollama is the current concrete model-runtime API integration. The repository
interacts with Ollama through its HTTP API but does not distribute Ollama or
model weights. Ollama installations, hosted services, and model artifacts
remain subject to their respective upstream licenses, terms, and usage
conditions.

## AI Models and Model Providers

No AI model is distributed with this repository at this stage.

Models, embedding models, rerankers, model providers, and related artifacts
must be documented when they become project dependencies or are used as named
validation artifacts.

For each applicable component, this project will track:

- component or model name;
- version or immutable reference when available;
- provider or upstream project;
- intended project use;
- applicable license or terms;
- whether artifacts are downloaded, hosted, or accessed remotely;
- relevant redistribution or commercial-use restrictions.

Model weights are not assumed to be covered by the repository license.

### Development Validation Model: Qwen3-Embedding-0.6B

Feature 4 development included an opt-in local embedding smoke validation
using `qwen3-embedding:0.6b` through a separately installed Ollama runtime.

Provenance recorded for that validation artifact:

- upstream model: `Qwen/Qwen3-Embedding-0.6B`;
- upstream organization: Qwen;
- intended project use: local development and embedding integration
  validation;
- upstream model metadata declares the Apache-2.0 license;
- model weights are installed outside this repository;
- the repository does not redistribute the model weights;
- the model is not required by CI;
- the model is not a mandatory runtime dependency of the platform;
- a different compatible embedding model may be used through the
  provider-neutral contract.

The upstream license, model terms, runtime terms, and any later changes remain
the responsibility of their respective third-party owners. Repository
licensing does not replace or modify those terms.

## Datasets

No third-party dataset is distributed with this repository at this stage.

Evaluation, test, demonstration, or knowledge-base datasets introduced
later must have documented provenance and usage rights.

Sensitive, confidential, proprietary, or unlawfully obtained data must
not be committed to this repository.

## Specifications and Standards

Technical specifications and standards may be referenced for
interoperability and implementation purposes.

Their inclusion or mention does not transfer ownership of those
specifications to this repository.

## Repository-Specific Material

Repository-specific source code, infrastructure definitions,
configuration, automation, tests, documentation, diagrams, architecture
decisions, procedures, and integrations are covered by the repository
LICENSE unless otherwise identified.

Third-party names, product names, and trademarks may be used solely for
identification, interoperability, compatibility, or documentation.

No ownership is claimed over such names or marks.

## Third-Party Source Code

No third-party source code is intentionally vendored or copied into this
repository unless explicitly documented.

If third-party source code is introduced later, its origin, applicable
license, and required notices must be recorded before merge.

## Maintenance

This file must be updated whenever a new third-party dependency, model,
dataset, service, specification, tool, or other material creates
licensing, attribution, redistribution, trademark, or usage obligations.

## Development, Security, and CI Tooling

The project also uses external development, security, and CI tooling.

| Component | Version / pin | Purpose |
| --- | --- | --- |
| uv | 0.12.17 | Python environment and dependency management |
| Gitleaks | 8.30.1 | Secret scanning |
| pip-audit | 2.10.1 | Dependency vulnerability auditing |
| pip-licenses | 5.5.5 | Dependency-license inventory |
| CycloneDX Python | 7.3.1 | SBOM generation |
| actionlint | 1.7.12 | GitHub Actions static validation |
| actions/checkout | v7.0.1, SHA pinned | Repository checkout in CI |
| astral-sh/setup-uv | v10.2.0, SHA pinned | uv setup in CI |
| actions/upload-artifact | v7.0.1, SHA pinned | CI evidence retention |

These tools and actions remain subject to their respective upstream
licenses, terms, copyrights, and ownership.

Repository licensing does not apply to these third-party components.

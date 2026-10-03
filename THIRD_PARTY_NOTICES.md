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
| pytest | 9.1.x | Automated testing | MIT |
| HTTPX2 | 2.13.x | Runtime HTTP client and adapter testing | BSD-3-Clause |
| Ruff | 0.16.x | Formatting and static linting | MIT |
| mypy | 2.4.x | Static type checking | MIT |

Exact resolved dependency versions, including transitive dependencies,
are recorded in `uv.lock`.

Automated SBOM and complete dependency-license inventory generation are
part of the repository software supply-chain controls.

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

Models, embedding models, rerankers, model providers, and related
artifacts will be documented individually before they are introduced.

For each applicable component, this project will track:

- component or model name;
- version or immutable reference when available;
- provider or upstream project;
- intended project use;
- applicable license or terms;
- whether artifacts are downloaded, hosted, or accessed remotely;
- relevant redistribution or commercial-use restrictions.

Model weights are not assumed to be covered by the repository license.

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

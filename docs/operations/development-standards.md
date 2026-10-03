# Development Standards

## Purpose

This document defines the engineering standards used throughout the
AI Engineering & Agent Platform.

The objective is to keep development secure, reproducible, reviewable,
testable, and understandable as the platform evolves.

## Development Model

Development should proceed incrementally.

Each feature should introduce the smallest coherent change that can be:

- understood;
- tested;
- reviewed;
- secured;
- documented;
- validated independently.

Large unrelated changes should not be combined into one feature.

## Main Branch

The `main` branch represents the latest validated project state.

Direct development on `main` should be avoided once the initial
repository bootstrap is complete.

Feature work should normally use dedicated branches.

Recommended naming patterns include:

    feature/<short-description>
    fix/<short-description>
    security/<short-description>
    docs/<short-description>
    refactor/<short-description>

Examples:

    feature/model-provider-interface
    feature/postgres-foundation
    security/tool-authorization
    docs/rag-architecture

## Branch Lifecycle

A typical branch lifecycle is:

    synchronized main
           |
           v
      create branch
           |
           v
       implement
           |
           v
        validate
           |
           v
         review
           |
           v
         commit
           |
           v
       push / CI
           |
           v
         merge
           |
           v
    synchronized main

## Commit Standards

Commits should be:

- focused;
- understandable;
- reviewable;
- free of secrets;
- internally consistent;
- validated before creation.

Preferred examples:

    feat: add model provider contract
    feat: add PostgreSQL development environment
    security: enforce tool execution policy
    test: add retrieval authorization coverage
    docs: document model gateway architecture
    refactor: isolate embedding provider adapter

Avoid vague messages such as:

    update
    changes
    fix stuff
    work
    final

## Implementation Principles

Code should favor:

- explicit behavior;
- clear interfaces;
- small cohesive modules;
- typed contracts where practical;
- deterministic behavior where AI is unnecessary;
- dependency injection at external boundaries;
- secure defaults;
- least privilege;
- structured errors;
- observable execution.

Complexity should be introduced only when it provides a measurable
architectural, operational, security, or maintenance benefit.

## Provider-Specific Code

Provider-specific behavior must remain behind adapters or integration
boundaries when practical.

Core domain logic should not directly depend on one particular:

- language model provider;
- embedding provider;
- reranker;
- vector implementation;
- workflow engine;
- telemetry backend;
- external AI service.

## Configuration

Configuration must be externalized where appropriate.

Secrets must never be stored in committed configuration.

Example configuration should use safe placeholder values.

Configuration should fail clearly when required settings are missing.

## Dependencies

A new dependency should only be added when it provides meaningful value.

Before introducing a dependency, review:

- purpose;
- maintenance status;
- license;
- security history when relevant;
- transitive dependencies;
- compatibility;
- whether the capability can reasonably be implemented without it.

Dependencies must be version-controlled through the appropriate package
manifest and lock mechanism.

## Third-Party Components

New third-party components may require updates to:

- `THIRD_PARTY_NOTICES.md`;
- dependency manifests;
- SBOM output;
- architecture documentation;
- security documentation.

Models and datasets require the same level of provenance consideration
as software packages.

## Testing

Testing should be introduced with the functionality being implemented.

Depending on the feature, testing may include:

- unit tests;
- integration tests;
- contract tests;
- API tests;
- security tests;
- authorization tests;
- migration tests;
- evaluation tests;
- adversarial AI tests;
- infrastructure validation;
- end-to-end tests.

Tests should validate behavior, not merely increase coverage numbers.

## AI Testing

AI-enabled functionality must not rely exclusively on manual prompting.

Where relevant, tests or evaluations should cover:

- retrieval relevance;
- grounding;
- citations;
- tool selection;
- authorization;
- prompt injection;
- indirect prompt injection;
- malformed model output;
- tool failure;
- timeout behavior;
- human approval;
- resource limits;
- regression against evaluation datasets.

Non-deterministic behavior should be evaluated using explicit acceptance
criteria rather than assuming exact text equality.

## Security Validation

Security checks should evolve with the repository.

Applicable checks may include:

- secret scanning;
- dependency vulnerability scanning;
- static analysis;
- container scanning;
- infrastructure-as-code scanning;
- SBOM generation;
- license review;
- API security tests;
- authorization tests;
- prompt-injection evaluations;
- tool-permission tests;
- supply-chain validation.

Scanner results must be reviewed rather than treated as automatically
correct.

## Documentation

Documentation must reflect implemented behavior.

Planned features must not be presented as already implemented.

Significant changes may require updates to:

- README;
- architecture overview;
- ADRs;
- threat model;
- security documentation;
- operational documentation;
- third-party notices.

## Architecture Decisions

A new ADR should be created when a decision has significant impact on:

- architecture;
- security;
- deployment;
- data ownership;
- interoperability;
- provider coupling;
- operational complexity;
- long-term maintainability.

Existing accepted ADRs should not be silently rewritten to change their
historical decision.

A superseding ADR should document major reversals.

## Generated Content and AI Assistance

AI tools may assist development, research, review, documentation, and
implementation.

AI-generated suggestions must not be assumed correct or original merely
because they were generated by a model.

Before accepting AI-assisted material, the repository owner should:

- understand the proposed change;
- review it;
- adapt it when necessary;
- verify that it fits the project architecture;
- test it;
- validate security implications;
- avoid intentional reproduction of protected third-party material.

AI assistance does not transfer ownership of third-party material into
this repository.

## Data Handling

Real production credentials, confidential information, private
corporate data, unlawfully obtained data, or sensitive personal
information must not be used as demonstration data.

Synthetic or clearly authorized data should be preferred for portfolio
examples.

## Reproducibility

A supported development environment should eventually be reproducible
from repository-controlled configuration.

Manual setup that affects runtime behavior should be documented.

Critical configuration should not depend solely on undocumented local
machine state.

## Validation Before Merge

Before merge, applicable validation should include:

    format
      |
      v
    lint
      |
      v
    type check
      |
      v
    unit tests
      |
      v
    integration tests
      |
      v
    security checks
      |
      v
    dependency / license review
      |
      v
    documentation review
      |
      v
    git diff review
      |
      v
    CI validation

Not every stage exists from the first feature.

Stages become mandatory when the repository contains the corresponding
technology or risk.

# Definition of Done

## Purpose

A feature is not complete solely because its primary function appears to
work.

This Definition of Done establishes the minimum engineering review used
throughout the AI Engineering & Agent Platform.

## Scope

A feature should have:

- a clearly defined purpose;
- no unrelated changes;
- architecture-aligned implementation;
- accurate representation of what is actually implemented.

## Functional Validation

- intended behavior works;
- expected failure behavior is validated;
- relevant edge cases are considered;
- integration failures are handled appropriately.

## Automated Testing

Applicable tests must be implemented and passing.

These may include:

- unit tests;
- integration tests;
- contract tests;
- API tests;
- security tests;
- evaluation tests;
- infrastructure tests;
- end-to-end tests.

## Static Quality

Where corresponding tooling exists:

- formatting passes;
- linting passes;
- type checking passes;
- configuration validation passes.

## Security

Applicable checks include:

- no committed secrets;
- authentication impact reviewed;
- authorization impact reviewed;
- untrusted input handling reviewed;
- AI trust boundaries reviewed;
- dependency vulnerabilities reviewed;
- sensitive telemetry exposure reviewed.

Security findings must either be remediated or explicitly understood
before merge.

## AI Safety and Agent Behavior

If the feature introduces or changes AI behavior:

- model output is treated as untrusted;
- retrieved content is treated as untrusted;
- prompt-injection exposure is considered;
- relevant evaluations are updated;
- tool permissions are explicitly bounded;
- sensitive actions use enforceable approval controls where required;
- execution limits are defined where relevant;
- failure behavior is understood.

## Data

If the feature stores or processes data:

- data ownership is defined;
- authorization boundaries are clear;
- sensitive-data handling is reviewed;
- migrations are tested when applicable.

## Dependencies

If dependencies are introduced or changed:

- their purpose is understood;
- applicable license or terms are reviewed;
- version changes are intentional;
- vulnerability findings are reviewed;
- `THIRD_PARTY_NOTICES.md` is updated when required;
- SBOM output is updated when SBOM generation exists.

## Models and Datasets

If a model, embedding model, reranker, or dataset is introduced:

- upstream source is documented;
- version or immutable reference is recorded when available;
- applicable license or terms are reviewed;
- redistribution implications are understood;
- commercial-use implications are understood where relevant;
- provenance is documented;
- repository licensing is not incorrectly applied to third-party
  artifacts.

## Documentation

Applicable documentation must be updated.

This may include:

- README;
- ADRs;
- architecture;
- threat model;
- security;
- operational procedures;
- third-party notices.

Documentation must distinguish planned behavior from implemented
behavior.

## Observability

When a feature introduces runtime behavior:

- important failures are observable;
- telemetry does not unnecessarily expose sensitive data;
- logs are structured where practical;
- relevant metrics or traces are considered.

## Git Review

Before commit or merge:

- `git status` is understood;
- staged files are intentional;
- `git diff --check` passes;
- staged diff is reviewed;
- no unrelated files are included;
- secret scanning passes;
- generated artifacts are intentional.

## CI

Once CI exists:

- required workflows pass;
- required security checks pass or have documented review;
- the branch is mergeable;
- CI configuration changes are themselves reviewed.

## Completion Rule

A feature is complete only when applicable implementation, testing,
security, documentation, and validation work is complete.

Passing functional tests alone is insufficient.

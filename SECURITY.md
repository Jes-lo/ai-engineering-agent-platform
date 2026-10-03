# Security Policy

## Project Scope

AI Engineering & Agent Platform is a portfolio project focused on
building and demonstrating secure AI engineering and AI platform
practices.

Security is treated as a design requirement throughout development.

## Reporting Security Issues

Do not create public issues containing:

- credentials;
- access tokens;
- API keys;
- private keys;
- personal information;
- exploitable vulnerability details;
- confidential information.

Security findings should be disclosed privately to the repository owner.

## Security Principles

The project follows these principles:

- least privilege;
- secure defaults;
- explicit authorization;
- defense in depth;
- input and output validation;
- secrets isolation;
- dependency and supply-chain validation;
- auditable privileged actions;
- human approval for sensitive AI actions;
- minimal data collection;
- sensitive-data minimization;
- bounded AI and agent execution;
- fail-safe behavior.

## AI-Specific Threats

The security model will explicitly consider threats including:

- prompt injection;
- indirect prompt injection;
- retrieval poisoning;
- knowledge-base poisoning;
- malicious document ingestion;
- unsafe model output;
- tool misuse;
- excessive agent permissions;
- unauthorized tool execution;
- human-approval bypass;
- MCP server impersonation;
- malicious or misleading MCP metadata;
- data exfiltration;
- cross-user or cross-tenant data exposure;
- secret leakage;
- sensitive telemetry exposure;
- unbounded agent loops;
- resource exhaustion;
- denial of service.

These threats will be refined as platform capabilities are introduced.

## Application and Infrastructure Threats

The project will also consider traditional software and infrastructure
threats including:

- authentication and authorization bypass;
- injection attacks;
- SSRF;
- path traversal;
- arbitrary code execution;
- insecure deserialization;
- credential exposure;
- insecure network configuration;
- vulnerable dependencies;
- dependency confusion;
- container vulnerabilities;
- infrastructure-as-code misconfiguration;
- CI/CD compromise;
- software supply-chain compromise.

## Secrets

Secrets must never be committed to the repository.

Runtime credentials must be provided using approved secret-management or
environment-injection mechanisms appropriate to the deployment target.

Example configuration files must contain placeholders only.

## Dependency Security

Dependencies must be reviewed before introduction.

Automated dependency, vulnerability, secret, license, container, and
supply-chain checks will be added incrementally as the corresponding
components are introduced.

## AI Models and Data

Models and datasets are not implicitly trusted.

Before use, applicable provenance, licenses, terms, source, integrity,
privacy implications, and security considerations should be evaluated.

Production secrets or confidential data must not be included in training,
evaluation, demonstration, or retrieval datasets.

## Security Testing

Security controls and tests will be introduced alongside the feature that
creates the relevant risk.

A feature is not considered complete solely because its functional tests
pass.

## Supported Versions

This project is under active portfolio development.

Security support currently applies only to the latest revision of the
main branch.

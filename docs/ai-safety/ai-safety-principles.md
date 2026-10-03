# AI Safety Principles

## Purpose

This document defines platform-level safety principles for AI-enabled
behavior within the AI Engineering & Agent Platform.

These principles complement traditional application and infrastructure
security controls.

## 1. Treat Model Output as Untrusted

Model-generated content must not automatically be treated as:

- executable code;
- authorization;
- trusted configuration;
- trusted database input;
- trusted tool parameters;
- factual truth.

Downstream systems must apply their own validation and authorization.

## 2. Treat Retrieved Content as Data

Documents, websites, database records, tool outputs, and other retrieved
content may contain malicious or misleading instructions.

Retrieved content must not automatically override platform policy or
trusted instructions.

## 3. Authorization Is Enforced Outside the Model

The model must not be the final authority for access control.

Authentication and authorization decisions must be enforced by
deterministic platform controls.

## 4. Tool Access Is Explicit

Agents must not automatically receive access to every available tool.

Tool availability should depend on:

- identity;
- permissions;
- execution context;
- risk;
- task requirements.

## 5. High-Impact Actions May Require Human Approval

Human approval should be required when an AI-driven operation has
sufficient impact or risk.

Approval must be enforced by platform logic.

A prompt saying "ask the user first" is not an enforceable approval
control.

## 6. Agent Execution Is Bounded

Agent execution should eventually have enforceable limits such as:

- maximum steps;
- maximum duration;
- bounded retries;
- tool restrictions;
- resource budgets;
- explicit termination conditions.

## 7. Minimize Sensitive Context

Only data required for a task should be exposed to:

- models;
- tools;
- workflows;
- telemetry;
- external providers.

Secrets and unrelated sensitive data should not be included in prompts
or retrieved context.

## 8. External AI Systems Are Trust Boundaries

Remote model providers, embedding providers, rerankers, MCP servers,
tools, and workflow systems must be treated as external trust
boundaries.

Connecting a service does not automatically authorize it to receive
all platform data.

## 9. Evaluate AI Behavior

AI behavior should be evaluated systematically.

Evaluation should evolve to include:

- correctness where measurable;
- grounding;
- retrieval quality;
- citation quality;
- policy compliance;
- tool selection;
- tool execution;
- adversarial inputs;
- regressions.

Manual demonstrations alone are insufficient evidence of reliability.

## 10. Prefer Deterministic Controls

Where deterministic software can reliably enforce a safety or security
requirement, deterministic enforcement should be preferred over relying
solely on model instructions.

## 11. Fail Safely

When authorization, validation, policy evaluation, tool execution, or
approval state is uncertain, the system should prefer a safe failure
over silently performing a privileged action.

## 12. Preserve Auditability

Privileged AI-driven operations should eventually provide enough
structured information to determine:

- what initiated the action;
- which identity or context was involved;
- which agent or workflow executed;
- which tool was called;
- what authorization was applied;
- whether approval was required;
- what result occurred.

Sensitive content should not be indiscriminately stored merely for
auditability.

## 13. Maintain Human Control

Users and operators should retain meaningful control over consequential
AI-driven operations.

Automation should not remove human decision points solely for the
purpose of making a demonstration appear more autonomous.

## 14. Safety Evolves With Capability

Controls should be introduced when the corresponding risk becomes real.

The project should not claim that a safety control exists until it is
implemented and, where practical, tested.

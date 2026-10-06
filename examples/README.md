# Golden Demo

This directory contains the deterministic portfolio demonstration for the
AI Engineering & Agent Platform.

The demo is intentionally designed to show platform engineering behavior
without depending on model randomness, model downloads, or external network
access for the AI response itself.

## Scenario

A production-change policy states that production changes require
authenticated human approval before execution.

The demonstration first retrieves that policy as evidence and produces a
grounded answer whose citation is resolved against platform-owned provenance.

It then evaluates an instruction-manipulation signal through the deterministic
runtime guardrail layer and proves that the configured high-severity signal is
blocked.

Finally, the runner invokes the project's existing isolated PostgreSQL
integration gate. That live stage validates the durable control plane,
including PostgreSQL persistence, workflow checkpoints, restart and resume,
approval authorization, durable agent continuations, controlled tool
execution, human-in-the-loop approval, and replay boundaries.

## Run

From the repository root run:

    ./scripts/demo/run-golden-demo.sh

Requirements:

- Git
- uv
- Docker Engine
- Docker Compose

The PostgreSQL validation environment is isolated and uses ephemeral external
credentials created by the project's existing integration tooling.

## Why deterministic providers are used

The purpose of this golden demo is to demonstrate the behavior of the
platform itself.

Using deterministic synthetic embedding/vector/model responses keeps the
demonstration repeatable and avoids confusing model randomness with platform
correctness.

The platform has real Ollama adapters, but Ollama is not required to execute
this deterministic portfolio demo.

## Security boundaries demonstrated

The demo and live integration gate demonstrate that retrieved content,
user-controlled content, model output, and tool proposals do not become
authorization merely because an AI model produced or consumed them.

ToolExecutionService remains the final tool execution authority.

ApprovalService remains the approval authority.

The durable HITL integration uses an AuthenticatedApprovalActor snapshot that
represents identity established by a trusted external authentication boundary.
Constructing that snapshot does not itself authenticate a person.

The project does not claim that prompt injection has been eliminated.

The project does not claim exactly-once external side effects.

The project does not claim byte-for-byte LLM output reproducibility.

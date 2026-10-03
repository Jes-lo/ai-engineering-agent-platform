# Local Ollama Smoke Tests

## Purpose

This procedure performs opt-in integration smoke tests against a real
Ollama installation.

The LLM procedure validates the complete local generation path:

    Settings
        |
        v
    runtime composition
        |
        v
    OllamaLLMProvider
        |
        v
    Ollama /api/chat
        |
        v
    LLMResponse

The smoke test is intentionally separate from the normal automated test suite
and CI. CI must not depend on a running Ollama instance or downloaded model
weights.

## Preconditions

Before running the smoke test:

1. install and start Ollama separately from this repository;
2. ensure an appropriate local model is already installed;
3. keep Ollama restricted to an intended local interface where possible;
4. do not expose Ollama to all network interfaces solely to make this test
   pass;
5. synchronize the project environment with `uv sync`;
6. ensure the repository contains no staged changes created by the smoke
   procedure.

Ollama and model weights remain third-party components and are not
distributed by this repository.

## Verify Runtime Availability

Confirm the local API is reachable:

    curl \
      --silent \
      --show-error \
      --fail \
      http://127.0.0.1:11434/api/version

List locally installed models:

    curl \
      --silent \
      --show-error \
      --fail \
      http://127.0.0.1:11434/api/tags

Choose an already-installed model and expose only its name to the smoke-test
process:

    export AI_PLATFORM_SMOKE_MODEL="<installed-model-name>"

The repository does not require a specific model for this procedure.

## Run the Platform Smoke Test

Run:

    uv run python - <<'PY'
    import asyncio
    import os

    from ai_engineering_agent_platform.config import Settings
    from ai_engineering_agent_platform.contracts import (
        LLMMessage,
        LLMRequest,
        MessageRole,
        ProviderKind,
    )
    from ai_engineering_agent_platform.runtime import (
        ollama_llm_runtime,
    )

    EXPECTED_MARKER = "PLATFORM_SMOKE_OK"


    async def main() -> None:
        model = os.environ["AI_PLATFORM_SMOKE_MODEL"]

        settings = Settings(
            ollama_base_url="http://127.0.0.1:11434",
            ollama_request_timeout_seconds=120.0,
        )

        request = LLMRequest(
            model=model,
            messages=(
                LLMMessage(
                    role=MessageRole.SYSTEM,
                    content=(
                        "Follow the user's instruction. "
                        "Do not add explanations."
                    ),
                ),
                LLMMessage(
                    role=MessageRole.USER,
                    content=(
                        "Reply with this exact marker: "
                        f"{EXPECTED_MARKER}"
                    ),
                ),
            ),
            temperature=0.0,
            max_output_tokens=64,
        )

        async with ollama_llm_runtime(settings) as provider:
            assert provider.descriptor.name == "ollama"
            assert provider.descriptor.kind is ProviderKind.LLM

            response = await provider.generate(request)

        content = response.message.content.strip()

        if EXPECTED_MARKER not in content:
            raise SystemExit(
                "FAIL: expected smoke marker absent"
            )

        if response.usage is None:
            raise SystemExit(
                "FAIL: provider omitted token usage"
            )

        if response.usage.input_tokens <= 0:
            raise SystemExit(
                "FAIL: invalid input token count"
            )

        if response.usage.output_tokens <= 0:
            raise SystemExit(
                "FAIL: invalid output token count"
            )

        print(f"provider={provider.descriptor.name}")
        print(f"model={response.model}")
        print(
            f"finish_reason={response.finish_reason.value}"
        )
        print(
            "usage="
            f"input:{response.usage.input_tokens},"
            f"output:{response.usage.output_tokens},"
            f"total:{response.usage.total_tokens}"
        )
        print(f"response_content={content!r}")
        print("PASS: real Ollama platform smoke test")


    asyncio.run(main())
    PY

## Expected Result

A successful run demonstrates that:

- validated settings can construct the local LLM runtime;
- runtime composition creates the HTTP client and provider;
- `OllamaLLMProvider` can invoke `/api/chat`;
- provider-specific responses normalize into `LLMResponse`;
- token usage is available when Ollama supplies it;
- runtime cleanup completes normally.

The exact generated content beyond the deterministic marker is not a quality
evaluation and must not be interpreted as evidence of model correctness.

## Security Notes

This smoke test must not:

- send production secrets;
- send confidential or personal data;
- require an Ollama listener exposed to untrusted networks;
- execute tools;
- invoke agents;
- alter repository configuration merely to make the test succeed.

Tool-capable Ollama responses now have a narrower boundary. When the request
explicitly exposes matching tool definitions, non-empty `tool_calls` are
normalized into inert `LLMToolCall` proposals and are not executed. A
response proposing a tool that was not explicitly requested still fails
closed. This smoke-test document does not claim deterministic tool selection
for every local model or implement an agent/tool-result round trip.

## Validation Record

Feature 3 development included a successful real-runtime smoke validation
against an already-installed local Ollama model.

The model used for a developer smoke run is not a runtime dependency of this
repository, is not distributed by the project, and is not required by CI.


## Embedding Runtime Smoke Test

Embedding validation is also opt-in and remains separate from CI.

Use an already-installed model that advertises Ollama's `embedding`
capability. Do not assume that a text-generation model can generate
embeddings.

Expose the selected local model name only to the smoke-test process:

    export AI_PLATFORM_EMBEDDING_SMOKE_MODEL="<installed-embedding-model>"

The repository does not require one specific embedding model.

Run:

    uv run python - <<'PY'
    import asyncio
    import math
    import os

    from ai_engineering_agent_platform.config import Settings
    from ai_engineering_agent_platform.contracts import (
        EmbeddingRequest,
        ProviderKind,
    )
    from ai_engineering_agent_platform.runtime import (
        ollama_embedding_runtime,
    )


    async def main() -> None:
        model = os.environ[
            "AI_PLATFORM_EMBEDDING_SMOKE_MODEL"
        ]

        settings = Settings(
            ollama_base_url="http://127.0.0.1:11434",
            ollama_request_timeout_seconds=120.0,
        )

        request = EmbeddingRequest(
            model=model,
            texts=(
                "Synthetic embedding smoke input alpha.",
                "Synthetic embedding smoke input beta.",
            ),
        )

        async with ollama_embedding_runtime(
            settings
        ) as provider:
            assert provider.descriptor.name == "ollama"
            assert (
                provider.descriptor.kind
                is ProviderKind.EMBEDDING
            )

            response = await provider.embed(request)

        if len(response.embeddings) != 2:
            raise SystemExit(
                "FAIL: expected one vector per input"
            )

        if response.dimensions <= 0:
            raise SystemExit(
                "FAIL: invalid embedding dimensionality"
            )

        for embedding in response.embeddings:
            if embedding.dimensions != response.dimensions:
                raise SystemExit(
                    "FAIL: inconsistent dimensions"
                )

            if not all(
                math.isfinite(value)
                for value in embedding.values
            ):
                raise SystemExit(
                    "FAIL: non-finite embedding value"
                )

        print(f"provider=ollama")
        print(f"model={response.model}")
        print(
            f"vectors={len(response.embeddings)}"
        )
        print(
            f"dimensions={response.dimensions}"
        )
        print(
            f"input_tokens={response.input_tokens}"
        )
        print(
            "PASS: real Ollama embedding smoke test"
        )


    asyncio.run(main())
    PY

A successful embedding run demonstrates that:

- validated settings can construct the embedding runtime;
- `OllamaEmbeddingProvider` can invoke `/api/embed`;
- batch inputs produce one normalized vector per input;
- vector values are finite;
- all returned vectors have consistent dimensions;
- token accounting is normalized when Ollama supplies it;
- runtime cleanup completes normally.

Optional requested dimensions can be tested separately by passing
`dimensions=<positive-integer>` to `EmbeddingRequest`. The returned vector
dimensions must match the requested value or the adapter rejects the response.

The embedding adapter deliberately sends `truncate=false`. Inputs that exceed
the provider/model context should therefore fail instead of being silently
truncated.

### Feature 4 Validation Record

Feature 4 development included a successful real-runtime smoke validation
using a separately installed local `qwen3-embedding:0.6b` model through
Ollama.

That development validation observed:

- Ollama advertised the `embedding` capability;
- batch embedding generation succeeded;
- the model returned 1024 dimensions with its default configuration;
- an explicit request for 256 dimensions returned 256-dimensional vectors;
- the complete
  `Settings -> runtime -> OllamaEmbeddingProvider -> EmbeddingResponse`
  path succeeded;
- the model weights remained outside the repository.

The named model records development evidence only. It is not required by CI,
is not distributed by this repository, and is not a mandatory runtime
dependency.

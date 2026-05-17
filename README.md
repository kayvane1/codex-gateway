# Codex Gateway

**Experimental:** this is an early local compatibility gateway for Codex app-server. Treat the HTTP contract and safety model as unstable, review changes carefully before relying on it, and do not expose it outside loopback.

Local FastAPI adapter that lets the official OpenAI Python SDK talk to `codex app-server` over stdio:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="<local-gateway-token>",
)

print(client.models.list())
print(client.chat.completions.create(
    model="gpt-5.5",
    messages=[{"role": "user", "content": "Say pong"}],
))
```

## Run

One-shot from PyPI:

```bash
uvx codex-gateway
```

When no token is configured, `codex-gateway` prints a generated local bearer token and the matching OpenAI SDK setup.
For a reusable shell session, `codex-gateway env` prints exports for `CODEX_GATEWAY_TOKEN`, `CODEX_GATEWAY_BASE_URL`, `CODEX_GATEWAY_HOST`, and `CODEX_GATEWAY_PORT`:

```bash
eval "$(uvx codex-gateway env)"
uvx codex-gateway
```

It does not write the generated local bearer token to disk. With those variables set, OpenAI SDK setup is three lines:

```python
import os
from openai import OpenAI

client = OpenAI(base_url=os.environ["CODEX_GATEWAY_BASE_URL"], api_key=os.environ["CODEX_GATEWAY_TOKEN"])
```

For scripts that only need a token value, use `codex-gateway token`.

For local development:

```bash
uv sync --group dev
uv run codex-gateway --port 8000
```

The `api_key` is a local gateway bearer token. It is not an OpenAI API key and is never proxied to Codex.
If `CODEX_GATEWAY_TOKEN` is not set, the gateway prints a generated local token at startup.

## Implemented

- `GET /v1/models`
- `POST /v1/chat/completions`
- `stream=True` SSE chunks compatible with `openai.OpenAI(...).chat.completions.create(..., stream=True)`
- Multi-turn text chat history via Codex `thread/inject_items`
- OpenAI `image_url` content parts in user messages, mapped to Codex image inputs/history. Image data URLs are materialized as temporary local files for the live Codex turn and cleaned up afterward.
- Local bearer-token authentication
- Codex app-server stdio transport

## Safety Defaults

- Binds to `127.0.0.1` by default.
- Starts ephemeral Codex threads with `sandbox="read-only"`, `approvalPolicy="never"`, no environments, and no dynamic tools.
- Unsubscribes from each per-request thread after the API response finishes.
- Accepts only `http://`, `https://`, and `data:image/...` image URLs; local file URLs are rejected.
- Denies app-server requests for command execution, patch approval, file-change approval, MCP elicitations, and dynamic tool calls.
- Does not expose app-server filesystem, shell, config, auth, or account endpoints through the OpenAI-compatible HTTP API.
- Does not print, persist, proxy, or expose Codex auth credentials.

## Architecture

- `chat_contract.py` owns the OpenAI-facing chat subset: validation, text/image message conversion, history item construction, completion payloads, streaming chunks, SSE framing, and OpenAI-style errors.
- `_app_server_stdio_session.py` owns the Codex app-server stdio transport: JSON-RPC request correlation, notification fan-out, stderr redaction, shutdown, and denial responses for privileged app-server requests.
- `_codex_turn_lifecycle.py` owns Codex turn behavior: starting turns, collecting final assistant output, streaming assistant deltas, draining idle notifications, and mapping token usage.
- `codex_client.py` is the high-level client used by FastAPI. It initializes the app-server session, lists models, starts ephemeral read-only Codex threads, injects prior text/image history, materializes image data URLs for live turns, delegates turn execution, and unsubscribes when finished.

## Unsupported OpenAI Features

This MVP returns a 400 instead of pretending to support features it cannot honor, including tools/functions, tool choice, response formats, JSON/schema-constrained output, audio, `n > 1`, logprobs, sampling controls, stop sequences, token limits, metadata storage, and non-chat APIs such as `/v1/responses`, `/v1/embeddings`, and legacy completions.

## Tests

```bash
uv sync --group dev
uv run --group dev pytest -m "not integration" tests
```

Live contract tests are opt-in because they start the FastAPI gateway, launch `codex app-server` over stdio, and require working Codex auth:

```bash
CODEX_GATEWAY_RUN_CONTRACT_TESTS=1 uv run --group dev pytest -m integration tests/test_contract.py
```

The contract tests instantiate the official OpenAI SDK with `base_url=.../v1`, call `/v1/models`, call non-streaming chat completions including a multi-turn message chain and image content part, call streaming SSE chat completions, and verify missing/invalid local bearer tokens are rejected.
The coverage gate is configured at 95% in `pyproject.toml`.

## Linting

```bash
uv run --group dev ruff check src tests
uv run --group dev ruff format --check src tests
uv run --group dev pre-commit install
uv run --group dev pre-commit run --all-files
```

The generated Codex app-server JSON Schema and TypeScript protocol references used for this implementation are in `generated/`.

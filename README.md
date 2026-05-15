# OpenAI Codex Shim

**Experimental:** this is an early local compatibility shim for Codex app-server. Treat the HTTP contract and safety model as unstable, review changes carefully before relying on it, and do not expose it outside loopback.

Local FastAPI adapter that lets the official OpenAI Python SDK talk to `codex app-server` over stdio:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="<local-shim-token>",
)

print(client.models.list())
print(client.chat.completions.create(
    model="gpt-5.5",
    messages=[{"role": "user", "content": "Say pong"}],
))
```

## Run

```bash
python -m pip install -e '.[test]'
CODEX_SHIM_TOKEN="$(python - <<'PY'
import secrets
print(secrets.token_urlsafe(32))
PY
)" codex-openai-shim --port 8000
```

The `api_key` is a local shim bearer token. It is not an OpenAI API key and is never proxied to Codex.

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
python -m pip install -e '.[test]'
pytest -m "not integration" tests
```

Live contract tests are opt-in because they start the FastAPI shim, launch `codex app-server` over stdio, and require working Codex auth:

```bash
CODEX_SHIM_RUN_CONTRACT_TESTS=1 pytest -m integration tests/test_contract.py
```

The contract tests instantiate the official OpenAI SDK with `base_url=.../v1`, call `/v1/models`, call non-streaming chat completions including a multi-turn message chain and image content part, call streaming SSE chat completions, and verify missing/invalid local bearer tokens are rejected.
The coverage gate is configured at 95% in `pyproject.toml`.

The generated Codex app-server JSON Schema and TypeScript protocol references used for this implementation are in `generated/`.

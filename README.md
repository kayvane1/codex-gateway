# Codex Gateway

Run OpenAI SDK examples locally using your Codex/ChatGPT subscription.

`codex-gateway` is a small local server that speaks a subset of the OpenAI API and forwards requests to `codex app-server`. That means many examples written for the official OpenAI Python SDK can run with only a `base_url` change, without creating or using an OpenAI API key.

It is useful for trying OpenAI cookbook snippets, SDK examples, prototypes, and small experiments when you already have Codex available through your ChatGPT account.

> Experimental: this is a local compatibility gateway, not a production OpenAI API replacement. Keep it bound to loopback and expect the supported API surface, HTTP contract, and safety model to evolve.

## What It Does

Normally an OpenAI SDK example looks like this:

```python
from openai import OpenAI

client = OpenAI(api_key="sk-...")
```

With `codex-gateway`, you point the same SDK at a local server instead:

```python
import os

from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key=os.environ["CODEX_GATEWAY_KEY"],
)
```

Your script still uses the official OpenAI SDK. The request goes to `codex-gateway`, which talks to your local `codex app-server` session.

The token above is only a local gateway bearer token. It is not an OpenAI API key and is never proxied to Codex.

## Why Use This?

Use this project when you want to:

- Try OpenAI cookbook examples without setting up OpenAI API billing.
- Run small SDK experiments through your existing Codex/ChatGPT subscription.
- Test code that uses `client.chat.completions.create(...)`.
- Try simple `client.responses.create(...)` calls.
- Keep example code close to the official OpenAI SDK shape.
- Experiment locally before deciding whether you need the full OpenAI API.

This is especially handy for learning, demos, and quick prototypes.

## Requirements

You need:

- Python 3.11 or newer.
- `uv` / `uvx`.
- The Codex CLI installed and authenticated.
- Access to `codex app-server`.
- The official OpenAI Python SDK in the project or example you are running.

## Quick Start

Initialize the gateway once:

```bash
uvx codex-gateway init
```

Start the local gateway:

```bash
uvx codex-gateway
```

The gateway listens on:

```text
http://127.0.0.1:8000/v1
```

`codex-gateway init` writes `~/.config/codex-gateway/config.json` with a generated local bearer token and `0600` permissions. The server reads that config automatically on future runs.

To print your SDK setup again later:

```bash
uvx codex-gateway show
```

For scripts that only need the token value:

```bash
uvx codex-gateway token
```

For notebooks or SDK examples, put the token in a gateway-specific environment variable:

```bash
export CODEX_GATEWAY_KEY="$(uvx codex-gateway token)"
```

For shell-based setup:

```bash
uvx codex-gateway env
```

If neither config nor `CODEX_GATEWAY_TOKEN` is set, the gateway prints a generated local token at startup.

## Example

```python
import os

from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key=os.environ["CODEX_GATEWAY_KEY"],
)

response = client.chat.completions.create(
    model="gpt-5.5",
    messages=[
        {"role": "user", "content": "Say pong"},
    ],
)

print(response.choices[0].message.content)
```

Responses API text calls are also supported:

```python
response = client.responses.create(
    model="gpt-5.5",
    input="Say pong",
)

print(response.output_text)
```

## Using With Cookbooks Or SDK Examples

Find where the example creates the OpenAI client:

```python
from openai import OpenAI

client = OpenAI()
```

Replace it with:

```python
import os

from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key=os.environ["CODEX_GATEWAY_KEY"],
)
```

Then run the example as usual.

Some examples use OpenAI API features this gateway does not support yet. When that happens, the gateway returns an explicit `400` error instead of pretending the feature worked.

## Compatibility

`codex-gateway` is tested as a local compatibility adapter, not as a production OpenAI API replacement.

| Surface | v0.2.0 expectation | Verification |
| --- | --- | --- |
| Python | Package metadata requires Python 3.11 or newer. CI targets 3.11, 3.12, and 3.13. | Unit tests and lint in GitHub Actions. |
| OpenAI Python SDK | Development and contract tests use `openai>=2.0` against the supported chat and Responses subsets. | Opt-in contract tests instantiate the official SDK with `base_url=.../v1`. |
| Codex CLI/app-server | Requires a local `codex app-server --listen stdio://` that supports the app-server methods used by this gateway. | Live contract tests are opt-in because they require Codex CLI/app-server and auth on the runner. |
| OpenAI HTTP API surface | Only `/v1/models` and the documented subsets of `/v1/chat/completions` and `/v1/responses` are supported. | Unsupported request features return explicit 400 errors. |

No `LICENSE` file is currently present, so the package metadata intentionally does not declare a license. Choosing and adding a license is a release blocker before public package distribution.

## Supported API Surface

Currently implemented:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `stream=True` SSE chunks compatible with `openai.OpenAI(...).chat.completions.create(..., stream=True)`
- Multi-turn text chat history via Codex `thread/inject_items`
- OpenAI `image_url` content parts in user messages, mapped to Codex image inputs/history
- Non-streaming Responses API text output for `client.responses.create(model=..., input="...")` and text-only message-style input
- Local bearer-token authentication
- Codex app-server stdio transport

## Not Supported Yet

This gateway intentionally supports only a small OpenAI-compatible subset. Unsupported features return a `400` instead of pretending to work.

Unsupported features include:

- Tools/functions
- Built-in Responses tools
- Tool choice
- Background mode
- Streaming Responses API calls
- Parallel tool calls
- Response formats
- JSON/schema-constrained output
- File inputs
- Audio
- `n > 1`
- Logprobs
- Sampling controls
- Stop sequences
- Token limits
- Stored response/conversation state
- Metadata storage
- Non-chat/non-Responses APIs such as `/v1/embeddings` and legacy completions

## Safety Defaults

`codex-gateway` is designed for local experimentation.

By default it:

- Binds only to `127.0.0.1`.
- Uses a local bearer token.
- Starts ephemeral Codex threads with `sandbox="read-only"`, `approvalPolicy="never"`, no environments, and no dynamic tools.
- Unsubscribes from each per-request thread after the API response finishes.
- Accepts only `http://`, `https://`, and `data:image/...` image URLs; local file URLs are rejected.
- Denies app-server requests for command execution, patch approval, file-change approval, MCP elicitations, and dynamic tool calls.
- Does not expose app-server filesystem, shell, config, auth, or account endpoints through the OpenAI-compatible HTTP API.
- Does not print, persist, proxy, or expose Codex auth credentials.

Do not expose this server publicly.

## Local Development

```bash
uv sync --group dev
uv run codex-gateway --port 8000
```

Run tests:

```bash
uv run --group dev pytest -m "not integration" tests
```

Run linting:

```bash
uv run --group dev ruff check src tests
uv run --group dev ruff format --check src tests
uv run --group dev pre-commit install
uv run --group dev pre-commit run --all-files
```

Live contract tests are opt-in because they start the FastAPI gateway, launch `codex app-server` over stdio, and require working Codex auth:

```bash
CODEX_GATEWAY_RUN_CONTRACT_TESTS=1 uv run --group dev pytest -m integration tests/test_contract.py
```

The contract tests instantiate the official OpenAI SDK with `base_url=.../v1`, call `/v1/models`, call non-streaming chat completions including a multi-turn message chain and image content part, call non-streaming Responses text output, call streaming SSE chat completions, and verify missing/invalid local bearer tokens are rejected.

The coverage gate is configured at 95% in `pyproject.toml`.

## Architecture

- `chat_contract.py` owns the OpenAI-facing chat subset: validation, text/image message conversion, history item construction, completion payloads, streaming chunks, SSE framing, and OpenAI-style errors.
- `responses_contract.py` owns the OpenAI-facing Responses subset: validation, text-only message input conversion, response payloads, usage mapping, and explicit unsupported-feature errors.
- `_app_server_stdio_session.py` owns the Codex app-server stdio transport: JSON-RPC request correlation, notification fan-out, stderr redaction, shutdown, and denial responses for privileged app-server requests.
- `_codex_turn_lifecycle.py` owns Codex turn behavior: starting turns, collecting final assistant output, streaming assistant deltas, draining idle notifications, and mapping token usage.
- `codex_client.py` is the high-level client used by FastAPI. It initializes the app-server session, lists models, starts ephemeral read-only Codex threads, injects prior text/image history, materializes image data URLs for live turns, delegates turn execution, and unsubscribes when finished.

## Protocol Artifacts

Codex app-server protocol references are intentionally not committed. Regenerate them locally when needed:

```bash
uv run codex-gateway generate-json-schema
uv run codex-gateway generate-ts
```

These write ignored files under `generated/` by default. Pass `--out <dir>` to use a different location.

## Release Checklist

- Update `CHANGELOG.md` with compatibility notes for the release.
- Confirm `pyproject.toml` metadata, classifiers, project URLs, and `MANIFEST.in` package exclusions are still accurate.
- Run `uv run --group dev pytest -m "not integration" tests`.
- Run `uv run --group dev ruff check src tests`.
- Run `uv build` and `uvx twine check dist/*.whl dist/*.tar.gz`.
- Run contract tests only when a runner has Codex CLI/app-server and auth: `CODEX_GATEWAY_RUN_CONTRACT_TESTS=1 uv run --group dev pytest -m integration tests/test_contract.py`.
- Do not commit `generated/` protocol artifacts; regenerate them locally when reviewing protocol changes.
- Do not publish publicly until an explicit repository license has been selected and added.

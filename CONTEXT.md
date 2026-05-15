# Project Context

This project is a local OpenAI-compatible HTTP shim for Codex app-server. It lets examples written for the official OpenAI Python SDK use `OpenAI(base_url="http://127.0.0.1:<port>/v1", api_key=<local-shim-token>)` while keeping Codex credentials and privileged app-server capabilities out of the OpenAI-style request surface.

## Domain Language

- OpenAI chat contract: the supported subset of `/v1/chat/completions`, including request validation, text and image message conversion, prior-history construction, OpenAI-shaped responses, streaming SSE framing, and explicit errors for unsupported OpenAI features.
- Local shim token: the bearer token accepted by this local HTTP server. It is not an OpenAI API key and must not be proxied, persisted, or confused with Codex auth.
- Codex app-server stdio session: the newline-delimited JSON-RPC transport to `codex app-server --listen stdio://`, including request correlation, notification fan-out, stderr redaction, and denial responses for privileged app-server requests.
- Codex turn lifecycle: the thread/turn interaction that starts a Codex turn, interprets app-server notifications, produces the final assistant text, streams assistant deltas, and maps Codex token usage into OpenAI-style usage.
- High-level Codex client: the narrow API used by the FastAPI server: initialize/stop the app-server session, list models, start ephemeral threads, inject prior OpenAI text/image history, materialize image data URLs for live turns, run one non-streaming or streaming chat turn at a time, and unsubscribe from the thread afterward.
- Unsupported feature policy: the shim returns explicit OpenAI-style errors for API features it cannot honor instead of silently ignoring or pretending to support them.

## Safety Boundaries

- The HTTP server binds to `127.0.0.1` by default.
- Normal OpenAI SDK calls must not expose arbitrary filesystem mutation, shell execution, Codex account APIs, or app-server auth credentials.
- Codex threads are started as ephemeral, read-only, no-approval turns with no dynamic tools or environments.
- Prior OpenAI user messages are injected as text/image model-visible history, prior assistant messages are injected as text output, and only the latest user message becomes the live Codex turn input.
- OpenAI image inputs are accepted only from user messages as remote HTTP(S) URLs or image data URLs; image data URLs are written to temporary local files only for the lifetime of the live Codex turn, and caller-supplied local file URLs are not passed through.
- App-server requests for command execution, patch/file approvals, tool calls, user input, MCP elicitation, and Codex auth refresh are declined or rejected by the stdio session layer.

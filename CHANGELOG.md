# Changelog

## 0.2.0 - 2026-05-17

### Release Posture

- Keeps `codex-gateway` experimental and local-only while tightening package metadata, CI checks, and release documentation.
- Expects Python 3.11 or newer, with CI intended to exercise Python 3.11, 3.12, and 3.13.
- Expects a Codex CLI/app-server install that supports `codex app-server --listen stdio://` and the app-server methods used by the committed JSON Schema protocol snapshot and gateway client.
- Leaves generated TypeScript protocol bindings out of git; use `codex-gateway generate-ts` to recreate them locally under ignored `generated/app-server-ts/`.
- Keeps live Codex app-server contract tests opt-in because they require a working local Codex CLI, app-server, and Codex authentication.

### Release Blockers

- No `LICENSE` file is present in the repository, so `pyproject.toml` intentionally does not declare a license. Select and add an explicit license before public package distribution.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]+"),
    re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.IGNORECASE),
    re.compile(r"(access[_-]?token[\"'=:\s]+)[A-Za-z0-9._\-]+", re.IGNORECASE),
    re.compile(r"(refresh[_-]?token[\"'=:\s]+)[A-Za-z0-9._\-]+", re.IGNORECASE),
)


class CodexAppServerError(RuntimeError):
    """Raised when Codex app-server returns an error or exits unexpectedly."""


class CodexAppServerProtocolError(CodexAppServerError):
    """Raised when Codex app-server returns a JSON-RPC protocol error."""

    def __init__(
        self,
        message: str,
        *,
        code: int | None = None,
        data: Any = None,
        method: str | None = None,
    ) -> None:
        self.code = code
        self.data = data
        self.method = method
        detail = message or "Codex app-server returned an error."
        if method is not None:
            detail = f"{method} failed: {detail}"
        if code is not None:
            detail = f"{detail} (JSON-RPC code {code})"
        super().__init__(detail)


class CodexProtocolCompatibilityError(CodexAppServerError):
    """Raised when the installed Codex app-server is not compatible enough for the gateway."""

    def __init__(self, message: str, *, phase: str, method: str | None = None) -> None:
        self.phase = phase
        self.method = method
        super().__init__(message)


class CodexTurnTimeout(CodexAppServerError):
    """Raised when a Codex turn does not produce a final assistant answer."""


@dataclass(frozen=True)
class CodexClientSettings:
    command: tuple[str, ...] = ("codex", "app-server", "--listen", "stdio://")
    cwd: Path = field(default_factory=lambda: Path.cwd())
    request_timeout_seconds: float = 30.0
    turn_timeout_seconds: float = 180.0
    reasoning_effort: str = "low"
    chat_max_active_turns: int = 1
    chat_max_pending_turns: int | None = None
    chat_admission_timeout_seconds: float | None = None


@dataclass(frozen=True)
class CodexChatResult:
    text: str
    usage: dict[str, int]


def build_text_input(text: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": text, "text_elements": []}]


def redact_secrets(value: str) -> str:
    redacted = value
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(lambda match: f"{match.group(1) if match.groups() else ''}[redacted]", redacted)
    return redacted

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class CodexAppServerError(RuntimeError):
    """Raised when Codex app-server returns an error or exits unexpectedly."""


class CodexTurnTimeout(CodexAppServerError):
    """Raised when a Codex turn does not produce a final assistant answer."""


@dataclass(frozen=True)
class CodexClientSettings:
    command: tuple[str, ...] = ("codex", "app-server", "--listen", "stdio://")
    cwd: Path = field(default_factory=lambda: Path.cwd())
    request_timeout_seconds: float = 30.0
    turn_timeout_seconds: float = 180.0
    reasoning_effort: str = "low"


@dataclass(frozen=True)
class CodexChatResult:
    text: str
    usage: dict[str, int]


def build_text_input(text: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": text, "text_elements": []}]

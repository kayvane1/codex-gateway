from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ._codex_shared import (
    CodexAppServerError,
    CodexAppServerProtocolError,
    CodexClientSettings,
    CodexProtocolCompatibilityError,
)

INITIALIZE_PARAMS: dict[str, Any] = {
    "clientInfo": {
        "name": "codex-gateway",
        "title": "Codex Gateway",
        "version": "0.2.0",
    },
    "capabilities": {
        "experimentalApi": True,
        "optOutNotificationMethods": [],
    },
}


class ProtocolPreflightSession(Protocol):
    async def start(self) -> None: ...

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any: ...


@dataclass(frozen=True)
class ProtocolCompatibilityReport:
    command: tuple[str, ...]
    cwd: Path
    model_count: int


class ProtocolCompatibilityPreflight:
    """Verifies the installed `codex app-server` runtime before chat turns can begin."""

    def __init__(
        self,
        *,
        session: ProtocolPreflightSession,
        settings: CodexClientSettings,
    ) -> None:
        self._session = session
        self._settings = settings

    async def run(self) -> ProtocolCompatibilityReport:
        await self._start_binary()
        initialize_result = await self._request(
            "initialize",
            INITIALIZE_PARAMS,
            phase="initialize",
            description="initialize the app-server protocol",
        )
        self._validate_initialize_response(initialize_result)
        model_list = await self._request(
            "model/list",
            {"includeHidden": False},
            phase="model/list",
            description="reach model/list",
        )
        models = self._model_data(model_list)
        return ProtocolCompatibilityReport(
            command=self._settings.command,
            cwd=self._settings.cwd,
            model_count=len(models),
        )

    async def _start_binary(self) -> None:
        try:
            await self._session.start()
        except FileNotFoundError as exc:
            raise self._compat_error(
                "binary-start",
                (
                    f"Unable to start Codex app-server command `{self._command_label()}`. "
                    "Install or update the Codex CLI and make sure the `codex` binary is on PATH."
                ),
            ) from exc
        except OSError as exc:
            raise self._compat_error(
                "binary-start",
                f"Unable to start Codex app-server command `{self._command_label()}`: {exc}",
            ) from exc

    async def _request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        phase: str,
        description: str,
    ) -> Any:
        try:
            return await self._session.request(
                method,
                params,
                timeout=self._settings.request_timeout_seconds,
            )
        except TimeoutError as exc:
            raise self._compat_error(
                phase,
                (
                    f"Timed out waiting for Codex app-server to {description} within "
                    f"{self._settings.request_timeout_seconds:g}s. The installed binary may not support the "
                    "stdio app-server protocol expected by this gateway."
                ),
                method=method,
            ) from exc
        except CodexAppServerProtocolError as exc:
            raise self._protocol_error(phase, method, exc) from exc
        except CodexAppServerError as exc:
            raise self._compat_error(
                phase,
                f"Codex app-server could not {description}: {exc}",
                method=method,
            ) from exc
        except OSError as exc:
            raise self._compat_error(
                phase,
                f"Codex app-server I/O failed while trying to {description}: {exc}",
                method=method,
            ) from exc

    def _validate_initialize_response(self, result: Any) -> None:
        if not isinstance(result, dict):
            raise self._compat_error(
                "initialize",
                (
                    "Codex app-server returned a malformed initialize response. Expected an object with app-server "
                    f"metadata, got {type(result).__name__}."
                ),
                method="initialize",
            )
        required_strings = ("codexHome", "platformFamily", "platformOs", "userAgent")
        missing = [key for key in required_strings if not isinstance(result.get(key), str) or not result[key]]
        if missing:
            raise self._compat_error(
                "initialize",
                "Codex app-server returned a malformed initialize response. Missing field(s): " + ", ".join(missing),
                method="initialize",
            )

    def _model_data(self, result: Any) -> list[Any]:
        if not isinstance(result, dict):
            raise self._compat_error(
                "model/list",
                (
                    "Codex app-server returned a malformed model/list response. Expected an object with a `data` "
                    f"array, got {type(result).__name__}."
                ),
                method="model/list",
            )
        data = result.get("data")
        if not isinstance(data, list):
            raise self._compat_error(
                "model/list",
                "Codex app-server returned a malformed model/list response. Expected `data` to be an array.",
                method="model/list",
            )
        for index, model in enumerate(data):
            if not isinstance(model, dict):
                raise self._compat_error(
                    "model/list",
                    f"Codex app-server returned a malformed model/list response. Expected `data[{index}]` to be an object.",
                    method="model/list",
                )
            model_id = model.get("id") or model.get("model")
            if not isinstance(model_id, str) or not model_id:
                raise self._compat_error(
                    "model/list",
                    (
                        "Codex app-server returned a malformed model/list response. "
                        f"Expected `data[{index}]` to include a non-empty model id."
                    ),
                    method="model/list",
                )
        return data

    def _protocol_error(
        self,
        phase: str,
        method: str,
        error: CodexAppServerProtocolError,
    ) -> CodexProtocolCompatibilityError:
        message = str(error)
        if error.code == -32601 or "method not found" in message.lower():
            return self._compat_error(
                phase,
                (
                    f"Codex app-server protocol is incompatible: `{method}` is not available from the installed "
                    "Codex runtime. Update the Codex CLI so `codex app-server --listen stdio://` supports the "
                    "gateway protocol."
                ),
                method=method,
            )
        if error.code == -32602 or "invalid params" in message.lower():
            return self._compat_error(
                phase,
                (
                    f"Codex app-server rejected the `{method}` preflight parameters. The installed Codex runtime "
                    f"and gateway protocol are out of sync. Details: {message}"
                ),
                method=method,
            )
        return self._compat_error(
            phase,
            f"Codex app-server returned a protocol error during `{method}` preflight. Details: {message}",
            method=method,
        )

    def _compat_error(
        self,
        phase: str,
        message: str,
        *,
        method: str | None = None,
    ) -> CodexProtocolCompatibilityError:
        return CodexProtocolCompatibilityError(message, phase=phase, method=method)

    def _command_label(self) -> str:
        return shlex.join(self._settings.command)

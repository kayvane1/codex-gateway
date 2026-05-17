from __future__ import annotations

import asyncio
import base64
import contextlib
import tempfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

from ._app_server_stdio_session import AppServerStdioSession
from ._codex_shared import (
    CodexAppServerError,
    CodexChatResult,
    CodexClientSettings,
    CodexTurnTimeout,
    build_text_input,
)
from ._codex_turn_lifecycle import CodexTurnLifecycle

__all__ = [
    "CodexAppServer",
    "CodexAppServerError",
    "CodexChatResult",
    "CodexClientSettings",
    "CodexTurnTimeout",
    "build_text_input",
]

DATA_IMAGE_PREFIX = "data:image/"
IMAGE_EXTENSION_BY_MEDIA_SUBTYPE = {
    "jpeg": "jpg",
    "jpg": "jpg",
    "png": "png",
    "gif": "gif",
    "webp": "webp",
}


class _SafetyBoundTurnSession:
    def __init__(self, session: AppServerStdioSession) -> None:
        self._session = session

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        if method == "turn/start":
            params = _turn_start_params_with_safety(params)
        return await self._session.request(method, params, timeout=timeout)

    def subscribe(self, thread_id: str) -> Any:
        return self._session.subscribe(thread_id)


def _thread_start_safety_params() -> dict[str, Any]:
    return {
        "approvalPolicy": "never",
        "approvalsReviewer": "user",
        "sandbox": "read-only",
        "environments": [],
        "dynamicTools": [],
    }


def _turn_start_params_with_safety(params: dict[str, Any] | None) -> dict[str, Any]:
    safe_params = dict(params or {})
    safe_params.update(
        {
            "approvalPolicy": "never",
            "approvalsReviewer": "user",
            "environments": [],
            "sandboxPolicy": {"type": "readOnly", "networkAccess": False},
        }
    )
    return safe_params


class CodexAppServer:
    """High-level Codex app-server client used by the OpenAI gateway."""

    def __init__(self, settings: CodexClientSettings | None = None) -> None:
        self.settings = settings or CodexClientSettings()
        self._session = AppServerStdioSession(
            command=self.settings.command,
            cwd=self.settings.cwd,
            request_timeout_seconds=self.settings.request_timeout_seconds,
        )
        self._turn_session = _SafetyBoundTurnSession(self._session)
        self._turn_lifecycle = CodexTurnLifecycle(
            session=self._turn_session,  # type: ignore[arg-type]
            request_timeout_seconds=self.settings.request_timeout_seconds,
            turn_timeout_seconds=self.settings.turn_timeout_seconds,
            reasoning_effort=self.settings.reasoning_effort,
        )
        self._chat_lock = asyncio.Lock()
        self._initialized = False

    async def start(self) -> None:
        if self._initialized:
            return
        await self._session.start()
        await self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "codex-gateway",
                    "title": "Codex Gateway",
                    "version": "0.1.0",
                },
                "capabilities": {
                    "experimentalApi": True,
                    "optOutNotificationMethods": [],
                },
            },
        )
        self._initialized = True

    async def stop(self) -> None:
        self._initialized = False
        await self._session.stop()

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        return await self._session.request(method, params, timeout=timeout)

    async def list_models(self) -> list[dict[str, Any]]:
        result = await self.request("model/list", {"includeHidden": False})
        return list(result.get("data", []))

    async def complete_chat(
        self,
        *,
        model: str,
        history_items: list[dict[str, Any]],
        input_items: list[dict[str, Any]],
        developer_instructions: str,
    ) -> CodexChatResult:
        async with self._chat_lock:
            thread_id = await self._start_thread(model, developer_instructions)
            try:
                await self._inject_history(thread_id, history_items)
                with _materialize_data_image_inputs(input_items) as prepared_input_items:
                    return await self._turn_lifecycle.complete(thread_id, prepared_input_items)
            finally:
                await self._unsubscribe_thread(thread_id)

    async def stream_chat(
        self,
        *,
        model: str,
        history_items: list[dict[str, Any]],
        input_items: list[dict[str, Any]],
        developer_instructions: str,
    ) -> AsyncIterator[str]:
        async with self._chat_lock:
            thread_id = await self._start_thread(model, developer_instructions)
            try:
                await self._inject_history(thread_id, history_items)
                with _materialize_data_image_inputs(input_items) as prepared_input_items:
                    async for delta in self._turn_lifecycle.stream(thread_id, prepared_input_items):
                        yield delta
            finally:
                await self._unsubscribe_thread(thread_id)

    async def _start_thread(self, model: str, developer_instructions: str) -> str:
        response = await self.request(
            "thread/start",
            {
                "model": model,
                "cwd": str(self.settings.cwd),
                **_thread_start_safety_params(),
                "baseInstructions": (
                    "You are a text-only assistant behind a local OpenAI-compatible "
                    "compatibility gateway. Answer directly in plain text. Do not execute "
                    "shell commands, read or write files, call tools, or request approvals."
                ),
                "developerInstructions": developer_instructions,
                "ephemeral": True,
                "experimentalRawEvents": False,
                "persistExtendedHistory": False,
            },
            timeout=self.settings.request_timeout_seconds,
        )
        return str(response["thread"]["id"])

    async def _inject_history(self, thread_id: str, history_items: list[dict[str, Any]]) -> None:
        if not history_items:
            return
        await self.request(
            "thread/inject_items",
            {
                "threadId": thread_id,
                "items": history_items,
            },
            timeout=self.settings.request_timeout_seconds,
        )

    async def _unsubscribe_thread(self, thread_id: str) -> None:
        try:
            await self.request(
                "thread/unsubscribe",
                {"threadId": thread_id},
                timeout=self.settings.request_timeout_seconds,
            )
        except Exception:
            # Cleanup is best-effort; preserve the original API response/error.
            return


@contextlib.contextmanager
def _materialize_data_image_inputs(input_items: list[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
    tempdir: tempfile.TemporaryDirectory[str] | None = None
    prepared_items: list[dict[str, Any]] = []
    image_index = 0
    try:
        for item in input_items:
            if (
                item.get("type") != "image"
                or not isinstance(item.get("url"), str)
                or not item["url"].startswith(DATA_IMAGE_PREFIX)
            ):
                prepared_items.append(item)
                continue

            tempdir = tempdir or tempfile.TemporaryDirectory(prefix="codex-gateway-images-")
            media_type, encoded = item["url"].split(",", 1)
            extension = _image_extension(media_type)
            image_path = Path(tempdir.name) / f"image-{image_index}.{extension}"
            image_path.write_bytes(base64.b64decode(encoded, validate=True))
            prepared_items.append({"type": "localImage", "path": str(image_path)})
            image_index += 1

        yield prepared_items
    finally:
        if tempdir is not None:
            tempdir.cleanup()


def _image_extension(media_type: str) -> str:
    subtype = media_type.removeprefix("data:image/").removesuffix(";base64").lower()
    return IMAGE_EXTENSION_BY_MEDIA_SUBTYPE.get(subtype, "img")

from __future__ import annotations

import contextlib
import tempfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

from ._app_server_stdio_session import AppServerStdioSession
from ._codex_shared import (
    CodexAppServerError,
    CodexAppServerProtocolError,
    CodexChatResult,
    CodexClientSettings,
    CodexProtocolCompatibilityError,
    CodexTurnTimeout,
    build_text_input,
)
from ._codex_turn_lifecycle import CodexTurnLifecycle
from .protocol_compat import ProtocolCompatibilityPreflight, ProtocolCompatibilityReport
from .safety_policy import (
    is_data_image_url,
    parse_data_image_url,
    thread_start_safety_params,
    turn_start_params_with_safety,
)
from .thread_lifetime import (
    ChatAdmissionPolicy,
    CodexChatAdmissionCancelled,
    CodexChatAdmissionError,
    CodexChatAdmissionTimeout,
    CodexChatOverloaded,
    ThreadLifetime,
)

__all__ = [
    "CodexAppServer",
    "CodexAppServerError",
    "CodexAppServerProtocolError",
    "CodexChatAdmissionCancelled",
    "CodexChatAdmissionError",
    "CodexChatAdmissionTimeout",
    "CodexChatResult",
    "CodexChatOverloaded",
    "CodexClientSettings",
    "CodexProtocolCompatibilityError",
    "CodexTurnTimeout",
    "build_text_input",
]


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
            params = turn_start_params_with_safety(params)
        return await self._session.request(method, params, timeout=timeout)

    def subscribe(self, thread_id: str) -> Any:
        return self._session.subscribe(thread_id)


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
        self._thread_lifetime = ThreadLifetime(
            ChatAdmissionPolicy(
                max_active_turns=self.settings.chat_max_active_turns,
                max_pending_turns=self.settings.chat_max_pending_turns,
                timeout_seconds=self._chat_admission_timeout_seconds(),
            )
        )
        self._initialized = False
        self._preflight_report: ProtocolCompatibilityReport | None = None

    async def start(self) -> None:
        if self._initialized:
            return
        try:
            self._preflight_report = await ProtocolCompatibilityPreflight(
                session=self._session,
                settings=self.settings,
            ).run()
        except Exception:
            with contextlib.suppress(Exception):
                await self._session.stop()
            raise
        self._initialized = True

    async def stop(self) -> None:
        self._initialized = False
        self._preflight_report = None
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
        async with self._thread_lifetime.chat_thread(
            start_thread=lambda: self._start_thread(model, developer_instructions),
            cleanup_thread=self._unsubscribe_thread,
        ) as thread_id:
            await self._inject_history(thread_id, history_items)
            with _materialize_data_image_inputs(input_items) as prepared_input_items:
                return await self._turn_lifecycle.complete(thread_id, prepared_input_items)

    async def stream_chat(
        self,
        *,
        model: str,
        history_items: list[dict[str, Any]],
        input_items: list[dict[str, Any]],
        developer_instructions: str,
    ) -> AsyncIterator[str]:
        async with self._thread_lifetime.chat_thread(
            start_thread=lambda: self._start_thread(model, developer_instructions),
            cleanup_thread=self._unsubscribe_thread,
        ) as thread_id:
            await self._inject_history(thread_id, history_items)
            with _materialize_data_image_inputs(input_items) as prepared_input_items:
                async for delta in self._turn_lifecycle.stream(thread_id, prepared_input_items):
                    yield delta

    def _chat_admission_timeout_seconds(self) -> float | None:
        if self.settings.chat_admission_timeout_seconds is not None:
            return self.settings.chat_admission_timeout_seconds
        return self.settings.request_timeout_seconds

    async def _start_thread(self, model: str, developer_instructions: str) -> str:
        response = await self.request(
            "thread/start",
            {
                "model": model,
                "cwd": str(self.settings.cwd),
                **thread_start_safety_params(),
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
                or not is_data_image_url(item["url"])
            ):
                prepared_items.append(item)
                continue

            tempdir = tempdir or tempfile.TemporaryDirectory(prefix="codex-gateway-images-")
            data_image = parse_data_image_url(item["url"], param="image_url.url")
            image_path = Path(tempdir.name) / f"image-{image_index}.{data_image.extension}"
            image_path.write_bytes(data_image.data)
            prepared_items.append({"type": "localImage", "path": str(image_path)})
            image_index += 1

        yield prepared_items
    finally:
        if tempdir is not None:
            tempdir.cleanup()

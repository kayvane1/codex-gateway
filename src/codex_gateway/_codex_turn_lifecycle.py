from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ._app_server_stdio_session import AppServerStdioSession
from ._codex_shared import CodexAppServerError, CodexChatResult, CodexTurnTimeout, redact_secrets


class _TurnTerminalState(Enum):
    ASSISTANT_MESSAGE_COMPLETED = "assistant_message_completed"
    TURN_COMPLETED = "turn_completed"
    IDLE = "idle"


class _TurnEventErrorKind(Enum):
    TURN_FAILED = "turn_failed"
    NOTIFICATION_ERROR = "notification_error"


@dataclass(frozen=True)
class _CodexTurnEventError:
    kind: _TurnEventErrorKind
    message: str

    def to_exception(self) -> CodexAppServerError:
        return CodexAppServerError(self.message)


@dataclass(frozen=True)
class _ReducedCodexTurnEvent:
    assistant_delta: str = ""
    completion_text: str | None = None
    usage: dict[str, int] | None = None
    terminal_state: _TurnTerminalState | None = None
    error: _CodexTurnEventError | None = None


class _CodexTurnEventReducer:
    """Concentrates Codex app-server notification interpretation for a turn."""

    def __init__(self) -> None:
        self.text = ""
        self.usage = _default_openai_usage()
        self.assistant_completed = False
        self.turn_completed = False
        self.idle = False
        self.events_seen = 0
        self.last_event_method: str | None = None

    def reduce(self, event: dict[str, Any]) -> _ReducedCodexTurnEvent:
        self.events_seen += 1
        method = str(event.get("method") or "")
        self.last_event_method = method or "<missing>"
        params = _dict_or_empty(event.get("params"))

        if method == "item/agentMessage/delta":
            delta = str(params.get("delta") or "")
            self.text += delta
            return _ReducedCodexTurnEvent(assistant_delta=delta)

        if method == "item/completed":
            return self._reduce_item_completed(params)

        if method == "thread/tokenUsage/updated":
            self.usage = openai_usage_from_codex(_dict_or_empty(params.get("tokenUsage")))
            return _ReducedCodexTurnEvent(usage=self.usage)

        if method == "thread/status/changed":
            status = _dict_or_empty(params.get("status"))
            if status.get("type") == "idle":
                self.idle = True
                return _ReducedCodexTurnEvent(terminal_state=_TurnTerminalState.IDLE)
            return _ReducedCodexTurnEvent()

        if method == "turn/completed":
            turn = _dict_or_empty(params.get("turn"))
            if turn.get("error"):
                return _ReducedCodexTurnEvent(
                    error=_CodexTurnEventError(
                        kind=_TurnEventErrorKind.TURN_FAILED,
                        message=f"Codex turn failed: {_format_app_server_error(turn['error'])}",
                    )
                )
            self.turn_completed = True
            return _ReducedCodexTurnEvent(terminal_state=_TurnTerminalState.TURN_COMPLETED)

        if method == "error":
            return _ReducedCodexTurnEvent(
                error=_CodexTurnEventError(
                    kind=_TurnEventErrorKind.NOTIFICATION_ERROR,
                    message=_format_error_notification(params),
                )
            )

        return _ReducedCodexTurnEvent()

    def timeout_message(self, waiting_for: str) -> str:
        observed_state = []
        if self.assistant_completed:
            observed_state.append("assistant_completed")
        if self.turn_completed:
            observed_state.append("turn_completed")
        if self.idle:
            observed_state.append("thread_idle")
        state = ", ".join(observed_state) or "awaiting_assistant_output"
        last_event = self.last_event_method or "none"
        return (
            f"Timed out waiting for Codex {waiting_for}. "
            f"Last event: {last_event}; events seen: {self.events_seen}; state: {state}."
        )

    def _reduce_item_completed(self, params: dict[str, Any]) -> _ReducedCodexTurnEvent:
        item = _dict_or_empty(params.get("item"))
        if item.get("type") != "agentMessage":
            return _ReducedCodexTurnEvent()

        prior_text = self.text
        completion_text = str(item.get("text") or "")
        assistant_delta = ""
        if completion_text:
            self.text = completion_text
            if completion_text != prior_text:
                assistant_delta = (
                    completion_text[len(prior_text) :] if completion_text.startswith(prior_text) else completion_text
                )
        self.assistant_completed = True
        return _ReducedCodexTurnEvent(
            assistant_delta=assistant_delta,
            completion_text=completion_text,
            terminal_state=_TurnTerminalState.ASSISTANT_MESSAGE_COMPLETED,
        )


class CodexTurnLifecycle:
    """Runs a Codex turn and interprets app-server notifications."""

    def __init__(
        self,
        *,
        session: AppServerStdioSession,
        request_timeout_seconds: float,
        turn_timeout_seconds: float,
        reasoning_effort: str,
    ) -> None:
        self._session = session
        self._request_timeout_seconds = request_timeout_seconds
        self._turn_timeout_seconds = turn_timeout_seconds
        self._reasoning_effort = reasoning_effort

    async def complete(self, thread_id: str, input_items: list[dict[str, Any]]) -> CodexChatResult:
        async with self._session.subscribe(thread_id) as events:
            await self._start_turn(thread_id, input_items)
            return await self._collect_completion(events)

    async def stream(self, thread_id: str, input_items: list[dict[str, Any]]) -> AsyncIterator[str]:
        async with self._session.subscribe(thread_id) as events:
            await self._start_turn(thread_id, input_items)
            async for delta in self._stream_completion(events):
                yield delta
            await self._drain_until_idle(events)

    async def _start_turn(self, thread_id: str, input_items: list[dict[str, Any]]) -> None:
        await self._session.request(
            "turn/start",
            {
                "threadId": thread_id,
                "input": input_items,
                "effort": self._reasoning_effort,
            },
            timeout=self._request_timeout_seconds,
        )

    async def _collect_completion(self, events: asyncio.Queue[dict[str, Any]]) -> CodexChatResult:
        reducer = _CodexTurnEventReducer()
        deadline = time.monotonic() + self._turn_timeout_seconds

        while True:
            timeout = min(1.0 if reducer.assistant_completed else 5.0, max(0.1, deadline - time.monotonic()))
            try:
                event = await asyncio.wait_for(events.get(), timeout=timeout)
            except asyncio.TimeoutError as exc:
                if reducer.assistant_completed:
                    return CodexChatResult(text=reducer.text, usage=reducer.usage)
                if time.monotonic() >= deadline:
                    raise CodexTurnTimeout(reducer.timeout_message("assistant output")) from exc
                continue

            reduced = reducer.reduce(event)
            if reduced.error:
                raise reduced.error.to_exception()
            if reducer.assistant_completed and reduced.terminal_state is _TurnTerminalState.IDLE:
                return CodexChatResult(text=reducer.text, usage=reducer.usage)

    async def _stream_completion(self, events: asyncio.Queue[dict[str, Any]]) -> AsyncIterator[str]:
        reducer = _CodexTurnEventReducer()
        deadline = time.monotonic() + self._turn_timeout_seconds

        while True:
            timeout = min(5.0, max(0.1, deadline - time.monotonic()))
            try:
                event = await asyncio.wait_for(events.get(), timeout=timeout)
            except asyncio.TimeoutError as exc:
                if time.monotonic() >= deadline:
                    raise CodexTurnTimeout(reducer.timeout_message("streaming output")) from exc
                continue

            reduced = reducer.reduce(event)
            if reduced.error:
                raise reduced.error.to_exception()
            if reduced.assistant_delta:
                yield reduced.assistant_delta
            if reduced.terminal_state is _TurnTerminalState.ASSISTANT_MESSAGE_COMPLETED:
                return
            if reduced.terminal_state is _TurnTerminalState.TURN_COMPLETED and reducer.text:
                return
            if reduced.terminal_state is _TurnTerminalState.IDLE and reducer.text:
                return

    async def _drain_until_idle(self, events: asyncio.Queue[dict[str, Any]]) -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                event = await asyncio.wait_for(events.get(), timeout=0.5)
            except asyncio.TimeoutError:
                return
            if event.get("method") == "thread/status/changed":
                status = (event.get("params") or {}).get("status") or {}
                if status.get("type") == "idle":
                    return


def openai_usage_from_codex(token_usage: dict[str, Any]) -> dict[str, int]:
    usage = token_usage.get("last") or token_usage.get("total") or {}
    prompt_tokens = int(usage.get("inputTokens") or 0)
    completion_tokens = int(usage.get("outputTokens") or 0)
    total_tokens = int(usage.get("totalTokens") or (prompt_tokens + completion_tokens))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _default_openai_usage() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _format_error_notification(params: dict[str, Any]) -> str:
    error = params.get("error")
    if not error and "willRetry" not in params:
        return redact_secrets(str(params.get("message") or params))
    if error:
        message = _format_app_server_error(error)
    else:
        message = redact_secrets(str(params.get("message") or params))
    if "willRetry" in params:
        message = f"{message}; willRetry={bool(params['willRetry'])}"
    return f"Codex error notification: {message}"


def _format_app_server_error(error: Any) -> str:
    if not isinstance(error, dict):
        return redact_secrets(str(error))

    message = redact_secrets(str(error.get("message") or error))
    details = []
    if error.get("codexErrorInfo"):
        details.append(redact_secrets(f"codexErrorInfo={error['codexErrorInfo']}"))
    if error.get("additionalDetails"):
        details.append(redact_secrets(f"additionalDetails={error['additionalDetails']}"))
    if not details:
        return message
    return f"{message}; {'; '.join(details)}"

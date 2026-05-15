from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from ._app_server_stdio_session import AppServerStdioSession
from ._codex_shared import CodexAppServerError, CodexChatResult, CodexTurnTimeout


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
        text = ""
        usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        assistant_completed = False
        deadline = time.monotonic() + self._turn_timeout_seconds

        while True:
            timeout = min(1.0 if assistant_completed else 5.0, max(0.1, deadline - time.monotonic()))
            try:
                event = await asyncio.wait_for(events.get(), timeout=timeout)
            except asyncio.TimeoutError:
                if assistant_completed:
                    return CodexChatResult(text=text, usage=usage)
                if time.monotonic() >= deadline:
                    raise CodexTurnTimeout("Timed out waiting for Codex assistant output.")
                continue

            method = event.get("method")
            params = event.get("params") or {}
            if method == "item/agentMessage/delta":
                text += str(params.get("delta") or "")
            elif method == "item/completed":
                item = params.get("item") or {}
                if item.get("type") == "agentMessage":
                    item_text = str(item.get("text") or "")
                    text = item_text or text
                    assistant_completed = True
            elif method == "thread/tokenUsage/updated":
                usage = openai_usage_from_codex(params.get("tokenUsage") or {})
            elif method == "thread/status/changed":
                status = params.get("status") or {}
                if assistant_completed and status.get("type") == "idle":
                    return CodexChatResult(text=text, usage=usage)
            elif method == "turn/completed":
                turn = params.get("turn") or {}
                if turn.get("error"):
                    raise CodexAppServerError(f"Codex turn failed: {turn['error']}")
            elif method == "error":
                raise CodexAppServerError(str(params.get("message") or params))

    async def _stream_completion(self, events: asyncio.Queue[dict[str, Any]]) -> AsyncIterator[str]:
        text = ""
        deadline = time.monotonic() + self._turn_timeout_seconds

        while True:
            timeout = min(5.0, max(0.1, deadline - time.monotonic()))
            try:
                event = await asyncio.wait_for(events.get(), timeout=timeout)
            except asyncio.TimeoutError as exc:
                if time.monotonic() >= deadline:
                    raise CodexTurnTimeout("Timed out waiting for Codex streaming output.") from exc
                continue

            method = event.get("method")
            params = event.get("params") or {}
            if method == "item/agentMessage/delta":
                delta = str(params.get("delta") or "")
                text += delta
                if delta:
                    yield delta
            elif method == "item/completed":
                item = params.get("item") or {}
                if item.get("type") == "agentMessage":
                    item_text = str(item.get("text") or "")
                    if item_text and item_text != text:
                        suffix = item_text[len(text) :] if item_text.startswith(text) else item_text
                        if suffix:
                            yield suffix
                    return
            elif method == "turn/completed":
                turn = params.get("turn") or {}
                if turn.get("error"):
                    raise CodexAppServerError(f"Codex turn failed: {turn['error']}")
                if text:
                    return
            elif method == "thread/status/changed":
                status = params.get("status") or {}
                if text and status.get("type") == "idle":
                    return
            elif method == "error":
                raise CodexAppServerError(str(params.get("message") or params))

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

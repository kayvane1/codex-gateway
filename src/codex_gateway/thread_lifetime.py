from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from ._codex_shared import CodexAppServerError


class CodexChatAdmissionError(CodexAppServerError):
    """Raised when a chat turn cannot be admitted locally."""


class CodexChatOverloaded(CodexChatAdmissionError):
    """Raised when the configured local chat queue is full."""


class CodexChatAdmissionTimeout(CodexChatAdmissionError):
    """Raised when a chat turn waits too long for local admission."""


class CodexChatAdmissionCancelled(CodexChatAdmissionError):
    """Raised when a chat turn is cancelled while waiting for admission."""


@dataclass(frozen=True)
class ChatAdmissionPolicy:
    max_active_turns: int = 1
    max_pending_turns: int | None = None
    timeout_seconds: float | None = 30.0


class ChatTurnAdmission:
    """Bounds concurrent chat turns and makes waiting behavior explicit."""

    def __init__(self, policy: ChatAdmissionPolicy) -> None:
        if policy.max_active_turns < 1:
            raise ValueError("max_active_turns must be at least 1.")
        if policy.max_pending_turns is not None and policy.max_pending_turns < 0:
            raise ValueError("max_pending_turns must be non-negative.")
        if policy.timeout_seconds is not None and policy.timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative.")

        self._policy = policy
        self._condition = asyncio.Condition()
        self._active_turns = 0
        self._pending_turns = 0

    @asynccontextmanager
    async def admit(self) -> AsyncIterator[None]:
        await self._acquire()
        try:
            yield
        finally:
            async with self._condition:
                self._active_turns -= 1
                self._condition.notify()

    async def _acquire(self) -> None:
        timeout = self._policy.timeout_seconds
        if timeout is not None and timeout <= 0:
            await self._acquire_slot(wait=False)
            return

        try:
            if timeout is None:
                await self._acquire_slot(wait=True)
            else:
                await asyncio.wait_for(self._acquire_slot(wait=True), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise CodexChatAdmissionTimeout("Timed out waiting for a free Codex chat slot.") from exc
        except asyncio.CancelledError as exc:
            raise CodexChatAdmissionCancelled("Cancelled while waiting for a free Codex chat slot.") from exc

    async def _acquire_slot(self, *, wait: bool) -> None:
        queued = False
        async with self._condition:
            if self._active_turns >= self._policy.max_active_turns:
                if not wait or self._queue_is_full():
                    raise CodexChatOverloaded("Codex chat capacity is full.")
                self._pending_turns += 1
                queued = True
            try:
                while self._active_turns >= self._policy.max_active_turns:
                    await self._condition.wait()
                self._active_turns += 1
            finally:
                if queued:
                    self._pending_turns -= 1

    def _queue_is_full(self) -> bool:
        return self._policy.max_pending_turns is not None and self._pending_turns >= self._policy.max_pending_turns


class ThreadLifetime:
    """Owns local chat admission and best-effort app-server thread cleanup."""

    def __init__(self, policy: ChatAdmissionPolicy) -> None:
        self._admission = ChatTurnAdmission(policy)

    @asynccontextmanager
    async def chat_thread(
        self,
        *,
        start_thread: Callable[[], Awaitable[str]],
        cleanup_thread: Callable[[str], Awaitable[None]],
    ) -> AsyncIterator[str]:
        async with self._admission.admit():
            thread_id = await start_thread()
            original_error: BaseException | None = None
            try:
                yield thread_id
            except BaseException as exc:
                original_error = exc
                raise
            finally:
                try:
                    await cleanup_thread(thread_id)
                except asyncio.CancelledError:
                    if original_error is None:
                        raise
                except Exception:
                    pass
                except BaseException:
                    if original_error is None:
                        raise

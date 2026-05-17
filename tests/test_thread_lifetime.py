from __future__ import annotations

import asyncio

import pytest

from codex_gateway.thread_lifetime import (
    ChatAdmissionPolicy,
    ChatTurnAdmission,
    CodexChatAdmissionCancelled,
    CodexChatAdmissionTimeout,
    CodexChatOverloaded,
    ThreadLifetime,
)


def run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize(
    "policy",
    [
        ChatAdmissionPolicy(max_active_turns=0),
        ChatAdmissionPolicy(max_pending_turns=-1),
        ChatAdmissionPolicy(timeout_seconds=-1),
    ],
)
def test_chat_admission_policy_rejects_invalid_bounds(policy: ChatAdmissionPolicy) -> None:
    with pytest.raises(ValueError):
        ChatTurnAdmission(policy)


def test_thread_lifetime_serializes_concurrent_chat_admission() -> None:
    async def scenario() -> None:
        lifetime = ThreadLifetime(ChatAdmissionPolicy(max_active_turns=1, timeout_seconds=0.5))
        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        second_entered = asyncio.Event()
        entries: list[str] = []
        cleanups: list[str] = []

        async def start_thread(thread_id: str) -> str:
            entries.append(f"start:{thread_id}")
            return thread_id

        async def cleanup_thread(thread_id: str) -> None:
            cleanups.append(thread_id)

        async def first_turn() -> None:
            async with lifetime.chat_thread(
                start_thread=lambda: start_thread("thread-1"),
                cleanup_thread=cleanup_thread,
            ) as thread_id:
                entries.append(f"enter:{thread_id}")
                first_entered.set()
                await release_first.wait()

        async def second_turn() -> None:
            async with lifetime.chat_thread(
                start_thread=lambda: start_thread("thread-2"),
                cleanup_thread=cleanup_thread,
            ) as thread_id:
                entries.append(f"enter:{thread_id}")
                second_entered.set()

        first = asyncio.create_task(first_turn())
        await asyncio.wait_for(first_entered.wait(), timeout=0.1)
        second = asyncio.create_task(second_turn())
        await asyncio.sleep(0.01)

        assert second_entered.is_set() is False
        assert entries == ["start:thread-1", "enter:thread-1"]

        release_first.set()
        await asyncio.gather(first, second)

        assert entries == ["start:thread-1", "enter:thread-1", "start:thread-2", "enter:thread-2"]
        assert cleanups == ["thread-1", "thread-2"]

    run(scenario())


def test_thread_lifetime_cleanup_runs_after_success_and_preserves_turn_error() -> None:
    async def scenario() -> None:
        lifetime = ThreadLifetime(ChatAdmissionPolicy(max_active_turns=1, timeout_seconds=0.1))
        cleanups: list[str] = []

        async def cleanup_thread(thread_id: str) -> None:
            cleanups.append(thread_id)
            raise RuntimeError("cleanup failed")

        async with lifetime.chat_thread(
            start_thread=lambda: asyncio.sleep(0, result="success-thread"),
            cleanup_thread=cleanup_thread,
        ):
            pass

        with pytest.raises(RuntimeError, match="turn failed"):
            async with lifetime.chat_thread(
                start_thread=lambda: asyncio.sleep(0, result="error-thread"),
                cleanup_thread=cleanup_thread,
            ):
                raise RuntimeError("turn failed")

        assert cleanups == ["success-thread", "error-thread"]

    run(scenario())


def test_thread_lifetime_overload_and_timeout_are_explicit() -> None:
    async def overload_scenario() -> None:
        lifetime = ThreadLifetime(ChatAdmissionPolicy(max_active_turns=1, max_pending_turns=0, timeout_seconds=0.5))
        release_first = asyncio.Event()

        async def first_turn() -> None:
            async with lifetime.chat_thread(
                start_thread=lambda: asyncio.sleep(0, result="thread-1"),
                cleanup_thread=lambda _thread_id: asyncio.sleep(0),
            ):
                await release_first.wait()

        first = asyncio.create_task(first_turn())
        await asyncio.sleep(0)

        with pytest.raises(CodexChatOverloaded):
            async with lifetime.chat_thread(
                start_thread=lambda: asyncio.sleep(0, result="thread-2"),
                cleanup_thread=lambda _thread_id: asyncio.sleep(0),
            ):
                pass

        release_first.set()
        await first

    async def timeout_scenario() -> None:
        lifetime = ThreadLifetime(ChatAdmissionPolicy(max_active_turns=1, timeout_seconds=0.01))
        release_first = asyncio.Event()

        async def first_turn() -> None:
            async with lifetime.chat_thread(
                start_thread=lambda: asyncio.sleep(0, result="thread-1"),
                cleanup_thread=lambda _thread_id: asyncio.sleep(0),
            ):
                await release_first.wait()

        first = asyncio.create_task(first_turn())
        await asyncio.sleep(0)

        with pytest.raises(CodexChatAdmissionTimeout):
            async with lifetime.chat_thread(
                start_thread=lambda: asyncio.sleep(0, result="thread-2"),
                cleanup_thread=lambda _thread_id: asyncio.sleep(0),
            ):
                pass

        release_first.set()
        await first

    async def no_wait_scenario() -> None:
        lifetime = ThreadLifetime(ChatAdmissionPolicy(max_active_turns=1, timeout_seconds=0))
        release_first = asyncio.Event()

        async def first_turn() -> None:
            async with lifetime.chat_thread(
                start_thread=lambda: asyncio.sleep(0, result="thread-1"),
                cleanup_thread=lambda _thread_id: asyncio.sleep(0),
            ):
                await release_first.wait()

        first = asyncio.create_task(first_turn())
        await asyncio.sleep(0)

        with pytest.raises(CodexChatOverloaded):
            async with lifetime.chat_thread(
                start_thread=lambda: asyncio.sleep(0, result="thread-2"),
                cleanup_thread=lambda _thread_id: asyncio.sleep(0),
            ):
                pass

        release_first.set()
        await first

    run(overload_scenario())
    run(timeout_scenario())
    run(no_wait_scenario())


def test_thread_lifetime_waiting_cancellation_is_explicit() -> None:
    async def scenario() -> None:
        lifetime = ThreadLifetime(ChatAdmissionPolicy(max_active_turns=1, timeout_seconds=0.5))
        first_entered = asyncio.Event()
        release_first = asyncio.Event()

        async def first_turn() -> None:
            async with lifetime.chat_thread(
                start_thread=lambda: asyncio.sleep(0, result="thread-1"),
                cleanup_thread=lambda _thread_id: asyncio.sleep(0),
            ):
                first_entered.set()
                await release_first.wait()

        async def waiting_turn() -> None:
            async with lifetime.chat_thread(
                start_thread=lambda: asyncio.sleep(0, result="thread-2"),
                cleanup_thread=lambda _thread_id: asyncio.sleep(0),
            ):
                pass

        first = asyncio.create_task(first_turn())
        await asyncio.wait_for(first_entered.wait(), timeout=0.1)
        waiting = asyncio.create_task(waiting_turn())
        await asyncio.sleep(0)
        waiting.cancel()

        with pytest.raises(CodexChatAdmissionCancelled):
            await waiting

        release_first.set()
        await first

    run(scenario())


def test_thread_lifetime_cleanup_cancellation_and_base_exception_propagate_without_turn_error() -> None:
    class CleanupFatal(BaseException):
        pass

    async def cancellation_scenario() -> None:
        lifetime = ThreadLifetime(ChatAdmissionPolicy(max_active_turns=1, timeout_seconds=0.1))

        async def cleanup_thread(_thread_id: str) -> None:
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            async with lifetime.chat_thread(
                start_thread=lambda: asyncio.sleep(0, result="thread-1"),
                cleanup_thread=cleanup_thread,
            ):
                pass

    async def base_exception_scenario() -> None:
        lifetime = ThreadLifetime(ChatAdmissionPolicy(max_active_turns=1, timeout_seconds=0.1))

        async def cleanup_thread(_thread_id: str) -> None:
            raise CleanupFatal("fatal cleanup")

        with pytest.raises(CleanupFatal):
            async with lifetime.chat_thread(
                start_thread=lambda: asyncio.sleep(0, result="thread-1"),
                cleanup_thread=cleanup_thread,
            ):
                pass

    run(cancellation_scenario())
    run(base_exception_scenario())

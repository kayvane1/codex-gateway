from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

import codex_gateway.codex_client as codex_client_module
from codex_gateway._app_server_stdio_session import AppServerStdioSession, _redact
from codex_gateway._codex_turn_lifecycle import CodexTurnLifecycle, openai_usage_from_codex
from codex_gateway.codex_client import (
    CodexAppServer,
    CodexAppServerError,
    CodexAppServerProtocolError,
    CodexChatResult,
    CodexClientSettings,
    CodexTurnTimeout,
    build_text_input,
)
from tests.support.stdio_process import FakeProcess, FakeStdin, FakeStream


class FakeTurnSession:
    def __init__(self, events_by_thread: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.events_by_thread = events_by_thread or {}
        self.requests: list[tuple[str, dict[str, Any] | None, float | None]] = []
        self.subscribed: list[str] = []
        self.unsubscribed: list[str] = []

    @asynccontextmanager
    async def subscribe(self, thread_id: str) -> AsyncIterator[asyncio.Queue[dict[str, Any]]]:
        self.subscribed.append(thread_id)
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        for event in self.events_by_thread.get(thread_id, []):
            queue.put_nowait(event)
        try:
            yield queue
        finally:
            self.unsubscribed.append(thread_id)

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        self.requests.append((method, params, timeout))
        return {"ok": True}


class FakeClientSession:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.requests: list[tuple[str, dict[str, Any] | None, float | None]] = []
        self.unsubscribe_error: Exception | None = None
        self.events_by_thread: dict[str, list[dict[str, Any]]] = {}
        self.subscribed: list[str] = []
        self.unsubscribed: list[str] = []

    async def start(self) -> None:
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        self.requests.append((method, params, timeout))
        if method == "thread/start":
            return {"thread": {"id": f"thread-{len(self.requests)}"}}
        if method == "thread/unsubscribe":
            if self.unsubscribe_error:
                raise self.unsubscribe_error
            return {"status": "unsubscribed"}
        if method == "model/list":
            return {"data": [{"id": "fake-model"}]}
        if method == "initialize":
            return {
                "codexHome": "/tmp/codex-home",
                "platformFamily": "unix",
                "platformOs": "macos",
                "userAgent": "codex-cli/0.128.0",
            }
        return {}

    @asynccontextmanager
    async def subscribe(self, thread_id: str) -> AsyncIterator[asyncio.Queue[dict[str, Any]]]:
        self.subscribed.append(thread_id)
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        for event in self.events_by_thread.get(thread_id, []):
            queue.put_nowait(event)
        try:
            yield queue
        finally:
            self.unsubscribed.append(thread_id)


class FakeClientTurnLifecycle:
    def __init__(self) -> None:
        self.completed: list[tuple[str, list[dict[str, Any]]]] = []
        self.streamed: list[tuple[str, list[dict[str, Any]]]] = []
        self.complete_error: Exception | None = None

    async def complete(self, thread_id: str, input_items: list[dict[str, Any]]) -> CodexChatResult:
        if self.complete_error:
            raise self.complete_error
        self.completed.append((thread_id, input_items))
        return CodexChatResult(
            text="delegated",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    async def stream(self, thread_id: str, input_items: list[dict[str, Any]]) -> AsyncIterator[str]:
        self.streamed.append((thread_id, input_items))
        yield "stream-"
        yield "delegated"


def run(coro):
    return asyncio.run(coro)


def session() -> AppServerStdioSession:
    return AppServerStdioSession(
        command=("codex", "app-server", "--listen", "stdio://"),
        cwd=Path.cwd(),
        request_timeout_seconds=0.05,
    )


def client(turn_timeout: float = 0.05) -> CodexAppServer:
    return CodexAppServer(
        CodexClientSettings(
            cwd=Path.cwd(),
            request_timeout_seconds=0.05,
            turn_timeout_seconds=turn_timeout,
        )
    )


def turn_lifecycle(
    *,
    events_by_thread: dict[str, list[dict[str, Any]]] | None = None,
    turn_timeout: float = 0.05,
) -> tuple[CodexTurnLifecycle, FakeTurnSession]:
    fake_session = FakeTurnSession(events_by_thread)
    lifecycle = CodexTurnLifecycle(
        session=fake_session,  # type: ignore[arg-type]
        request_timeout_seconds=0.05,
        turn_timeout_seconds=turn_timeout,
        reasoning_effort="low",
    )
    return lifecycle, fake_session


def queue_with(events: list[dict[str, Any]]) -> asyncio.Queue[dict[str, Any]]:
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    for event in events:
        queue.put_nowait(event)
    return queue


def test_redacts_common_secret_shapes_and_builds_helpers() -> None:
    assert "key [redacted]" == _redact("key sk-abc123")
    assert "Bearer [redacted]" in _redact("Authorization: Bearer token.value")
    assert "access_token=[redacted]" in _redact("access_token=abc")
    assert "refresh-token: [redacted]" in _redact("refresh-token: abc")
    assert build_text_input("hi") == [{"type": "text", "text": "hi", "text_elements": []}]
    assert (
        openai_usage_from_codex({"last": {"inputTokens": 1, "outputTokens": 2, "totalTokens": 3}})["total_tokens"] == 3
    )
    assert openai_usage_from_codex({"total": {"inputTokens": 4, "outputTokens": 5}})["total_tokens"] == 9


def test_subscriptions_and_response_handling() -> None:
    async def scenario() -> None:
        stdio = session()
        subscription = stdio._subscribe("thread-1")
        stdio._dispatch_notification({"method": "x", "params": {"threadId": "thread-1"}})
        assert await subscription.queue.get() == {"method": "x", "params": {"threadId": "thread-1"}}
        stdio._dispatch_notification({"method": "x", "params": {"thread": {"id": "thread-1"}}})
        assert (await subscription.queue.get())["params"]["thread"]["id"] == "thread-1"
        stdio._dispatch_notification({"method": "x", "params": {}})
        stdio._unsubscribe(subscription)
        stdio._unsubscribe(subscription)
        assert "thread-1" not in stdio._subscriptions

        loop = asyncio.get_running_loop()
        result_future = loop.create_future()
        stdio._pending[1] = result_future
        stdio._handle_response({"id": 1, "result": {"ok": True}})
        assert result_future.result() == {"ok": True}
        stdio._handle_response({"id": 1, "result": {"ignored": True}})

        error_future = loop.create_future()
        stdio._pending[2] = error_future
        stdio._pending_methods[2] = "model/list"
        stdio._handle_response({"id": 2, "error": {"message": "bad"}})
        with pytest.raises(CodexAppServerProtocolError) as exc_info:
            error_future.result()
        assert exc_info.value.method == "model/list"
        assert "model/list failed: bad" in str(exc_info.value)

        stdio._handle_response({"id": 999, "result": {}})

    run(scenario())


def test_request_writes_json_and_handles_missing_stdin() -> None:
    async def scenario() -> None:
        stdio = session()
        stdin = FakeStdin()
        stdio._process = FakeProcess(stdin=stdin)

        async def complete_request() -> None:
            await asyncio.sleep(0)
            stdio._handle_response({"id": 1, "result": {"ok": 1}})

        asyncio.create_task(complete_request())
        assert await stdio.request("model/list") == {"ok": 1}
        assert json.loads(stdin.writes[0]) == {"id": 1, "method": "model/list"}

        stdio._process = FakeProcess(stdin=None)
        with pytest.raises(CodexAppServerError):
            await stdio.request("model/list")

    run(scenario())


def test_list_models_and_start_thread_request_shape() -> None:
    async def scenario() -> None:
        codex = client()
        calls: list[tuple[str, dict[str, Any] | None]] = []

        async def fake_request(method: str, params: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
            calls.append((method, params))
            if method == "model/list":
                return {"data": [{"id": "m"}]}
            return {"thread": {"id": "thread-id"}}

        codex.request = fake_request  # type: ignore[method-assign]
        assert await codex.list_models() == [{"id": "m"}]
        assert await codex._start_thread("m", "dev") == "thread-id"
        assert calls[0] == ("model/list", {"includeHidden": False})
        thread_params = calls[1][1]
        assert thread_params is not None
        assert thread_params["approvalPolicy"] == "never"
        assert thread_params["approvalsReviewer"] == "user"
        assert thread_params["sandbox"] == "read-only"
        assert thread_params["dynamicTools"] == []
        assert thread_params["environments"] == []
        assert "Do not execute" in thread_params["baseInstructions"]
        assert thread_params["developerInstructions"] == "dev"
        assert "Do not execute" not in thread_params["developerInstructions"]

    run(scenario())


def test_complete_chat_turn_start_restates_safety_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        fake_session = FakeClientSession()
        fake_session.events_by_thread["thread-1"] = [
            {"method": "item/completed", "params": {"item": {"type": "agentMessage", "text": "done"}}},
            {"method": "thread/status/changed", "params": {"status": {"type": "idle"}}},
        ]
        monkeypatch.setattr(codex_client_module, "AppServerStdioSession", lambda **_: fake_session)
        codex = client()

        input_items = build_text_input("hi")
        result = await codex.complete_chat(
            model="m",
            history_items=[],
            input_items=input_items,
            developer_instructions="dev",
        )

        assert result.text == "done"
        turn_calls = [call for call in fake_session.requests if call[0] == "turn/start"]
        assert len(turn_calls) == 1
        turn_params = turn_calls[0][1]
        assert turn_params == {
            "threadId": "thread-1",
            "input": input_items,
            "effort": "low",
            "approvalPolicy": "never",
            "approvalsReviewer": "user",
            "environments": [],
            "sandboxPolicy": {"type": "readOnly", "networkAccess": False},
        }
        assert [call[0] for call in fake_session.requests] == [
            "thread/start",
            "turn/start",
            "thread/unsubscribe",
        ]

    run(scenario())


def test_turn_session_passthrough_leaves_non_turn_requests_unchanged() -> None:
    async def scenario() -> None:
        fake_session = FakeClientSession()
        codex = client()
        codex._session = fake_session  # type: ignore[assignment]
        codex._turn_session = codex_client_module._SafetyBoundTurnSession(fake_session)  # type: ignore[arg-type]

        await codex._turn_session.request("model/list", {"includeHidden": False}, timeout=0.5)

        assert fake_session.requests == [("model/list", {"includeHidden": False}, 0.5)]

    run(scenario())


def test_materialize_data_images_writes_local_temp_files_and_cleans_up() -> None:
    input_items = [
        {"type": "text", "text": "look", "text_elements": []},
        {"type": "image", "url": "https://example.com/image.png"},
        {"type": "image", "url": "data:image/png;base64,QUJD"},
        {"type": "image", "url": "data:image/tiff;base64,AAEC"},
    ]

    with codex_client_module._materialize_data_image_inputs(input_items) as prepared:
        local_images = [item for item in prepared if item["type"] == "localImage"]
        image_paths = [Path(item["path"]) for item in local_images]

        assert prepared[:2] == input_items[:2]
        assert [path.name for path in image_paths] == ["image-0.png", "image-1.img"]
        assert [path.read_bytes() for path in image_paths] == [b"ABC", b"\x00\x01\x02"]
        assert all(path.exists() for path in image_paths)

    assert all(not path.exists() for path in image_paths)
    assert input_items[2]["url"].startswith("data:image/png")


def test_client_start_stop_and_chat_delegation_are_bounded_to_high_level_api() -> None:
    async def scenario() -> None:
        codex = client()
        fake_session = FakeClientSession()
        fake_lifecycle = FakeClientTurnLifecycle()
        codex._session = fake_session  # type: ignore[assignment]
        codex._turn_lifecycle = fake_lifecycle  # type: ignore[assignment]

        await codex.start()
        await codex.start()
        assert fake_session.started == 1
        assert [call[0] for call in fake_session.requests] == ["initialize", "model/list"]

        input_items = build_text_input("hi")
        result = await codex.complete_chat(
            model="m",
            history_items=[],
            input_items=input_items,
            developer_instructions="dev",
        )
        assert result.text == "delegated"
        assert fake_lifecycle.completed == [("thread-3", input_items)]

        chunks = [
            chunk
            async for chunk in codex.stream_chat(
                model="m",
                history_items=[],
                input_items=input_items,
                developer_instructions="dev",
            )
        ]
        assert chunks == ["stream-", "delegated"]
        assert fake_lifecycle.streamed == [("thread-5", input_items)]
        assert [call[0] for call in fake_session.requests] == [
            "initialize",
            "model/list",
            "thread/start",
            "thread/unsubscribe",
            "thread/start",
            "thread/unsubscribe",
        ]

        await codex.stop()
        assert fake_session.stopped == 1
        assert codex._initialized is False

    run(scenario())


def test_client_injects_history_and_unsubscribes_after_success_and_errors() -> None:
    async def scenario() -> None:
        codex = client()
        fake_session = FakeClientSession()
        fake_lifecycle = FakeClientTurnLifecycle()
        codex._session = fake_session  # type: ignore[assignment]
        codex._turn_lifecycle = fake_lifecycle  # type: ignore[assignment]

        history_items = [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "old"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "older"}]},
        ]
        input_items = build_text_input("new")

        await codex.complete_chat(
            model="m",
            history_items=history_items,
            input_items=input_items,
            developer_instructions="dev",
        )
        assert [call[0] for call in fake_session.requests] == [
            "thread/start",
            "thread/inject_items",
            "thread/unsubscribe",
        ]
        assert fake_session.requests[1][1] == {"threadId": "thread-1", "items": history_items}
        assert fake_lifecycle.completed == [("thread-1", input_items)]

        fake_session.unsubscribe_error = CodexAppServerError("cleanup failed")
        cleanup_result = await codex.complete_chat(
            model="m",
            history_items=[],
            input_items=input_items,
            developer_instructions="dev",
        )
        assert cleanup_result.text == "delegated"
        fake_session.unsubscribe_error = None

        fake_session.unsubscribe_error = CodexAppServerError("cleanup failed")
        fake_lifecycle.complete_error = CodexAppServerError("turn failed")
        with pytest.raises(CodexAppServerError, match="turn failed"):
            await codex.complete_chat(
                model="m",
                history_items=[],
                input_items=input_items,
                developer_instructions="dev",
            )
        assert [call[0] for call in fake_session.requests[-2:]] == ["thread/start", "thread/unsubscribe"]

    run(scenario())


def test_turn_lifecycle_public_methods_start_turns_and_unsubscribe() -> None:
    async def scenario() -> None:
        events_by_thread = {
            "complete-thread": [
                {"method": "item/completed", "params": {"item": {"type": "agentMessage", "text": "done"}}},
                {"method": "thread/status/changed", "params": {"status": {"type": "idle"}}},
            ],
            "stream-thread": [
                {"method": "item/agentMessage/delta", "params": {"delta": "streamed"}},
                {"method": "item/completed", "params": {"item": {"type": "agentMessage", "text": "streamed"}}},
                {"method": "thread/status/changed", "params": {"status": {"type": "idle"}}},
            ],
        }
        lifecycle, fake_session = turn_lifecycle(events_by_thread=events_by_thread)

        input_items = build_text_input("hi")
        result = await lifecycle.complete("complete-thread", input_items)
        assert result.text == "done"

        chunks = [chunk async for chunk in lifecycle.stream("stream-thread", input_items)]
        assert chunks == ["streamed"]

        assert fake_session.requests == [
            (
                "turn/start",
                {"threadId": "complete-thread", "input": input_items, "effort": "low"},
                0.05,
            ),
            (
                "turn/start",
                {"threadId": "stream-thread", "input": input_items, "effort": "low"},
                0.05,
            ),
        ]
        assert fake_session.subscribed == ["complete-thread", "stream-thread"]
        assert fake_session.unsubscribed == ["complete-thread", "stream-thread"]

    run(scenario())


def test_collect_completion_success_timeout_and_errors() -> None:
    async def success() -> None:
        lifecycle, _ = turn_lifecycle()
        result = await lifecycle._collect_completion(
            queue_with(
                [
                    {"method": "item/agentMessage/delta", "params": {"delta": "hel"}},
                    {
                        "method": "thread/tokenUsage/updated",
                        "params": {"tokenUsage": {"last": {"inputTokens": 1, "outputTokens": 2, "totalTokens": 3}}},
                    },
                    {"method": "item/completed", "params": {"item": {"type": "agentMessage", "text": "hello"}}},
                    {"method": "thread/status/changed", "params": {"status": {"type": "idle"}}},
                ]
            )
        )
        assert result.text == "hello"
        assert result.usage["total_tokens"] == 3

    async def success_after_short_timeout() -> None:
        lifecycle, _ = turn_lifecycle(turn_timeout=0.02)
        result = await lifecycle._collect_completion(
            queue_with([{"method": "item/completed", "params": {"item": {"type": "agentMessage", "text": "done"}}}])
        )
        assert result.text == "done"

    async def turn_completed_success() -> None:
        lifecycle, _ = turn_lifecycle()
        result = await lifecycle._collect_completion(
            queue_with(
                [
                    {"method": "item/completed", "params": {"item": {"type": "agentMessage", "text": "done"}}},
                    {"method": "turn/completed", "params": {"turn": {"error": None}}},
                ]
            )
        )
        assert result.text == "done"

    async def turn_completed_waits_for_late_usage() -> None:
        lifecycle, _ = turn_lifecycle()
        result = await lifecycle._collect_completion(
            queue_with(
                [
                    {"method": "item/completed", "params": {"item": {"type": "agentMessage", "text": "done"}}},
                    {"method": "turn/completed", "params": {"turn": {"error": None}}},
                    {
                        "method": "thread/tokenUsage/updated",
                        "params": {"tokenUsage": {"last": {"inputTokens": 4, "outputTokens": 5, "totalTokens": 9}}},
                    },
                    {"method": "thread/status/changed", "params": {"status": {"type": "idle"}}},
                ]
            )
        )
        assert result.text == "done"
        assert result.usage == {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9}

    async def timeout() -> None:
        lifecycle, _ = turn_lifecycle(turn_timeout=0.001)
        with pytest.raises(CodexTurnTimeout):
            await lifecycle._collect_completion(asyncio.Queue())

    async def turn_error() -> None:
        lifecycle, _ = turn_lifecycle()
        with pytest.raises(CodexAppServerError):
            await lifecycle._collect_completion(
                queue_with([{"method": "turn/completed", "params": {"turn": {"error": "bad"}}}])
            )

    async def notification_error() -> None:
        lifecycle, _ = turn_lifecycle()
        with pytest.raises(CodexAppServerError):
            await lifecycle._collect_completion(queue_with([{"method": "error", "params": {"message": "bad"}}]))

    run(success())
    run(success_after_short_timeout())
    run(turn_completed_success())
    run(turn_completed_waits_for_late_usage())
    run(timeout())
    run(turn_error())
    run(notification_error())


def test_stream_completion_success_suffix_and_errors() -> None:
    async def collect(queue_events: list[dict[str, Any]], *, turn_timeout: float = 0.05) -> list[str]:
        lifecycle, _ = turn_lifecycle(turn_timeout=turn_timeout)
        return [chunk async for chunk in lifecycle._stream_completion(queue_with(queue_events))]

    assert run(
        collect(
            [
                {"method": "item/agentMessage/delta", "params": {"delta": "hel"}},
                {"method": "item/agentMessage/delta", "params": {"delta": ""}},
                {"method": "item/completed", "params": {"item": {"type": "agentMessage", "text": "hello"}}},
            ]
        )
    ) == ["hel", "lo"]

    assert run(
        collect(
            [
                {"method": "item/agentMessage/delta", "params": {"delta": "done"}},
                {"method": "turn/completed", "params": {"turn": {"error": None}}},
            ]
        )
    ) == ["done"]

    assert run(
        collect(
            [
                {"method": "item/agentMessage/delta", "params": {"delta": "done"}},
                {"method": "thread/status/changed", "params": {"status": {"type": "idle"}}},
            ]
        )
    ) == ["done"]

    with pytest.raises(CodexTurnTimeout):
        run(collect([], turn_timeout=0.001))

    with pytest.raises(CodexAppServerError):
        run(collect([{"method": "turn/completed", "params": {"turn": {"error": "bad"}}}]))

    with pytest.raises(CodexAppServerError):
        run(collect([{"method": "error", "params": {"message": "bad"}}]))


def test_drain_until_idle_handles_idle_and_timeout() -> None:
    async def scenario() -> None:
        lifecycle, _ = turn_lifecycle()
        await lifecycle._drain_until_idle(
            queue_with([{"method": "thread/status/changed", "params": {"status": {"type": "idle"}}}])
        )
        await lifecycle._drain_until_idle(asyncio.Queue())

    run(scenario())


def test_server_requests_are_declined_without_exposing_privileged_actions() -> None:
    async def scenario() -> None:
        stdio = session()
        sent: list[dict[str, Any]] = []

        async def fake_send_raw(message: dict[str, Any]) -> None:
            sent.append(message)

        stdio._send_raw = fake_send_raw  # type: ignore[method-assign]
        methods = [
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
            "item/permissions/requestApproval",
            "applyPatchApproval",
            "execCommandApproval",
            "item/tool/requestUserInput",
            "mcpServer/elicitation/request",
            "item/tool/call",
            "account/chatgptAuthTokens/refresh",
            "unknown/method",
        ]
        for index, method in enumerate(methods, start=1):
            await stdio._handle_server_request({"id": index, "method": method, "params": {}})

        assert sent[0]["result"]["decision"] == "decline"
        assert sent[2]["result"] == {"permissions": {}, "scope": "turn", "strictAutoReview": True}
        assert sent[3]["result"]["decision"] == "denied"
        assert sent[5]["result"] == {"answers": {}}
        assert sent[6]["result"]["action"] == "decline"
        assert sent[7]["result"]["success"] is False
        assert "error" in sent[8]
        assert "error" in sent[9]

    run(scenario())


def test_send_raw_require_process_and_fail_pending_paths() -> None:
    async def scenario() -> None:
        stdio = session()
        with pytest.raises(CodexAppServerError):
            stdio._require_process()

        stdio._process = FakeProcess(stdin=None)
        await stdio._send_raw({"id": 1, "result": {}})

        stdin = FakeStdin()
        stdio._process = FakeProcess(stdin=stdin)
        await stdio._send_raw({"id": 2, "result": {"ok": True}})
        assert json.loads(stdin.writes[0])["id"] == 2

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        stdio._pending[1] = future
        stdio._fail_pending(CodexAppServerError("stopped"))
        with pytest.raises(CodexAppServerError):
            future.result()

    run(scenario())


def test_stdout_and_stderr_readers_route_protocol_messages() -> None:
    async def scenario() -> None:
        stdio = session()
        stdin = FakeStdin()
        subscription = stdio._subscribe("thread-1")
        loop = asyncio.get_running_loop()
        response_future = loop.create_future()
        stdio._pending[2] = response_future
        lines = [
            b"not-json\n",
            json.dumps({"id": 1, "method": "item/tool/requestUserInput", "params": {}}).encode() + b"\n",
            json.dumps({"id": 2, "result": {"ok": True}}).encode() + b"\n",
            json.dumps({"method": "x", "params": {"threadId": "thread-1"}}).encode() + b"\n",
            b"",
        ]
        stdio._process = FakeProcess(stdin=stdin, stdout=FakeStream(lines), returncode=None)
        await stdio._read_stdout()
        assert response_future.result() == {"ok": True}
        assert json.loads(stdin.writes[0]) == {"id": 1, "result": {"answers": {}}}
        assert (await subscription.queue.get())["method"] == "x"

        stdio._process = FakeProcess(stdout=None)
        await stdio._read_stdout()

        stdio._process = FakeProcess(stderr=FakeStream([b"Authorization: Bearer secret\n", b""]))
        await stdio._read_stderr()
        assert stdio._stderr_tail[-1] == "Authorization: Bearer [redacted]"

        stdio._process = FakeProcess(stderr=None)
        await stdio._read_stderr()

    run(scenario())


def test_start_and_stop_manage_stdio_process_lifetime(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        stdio = session()
        process = FakeProcess(stdin=FakeStdin(), stdout=None, stderr=None, returncode=None)
        created: list[tuple[tuple[str, ...], str]] = []

        async def fake_create_subprocess_exec(
            *command: str,
            stdin: Any,
            stdout: Any,
            stderr: Any,
            cwd: str,
        ) -> FakeProcess:
            created.append((command, cwd))
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
        await stdio.start()
        await stdio.start()

        assert created == [(("codex", "app-server", "--listen", "stdio://"), str(Path.cwd()))]

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        stdio._pending[1] = future
        stdio._subscribe("thread-1")
        await stdio.stop()

        assert process.terminated is True
        assert stdio._pending == {}
        assert stdio._subscriptions == {}
        assert stdio._stdout_task is None
        assert stdio._stderr_task is None
        with pytest.raises(CodexAppServerError):
            future.result()

    run(scenario())

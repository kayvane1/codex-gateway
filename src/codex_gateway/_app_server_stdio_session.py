from __future__ import annotations

import asyncio
import contextlib
import json
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._codex_shared import CodexAppServerError, CodexAppServerProtocolError, redact_secrets
from .safety_policy import app_server_request_denial


@dataclass(frozen=True)
class AppServerSubscription:
    thread_id: str
    queue: asyncio.Queue[dict[str, Any]]


def _redact(value: str) -> str:
    return redact_secrets(value)


class AppServerStdioSession:
    """Newline-delimited JSON-RPC session over `codex app-server` stdio."""

    def __init__(
        self,
        *,
        command: tuple[str, ...],
        cwd: Path,
        request_timeout_seconds: float,
    ) -> None:
        self.command = command
        self.cwd = cwd
        self.request_timeout_seconds = request_timeout_seconds
        self._process: asyncio.subprocess.Process | None = None
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._pending_methods: dict[int, str] = {}
        self._subscriptions: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}
        self._send_lock = asyncio.Lock()
        self._next_id = 1
        self._stderr_tail: deque[str] = deque(maxlen=20)

    async def start(self) -> None:
        if self._process is not None:
            return

        self._process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.cwd),
        )
        self._stdout_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())

    async def stop(self) -> None:
        reader_tasks = (self._stdout_task, self._stderr_task)
        for task in reader_tasks:
            if task is not None:
                task.cancel()

        process = self._process
        self._process = None
        for future in self._pending.values():
            if not future.done():
                future.set_exception(CodexAppServerError("Codex app-server stopped."))
        self._pending.clear()
        self._pending_methods.clear()
        self._subscriptions.clear()

        if process is not None and process.returncode is None:
            process.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=5)
            if process.returncode is None:
                process.kill()
                await process.wait()

        for task in reader_tasks:
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError, CodexAppServerError):
                    await task
        self._stdout_task = None
        self._stderr_task = None

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        process = self._require_process()
        if process.stdin is None:
            raise CodexAppServerError("Codex app-server stdin is unavailable.")

        loop = asyncio.get_running_loop()
        request_id = self._next_id
        self._next_id += 1
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = future
        self._pending_methods[request_id] = method

        message: dict[str, Any] = {"id": request_id, "method": method}
        if params is not None:
            message["params"] = params

        async with self._send_lock:
            process.stdin.write(json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n")
            await process.stdin.drain()

        try:
            return await asyncio.wait_for(
                future,
                timeout=timeout or self.request_timeout_seconds,
            )
        finally:
            self._pending.pop(request_id, None)
            self._pending_methods.pop(request_id, None)

    @asynccontextmanager
    async def subscribe(self, thread_id: str) -> AsyncIterator[asyncio.Queue[dict[str, Any]]]:
        subscription = self._subscribe(thread_id)
        try:
            yield subscription.queue
        finally:
            self._unsubscribe(subscription)

    def _subscribe(self, thread_id: str) -> AppServerSubscription:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscriptions.setdefault(thread_id, set()).add(queue)
        return AppServerSubscription(thread_id=thread_id, queue=queue)

    def _unsubscribe(self, subscription: AppServerSubscription) -> None:
        subscribers = self._subscriptions.get(subscription.thread_id)
        if not subscribers:
            return
        subscribers.discard(subscription.queue)
        if not subscribers:
            self._subscriptions.pop(subscription.thread_id, None)

    async def _read_stdout(self) -> None:
        process = self._require_process()
        if process.stdout is None:
            return

        while True:
            line = await process.stdout.readline()
            if not line:
                break
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "id" in message and "method" in message:
                await self._handle_server_request(message)
            elif "id" in message:
                self._handle_response(message)
            elif "method" in message:
                self._dispatch_notification(message)

        if self._process is process and process.returncode is None:
            await process.wait()
        self._fail_pending(
            CodexAppServerError(
                f"Codex app-server exited unexpectedly with code {process.returncode}. Recent stderr: {list(self._stderr_tail)}"
            )
        )

    async def _read_stderr(self) -> None:
        process = self._require_process()
        if process.stderr is None:
            return
        while True:
            line = await process.stderr.readline()
            if not line:
                return
            self._stderr_tail.append(_redact(line.decode("utf-8", errors="replace").strip()))

    def _handle_response(self, message: dict[str, Any]) -> None:
        future = self._pending.get(message["id"])
        if future is None or future.done():
            return
        if "error" in message:
            error = message["error"]
            method = self._pending_methods.get(message["id"])
            if isinstance(error, dict):
                code = error.get("code")
                future.set_exception(
                    CodexAppServerProtocolError(
                        str(error.get("message") or error),
                        code=code if isinstance(code, int) else None,
                        data=error.get("data"),
                        method=method,
                    )
                )
            else:
                future.set_exception(CodexAppServerProtocolError(str(error), method=method))
        else:
            future.set_result(message.get("result"))

    def _dispatch_notification(self, message: dict[str, Any]) -> None:
        params = message.get("params") or {}
        thread_id = params.get("threadId")
        if thread_id is None and isinstance(params.get("thread"), dict):
            thread_id = params["thread"].get("id")
        if thread_id is None:
            return
        for queue in list(self._subscriptions.get(str(thread_id), ())):
            queue.put_nowait(message)

    async def _handle_server_request(self, message: dict[str, Any]) -> None:
        await self._send_raw(app_server_request_denial(message))

    async def _send_result(self, request_id: Any, result: dict[str, Any]) -> None:
        await self._send_raw({"id": request_id, "result": result})

    async def _send_error(self, request_id: Any, message: str) -> None:
        await self._send_raw({"id": request_id, "error": {"code": -32000, "message": message}})

    async def _send_raw(self, message: dict[str, Any]) -> None:
        process = self._require_process()
        if process.stdin is None:
            return
        async with self._send_lock:
            process.stdin.write(json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n")
            await process.stdin.drain()

    def _require_process(self) -> asyncio.subprocess.Process:
        if self._process is None:
            raise CodexAppServerError("Codex app-server has not been started.")
        return self._process

    def _fail_pending(self, error: BaseException) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)

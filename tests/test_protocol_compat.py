from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from codex_gateway._codex_shared import (
    CodexAppServerError,
    CodexAppServerProtocolError,
    CodexClientSettings,
    CodexProtocolCompatibilityError,
)
from codex_gateway.protocol_compat import INITIALIZE_PARAMS, ProtocolCompatibilityPreflight

INITIALIZE_RESPONSE = {
    "codexHome": "/tmp/codex-home",
    "platformFamily": "unix",
    "platformOs": "macos",
    "userAgent": "codex-cli/0.128.0",
}


class FakePreflightSession:
    def __init__(
        self,
        responses: dict[str, Any] | None = None,
        *,
        start_error: Exception | None = None,
        request_errors: dict[str, Exception] | None = None,
    ) -> None:
        self.responses = responses or {}
        self.start_error = start_error
        self.request_errors = request_errors or {}
        self.started = 0
        self.requests: list[tuple[str, dict[str, Any] | None, float | None]] = []

    async def start(self) -> None:
        self.started += 1
        if self.start_error is not None:
            raise self.start_error

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        self.requests.append((method, params, timeout))
        if method in self.request_errors:
            raise self.request_errors[method]
        return self.responses.get(method, {})


def run(coro):
    return asyncio.run(coro)


def settings() -> CodexClientSettings:
    return CodexClientSettings(
        command=("codex", "app-server", "--listen", "stdio://"),
        cwd=Path.cwd(),
        request_timeout_seconds=0.05,
    )


def test_preflight_starts_initializes_and_reaches_model_list() -> None:
    async def scenario() -> None:
        fake_session = FakePreflightSession(
            {
                "initialize": INITIALIZE_RESPONSE,
                "model/list": {"data": [{"id": "gpt-5-codex"}, {"id": "o4-mini"}]},
            }
        )

        report = await ProtocolCompatibilityPreflight(session=fake_session, settings=settings()).run()

        assert fake_session.started == 1
        assert fake_session.requests == [
            ("initialize", INITIALIZE_PARAMS, 0.05),
            ("model/list", {"includeHidden": False}, 0.05),
        ]
        assert report.model_count == 2
        assert report.command == ("codex", "app-server", "--listen", "stdio://")

    run(scenario())


def test_preflight_missing_codex_binary_becomes_local_compatibility_error() -> None:
    async def scenario() -> None:
        fake_session = FakePreflightSession(start_error=FileNotFoundError("missing"))

        with pytest.raises(CodexProtocolCompatibilityError) as exc_info:
            await ProtocolCompatibilityPreflight(session=fake_session, settings=settings()).run()

        assert exc_info.value.phase == "binary-start"
        assert "Unable to start Codex app-server command" in str(exc_info.value)
        assert "`codex` binary is on PATH" in str(exc_info.value)

    run(scenario())


def test_preflight_start_os_error_becomes_local_compatibility_error() -> None:
    async def scenario() -> None:
        fake_session = FakePreflightSession(start_error=OSError("permission denied"))

        with pytest.raises(CodexProtocolCompatibilityError) as exc_info:
            await ProtocolCompatibilityPreflight(session=fake_session, settings=settings()).run()

        assert exc_info.value.phase == "binary-start"
        assert "permission denied" in str(exc_info.value)

    run(scenario())


def test_preflight_method_not_found_points_to_installed_runtime_protocol() -> None:
    async def scenario() -> None:
        fake_session = FakePreflightSession(
            {"initialize": INITIALIZE_RESPONSE},
            request_errors={
                "model/list": CodexAppServerProtocolError(
                    "Method not found",
                    code=-32601,
                    method="model/list",
                )
            },
        )

        with pytest.raises(CodexProtocolCompatibilityError) as exc_info:
            await ProtocolCompatibilityPreflight(session=fake_session, settings=settings()).run()

        assert exc_info.value.phase == "model/list"
        assert exc_info.value.method == "model/list"
        assert "`model/list` is not available" in str(exc_info.value)
        assert "installed Codex runtime" in str(exc_info.value)

    run(scenario())


def test_preflight_wraps_timeout_and_generic_app_server_errors() -> None:
    async def timeout_case() -> None:
        fake_session = FakePreflightSession(request_errors={"initialize": TimeoutError()})

        with pytest.raises(CodexProtocolCompatibilityError) as exc_info:
            await ProtocolCompatibilityPreflight(session=fake_session, settings=settings()).run()

        assert exc_info.value.phase == "initialize"
        assert "Timed out waiting" in str(exc_info.value)

    async def generic_case() -> None:
        fake_session = FakePreflightSession(request_errors={"initialize": CodexAppServerError("exited")})

        with pytest.raises(CodexProtocolCompatibilityError) as exc_info:
            await ProtocolCompatibilityPreflight(session=fake_session, settings=settings()).run()

        assert exc_info.value.phase == "initialize"
        assert "could not initialize" in str(exc_info.value)

    run(timeout_case())
    run(generic_case())


def test_preflight_wraps_request_os_errors() -> None:
    async def scenario() -> None:
        fake_session = FakePreflightSession(
            request_errors={"initialize": OSError("broken pipe")},
        )

        with pytest.raises(CodexProtocolCompatibilityError) as exc_info:
            await ProtocolCompatibilityPreflight(session=fake_session, settings=settings()).run()

        assert exc_info.value.phase == "initialize"
        assert "I/O failed" in str(exc_info.value)

    run(scenario())


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (CodexAppServerProtocolError("Invalid params", code=-32602, method="model/list"), "rejected"),
        (CodexAppServerProtocolError("Unexpected failure", code=-32000, method="model/list"), "protocol error"),
    ],
)
def test_preflight_protocol_error_variants(error: CodexAppServerProtocolError, expected: str) -> None:
    async def scenario() -> None:
        fake_session = FakePreflightSession(
            {"initialize": INITIALIZE_RESPONSE},
            request_errors={"model/list": error},
        )

        with pytest.raises(CodexProtocolCompatibilityError) as exc_info:
            await ProtocolCompatibilityPreflight(session=fake_session, settings=settings()).run()

        assert exc_info.value.phase == "model/list"
        assert expected in str(exc_info.value)

    run(scenario())


def test_preflight_rejects_malformed_model_list_response() -> None:
    async def scenario() -> None:
        fake_session = FakePreflightSession(
            {
                "initialize": INITIALIZE_RESPONSE,
                "model/list": {"models": []},
            }
        )

        with pytest.raises(CodexProtocolCompatibilityError) as exc_info:
            await ProtocolCompatibilityPreflight(session=fake_session, settings=settings()).run()

        assert exc_info.value.phase == "model/list"
        assert "Expected `data` to be an array" in str(exc_info.value)

    run(scenario())


def test_preflight_rejects_malformed_initialize_response() -> None:
    async def scenario() -> None:
        fake_session = FakePreflightSession(
            {
                "initialize": {"capabilities": {}},
                "model/list": {"data": [{"id": "gpt-5-codex"}]},
            }
        )

        with pytest.raises(CodexProtocolCompatibilityError) as exc_info:
            await ProtocolCompatibilityPreflight(session=fake_session, settings=settings()).run()

        assert exc_info.value.phase == "initialize"
        assert exc_info.value.method == "initialize"
        assert "Missing field(s)" in str(exc_info.value)

    run(scenario())


def test_preflight_rejects_non_object_initialize_response() -> None:
    async def scenario() -> None:
        fake_session = FakePreflightSession(
            {
                "initialize": [],
                "model/list": {"data": [{"id": "gpt-5-codex"}]},
            }
        )

        with pytest.raises(CodexProtocolCompatibilityError) as exc_info:
            await ProtocolCompatibilityPreflight(session=fake_session, settings=settings()).run()

        assert exc_info.value.phase == "initialize"
        assert "got list" in str(exc_info.value)

    run(scenario())


def test_preflight_rejects_non_object_model_list_response() -> None:
    async def scenario() -> None:
        fake_session = FakePreflightSession(
            {
                "initialize": INITIALIZE_RESPONSE,
                "model/list": [],
            }
        )

        with pytest.raises(CodexProtocolCompatibilityError) as exc_info:
            await ProtocolCompatibilityPreflight(session=fake_session, settings=settings()).run()

        assert exc_info.value.phase == "model/list"
        assert "got list" in str(exc_info.value)

    run(scenario())


@pytest.mark.parametrize(
    "model_data",
    [
        ["not-a-model"],
        [{"name": "missing-id"}],
    ],
)
def test_preflight_rejects_malformed_model_entries(model_data: list[Any]) -> None:
    async def scenario() -> None:
        fake_session = FakePreflightSession(
            {
                "initialize": INITIALIZE_RESPONSE,
                "model/list": {"data": model_data},
            }
        )

        with pytest.raises(CodexProtocolCompatibilityError) as exc_info:
            await ProtocolCompatibilityPreflight(session=fake_session, settings=settings()).run()

        assert exc_info.value.phase == "model/list"
        assert exc_info.value.method == "model/list"
        assert "data[0]" in str(exc_info.value)

    run(scenario())

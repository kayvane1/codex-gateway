from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from codex_openai_shim import server
from codex_openai_shim.chat_contract import (
    _extract_content_parts,
    _extract_text_content,
    _messages_to_codex_history_and_input,
    _validate_chat_body,
    sse,
)
from codex_openai_shim.codex_client import CodexAppServerError, CodexChatResult
from codex_openai_shim.server import (
    LOCAL_TOKEN_PREFIX,
    OpenAIHTTPError,
    ShimSettings,
    _settings_from_args,
    _shell_exports,
    create_app,
    main,
)


class FakeCodex:
    def __init__(self) -> None:
        self.models_error: Exception | None = None
        self.complete_error: Exception | None = None
        self.stream_error: Exception | None = None
        self.completed: list[dict[str, Any]] = []
        self.streamed: list[dict[str, Any]] = []

    async def list_models(self) -> list[dict[str, str]]:
        if self.models_error:
            raise self.models_error
        return [{"id": "codex-test-model"}]

    async def complete_chat(
        self,
        *,
        model: str,
        history_items: list[dict[str, Any]],
        input_items: list[dict[str, Any]],
        developer_instructions: str,
    ) -> CodexChatResult:
        if self.complete_error:
            raise self.complete_error
        self.completed.append(
            {
                "model": model,
                "history_items": history_items,
                "input_items": input_items,
                "developer_instructions": developer_instructions,
            }
        )
        return CodexChatResult(
            text="unit-pong",
            usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        )

    async def stream_chat(
        self,
        *,
        model: str,
        history_items: list[dict[str, Any]],
        input_items: list[dict[str, Any]],
        developer_instructions: str,
    ):
        if self.stream_error:
            raise self.stream_error
        self.streamed.append(
            {
                "model": model,
                "history_items": history_items,
                "input_items": input_items,
                "developer_instructions": developer_instructions,
            }
        )
        yield "stream-"
        yield "pong"


def test_validate_chat_body_rejects_unsupported_shapes() -> None:
    with pytest.raises(OpenAIHTTPError) as non_object:
        _validate_chat_body([])
    assert non_object.value.code is None

    with pytest.raises(OpenAIHTTPError) as missing_model:
        _validate_chat_body({"messages": [{"role": "user", "content": "hi"}]})
    assert missing_model.value.code == "missing_model"

    with pytest.raises(OpenAIHTTPError) as bad_model:
        _validate_chat_body({"model": 1, "messages": [{"role": "user", "content": "hi"}]})
    assert bad_model.value.code == "invalid_model"

    with pytest.raises(OpenAIHTTPError) as empty_model:
        _validate_chat_body({"model": "", "messages": [{"role": "user", "content": "hi"}]})
    assert empty_model.value.code == "invalid_model"

    with pytest.raises(OpenAIHTTPError) as bad_messages:
        _validate_chat_body({"model": "m", "messages": []})
    assert bad_messages.value.code == "invalid_messages"

    with pytest.raises(OpenAIHTTPError) as unknown_field:
        _validate_chat_body({"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 0})
    assert unknown_field.value.code == "unsupported_feature"

    with pytest.raises(OpenAIHTTPError) as bad_stream:
        _validate_chat_body({"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": "true"})
    assert bad_stream.value.code == "invalid_stream"

    with pytest.raises(OpenAIHTTPError) as unsupported:
        _validate_chat_body({"model": "m", "messages": [{"role": "user", "content": "hi"}], "n": 2})
    assert unsupported.value.code == "unsupported_feature"

    with pytest.raises(OpenAIHTTPError) as bool_n:
        _validate_chat_body({"model": "m", "messages": [{"role": "user", "content": "hi"}], "n": True})
    assert bool_n.value.code == "unsupported_feature"

    _validate_chat_body({"model": "m", "messages": [{"role": "user", "content": "hi"}], "n": 1})
    _validate_chat_body({"model": "m", "messages": [{"role": "user", "content": "hi"}], "n": None, "stream": False})


def test_message_conversion_accepts_only_text_chat() -> None:
    history_items, input_items, instructions = _messages_to_codex_history_and_input(
        [
            {"role": "system", "content": "system instruction"},
            {"role": "developer", "content": "developer instruction"},
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": "previous answer"},
            {"role": "user", "content": None},
        ]
    )

    assert instructions == (
        "Caller-supplied system message:\nsystem instruction\n\nCaller-supplied developer message:\ndeveloper instruction"
    )
    assert "Do not execute" not in instructions
    assert "OpenAI compatibility request" not in instructions
    assert history_items == [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "previous answer"}]},
    ]
    assert input_items == [{"type": "text", "text": "", "text_elements": []}]

    with pytest.raises(OpenAIHTTPError) as non_dict:
        _messages_to_codex_history_and_input(["not-a-message"])
    assert non_dict.value.code == "invalid_messages"

    with pytest.raises(OpenAIHTTPError) as bad_role:
        _messages_to_codex_history_and_input([{"role": "tool", "content": "nope"}])
    assert bad_role.value.code == "unsupported_message_role"

    with pytest.raises(OpenAIHTTPError) as no_user:
        _messages_to_codex_history_and_input([{"role": "system", "content": "only instructions"}])
    assert no_user.value.code == "invalid_messages"

    with pytest.raises(OpenAIHTTPError) as assistant_last:
        _messages_to_codex_history_and_input([{"role": "assistant", "content": "prefill"}])
    assert assistant_last.value.code == "unsupported_message_sequence"

    assert _extract_content_parts(
        [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA==", "detail": "high"}}], 0
    ) == [{"type": "image_url", "image_url": "data:image/png;base64,AA==", "detail": "high"}]
    with pytest.raises(OpenAIHTTPError) as image_in_text_only:
        _extract_text_content([{"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}], 0)
    assert image_in_text_only.value.code == "unsupported_content_part"
    with pytest.raises(OpenAIHTTPError) as bad_image_url:
        _extract_content_parts([{"type": "image_url", "image_url": {"url": ""}}], 0)
    assert bad_image_url.value.code == "invalid_image_url"
    with pytest.raises(OpenAIHTTPError) as local_image_url:
        _extract_content_parts([{"type": "image_url", "image_url": {"url": "file:///private/tmp/image.png"}}], 0)
    assert local_image_url.value.code == "invalid_image_url"
    with pytest.raises(OpenAIHTTPError) as bad_detail:
        _extract_content_parts(
            [{"type": "image_url", "image_url": {"url": "https://example.com/image.png", "detail": "original"}}], 0
        )
    assert bad_detail.value.code == "unsupported_image_detail"

    with pytest.raises(OpenAIHTTPError) as bad_content:
        _extract_text_content({"type": "text", "text": "hi"}, 0)
    assert bad_content.value.code == "unsupported_content"


def test_image_messages_become_codex_inputs_and_history() -> None:
    history_items, input_items, instructions = _messages_to_codex_history_and_input(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "old image"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,OLD=", "detail": "low"}},
                ],
            },
            {"role": "assistant", "content": "old answer"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "new image"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,NEW=", "detail": "high"}},
                ],
            },
        ]
    )

    assert instructions == ""
    assert history_items == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "old image"},
                {"type": "input_image", "image_url": "data:image/png;base64,OLD=", "detail": "low"},
            ],
        },
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "old answer"}]},
    ]
    assert input_items == [
        {"type": "text", "text": "new image", "text_elements": []},
        {"type": "image", "url": "data:image/png;base64,NEW="},
    ]

    with pytest.raises(OpenAIHTTPError) as assistant_image:
        _messages_to_codex_history_and_input(
            [
                {
                    "role": "assistant",
                    "content": [{"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}],
                },
                {"role": "user", "content": "next"},
            ]
        )
    assert assistant_image.value.code == "unsupported_content_part"


def test_sse_uses_openai_data_framing() -> None:
    assert sse({"hello": "world"}) == b'data: {"hello":"world"}\n\n'


@pytest.mark.parametrize("endpoint", ["/v1/models", "/v1/chat/completions"])
def test_http_auth_rejects_missing_token(endpoint: str) -> None:
    async def run() -> None:
        app = create_app(ShimSettings(token="token"))
        app.state.codex = FakeCodex()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            if endpoint.endswith("models"):
                response = await client.get(endpoint)
            else:
                response = await client.post(
                    endpoint, json={"model": "m", "messages": [{"role": "user", "content": "hi"}]}
                )
        assert response.status_code == 401
        assert response.json()["error"]["type"] == "authentication_error"

    import asyncio

    asyncio.run(run())


def test_lifespan_starts_codex_assigns_state_and_healthz_is_public(monkeypatch: pytest.MonkeyPatch) -> None:
    class LifespanCodex(FakeCodex):
        def __init__(self, settings: Any) -> None:
            super().__init__()
            self.settings = settings
            self.started = 0
            self.stopped = 0

        async def start(self) -> None:
            self.started += 1

        async def stop(self) -> None:
            self.stopped += 1

    created: list[LifespanCodex] = []

    def fake_codex_app_server(settings: Any) -> LifespanCodex:
        fake = LifespanCodex(settings)
        created.append(fake)
        return fake

    async def run() -> None:
        monkeypatch.setattr(server, "CodexAppServer", fake_codex_app_server)
        app = create_app(
            ShimSettings(
                token="token",
                codex_command=("fake-codex", "app-server"),
                cwd=Path.cwd(),
                request_timeout_seconds=1.5,
                turn_timeout_seconds=2.5,
                reasoning_effort="medium",
            )
        )

        assert len(created) == 1
        fake = created[0]
        assert fake.settings.command == ("fake-codex", "app-server")
        assert fake.settings.request_timeout_seconds == 1.5
        assert fake.settings.turn_timeout_seconds == 2.5
        assert fake.settings.reasoning_effort == "medium"

        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            assert app.state.codex is fake
            assert fake.started == 1
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/healthz")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert fake.stopped == 1

    import asyncio

    asyncio.run(run())


def test_http_endpoints_cover_success_and_error_branches() -> None:
    async def run() -> None:
        app = create_app(ShimSettings(token="token"))
        fake = FakeCodex()
        app.state.codex = fake
        headers = {"Authorization": "Bearer token"}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            models = await client.get("/v1/models", headers=headers)
            assert models.status_code == 200
            assert models.json()["data"][0]["id"] == "codex-test-model"

            fake.models_error = CodexAppServerError("model list failed")
            models_error = await client.get("/v1/models", headers=headers)
            assert models_error.status_code == 502
            assert models_error.json()["error"]["code"] == "codex_error"
            fake.models_error = None

            invalid_json = await client.post(
                "/v1/chat/completions",
                headers={**headers, "content-type": "application/json"},
                content=b"{",
            )
            assert invalid_json.status_code == 400
            assert invalid_json.json()["error"]["code"] == "invalid_json"

            completion = await client.post(
                "/v1/chat/completions",
                headers=headers,
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert completion.status_code == 200
            assert completion.json()["choices"][0]["message"]["content"] == "unit-pong"
            assert fake.completed[0]["history_items"] == []
            assert fake.completed[0]["input_items"][0]["text"] == "hi"

            multi_turn = await client.post(
                "/v1/chat/completions",
                headers=headers,
                json={
                    "model": "m",
                    "messages": [
                        {"role": "system", "content": "system note"},
                        {"role": "user", "content": "first"},
                        {"role": "assistant", "content": "second"},
                        {"role": "user", "content": "third"},
                    ],
                },
            )
            assert multi_turn.status_code == 200
            assert fake.completed[1]["history_items"] == [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "first"}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "second"}]},
            ]
            assert fake.completed[1]["input_items"] == [{"type": "text", "text": "third", "text_elements": []}]
            assert "system note" in fake.completed[1]["developer_instructions"]
            assert "Caller-supplied system message:" in fake.completed[1]["developer_instructions"]
            assert "Do not execute" not in fake.completed[1]["developer_instructions"]

            fake.complete_error = CodexAppServerError("turn failed")
            completion_error = await client.post(
                "/v1/chat/completions",
                headers=headers,
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert completion_error.status_code == 502
            assert completion_error.json()["error"]["type"] == "codex_app_server_error"
            fake.complete_error = None

            stream = await client.post(
                "/v1/chat/completions",
                headers=headers,
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            )
            assert stream.status_code == 200
            assert '"content":"stream-"' in stream.text
            assert '"content":"pong"' in stream.text
            assert stream.text.endswith("data: [DONE]\n\n")

            fake.stream_error = CodexAppServerError("stream failed")
            stream_error = await client.post(
                "/v1/chat/completions",
                headers=headers,
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            )
            assert "stream failed" in stream_error.text
            assert stream_error.text.endswith("data: [DONE]\n\n")

            image_completion = await client.post(
                "/v1/chat/completions",
                headers=headers,
                json={
                    "model": "m",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "look"},
                                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                            ],
                        }
                    ],
                },
            )
            assert image_completion.status_code == 200
            assert fake.completed[-1]["input_items"] == [
                {"type": "text", "text": "look", "text_elements": []},
                {"type": "image", "url": "data:image/png;base64,AA=="},
            ]

    import asyncio

    asyncio.run(run())


def test_settings_and_main_cli_paths(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("CODEX_SHIM_TOKEN", raising=False)
    monkeypatch.delenv("CODEX_SHIM_HOST", raising=False)
    monkeypatch.delenv("CODEX_SHIM_PORT", raising=False)
    settings = _settings_from_args(["--token", "local", "--port", "8123", "--cwd", str(Path.cwd())])
    assert settings.token == "local"
    assert settings.port == 8123
    assert settings.generated_token is False

    generated = _settings_from_args(["--port", "8124"])
    assert generated.token.startswith(LOCAL_TOKEN_PREFIX)
    assert generated.generated_token is True

    env_settings = _settings_from_args(["env", "--token", "local", "--port", "8125"])
    assert env_settings.token == "local"
    assert env_settings.port == 8125
    assert _shell_exports(env_settings).splitlines() == [
        "export CODEX_SHIM_HOST=127.0.0.1",
        "export CODEX_SHIM_PORT=8125",
        "export CODEX_SHIM_BASE_URL=http://127.0.0.1:8125/v1",
        "export CODEX_SHIM_TOKEN=local",
    ]

    with pytest.raises(SystemExit):
        _settings_from_args(["--host", "0.0.0.0"])
    with pytest.raises(SystemExit):
        _settings_from_args(["--token", "sk-not-a-local-shim-token"])

    calls: list[dict[str, Any]] = []

    def fake_run(app: Any, *, host: str, port: int, log_level: str) -> None:
        calls.append({"app": app, "host": host, "port": port, "log_level": log_level})

    monkeypatch.setattr(server.uvicorn, "run", fake_run)
    main(["--port", "8130"])
    captured = capsys.readouterr()
    assert "Generated a local shim token" in captured.out
    assert "client = OpenAI(base_url=" in captured.out
    assert LOCAL_TOKEN_PREFIX in captured.out
    assert calls[-1]["host"] == "127.0.0.1"
    assert calls[-1]["port"] == 8130

    main(["--token", "explicit", "--port", "8131"])
    captured = capsys.readouterr()
    assert "Generated a local shim token" not in captured.out
    assert calls[-1]["port"] == 8131

    main(["token", "--token", "only-token"])
    captured = capsys.readouterr()
    assert captured.out == "only-token\n"
    assert calls[-1]["port"] == 8131

    main(["env", "--token", "env-token", "--port", "8132"])
    captured = capsys.readouterr()
    assert "export CODEX_SHIM_BASE_URL=http://127.0.0.1:8132/v1\n" in captured.out
    assert "export CODEX_SHIM_TOKEN=env-token\n" in captured.out
    assert "OPENAI_API_KEY" not in captured.out
    assert calls[-1]["port"] == 8131

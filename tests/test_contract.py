from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path

import httpx
import openai
import pytest
import uvicorn

from codex_gateway.server import GatewaySettings, create_app

TEST_TOKEN = "local-contract-test-token"
RUN_CONTRACT_TESTS_ENV = "CODEX_GATEWAY_RUN_CONTRACT_TESTS"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get(RUN_CONTRACT_TESTS_ENV) != "1",
        reason=f"set {RUN_CONTRACT_TESTS_ENV}=1 to run live Codex app-server contract tests",
    ),
]


@pytest.fixture(scope="module")
def gateway_server() -> str:
    port = _free_port()
    settings = GatewaySettings(
        token=TEST_TOKEN,
        host="127.0.0.1",
        port=port,
        cwd=Path.cwd(),
        request_timeout_seconds=60,
        turn_timeout_seconds=240,
        reasoning_effort="low",
    )
    app = create_app(settings)
    config = uvicorn.Config(app, host=settings.host, port=settings.port, log_level="warning", lifespan="on")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://{settings.host}:{settings.port}"
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{base_url}/healthz", timeout=2)
            if response.status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    else:
        server.should_exit = True
        thread.join(timeout=10)
        raise RuntimeError("FastAPI gateway did not become ready.")

    yield base_url

    server.should_exit = True
    thread.join(timeout=20)


def test_openai_sdk_models_non_streaming_and_streaming_chat(gateway_server: str) -> None:
    client = openai.OpenAI(base_url=f"{gateway_server}/v1", api_key=TEST_TOKEN, timeout=240)

    models = client.models.list()
    assert models.data
    model = models.data[0].id

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: contract-pong"}],
    )
    assert completion.object == "chat.completion"
    assert completion.choices[0].message.role == "assistant"
    assert "contract-pong" in (completion.choices[0].message.content or "").lower()

    multi_turn_completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "The earlier answer token is history-pong."},
            {"role": "assistant", "content": "I will use history-pong as the answer token."},
            {"role": "user", "content": "Reply with exactly the earlier answer token."},
        ],
    )
    assert multi_turn_completion.object == "chat.completion"
    assert multi_turn_completion.choices[0].message.role == "assistant"
    assert "history-pong" in (multi_turn_completion.choices[0].message.content or "").lower()

    image_completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Reply with exactly: image-pong"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
                        },
                    },
                ],
            }
        ],
    )
    assert image_completion.object == "chat.completion"
    assert image_completion.choices[0].message.role == "assistant"
    assert "image-pong" in (image_completion.choices[0].message.content or "").lower()

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: stream-pong"}],
        stream=True,
    )
    streamed_text = "".join(chunk.choices[0].delta.content or "" for chunk in stream)
    assert "stream-pong" in streamed_text.lower()


def test_rejects_missing_or_invalid_local_bearer_tokens(gateway_server: str) -> None:
    missing = httpx.get(f"{gateway_server}/v1/models", timeout=10)
    assert missing.status_code == 401
    assert missing.json()["error"]["type"] == "authentication_error"

    invalid = httpx.get(
        f"{gateway_server}/v1/models",
        headers={"Authorization": "Bearer not-the-local-token"},
        timeout=10,
    )
    assert invalid.status_code == 401
    assert invalid.json()["error"]["code"] == "invalid_api_key"

    bad_client = openai.OpenAI(base_url=f"{gateway_server}/v1", api_key="not-the-local-token", timeout=10)
    with pytest.raises(openai.AuthenticationError):
        bad_client.models.list()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])

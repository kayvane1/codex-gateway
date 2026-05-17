from __future__ import annotations

from typing import Any

import httpx
import pytest
from openai.types.responses import Response

from codex_gateway.chat_contract import OpenAIContractError
from codex_gateway.codex_client import CodexAppServerError, CodexChatResult
from codex_gateway.responses_contract import (
    _extract_response_text_content,
    _response_input_to_codex_history_and_input,
    _validate_response_body,
    prepare_response_turn,
    response_payload,
)
from codex_gateway.server import GatewaySettings, create_app
from tests.test_server_unit import FakeCodex


def test_prepare_response_turn_accepts_text_input() -> None:
    turn = prepare_response_turn({"model": "m", "input": "hello"})

    assert turn.model == "m"
    assert turn.history_items == []
    assert turn.input_items == [{"type": "text", "text": "hello", "text_elements": []}]
    assert turn.developer_instructions == ""
    assert turn.instructions is None


def test_prepare_response_turn_accepts_message_input() -> None:
    turn = prepare_response_turn(
        {
            "model": "m",
            "instructions": "top-level instruction",
            "input": [
                {"role": "system", "content": "system note"},
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "dev note"}]},
                {"role": "user", "content": [{"type": "input_text", "text": "first"}]},
                {"role": "assistant", "content": [{"type": "output_text", "text": "second"}]},
                {"role": "user", "content": "third"},
            ],
            "stream": False,
            "store": False,
            "parallel_tool_calls": False,
            "tools": [],
            "text": {"format": {"type": "text"}},
        }
    )

    assert turn.model == "m"
    assert turn.instructions == "top-level instruction"
    assert turn.history_items == [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "first"}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "second"}]},
    ]
    assert turn.input_items == [{"type": "text", "text": "third", "text_elements": []}]
    assert "Caller-supplied response instructions:\ntop-level instruction" in turn.developer_instructions
    assert "Caller-supplied system message:\nsystem note" in turn.developer_instructions
    assert "Caller-supplied developer message:\ndev note" in turn.developer_instructions


@pytest.mark.parametrize(
    ("body", "code", "message"),
    [
        ([], None, "JSON object"),
        ({"input": "hi"}, "missing_model", "`model` is required"),
        ({"model": "", "input": "hi"}, "invalid_model", "non-empty string"),
        ({"model": "m"}, "missing_input", "`input` is required"),
        ({"model": "m", "input": {}}, "invalid_input", "`input` must be a string"),
        ({"model": "m", "input": "hi", "stream": True}, "unsupported_feature", "Streaming responses"),
        ({"model": "m", "input": "hi", "background": True}, "unsupported_feature", "Background responses"),
        (
            {"model": "m", "input": "hi", "parallel_tool_calls": True},
            "unsupported_feature",
            "Parallel tool calls",
        ),
        ({"model": "m", "input": "hi", "store": True}, "unsupported_feature", "Stored response state"),
        (
            {"model": "m", "input": "hi", "tools": [{"type": "web_search_preview"}]},
            "unsupported_feature",
            "built-in tools",
        ),
        (
            {"model": "m", "input": "hi", "text": {"format": {"type": "json_schema", "schema": {}}}},
            "unsupported_feature",
            "JSON schema output",
        ),
        ({"model": "m", "input": "hi", "instructions": []}, "invalid_instructions", "`instructions`"),
        ({"model": "m", "input": "hi", "stream": "true"}, "invalid_stream", "`stream` must be a boolean"),
        ({"model": "m", "input": "hi", "tools": {}}, "invalid_tools", "`tools` must be an array"),
        ({"model": "m", "input": "hi", "text": []}, "invalid_text", "`text` must be an object"),
        ({"model": "m", "input": "hi", "text": {"verbosity": "low"}}, "unsupported_feature", "verbosity"),
        ({"model": "m", "input": "hi", "text": {"format": []}}, "invalid_text", "`text.format`"),
        ({"model": "m", "input": "hi", "temperature": 0}, "unsupported_feature", "temperature"),
    ],
)
def test_validate_response_body_rejects_unsupported_shapes(body: Any, code: str | None, message: str) -> None:
    with pytest.raises(OpenAIContractError) as exc_info:
        _validate_response_body(body)

    assert exc_info.value.code == code
    assert message in exc_info.value.message


def test_response_input_conversion_rejects_unsupported_items_and_content() -> None:
    with pytest.raises(OpenAIContractError) as empty_input:
        _response_input_to_codex_history_and_input([])
    assert empty_input.value.code == "invalid_input"

    with pytest.raises(OpenAIContractError) as non_object:
        _response_input_to_codex_history_and_input(["not-a-message"])
    assert non_object.value.code == "invalid_input"

    with pytest.raises(OpenAIContractError) as bad_input_type:
        _response_input_to_codex_history_and_input({})
    assert bad_input_type.value.code == "invalid_input"

    with pytest.raises(OpenAIContractError) as tool_item:
        _response_input_to_codex_history_and_input([{"type": "function_call", "name": "tool"}])
    assert tool_item.value.code == "unsupported_response_input_item"

    with pytest.raises(OpenAIContractError) as bad_role:
        _response_input_to_codex_history_and_input([{"role": "tool", "content": "nope"}])
    assert bad_role.value.code == "unsupported_message_role"

    with pytest.raises(OpenAIContractError) as unsupported_message_field:
        _response_input_to_codex_history_and_input(
            [{"id": "msg_123", "status": "completed", "role": "user", "content": "hi"}]
        )
    assert unsupported_message_field.value.code == "unsupported_feature"
    assert "id, status" in unsupported_message_field.value.message

    with pytest.raises(OpenAIContractError) as no_user:
        _response_input_to_codex_history_and_input([{"role": "system", "content": "only instructions"}])
    assert no_user.value.code == "invalid_input"

    with pytest.raises(OpenAIContractError) as assistant_last:
        _response_input_to_codex_history_and_input([{"role": "assistant", "content": "prefill"}])
    assert assistant_last.value.code == "unsupported_message_sequence"

    with pytest.raises(OpenAIContractError) as file_input:
        _response_input_to_codex_history_and_input(
            [{"role": "user", "content": [{"type": "input_file", "file_id": "file_123"}]}]
        )
    assert file_input.value.code == "unsupported_file_input"

    with pytest.raises(OpenAIContractError) as image_input:
        _response_input_to_codex_history_and_input(
            [{"role": "user", "content": [{"type": "input_image", "image_url": "https://example.com/image.png"}]}]
        )
    assert image_input.value.code == "unsupported_content_part"

    with pytest.raises(OpenAIContractError) as output_text_for_user:
        _extract_response_text_content([{"type": "output_text", "text": "wrong side"}], 0, "user")
    assert output_text_for_user.value.code == "unsupported_content_part"

    assert _extract_response_text_content(None, 0, "user") == ""

    with pytest.raises(OpenAIContractError) as non_object_part:
        _extract_response_text_content(["not-a-part"], 0, "user")
    assert non_object_part.value.code == "unsupported_content_part"

    with pytest.raises(OpenAIContractError) as unsupported_content_field:
        _extract_response_text_content(
            [{"type": "output_text", "text": "answer", "annotations": [], "logprobs": []}],
            0,
            "assistant",
        )
    assert unsupported_content_field.value.code == "unsupported_feature"
    assert "annotations, logprobs" in unsupported_content_field.value.message

    with pytest.raises(OpenAIContractError) as non_text_content:
        _extract_response_text_content({"type": "input_text", "text": "hi"}, 0, "user")
    assert non_text_content.value.code == "unsupported_content"


def test_response_payload_matches_openai_sdk_response_model() -> None:
    payload = response_payload(
        response_id="resp_codex_test",
        message_id="msg_codex_test",
        created_at=123,
        model="m",
        result=CodexChatResult(
            text="unit-pong",
            usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        ),
        instructions="be brief",
    )

    response = Response.model_validate(payload)
    assert response.object == "response"
    assert response.status == "completed"
    assert response.output_text == "unit-pong"
    assert response.instructions == "be brief"
    assert response.usage is not None
    assert response.usage.input_tokens == 1
    assert response.usage.output_tokens == 2
    assert response.usage.total_tokens == 3


def test_responses_endpoint_uses_codex_turn_lifecycle() -> None:
    async def run() -> None:
        app = create_app(GatewaySettings(token="token"))
        fake = FakeCodex()
        app.state.codex = fake
        headers = {"Authorization": "Bearer token"}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/responses",
                headers=headers,
                json={
                    "model": "m",
                    "instructions": "answer plainly",
                    "input": [
                        {"role": "user", "content": [{"type": "input_text", "text": "first"}]},
                        {"role": "assistant", "content": [{"type": "output_text", "text": "second"}]},
                        {"role": "user", "content": "third"},
                    ],
                },
            )

            assert response.status_code == 200
            payload = response.json()
            assert payload["object"] == "response"
            assert payload["output"][0]["content"][0]["text"] == "unit-pong"
            assert payload["usage"]["input_tokens"] == 1
            assert fake.completed == [
                {
                    "model": "m",
                    "history_items": [
                        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "first"}]},
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "second"}],
                        },
                    ],
                    "input_items": [{"type": "text", "text": "third", "text_elements": []}],
                    "developer_instructions": "Caller-supplied response instructions:\nanswer plainly",
                }
            ]

            invalid_json = await client.post(
                "/v1/responses",
                headers={**headers, "content-type": "application/json"},
                content=b"{",
            )
            assert invalid_json.status_code == 400
            assert invalid_json.json()["error"]["code"] == "invalid_json"

            fake.complete_error = CodexAppServerError("response turn failed")
            codex_error = await client.post(
                "/v1/responses",
                headers=headers,
                json={"model": "m", "input": "hi"},
            )
            assert codex_error.status_code == 502
            assert codex_error.json()["error"]["type"] == "codex_app_server_error"

    import asyncio

    asyncio.run(run())

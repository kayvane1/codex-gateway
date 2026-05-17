from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ._codex_shared import CodexChatResult, build_text_input
from .safety_policy import SafetyPolicyViolation, parse_data_image_url, validate_image_url


class OpenAIContractError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        error_type: str,
        code: str | None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.code = code
        self.headers = headers


@dataclass(frozen=True)
class ChatTurn:
    model: str
    history_items: list[dict[str, Any]]
    input_items: list[dict[str, Any]]
    developer_instructions: str


SUPPORTED_CHAT_FIELDS = {"model", "messages", "stream", "n"}


OPENAI_IMAGE_DETAILS = {"auto", "low", "high"}


def prepare_chat_turn(body: dict[str, Any]) -> ChatTurn:
    _validate_chat_body(body)
    history_items, input_items, developer_instructions = _messages_to_codex_history_and_input(body["messages"])
    return ChatTurn(
        model=str(body["model"]),
        history_items=history_items,
        input_items=input_items,
        developer_instructions=developer_instructions,
    )


def completion_payload(
    *,
    completion_id: str,
    created: int,
    model: str,
    result: CodexChatResult,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": "stop",
            }
        ],
        "usage": result.usage,
    }


def stream_start_payload(*, completion_id: str, created: int, model: str) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }


def stream_delta_payload(*, completion_id: str, created: int, model: str, delta: str) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
    }


def stream_stop_payload(*, completion_id: str, created: int, model: str) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }


def stream_error_payload(
    *,
    message: str,
    error_type: str = "codex_app_server_error",
    code: str | None = "codex_error",
) -> dict[str, Any]:
    return {"error": {"message": message, "type": error_type, "code": code}}


def sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode("utf-8")


def sse_done() -> bytes:
    return b"data: [DONE]\n\n"


def openai_error_payload(message: str, error_type: str, code: str | None) -> dict[str, dict[str, str | None]]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code,
        }
    }


def _validate_chat_body(body: dict[str, Any]) -> None:
    if not isinstance(body, dict):
        raise_openai_error(400, "Request body must be a JSON object.", "invalid_request_error", None)

    unsupported = sorted(set(body) - SUPPORTED_CHAT_FIELDS)
    if unsupported:
        raise_openai_error(
            400,
            "Unsupported OpenAI chat.completions field(s): " + ", ".join(unsupported),
            "invalid_request_error",
            "unsupported_feature",
        )

    model = body.get("model")
    if "model" not in body:
        raise_openai_error(400, "`model` is required.", "invalid_request_error", "missing_model")
    if not isinstance(model, str) or not model:
        raise_openai_error(400, "`model` must be a non-empty string.", "invalid_request_error", "invalid_model")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise_openai_error(400, "`messages` must be a non-empty list.", "invalid_request_error", "invalid_messages")

    stream = body.get("stream")
    if "stream" in body and not isinstance(stream, bool):
        raise_openai_error(400, "`stream` must be a boolean.", "invalid_request_error", "invalid_stream")

    n = body.get("n")
    if "n" in body and not (n is None or (type(n) is int and n == 1)):
        raise_openai_error(
            400,
            "Unsupported OpenAI chat.completions field(s): n",
            "invalid_request_error",
            "unsupported_feature",
        )


def _messages_to_codex_history_and_input(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    instruction_parts: list[str] = []
    dialogue: list[tuple[str, list[dict[str, Any]]]] = []

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise_openai_error(
                400, f"`messages[{index}]` must be an object.", "invalid_request_error", "invalid_messages"
            )
        role = message.get("role")
        content_parts = _extract_content_parts(message.get("content"), index)

        if role in {"system", "developer"}:
            instruction_parts.append(f"Caller-supplied {role} message:\n{_content_parts_to_text(content_parts, index)}")
        elif role in {"user", "assistant"}:
            dialogue.append((role, content_parts))
        else:
            raise_openai_error(
                400,
                f"Unsupported message role at messages[{index}]: {role!r}",
                "invalid_request_error",
                "unsupported_message_role",
            )

    if not dialogue:
        raise_openai_error(400, "At least one user message is required.", "invalid_request_error", "invalid_messages")
    if dialogue[-1][0] != "user":
        raise_openai_error(
            400,
            "The final user/assistant message must be a user message for this OpenAI compatibility MVP.",
            "invalid_request_error",
            "unsupported_message_sequence",
        )

    history_items = [
        _message_to_response_item(role, content_parts, index)
        for index, (role, content_parts) in enumerate(dialogue[:-1])
    ]
    input_items = _content_parts_to_user_input(dialogue[-1][1])
    return history_items, input_items, "\n\n".join(instruction_parts)


def _message_to_response_item(role: str, content_parts: list[dict[str, Any]], message_index: int) -> dict[str, Any]:
    if role == "assistant":
        content = [{"type": "output_text", "text": _content_parts_to_text(content_parts, message_index)}]
    else:
        content = [_content_part_to_response_content(part) for part in content_parts]
    return {
        "type": "message",
        "role": role,
        "content": content,
    }


def _content_part_to_response_content(part: dict[str, Any]) -> dict[str, Any]:
    if part["type"] == "text":
        return {"type": "input_text", "text": part["text"]}
    image_content = {"type": "input_image", "image_url": part["image_url"]}
    if part.get("detail") is not None:
        image_content["detail"] = part["detail"]
    return image_content


def _content_parts_to_user_input(content_parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    input_items: list[dict[str, Any]] = []
    for part in content_parts:
        if part["type"] == "text":
            input_items.extend(build_text_input(part["text"]))
        else:
            input_items.append({"type": "image", "url": part["image_url"]})
    return input_items


def _content_parts_to_text(content_parts: list[dict[str, Any]], index: int) -> str:
    chunks: list[str] = []
    for part in content_parts:
        if part["type"] != "text":
            raise_openai_error(
                400,
                f"Images are supported only in user messages at messages[{index}].content.",
                "invalid_request_error",
                "unsupported_content_part",
            )
        chunks.append(part["text"])
    return "\n".join(chunks)


def _extract_content_parts(content: Any, index: int) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for part_index, part in enumerate(content):
            if not isinstance(part, dict):
                raise_openai_error(
                    400,
                    f"Only text and image_url content parts are supported at messages[{index}].content[{part_index}].",
                    "invalid_request_error",
                    "unsupported_content_part",
                )
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image_url":
                parts.append(_extract_image_content_part(part, index, part_index))
            else:
                raise_openai_error(
                    400,
                    f"Only text and image_url content parts are supported at messages[{index}].content[{part_index}].",
                    "invalid_request_error",
                    "unsupported_content_part",
                )
        return parts or [{"type": "text", "text": ""}]
    if content is None:
        return [{"type": "text", "text": ""}]
    raise_openai_error(
        400,
        f"Only string or text/image content parts are supported at messages[{index}].content.",
        "invalid_request_error",
        "unsupported_content",
    )


def _extract_image_content_part(part: dict[str, Any], message_index: int, part_index: int) -> dict[str, Any]:
    image_url = part.get("image_url")
    if not isinstance(image_url, dict) or not isinstance(image_url.get("url"), str) or not image_url["url"]:
        raise_openai_error(
            400,
            f"`messages[{message_index}].content[{part_index}].image_url.url` must be a non-empty string.",
            "invalid_request_error",
            "invalid_image_url",
        )
    _validate_image_url(image_url["url"], message_index, part_index)
    detail = image_url.get("detail")
    if detail is not None and detail not in OPENAI_IMAGE_DETAILS:
        raise_openai_error(
            400,
            f"`messages[{message_index}].content[{part_index}].image_url.detail` must be one of: auto, low, high.",
            "invalid_request_error",
            "unsupported_image_detail",
        )
    normalized = {"type": "image_url", "image_url": image_url["url"]}
    if detail is not None:
        normalized["detail"] = detail
    return normalized


def _validate_data_image_url(url: str, message_index: int, part_index: int) -> None:
    param = f"messages[{message_index}].content[{part_index}].image_url.url"
    try:
        parse_data_image_url(url, param=param)
    except SafetyPolicyViolation as exc:
        _raise_safety_policy_violation(exc)


def _validate_image_url(url: str, message_index: int, part_index: int) -> None:
    param = f"messages[{message_index}].content[{part_index}].image_url.url"
    try:
        validate_image_url(url, param=param)
    except SafetyPolicyViolation as exc:
        _raise_safety_policy_violation(exc)


def _raise_safety_policy_violation(exc: SafetyPolicyViolation) -> None:
    raise_openai_error(exc.status_code, exc.message, exc.error_type, exc.code)


def _extract_text_content(content: Any, index: int) -> str:
    return _content_parts_to_text(_extract_content_parts(content, index), index)


def raise_openai_error(
    status_code: int,
    message: str,
    error_type: str,
    code: str | None,
    headers: dict[str, str] | None = None,
) -> None:
    raise OpenAIContractError(status_code, message, error_type, code, headers)

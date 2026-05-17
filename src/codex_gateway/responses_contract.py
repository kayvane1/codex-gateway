from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._codex_shared import CodexChatResult, build_text_input
from .chat_contract import raise_openai_error


@dataclass(frozen=True)
class ResponseTurn:
    model: str
    history_items: list[dict[str, Any]]
    input_items: list[dict[str, Any]]
    developer_instructions: str
    instructions: str | None


SUPPORTED_RESPONSE_FIELDS = {
    "background",
    "input",
    "instructions",
    "model",
    "parallel_tool_calls",
    "store",
    "stream",
    "text",
    "tools",
}

RESPONSE_TEXT_FORMAT_KEYS = {"type"}
RESPONSE_TEXT_KEYS = {"format"}
RESPONSE_MESSAGE_KEYS = {"content", "role", "type"}
RESPONSE_TEXT_CONTENT_KEYS = {"text", "type"}
MESSAGE_ITEM_TYPES = {None, "message"}
MESSAGE_ROLES = {"system", "developer", "user", "assistant"}
TEXT_CONTENT_PART_TYPES = {"input_text", "output_text"}
FILE_CONTENT_KEYS = {"file_data", "file_id", "file_url", "filename"}


def prepare_response_turn(body: dict[str, Any]) -> ResponseTurn:
    _validate_response_body(body)
    history_items, input_items, message_instructions = _response_input_to_codex_history_and_input(body["input"])
    instructions = body.get("instructions")
    instruction_parts = []
    if instructions:
        instruction_parts.append(f"Caller-supplied response instructions:\n{instructions}")
    if message_instructions:
        instruction_parts.append(message_instructions)
    return ResponseTurn(
        model=str(body["model"]),
        history_items=history_items,
        input_items=input_items,
        developer_instructions="\n\n".join(instruction_parts),
        instructions=instructions,
    )


def response_payload(
    *,
    response_id: str,
    message_id: str,
    created_at: int,
    model: str,
    result: CodexChatResult,
    instructions: str | None,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "completed_at": created_at,
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": instructions,
        "max_output_tokens": None,
        "max_tool_calls": None,
        "model": model,
        "output": [
            {
                "id": message_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": result.text,
                        "annotations": [],
                    }
                ],
            }
        ],
        "parallel_tool_calls": False,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": False,
        "temperature": None,
        "text": {"format": {"type": "text"}},
        "tool_choice": "none",
        "tools": [],
        "top_logprobs": 0,
        "top_p": None,
        "truncation": "disabled",
        "usage": _responses_usage(result.usage),
        "user": None,
        "metadata": {},
    }


def _validate_response_body(body: dict[str, Any]) -> None:
    if not isinstance(body, dict):
        raise_openai_error(400, "Request body must be a JSON object.", "invalid_request_error", None)

    unsupported = sorted(set(body) - SUPPORTED_RESPONSE_FIELDS)
    if unsupported:
        raise_openai_error(
            400,
            "Unsupported OpenAI responses field(s): " + ", ".join(unsupported),
            "invalid_request_error",
            "unsupported_feature",
        )

    model = body.get("model")
    if "model" not in body:
        raise_openai_error(400, "`model` is required.", "invalid_request_error", "missing_model")
    if not isinstance(model, str) or not model:
        raise_openai_error(400, "`model` must be a non-empty string.", "invalid_request_error", "invalid_model")

    if "input" not in body:
        raise_openai_error(400, "`input` is required.", "invalid_request_error", "missing_input")
    response_input = body["input"]
    if not isinstance(response_input, (str, list)) or response_input == []:
        raise_openai_error(
            400,
            "`input` must be a string or a non-empty array of message input items.",
            "invalid_request_error",
            "invalid_input",
        )

    instructions = body.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise_openai_error(
            400, "`instructions` must be a string when provided.", "invalid_request_error", "invalid_instructions"
        )

    _validate_optional_false(body, "stream", "Streaming responses are not supported by this gateway.")
    _validate_optional_false(body, "background", "Background responses are not supported by this gateway.")
    _validate_optional_false(
        body,
        "parallel_tool_calls",
        "Parallel tool calls are not supported by this gateway.",
    )
    _validate_optional_false(body, "store", "Stored response state is not supported by this gateway.")
    _validate_tools(body)
    _validate_text_format(body)


def _validate_optional_false(body: dict[str, Any], field: str, unsupported_message: str) -> None:
    if field not in body or body[field] is None:
        return
    value = body[field]
    if type(value) is not bool:
        raise_openai_error(
            400,
            f"`{field}` must be a boolean when provided.",
            "invalid_request_error",
            f"invalid_{field}",
        )
    if value:
        raise_openai_error(400, unsupported_message, "invalid_request_error", "unsupported_feature")


def _validate_tools(body: dict[str, Any]) -> None:
    if "tools" not in body or body["tools"] is None:
        return
    tools = body["tools"]
    if not isinstance(tools, list):
        raise_openai_error(400, "`tools` must be an array when provided.", "invalid_request_error", "invalid_tools")
    if tools:
        raise_openai_error(
            400,
            "Responses tools, including built-in tools, are not supported by this gateway.",
            "invalid_request_error",
            "unsupported_feature",
        )


def _validate_text_format(body: dict[str, Any]) -> None:
    if "text" not in body or body["text"] is None:
        return
    text = body["text"]
    if not isinstance(text, dict):
        raise_openai_error(400, "`text` must be an object when provided.", "invalid_request_error", "invalid_text")
    unsupported_text_keys = sorted(set(text) - RESPONSE_TEXT_KEYS)
    if unsupported_text_keys:
        raise_openai_error(
            400,
            "Unsupported OpenAI responses text field(s): " + ", ".join(unsupported_text_keys),
            "invalid_request_error",
            "unsupported_feature",
        )
    text_format = text.get("format")
    if text_format is None:
        return
    if not isinstance(text_format, dict):
        raise_openai_error(
            400, "`text.format` must be an object when provided.", "invalid_request_error", "invalid_text"
        )
    unsupported_format_keys = sorted(set(text_format) - RESPONSE_TEXT_FORMAT_KEYS)
    if text_format.get("type") != "text" or unsupported_format_keys:
        raise_openai_error(
            400,
            "Only `text.format.type = 'text'` is supported; JSON schema output is not supported by this gateway.",
            "invalid_request_error",
            "unsupported_feature",
        )


def _response_input_to_codex_history_and_input(
    response_input: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    if isinstance(response_input, str):
        return [], build_text_input(response_input), ""
    if isinstance(response_input, list):
        return _response_messages_to_codex_history_and_input(response_input)
    raise_openai_error(
        400,
        "`input` must be a string or a non-empty array of message input items.",
        "invalid_request_error",
        "invalid_input",
    )


def _response_messages_to_codex_history_and_input(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    if not messages:
        raise_openai_error(
            400,
            "`input` must be a string or a non-empty array of message input items.",
            "invalid_request_error",
            "invalid_input",
        )

    instruction_parts: list[str] = []
    dialogue: list[tuple[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise_openai_error(
                400,
                f"`input[{index}]` must be an object.",
                "invalid_request_error",
                "invalid_input",
            )
        item_type = message.get("type")
        if item_type not in MESSAGE_ITEM_TYPES:
            _raise_unsupported_input_item(index, item_type)
        _validate_response_message_fields(message, index)

        role = message.get("role")
        if role not in MESSAGE_ROLES:
            raise_openai_error(
                400,
                f"Unsupported response input role at input[{index}]: {role!r}",
                "invalid_request_error",
                "unsupported_message_role",
            )
        text = _extract_response_text_content(message.get("content"), index, role)
        if role in {"system", "developer"}:
            instruction_parts.append(f"Caller-supplied {role} message:\n{text}")
        else:
            dialogue.append((role, text))

    if not dialogue:
        raise_openai_error(
            400, "At least one user input message is required.", "invalid_request_error", "invalid_input"
        )
    if dialogue[-1][0] != "user":
        raise_openai_error(
            400,
            "The final user/assistant response input message must be a user message for this compatibility MVP.",
            "invalid_request_error",
            "unsupported_message_sequence",
        )

    history_items = [_response_message_to_history_item(role, text) for role, text in dialogue[:-1]]
    input_items = build_text_input(dialogue[-1][1])
    return history_items, input_items, "\n\n".join(instruction_parts)


def _response_message_to_history_item(role: str, text: str) -> dict[str, Any]:
    content_type = "output_text" if role == "assistant" else "input_text"
    return {
        "type": "message",
        "role": role,
        "content": [{"type": content_type, "text": text}],
    }


def _extract_response_text_content(content: Any, message_index: int, role: str) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        chunks: list[str] = []
        for part_index, part in enumerate(content):
            if not isinstance(part, dict):
                _raise_unsupported_content_part(message_index, part_index)
            part_type = part.get("type")
            if _is_file_content_part(part):
                raise_openai_error(
                    400,
                    f"File inputs are not supported at input[{message_index}].content[{part_index}].",
                    "invalid_request_error",
                    "unsupported_file_input",
                )
            if part_type == "input_image":
                raise_openai_error(
                    400,
                    f"Image inputs are not supported at input[{message_index}].content[{part_index}].",
                    "invalid_request_error",
                    "unsupported_content_part",
                )
            unsupported = sorted(set(part) - RESPONSE_TEXT_CONTENT_KEYS)
            if unsupported:
                raise_openai_error(
                    400,
                    (
                        f"Unsupported OpenAI responses input[{message_index}].content[{part_index}] field(s): "
                        + ", ".join(unsupported)
                    ),
                    "invalid_request_error",
                    "unsupported_feature",
                )
            if part_type not in TEXT_CONTENT_PART_TYPES or not isinstance(part.get("text"), str):
                _raise_unsupported_content_part(message_index, part_index)
            if role != "assistant" and part_type != "input_text":
                _raise_unsupported_content_part(message_index, part_index)
            chunks.append(part["text"])
        return "\n".join(chunks)
    raise_openai_error(
        400,
        f"Only string or text content parts are supported at input[{message_index}].content.",
        "invalid_request_error",
        "unsupported_content",
    )


def _is_file_content_part(part: dict[str, Any]) -> bool:
    return part.get("type") == "input_file" or bool(FILE_CONTENT_KEYS & set(part))


def _validate_response_message_fields(message: dict[str, Any], index: int) -> None:
    unsupported = sorted(set(message) - RESPONSE_MESSAGE_KEYS)
    if unsupported:
        raise_openai_error(
            400,
            f"Unsupported OpenAI responses input[{index}] field(s): " + ", ".join(unsupported),
            "invalid_request_error",
            "unsupported_feature",
        )


def _raise_unsupported_input_item(index: int, item_type: Any) -> None:
    raise_openai_error(
        400,
        f"Only message input items are supported at input[{index}]; got {item_type!r}.",
        "invalid_request_error",
        "unsupported_response_input_item",
    )


def _raise_unsupported_content_part(message_index: int, part_index: int) -> None:
    raise_openai_error(
        400,
        f"Only input_text content parts are supported at input[{message_index}].content[{part_index}].",
        "invalid_request_error",
        "unsupported_content_part",
    )


def _responses_usage(usage: dict[str, int]) -> dict[str, Any]:
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": total_tokens,
    }

from __future__ import annotations

from codex_gateway._codex_turn_lifecycle import (
    _CodexTurnEventReducer,
    _TurnEventErrorKind,
    _TurnTerminalState,
)
from codex_gateway.codex_client import CodexAppServerError


def test_turn_event_reducer_maps_text_completion_late_usage_and_idle() -> None:
    reducer = _CodexTurnEventReducer()

    delta = reducer.reduce({"method": "item/agentMessage/delta", "params": {"delta": "hel"}})
    assert delta.assistant_delta == "hel"
    assert reducer.text == "hel"

    completed = reducer.reduce(
        {"method": "item/completed", "params": {"item": {"type": "agentMessage", "text": "hello"}}}
    )
    assert completed.assistant_delta == "lo"
    assert completed.completion_text == "hello"
    assert completed.terminal_state is _TurnTerminalState.ASSISTANT_MESSAGE_COMPLETED
    assert reducer.text == "hello"
    assert reducer.assistant_completed

    turn_completed = reducer.reduce({"method": "turn/completed", "params": {"turn": {"error": None}}})
    assert turn_completed.terminal_state is _TurnTerminalState.TURN_COMPLETED
    assert reducer.turn_completed

    late_usage = reducer.reduce(
        {
            "method": "thread/tokenUsage/updated",
            "params": {"tokenUsage": {"last": {"inputTokens": 4, "outputTokens": 5, "totalTokens": 9}}},
        }
    )
    assert late_usage.usage == {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9}
    assert reducer.usage == {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9}

    idle = reducer.reduce({"method": "thread/status/changed", "params": {"status": {"type": "idle"}}})
    assert idle.terminal_state is _TurnTerminalState.IDLE
    assert reducer.idle


def test_turn_event_reducer_distinguishes_turn_and_notification_errors() -> None:
    turn_error = _CodexTurnEventReducer().reduce(
        {
            "method": "turn/completed",
            "params": {
                "turn": {
                    "error": {
                        "message": "usage limit reached",
                        "codexErrorInfo": "usageLimitExceeded",
                        "additionalDetails": "try later",
                    }
                }
            },
        }
    )
    assert turn_error.error is not None
    assert turn_error.error.kind is _TurnEventErrorKind.TURN_FAILED
    assert "usage limit reached" in turn_error.error.message
    assert "usageLimitExceeded" in turn_error.error.message
    assert isinstance(turn_error.error.to_exception(), CodexAppServerError)

    notification_error = _CodexTurnEventReducer().reduce(
        {
            "method": "error",
            "params": {
                "error": {"message": "provider disconnected", "codexErrorInfo": "responseStreamDisconnected"},
                "willRetry": False,
            },
        }
    )
    assert notification_error.error is not None
    assert notification_error.error.kind is _TurnEventErrorKind.NOTIFICATION_ERROR
    assert "provider disconnected" in notification_error.error.message
    assert "willRetry=False" in notification_error.error.message


def test_turn_event_reducer_ignores_nonterminal_events_and_formats_simple_errors() -> None:
    reducer = _CodexTurnEventReducer()

    status = reducer.reduce({"method": "thread/status/changed", "params": {"status": {"type": "busy"}}})
    assert status.terminal_state is None

    non_agent = reducer.reduce({"method": "item/completed", "params": {"item": {"type": "commandExecution"}}})
    assert non_agent.terminal_state is None

    empty_agent = reducer.reduce({"method": "item/completed", "params": {"item": {"type": "agentMessage"}}})
    assert empty_agent.assistant_delta == ""
    assert empty_agent.terminal_state is _TurnTerminalState.ASSISTANT_MESSAGE_COMPLETED

    notification_message = _CodexTurnEventReducer().reduce(
        {"method": "error", "params": {"message": "Authorization: Bearer secret"}}
    )
    assert notification_message.error is not None
    assert "Bearer [redacted]" in notification_message.error.message

    scalar_turn_error = _CodexTurnEventReducer().reduce(
        {"method": "turn/completed", "params": {"turn": {"error": "sk-scalarsecret"}}}
    )
    assert scalar_turn_error.error is not None
    assert "sk-scalarsecret" not in scalar_turn_error.error.message


def test_turn_event_reducer_redacts_secrets_from_errors() -> None:
    turn_error = _CodexTurnEventReducer().reduce(
        {
            "method": "turn/completed",
            "params": {
                "turn": {
                    "error": {
                        "message": "upstream Authorization: Bearer abc.def",
                        "additionalDetails": "sk-testsecret access_token='tok123' refresh_token=ref456",
                    }
                }
            },
        }
    )
    assert turn_error.error is not None
    assert "Bearer [redacted]" in turn_error.error.message
    assert "[redacted]" in turn_error.error.message
    assert "sk-testsecret" not in turn_error.error.message
    assert "access_token='[redacted]" in turn_error.error.message
    assert "refresh_token=[redacted]" in turn_error.error.message
    assert "abc.def" not in turn_error.error.message
    assert "tok123" not in turn_error.error.message
    assert "ref456" not in turn_error.error.message


def test_turn_event_reducer_timeout_message_includes_last_event_and_state() -> None:
    reducer = _CodexTurnEventReducer()
    reducer.reduce({"method": "item/completed", "params": {"item": {"type": "agentMessage", "text": "done"}}})

    message = reducer.timeout_message("assistant output")

    assert "Last event: item/completed" in message
    assert "events seen: 1" in message
    assert "assistant_completed" in message

from __future__ import annotations

import pytest

import codex_gateway.safety_policy as safety_policy
from codex_gateway.safety_policy import (
    SafetyPolicyViolation,
    app_server_request_denial,
    parse_data_image_url,
    thread_start_safety_params,
    turn_start_params_with_safety,
    validate_image_url,
)


def test_thread_start_safety_params_are_locked_down() -> None:
    params = thread_start_safety_params()

    assert params["approvalPolicy"] == "never"
    assert params["approvalsReviewer"] == "user"
    assert params["sandbox"] == "read-only"
    assert params["environments"] == []
    assert params["dynamicTools"] == []
    assert "Do not execute shell commands" in params["baseInstructions"]

    params["dynamicTools"].append({"name": "unsafe"})
    assert thread_start_safety_params()["dynamicTools"] == []


def test_turn_start_safety_overrides_hostile_params_without_mutating_input() -> None:
    original = {
        "threadId": "thread-1",
        "input": [{"type": "text", "text": "hi", "text_elements": []}],
        "effort": "low",
        "approvalPolicy": "on-request",
        "approvalsReviewer": "auto",
        "cwd": "/private/tmp/unsafe",
        "permissions": {"shell": True},
        "collaborationMode": "dangerous",
        "environments": [{"id": "unsafe"}],
        "sandboxPolicy": {"type": "workspaceWrite", "networkAccess": True},
    }

    params = turn_start_params_with_safety(original)

    assert params == {
        "threadId": "thread-1",
        "input": [{"type": "text", "text": "hi", "text_elements": []}],
        "effort": "low",
        "approvalPolicy": "never",
        "approvalsReviewer": "user",
        "environments": [],
        "sandboxPolicy": {"type": "readOnly", "networkAccess": False},
    }
    assert original["approvalPolicy"] == "on-request"
    assert original["environments"] == [{"id": "unsafe"}]
    assert "cwd" not in params
    assert "permissions" not in params
    assert "collaborationMode" not in params


@pytest.mark.parametrize(
    ("method", "result"),
    [
        ("item/commandExecution/requestApproval", {"decision": "decline"}),
        ("item/fileChange/requestApproval", {"decision": "decline"}),
        ("item/permissions/requestApproval", {"permissions": {}, "scope": "turn", "strictAutoReview": True}),
        ("applyPatchApproval", {"decision": "denied"}),
        ("execCommandApproval", {"decision": "denied"}),
        ("item/tool/requestUserInput", {"answers": {}}),
        ("mcpServer/elicitation/request", {"action": "decline", "content": None, "_meta": None}),
        ("item/tool/call", {"contentItems": [], "success": False}),
    ],
)
def test_app_server_privileged_requests_are_denied_by_policy(method: str, result: dict[str, object]) -> None:
    assert app_server_request_denial({"id": 7, "method": method, "params": {}}) == {"id": 7, "result": result}


def test_app_server_auth_refresh_and_unknown_requests_are_errors() -> None:
    auth_refresh = app_server_request_denial(
        {"id": "refresh", "method": "account/chatgptAuthTokens/refresh", "params": {}}
    )
    unknown = app_server_request_denial({"id": "unknown", "method": "unknown/method", "params": {}})

    assert auth_refresh["error"]["code"] == -32000
    assert "auth token refresh is not supported" in auth_refresh["error"]["message"]
    assert unknown["error"]["code"] == -32000
    assert "unknown/method" in unknown["error"]["message"]


def test_image_url_policy_allows_http_and_valid_data_images() -> None:
    validate_image_url("http://example.com/image.png", param="image_url.url")
    validate_image_url("https://example.com/image.png", param="image_url.url")

    data_image = parse_data_image_url("data:image/png;base64,QUJD", param="image_url.url")

    assert data_image.media_type == "data:image/png;base64"
    assert data_image.encoded == "QUJD"
    assert data_image.data == b"ABC"
    assert data_image.extension == "png"


def test_image_url_policy_rejects_file_urls_and_bad_data_images(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SafetyPolicyViolation) as file_url:
        validate_image_url("file:///private/tmp/image.png", param="image_url.url")
    assert file_url.value.code == "invalid_image_url"

    with pytest.raises(SafetyPolicyViolation) as bad_shape:
        validate_image_url("data:image/png,QUJD", param="image_url.url")
    assert bad_shape.value.code == "invalid_image_url"

    with pytest.raises(SafetyPolicyViolation) as bad_base64:
        validate_image_url("data:image/png;base64,not base64", param="image_url.url")
    assert bad_base64.value.code == "invalid_image_url"

    monkeypatch.setattr(safety_policy, "MAX_DATA_IMAGE_BYTES", 1)
    with pytest.raises(SafetyPolicyViolation) as too_large:
        validate_image_url("data:image/png;base64,QUJD", param="image_url.url")
    assert too_large.value.code == "image_too_large"

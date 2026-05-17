from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from typing import Any

DATA_IMAGE_PREFIX = "data:image/"
MAX_DATA_IMAGE_BYTES = 20 * 1024 * 1024
SUPPORTED_IMAGE_URL_PREFIXES = ("http://", "https://", DATA_IMAGE_PREFIX)
TURN_START_ALLOWED_FIELDS = {"threadId", "input", "effort"}

THREAD_START_BASE_INSTRUCTIONS = (
    "You are a text-only assistant behind a local OpenAI-compatible "
    "compatibility gateway. Answer directly in plain text. Do not execute "
    "shell commands, read or write files, call tools, or request approvals."
)

IMAGE_EXTENSION_BY_MEDIA_SUBTYPE = {
    "jpeg": "jpg",
    "jpg": "jpg",
    "png": "png",
    "gif": "gif",
    "webp": "webp",
}


class SafetyPolicyViolation(ValueError):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.error_type = error_type


@dataclass(frozen=True)
class DataImage:
    media_type: str
    encoded: str
    data: bytes
    extension: str


def thread_start_safety_params() -> dict[str, Any]:
    return {
        "approvalPolicy": "never",
        "approvalsReviewer": "user",
        "sandbox": "read-only",
        "environments": [],
        "dynamicTools": [],
        "baseInstructions": THREAD_START_BASE_INSTRUCTIONS,
    }


def turn_start_params_with_safety(params: dict[str, Any] | None) -> dict[str, Any]:
    safe_params = {key: value for key, value in (params or {}).items() if key in TURN_START_ALLOWED_FIELDS}
    safe_params.update(
        {
            "approvalPolicy": "never",
            "approvalsReviewer": "user",
            "environments": [],
            "sandboxPolicy": {"type": "readOnly", "networkAccess": False},
        }
    )
    return safe_params


def validate_image_url(url: str, *, param: str) -> None:
    if not url.startswith(SUPPORTED_IMAGE_URL_PREFIXES):
        raise SafetyPolicyViolation(
            f"`{param}` must be an http(s) URL or image data URL.",
            code="invalid_image_url",
        )
    if is_data_image_url(url):
        parse_data_image_url(url, param=param)


def is_data_image_url(url: str) -> bool:
    return url.startswith(DATA_IMAGE_PREFIX)


def parse_data_image_url(url: str, *, param: str) -> DataImage:
    prefix, separator, encoded = url.partition(",")
    if separator != "," or not prefix.startswith(DATA_IMAGE_PREFIX) or not prefix.endswith(";base64") or not encoded:
        raise SafetyPolicyViolation(
            f"`{param}` must be a base64 image data URL.",
            code="invalid_image_url",
        )
    if len(encoded) * 3 // 4 > MAX_DATA_IMAGE_BYTES:
        raise SafetyPolicyViolation(
            f"`{param}` exceeds the 20MB image limit.",
            code="image_too_large",
        )
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise SafetyPolicyViolation(
            f"`{param}` contains invalid base64 image data.",
            code="invalid_image_url",
        ) from exc

    return DataImage(
        media_type=prefix,
        encoded=encoded,
        data=data,
        extension=_image_extension(prefix),
    )


def app_server_request_denial(message: dict[str, Any]) -> dict[str, Any]:
    method = message.get("method")
    request_id = message["id"]
    if method in {"item/commandExecution/requestApproval", "item/fileChange/requestApproval"}:
        return _result(request_id, {"decision": "decline"})
    if method == "item/permissions/requestApproval":
        return _result(request_id, {"permissions": {}, "scope": "turn", "strictAutoReview": True})
    if method in {"applyPatchApproval", "execCommandApproval"}:
        return _result(request_id, {"decision": "denied"})
    if method == "item/tool/requestUserInput":
        return _result(request_id, {"answers": {}})
    if method == "mcpServer/elicitation/request":
        return _result(request_id, {"action": "decline", "content": None, "_meta": None})
    if method == "item/tool/call":
        return _result(request_id, {"contentItems": [], "success": False})
    if method == "account/chatgptAuthTokens/refresh":
        return _error(
            request_id,
            "Codex auth token refresh is not supported by the local OpenAI gateway.",
        )
    return _error(request_id, f"Server request {method!r} is not supported.")


def _image_extension(media_type: str) -> str:
    subtype = media_type.removeprefix(DATA_IMAGE_PREFIX).removesuffix(";base64").lower()
    return IMAGE_EXTENSION_BY_MEDIA_SUBTYPE.get(subtype, "img")


def _result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"id": request_id, "result": result}


def _error(request_id: Any, message: str) -> dict[str, Any]:
    return {"id": request_id, "error": {"code": -32000, "message": message}}

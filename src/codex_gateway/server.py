from __future__ import annotations

import argparse
import json
import os
import secrets
import shlex
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .chat_contract import (
    ChatTurn,
    completion_payload,
    openai_error_payload,
    prepare_chat_turn,
    raise_openai_error,
    sse,
    sse_done,
    stream_delta_payload,
    stream_error_payload,
    stream_start_payload,
    stream_stop_payload,
)
from .chat_contract import (
    OpenAIContractError as OpenAIHTTPError,
)
from .codex_client import (
    CodexAppServer,
    CodexAppServerError,
    CodexChatAdmissionCancelled,
    CodexChatAdmissionTimeout,
    CodexChatOverloaded,
    CodexClientSettings,
    CodexTurnTimeout,
)
from .responses_contract import prepare_response_turn, response_payload

LOCAL_TOKEN_PREFIX = "codex-gateway-local-"
CONFIG_ENV_VAR = "CODEX_GATEWAY_CONFIG"
CONFIG_DIR_NAME = "codex-gateway"
CONFIG_FILE_NAME = "config.json"
CONFIG_FILE_MODE = 0o600


@dataclass(frozen=True)
class GatewaySettings:
    token: str
    host: str = "127.0.0.1"
    port: int = 8000
    codex_command: tuple[str, ...] = ("codex", "app-server", "--listen", "stdio://")
    cwd: Path = Path.cwd()
    request_timeout_seconds: float = 30.0
    turn_timeout_seconds: float = 180.0
    reasoning_effort: str = "low"
    chat_max_active_turns: int = 1
    chat_max_pending_turns: int | None = None
    chat_admission_timeout_seconds: float | None = None
    generated_token: bool = False


security = HTTPBearer(auto_error=False)


def create_app(settings: GatewaySettings) -> FastAPI:
    codex = CodexAppServer(
        CodexClientSettings(
            command=settings.codex_command,
            cwd=settings.cwd,
            request_timeout_seconds=settings.request_timeout_seconds,
            turn_timeout_seconds=settings.turn_timeout_seconds,
            reasoning_effort=settings.reasoning_effort,
            chat_max_active_turns=settings.chat_max_active_turns,
            chat_max_pending_turns=settings.chat_max_pending_turns,
            chat_admission_timeout_seconds=settings.chat_admission_timeout_seconds,
        )
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await codex.start()
        app.state.codex = codex
        try:
            yield
        finally:
            await codex.stop()

    app = FastAPI(
        title="Codex Gateway",
        version="0.2.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )

    @app.exception_handler(OpenAIHTTPError)
    async def openai_http_error_handler(_request: Request, exc: OpenAIHTTPError) -> JSONResponse:
        return _openai_error_response(exc.status_code, exc.message, exc.error_type, exc.code, exc.headers)

    async def require_auth(
        credentials: HTTPAuthorizationCredentials | None = Depends(security),
    ) -> None:
        supplied = credentials.credentials if credentials and credentials.scheme.lower() == "bearer" else ""
        if not supplied or not secrets.compare_digest(supplied, settings.token):
            return raise_openai_error(
                401,
                "Missing or invalid local gateway bearer token.",
                "authentication_error",
                "invalid_api_key",
                {"WWW-Authenticate": "Bearer"},
            )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models", dependencies=[Depends(require_auth)])
    async def list_models(request: Request) -> dict[str, Any]:
        try:
            models = await request.app.state.codex.list_models()
        except CodexAppServerError as exc:
            return raise_openai_error(*_codex_openai_error_args(exc))
        return {
            "object": "list",
            "data": [
                {
                    "id": model.get("id") or model.get("model"),
                    "object": "model",
                    "created": 0,
                    "owned_by": "codex",
                }
                for model in models
            ],
        }

    @app.post("/v1/chat/completions", dependencies=[Depends(require_auth)], response_model=None)
    async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return _openai_error_response(
                400, "Request body must be valid JSON.", "invalid_request_error", "invalid_json"
            )
        turn = prepare_chat_turn(body)

        if bool(body.get("stream")):
            return StreamingResponse(
                _stream_openai_chunks(request.app.state.codex, turn),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        try:
            result = await request.app.state.codex.complete_chat(
                model=turn.model,
                history_items=turn.history_items,
                input_items=turn.input_items,
                developer_instructions=turn.developer_instructions,
            )
        except CodexAppServerError as exc:
            return _codex_error_response(exc)

        created = int(time.time())
        return JSONResponse(
            completion_payload(
                completion_id=f"chatcmpl-codex-{secrets.token_urlsafe(12)}",
                created=created,
                model=turn.model,
                result=result,
            )
        )

    @app.post("/v1/responses", dependencies=[Depends(require_auth)], response_model=None)
    async def responses(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return _openai_error_response(
                400, "Request body must be valid JSON.", "invalid_request_error", "invalid_json"
            )
        turn = prepare_response_turn(body)

        try:
            result = await request.app.state.codex.complete_chat(
                model=turn.model,
                history_items=turn.history_items,
                input_items=turn.input_items,
                developer_instructions=turn.developer_instructions,
            )
        except CodexAppServerError as exc:
            return _codex_error_response(exc)

        created = int(time.time())
        return JSONResponse(
            response_payload(
                response_id=f"resp_codex_{secrets.token_urlsafe(12)}",
                message_id=f"msg_codex_{secrets.token_urlsafe(12)}",
                created_at=created,
                model=turn.model,
                result=result,
                instructions=turn.instructions,
            )
        )

    return app


async def _stream_openai_chunks(
    codex: CodexAppServer,
    turn: ChatTurn,
) -> AsyncIterator[bytes]:
    completion_id = f"chatcmpl-codex-{secrets.token_urlsafe(12)}"
    created = int(time.time())

    yield sse(stream_start_payload(completion_id=completion_id, created=created, model=turn.model))

    try:
        async for delta in codex.stream_chat(
            model=turn.model,
            history_items=turn.history_items,
            input_items=turn.input_items,
            developer_instructions=turn.developer_instructions,
        ):
            yield sse(stream_delta_payload(completion_id=completion_id, created=created, model=turn.model, delta=delta))
    except CodexAppServerError as exc:
        _status_code, message, error_type, code, _headers = _codex_openai_error_args(exc)
        yield sse(stream_error_payload(message=message, error_type=error_type, code=code))
        yield sse_done()
        return

    yield sse(stream_stop_payload(completion_id=completion_id, created=created, model=turn.model))
    yield sse_done()


def _openai_error_response(
    status_code: int,
    message: str,
    error_type: str,
    code: str | None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content=openai_error_payload(message, error_type, code),
    )


def _codex_error_response(exc: CodexAppServerError) -> JSONResponse:
    return _openai_error_response(*_codex_openai_error_args(exc))


def _codex_openai_error_args(
    exc: CodexAppServerError,
) -> tuple[int, str, str, str | None, dict[str, str] | None]:
    if isinstance(exc, CodexChatOverloaded):
        return 429, str(exc), "rate_limit_error", "codex_chat_overloaded", {"Retry-After": "1"}
    if isinstance(exc, CodexChatAdmissionTimeout):
        return 504, str(exc), "codex_timeout_error", "codex_chat_admission_timeout", None
    if isinstance(exc, CodexTurnTimeout):
        return 504, str(exc), "codex_timeout_error", "codex_turn_timeout", None
    if isinstance(exc, CodexChatAdmissionCancelled):
        return 499, str(exc), "request_cancelled", "codex_chat_admission_cancelled", None
    return 502, str(exc), "codex_app_server_error", "codex_error", None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local OpenAI-compatible gateway for Codex app-server.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["env", "init", "show", "token"],
        help=(
            "Use `init` to create local config, `show` to print SDK setup, `env` to print shell exports, "
            "or `token` to print a bearer token instead of starting the server."
        ),
    )
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--token")
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--config", type=Path, help=f"Config path. Defaults to ${CONFIG_ENV_VAR} or XDG config.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing config when used with `init`.")
    return parser


def _settings_from_namespace(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    generate_token: bool = True,
    load_config: bool = True,
) -> GatewaySettings:
    config_path = _config_path(args.config)
    try:
        config = _read_config(config_path) if load_config else {}
    except ValueError as exc:
        parser.error(str(exc))

    token = _value_from_sources(args.token, "CODEX_GATEWAY_TOKEN", config, "token")
    generated_token = token is None
    if token is None:
        if not generate_token:
            parser.error("No local gateway token configured. Run `codex-gateway init` or pass --token.")
        token = _new_local_token()
    if token.startswith("sk-"):
        parser.error("Use a local gateway bearer token, not an OpenAI API key, for --token/CODEX_GATEWAY_TOKEN.")
    host = str(_value_from_sources(args.host, "CODEX_GATEWAY_HOST", config, "host") or "127.0.0.1")
    if host != "127.0.0.1":
        parser.error("The gateway binds to 127.0.0.1 by default; pass a loopback host only for this MVP.")

    try:
        port = int(_value_from_sources(args.port, "CODEX_GATEWAY_PORT", config, "port") or 8000)
    except (TypeError, ValueError):
        parser.error("Gateway port must be an integer.")

    cwd_value = _value_from_sources(args.cwd, "CODEX_GATEWAY_CWD", config, "cwd")
    reasoning_effort = str(
        _value_from_sources(args.reasoning_effort, "CODEX_GATEWAY_REASONING_EFFORT", config, "reasoning_effort")
        or "low"
    )

    return GatewaySettings(
        token=token,
        host=host,
        port=port,
        cwd=Path(cwd_value) if cwd_value is not None else Path.cwd(),
        reasoning_effort=reasoning_effort,
        generated_token=generated_token,
    )


def _settings_from_args(argv: list[str] | None = None) -> GatewaySettings:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return _settings_from_namespace(args, parser)


def _openai_base_url(settings: GatewaySettings) -> str:
    return f"http://{settings.host}:{settings.port}/v1"


def _new_local_token() -> str:
    return f"{LOCAL_TOKEN_PREFIX}{secrets.token_urlsafe(32)}"


def _value_from_sources(
    arg_value: Any,
    env_name: str,
    config: dict[str, Any],
    config_key: str,
) -> Any:
    if arg_value is not None:
        return arg_value
    if env_name in os.environ:
        return os.environ[env_name]
    return config.get(config_key)


def _config_path(arg_path: Path | None = None) -> Path:
    if arg_path is not None:
        return arg_path.expanduser()
    if CONFIG_ENV_VAR in os.environ:
        return Path(os.environ[CONFIG_ENV_VAR]).expanduser()
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")).expanduser()
    return config_home / CONFIG_DIR_NAME / CONFIG_FILE_NAME


def _read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Config file is not valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return data


def _write_config(path: Path, settings: GatewaySettings) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    config = {
        "token": settings.token,
        "host": settings.host,
        "port": settings.port,
        "reasoning_effort": settings.reasoning_effort,
    }
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, CONFIG_FILE_MODE)
    try:
        with os.fdopen(fd, "w") as file:
            json.dump(config, file, indent=2)
            file.write("\n")
    finally:
        os.chmod(path, CONFIG_FILE_MODE)


def _shell_exports(settings: GatewaySettings) -> str:
    values = {
        "CODEX_GATEWAY_HOST": settings.host,
        "CODEX_GATEWAY_PORT": str(settings.port),
        "CODEX_GATEWAY_BASE_URL": _openai_base_url(settings),
        "CODEX_GATEWAY_TOKEN": settings.token,
    }
    return "\n".join(f"export {name}={shlex.quote(value)}" for name, value in values.items())


def _sdk_setup_snippet(settings: GatewaySettings) -> str:
    return "\n".join(
        [
            "Use this OpenAI SDK setup:",
            "from openai import OpenAI",
            f'client = OpenAI(base_url="{_openai_base_url(settings)}", api_key="{settings.token}")',
        ]
    )


def _init_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    path = _config_path(args.config)
    if path.exists() and not args.force:
        settings = _settings_from_namespace(args, parser, generate_token=False)
        print(f"Codex Gateway is already initialized at: {path}\n\n{_sdk_setup_snippet(settings)}")
        return

    settings = _settings_from_namespace(args, parser)
    if args.force and args.token is None and "CODEX_GATEWAY_TOKEN" not in os.environ:
        settings = replace(settings, token=_new_local_token(), generated_token=True)
    _write_config(path, settings)
    print(f"Wrote Codex Gateway config to: {path}\n\n{_sdk_setup_snippet(settings)}")


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "init":
        _init_config(args, parser)
        return
    settings = _settings_from_namespace(args, parser, generate_token=args.command != "show")
    if args.command == "token":
        print(settings.token)
        return
    if args.command == "env":
        print(_shell_exports(settings))
        return
    if args.command == "show":
        print(_sdk_setup_snippet(settings))
        return
    if settings.generated_token:
        print(
            f"Generated a local gateway token for this process.\n\n{_sdk_setup_snippet(settings)}",
            flush=True,
        )
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()

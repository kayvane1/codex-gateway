from __future__ import annotations

import argparse
import json
import os
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .chat_contract import (
    ChatTurn,
    OpenAIContractError as OpenAIHTTPError,
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
from .codex_client import (
    CodexAppServer,
    CodexAppServerError,
    CodexClientSettings,
)


@dataclass(frozen=True)
class ShimSettings:
    token: str
    host: str = "127.0.0.1"
    port: int = 8000
    codex_command: tuple[str, ...] = ("codex", "app-server", "--listen", "stdio://")
    cwd: Path = Path.cwd()
    request_timeout_seconds: float = 30.0
    turn_timeout_seconds: float = 180.0
    reasoning_effort: str = "low"
    generated_token: bool = False


security = HTTPBearer(auto_error=False)


def create_app(settings: ShimSettings) -> FastAPI:
    codex = CodexAppServer(
        CodexClientSettings(
            command=settings.codex_command,
            cwd=settings.cwd,
            request_timeout_seconds=settings.request_timeout_seconds,
            turn_timeout_seconds=settings.turn_timeout_seconds,
            reasoning_effort=settings.reasoning_effort,
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
        title="OpenAI Codex Shim",
        version="0.1.0",
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
                "Missing or invalid local shim bearer token.",
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
            return raise_openai_error(502, str(exc), "codex_app_server_error", "codex_error")
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
            return _openai_error_response(400, "Request body must be valid JSON.", "invalid_request_error", "invalid_json")
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
            return _openai_error_response(502, str(exc), "codex_app_server_error", "codex_error")

        created = int(time.time())
        return JSONResponse(
            completion_payload(
                completion_id=f"chatcmpl-codex-{secrets.token_urlsafe(12)}",
                created=created,
                model=turn.model,
                result=result,
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
        yield sse(stream_error_payload(message=str(exc)))
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


def _settings_from_args(argv: list[str] | None = None) -> ShimSettings:
    parser = argparse.ArgumentParser(description="Run a local OpenAI-compatible shim for Codex app-server.")
    parser.add_argument("--host", default=os.getenv("CODEX_SHIM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("CODEX_SHIM_PORT", "8000")))
    parser.add_argument("--token", default=os.getenv("CODEX_SHIM_TOKEN"))
    parser.add_argument("--cwd", type=Path, default=Path(os.getenv("CODEX_SHIM_CWD", str(Path.cwd()))))
    parser.add_argument("--reasoning-effort", default=os.getenv("CODEX_SHIM_REASONING_EFFORT", "low"))
    args = parser.parse_args(argv)

    generated_token = args.token is None
    token = args.token or secrets.token_urlsafe(32)
    if args.host != "127.0.0.1":
        parser.error("The shim binds to 127.0.0.1 by default; pass a loopback host only for this MVP.")

    return ShimSettings(
        token=token,
        host=args.host,
        port=args.port,
        cwd=args.cwd,
        reasoning_effort=args.reasoning_effort,
        generated_token=generated_token,
    )


def main(argv: list[str] | None = None) -> None:
    settings = _settings_from_args(argv)
    if settings.generated_token:
        print(
            "Generated a local shim token for this process. Use it as the OpenAI SDK api_key: "
            f"{settings.token}",
            flush=True,
        )
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()

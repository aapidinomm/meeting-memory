"""In-gateway HTTP routes for meeting-memory, mounted at ``/api/apps/meeting-memory/``.

No socket of our own is opened: the gateway serves these handlers behind its own authentication. Every route is
owner-only (meetings are private recordings): a dashboard caller must pass ``ctx.require_owner``; an app token is
accepted only when it is this app's own token, which the gateway's token middleware confines to
``permissions.api``. A host without ``require_owner`` is refused (fail closed).
"""

from __future__ import annotations

import asyncio
import functools
from dataclasses import dataclass
from typing import Any, Callable

from . import core


@dataclass(frozen=True)
class Route:
    method: str
    path: str
    handler: Callable


def _json(data: Any, status: int = 200):
    from aiohttp import web  # provided by the gateway process

    return web.json_response(data, status=status)


async def _authorize(request, ctx, operation: str):
    try:
        app_claim = request.get("app")
    except Exception:  # pragma: no cover - defensive
        app_claim = None
    if app_claim:
        name = app_claim if isinstance(app_claim, str) else getattr(app_claim, "name", None)
        if name == core.APP_NAME:
            return None
        return _json({"error": "forbidden", "code": "wrong_app"}, status=403)
    gate = getattr(ctx, "require_owner", None)
    if gate is None:
        return _json({"error": "owner authorization required", "code": "owner_only"}, status=403)
    return await gate(request, operation)


def _guard(operation: str):
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(request, ctx):
            denied = await _authorize(request, ctx, operation)
            if denied is not None:
                return denied
            try:
                return await fn(request, ctx)
            except core.AppError as exc:
                return _json(exc.as_dict(), status=exc.status)

        return wrapper

    return deco


def _run(fn, *args, **kwargs):
    return asyncio.get_running_loop().run_in_executor(None, functools.partial(fn, *args, **kwargs))


async def _body(request) -> dict:
    if not request.can_read_body:
        return {}
    try:
        data = await request.json()
    except Exception:
        raise core.AppError("bad_json", "request body must be JSON")
    if not isinstance(data, dict):
        raise core.AppError("bad_json", "request body must be a JSON object")
    return data


def _int(request, name, default=None):
    raw = request.query.get(name)
    if raw is None:
        if default is None:
            raise core.AppError("bad_param", "missing query parameter %s" % name)
        return default
    try:
        return int(raw)
    except ValueError:
        raise core.AppError("bad_param", "%s must be an integer" % name)


def _cfg(ctx):
    return core.load_settings(ctx.data_dir)


def _mid(request) -> str:
    return request.match_info.get("meeting_id", "")


# ---- handlers -------------------------------------------------------------


@_guard("status")
async def get_status(request, ctx):
    return _json(await _run(core.status, ctx.data_dir, _cfg(ctx)))


@_guard("settings_read")
async def get_settings(request, ctx):
    return _json(_cfg(ctx))


@_guard("settings_write")
async def put_settings(request, ctx):
    return _json(await _run(core.save_settings, ctx.data_dir, await _body(request)))


@_guard("settings_read")
async def get_stt_models(request, ctx):
    return _json(await _run(core.stt_options, ctx.data_dir))


@_guard("settings_test")
async def post_settings_test(request, ctx):
    return _json(await _run(core.check_connection, _cfg(ctx)))


@_guard("meetings_list")
async def get_meetings(request, ctx):
    return _json({"meetings": await _run(core.list_meetings, ctx.data_dir, _int(request, "limit", 200))})


@_guard("meeting_create")
async def post_meeting(request, ctx):
    settings = _cfg(ctx)
    if not core.configured(settings):
        raise core.AppError("not_configured", "Set the AWS profile and region in Settings first", 409)
    body = await _body(request)
    return _json(await _run(core.create_meeting, ctx.data_dir, str(body.get("title") or "")), status=201)


@_guard("meeting_read")
async def get_meeting(request, ctx):
    return _json(await _run(core.get_meeting, ctx.data_dir, _mid(request), request.query.get("transcript") != "0"))


@_guard("meeting_write")
async def put_meeting(request, ctx):
    return _json(await _run(core.update_meeting, ctx.data_dir, _mid(request), await _body(request)))


@_guard("chunk_write")
async def post_chunk(request, ctx):
    if request.content_length is not None and request.content_length > core.MAX_CHUNK_BYTES:
        raise core.AppError("too_large", "audio chunk is too large", 413)
    data = await request.read()
    return _json(await _run(core.add_chunk, ctx.data_dir, _cfg(ctx), _mid(request), _int(request, "seq"),
                            _int(request, "start_ms"), _int(request, "duration_ms"), data), status=201)


@_guard("meeting_stop")
async def post_stop(request, ctx):
    return _json(await _run(core.stop_meeting, ctx.data_dir, _cfg(ctx), _mid(request)))


@_guard("meeting_summarize")
async def post_summarize(request, ctx):
    return _json(await _run(core.stop_meeting, ctx.data_dir, _cfg(ctx), _mid(request)))


@_guard("meeting_delete")
async def post_delete(request, ctx):
    return _json(await _run(core.delete_meeting, ctx.data_dir, _mid(request)))


@_guard("search")
async def post_search(request, ctx):
    body = await _body(request)
    return _json(await _run(core.search, ctx.data_dir, _cfg(ctx), str(body.get("query") or ""), body.get("k") or 8))


@_guard("ask")
async def post_ask(request, ctx):
    body = await _body(request)
    return _json(await _run(core.ask, ctx.data_dir, _cfg(ctx), str(body.get("question") or ""), body.get("k") or 8))


def register_routes(ctx):
    return [
        Route("GET", "/status", get_status),
        Route("GET", "/settings", get_settings),
        Route("PUT", "/settings", put_settings),
        Route("POST", "/settings/test", post_settings_test),
        Route("GET", "/stt-models", get_stt_models),
        Route("GET", "/meetings", get_meetings),
        Route("POST", "/meetings", post_meeting),
        Route("GET", "/meetings/{meeting_id}", get_meeting),
        Route("PUT", "/meetings/{meeting_id}", put_meeting),
        Route("POST", "/meetings/{meeting_id}/chunks", post_chunk),
        Route("POST", "/meetings/{meeting_id}/stop", post_stop),
        Route("POST", "/meetings/{meeting_id}/summarize", post_summarize),
        Route("POST", "/meetings/{meeting_id}/delete", post_delete),
        Route("POST", "/search", post_search),
        Route("POST", "/ask", post_ask),
    ]

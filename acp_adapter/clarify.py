"""Ask Hermes clarify questions through ACP permission cards, one at a time."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from itertools import count
from typing import Callable

import acp
from acp.schema import PermissionOption

from acp_adapter.permissions import await_permission

_REQUEST_IDS = count(1)
_CHAT_OPTION = "answer_in_chat"
_HOLD = "No answer (hold). Ask again later or put it on the Desk."
_CHAT_NOTICE = "User will answer in chat"


def resolve_clarify_timeout() -> float | None:
    """ACP defaults to 15 minutes; an explicit clarify.timeout keeps its usual <=0 semantics."""
    from hermes_cli.config import load_config

    raw = ((load_config() or {}).get("clarify") or {}).get("timeout", 900)
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        seconds = 900
    return seconds if seconds > 0 else None


def _card(question: dict):
    text = question["question"]
    if header := question.get("header"):
        text = f"{header}\n{text}"
    choices = question.get("choices") or []
    options = [PermissionOption(option_id=str(index), name=label[:40], kind="allow_once")
               for index, label in enumerate(choices)]
    options.append(PermissionOption(option_id=_CHAT_OPTION, name="Answer in chat", kind="reject_once"))
    # Full labels remain readable even when the compact button label is truncated.
    content = text + ("\n\n" + "\n".join(f"{i + 1}. {label}" for i, label in enumerate(choices)) if choices else "")
    return acp.update_tool_call(
        f"clarify-{next(_REQUEST_IDS)}", title=text, kind="other", status="pending",
        content=[acp.tool_content(acp.text_block(content))],
        raw_input={"tool": "clarify", "question": question["question"], "choices": choices},
    ), options


async def _request_until_cancelled(request_permission_fn: Callable, cancel_event: threading.Event | None, **kwargs):
    """ACP session/cancel must release a waiting worker even if the client never replies."""
    if cancel_event is None:
        return await request_permission_fn(**kwargs)
    task = asyncio.create_task(request_permission_fn(**kwargs))
    try:
        while not cancel_event.is_set():
            done, _ = await asyncio.wait({task}, timeout=0.1)
            if done:
                return task.result()
        return None
    finally:
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def make_clarify_callback(
    request_permission_fn: Callable, loop: asyncio.AbstractEventLoop, session_id: str, *,
    cancel_event: threading.Event | None = None, send_update: Callable[[object], None] | None = None,
) -> Callable:
    """Return the session's blocking clarify callback; only explicit choices become answers."""
    async def request(**kwargs):
        return await _request_until_cancelled(request_permission_fn, cancel_event, **kwargs)

    def callback(questions: list[dict]) -> dict:
        answers = {}
        chat_requested = False
        timeout = resolve_clarify_timeout()
        for question in questions:
            if cancel_event is not None and cancel_event.is_set():
                return {"answers": answers, "outcome": "cancelled", "notice": _HOLD}
            tool_call, options = _card(question)
            response, timed_out = await_permission(
                request, loop, session_id, tool_call=tool_call, options=options, timeout=timeout,
                what="Clarify request",
            )
            outcome = getattr(response, "outcome", None)
            selected = getattr(outcome, "option_id", None)
            valid = (getattr(outcome, "outcome", None) == "selected"
                     and any(option.option_id == selected for option in options))
            if send_update is not None:
                send_update(acp.update_tool_call(tool_call.tool_call_id, status="completed" if valid else "failed"))
            if not valid:
                return {"answers": answers, "outcome": "timed_out" if timed_out else "cancelled", "notice": _HOLD}
            if selected == _CHAT_OPTION:
                chat_requested = True
            else:
                answers[question["qid"]] = question["choices"][int(selected)]
        result = {"answers": answers, "outcome": "submitted"}
        if chat_requested:
            result["notice"] = _CHAT_NOTICE
        return result

    return callback

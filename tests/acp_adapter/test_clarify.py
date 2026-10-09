"""ACP clarify must wait for explicit input, scoped to the requesting session."""

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
from acp.schema import AllowedOutcome, DeniedOutcome, RequestPermissionResponse

from acp_adapter.server import HermesACPAgent
from tools.clarify_tool import clarify_tool


class Connection:
    def __init__(self, selection="0"):
        self.selection = selection
        self.requests = []
        self.updates = []

    async def request_permission(self, **kwargs):
        self.requests.append(kwargs)
        outcome = (DeniedOutcome(outcome="cancelled") if self.selection is None else
                   AllowedOutcome(outcome="selected", option_id=self.selection))
        return RequestPermissionResponse(outcome=outcome)

    async def session_update(self, session_id, update):
        self.updates.append((session_id, update))


def wire(conn, session_id):
    state = SimpleNamespace(agent=SimpleNamespace(clarify_callback=None), message_ids=None,
                            cancel_event=threading.Event())
    HermesACPAgent()._wire_turn_callbacks(state, session_id, conn, asyncio.get_running_loop())
    return state


def test_clarify_is_available_to_acp():
    from toolsets import resolve_toolset

    assert "clarify" in resolve_toolset("hermes-acp")


@pytest.mark.asyncio
async def test_clarify_sessions_route_choices_and_chat_answers_independently():
    first, second = Connection("1"), Connection("answer_in_chat")
    a, b = wire(first, "session-a"), wire(second, "session-b")
    assert callable(a.agent.clarify_callback)
    assert callable(b.agent.clarify_callback)
    long_choice = "A full choice label " * 5
    questions = [{"question": "Runtime\nWhich runtime?", "choices": ["Python", long_choice]},
                 {"question": "Add a note?"}]
    first.selection = "1"

    async def select(**kwargs):
        first.selection = "1" if not first.requests else "answer_in_chat"
        return await Connection.request_permission(first, **kwargs)

    first.request_permission = select
    # Rewire after replacing the connection's method.
    a = wire(first, "session-a")
    result_a, result_b = await asyncio.gather(
        asyncio.to_thread(clarify_tool, questions, a.agent.clarify_callback),
        asyncio.to_thread(clarify_tool, questions[:1], b.agent.clarify_callback),
    )
    result_a, result_b = json.loads(result_a), json.loads(result_b)
    assert result_a["responses"][0]["user_response"] == long_choice.strip()
    assert result_a["notice"] == result_b["notice"] == "User will answer in chat"
    assert result_a["responses"][1]["user_response"] is None
    assert result_b["responses"][0]["user_response"] is None
    request = first.requests[0]
    assert request["tool_call"].title == "Runtime\nWhich runtime?"
    assert long_choice.strip() in request["tool_call"].content[0].content.text
    assert [(o.option_id, o.kind) for o in request["options"]] == [
        ("0", "allow_once"), ("1", "allow_once"), ("answer_in_chat", "reject_once")]
    assert all(len(o.name) <= 40 for o in request["options"])
    assert [o.name for o in first.requests[1]["options"]] == ["Answer in chat"]
    assert all(r["session_id"] == "session-a" for r in first.requests)
    assert all(r["session_id"] == "session-b" for r in second.requests)
    assert len({r["tool_call"].tool_call_id for r in first.requests + second.requests}) == 3
    assert all(update.status == "completed" for _, update in first.updates + second.updates)


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", [None, "unknown", "-1"])
async def test_clarify_cancel_or_invalid_selection_holds_without_asking_next_question(selection):
    conn = Connection(selection)
    state = wire(conn, "session")
    assert callable(state.agent.clarify_callback)
    result = json.loads(await asyncio.to_thread(clarify_tool, [
        {"question": "First?", "choices": ["One", "Two"]}, {"question": "Second?"},
    ], state.agent.clarify_callback))
    assert result["outcome"] == "cancelled"
    assert result["notice"] == "No answer (hold). Ask again later or put it on the Desk."
    assert all(r["status"] == "unanswered" and r["user_response"] is None for r in result["responses"])
    assert len(conn.requests) == 1
    assert conn.updates[-1][1].status == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["timeout", "cancel", "disconnect"])
async def test_clarify_wait_releases_pending_request_on_timeout_cancel_or_disconnect(stop):
    from acp_adapter.clarify import make_clarify_callback
    from hermes_constants import get_hermes_home

    (get_hermes_home() / "config.yaml").write_text("clarify:\n  timeout: 0.2\n")
    started, released = asyncio.Event(), asyncio.Event()
    cancel_event = threading.Event()
    updates = []

    async def request_permission(**kwargs):
        started.set()
        try:
            if stop == "disconnect":
                raise ConnectionError("client disconnected")
            await asyncio.Future()
        finally:
            released.set()

    callback = make_clarify_callback(request_permission, asyncio.get_running_loop(), "session",
                                    cancel_event=cancel_event, send_update=updates.append)
    task = asyncio.create_task(asyncio.to_thread(clarify_tool, [{"question": "Wait?"}], callback))
    await asyncio.wait_for(started.wait(), 5)
    if stop == "cancel":
        cancel_event.set()
    result = json.loads(await asyncio.wait_for(task, 5))
    await asyncio.wait_for(released.wait(), 5)
    assert result["outcome"] == ("timed_out" if stop == "timeout" else "cancelled")
    assert result["notice"] == "No answer (hold). Ask again later or put it on the Desk."
    assert result["responses"][0]["user_response"] is None
    assert updates[-1].status == "failed"


@pytest.mark.parametrize("config, expected", [({}, 900), ({"agent": {"clarify_timeout": 3600}}, 900),
                                           ({"clarify": {"timeout": 120}}, 120),
                                           ({"clarify": {"timeout": 0}}, None)])
def test_clarify_timeout_uses_acp_default_or_explicit_config(config, expected, monkeypatch):
    from acp_adapter.clarify import resolve_clarify_timeout

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    assert resolve_clarify_timeout() == expected

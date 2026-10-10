"""Busy ACP prompts and cancel+prompt must reach the model exactly once.

Use a real AIAgent, SessionDB and loopback provider so stale interrupt flags exercise
durable lease admission, including a cancel arriving after the core has finalized.
"""

import asyncio
import os
import threading

import pytest
from acp.schema import ImageContentBlock, TextContentBlock

from tests.acp_adapter import test_failed_turn_closure

acp = test_failed_turn_closure.acp
_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j1ZkAAAAASUVORK5CYII="


class _GatedReplies(list):
    def __init__(self):
        super().__init__([
            {"finish_reason": "stop", "content": "first reply"},
            {"finish_reason": "stop", "content": "follow-up reply"},
        ])
        self.started = threading.Event()
        self.release = threading.Event()

    def pop(self, index=0):
        reply = super().pop(index)
        if reply["content"] == "first reply":
            self.started.set()
            assert self.release.wait(20), "test did not release the first model request"
        return reply


def _text(text):
    return [TextContentBlock(type="text", text=text)]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["text", "image", "steer", "late_steer"])
async def test_busy_prompt_runs_after_current_turn_with_complete_content(acp, kind, monkeypatch):
    provider, _, rows, _, sid, conn, server = acp
    from hermes_cli import config

    cfg = config.load_config()
    cfg["model"]["supports_vision"] = True
    monkeypatch.setattr(config, "load_config", lambda *a, **k: cfg)
    if kind == "late_steer":
        agent = server.session_manager.get_session(sid).agent
        run = agent.run_conversation
        loop = asyncio.get_running_loop()

        def steer_after_core_finalizer(**kwargs):
            result = run(**kwargs)
            if kwargs["user_message"] == "first request":
                asyncio.run_coroutine_threadsafe(server.prompt(
                    prompt=_text("/steer follow-up request"), session_id=sid,
                ), loop).result(timeout=10)
            return result

        monkeypatch.setattr(agent, "run_conversation", steer_after_core_finalizer)
    provider.requests.clear()  # Session construction probes model metadata before prompting.
    script = provider.script = _GatedReplies()
    first = asyncio.create_task(server.prompt(prompt=_text("first request"), session_id=sid))
    follow_up = _text("/steer follow-up request" if kind == "steer" else "follow-up request")
    if kind == "image":
        follow_up.append(ImageContentBlock(type="image", data=_PNG, mimeType="image/png"))
    try:
        assert await asyncio.to_thread(script.started.wait, 10)
        if kind != "late_steer":
            await server.prompt(prompt=follow_up, session_id=sid)
        # Plain ACP prompts are queued; only explicit /steer injects into a turn.
        assert len(provider.requests) == 1
    finally:
        script.release.set()
        await asyncio.wait_for(first, 20)

    assert len(provider.requests) == 2
    users = [m["content"] for m in provider.requests[-1]["messages"] if m["role"] == "user"]
    assert users[0] == "first request"
    if kind == "image":
        assert users[-1] == [
            {"type": "text", "text": "follow-up request"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _PNG}},
        ]
    else:
        assert users[-1] == "follow-up request"
    assert [role for role, _ in rows()] == ["user", "assistant", "user", "assistant"]
    assert "not processed" not in str(conn.updates)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_wait", [False, True])
async def test_busy_durable_lease_retains_input_until_admission_or_cancel_follow_up(acp, monkeypatch, cancel_wait):
    provider, _, rows, db, sid, conn, server = acp
    provider.requests.clear()
    provider.script = [{"finish_reason": "stop", "content": "admitted reply"}]
    holder = f"pid={os.getpid()}:other-acp-turn"
    assert db.try_acquire_session_turn_lease(sid, holder)
    expired = threading.Event()
    state = server.session_manager.get_session(sid)
    monkeypatch.setattr(state.agent, "_emit_warning", lambda _text: expired.set())
    monkeypatch.setattr("agent.turn_facade_lease.LEASE_WAIT_SECONDS", 0)
    pending = asyncio.create_task(server.prompt(prompt=_text("waiting request"), session_id=sid))
    try:
        assert await asyncio.to_thread(expired.wait, 10)
        if cancel_wait:
            await server.cancel(sid)
            assert (await asyncio.wait_for(pending, 20)).stop_reason == "cancelled"
    finally:
        db.release_session_turn_lease(sid, holder)
        response = await asyncio.wait_for(pending, 20)
    if cancel_wait:
        response = await server.prompt(prompt=_text("follow-up request"), session_id=sid)
    assert response.stop_reason == "end_turn"
    assert len(provider.requests) == 1
    assert "waiting request" in str(provider.requests[0]["messages"][-1])
    assert rows()[-1] == ("assistant", "admitted reply")
    assert "not processed" not in str(conn.updates)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_at", ["model", "overlap", "delivery", "idle"])
async def test_cancel_stops_only_current_turn_and_follow_up_reaches_model(acp, cancel_at):
    provider, _, rows, _, sid, conn, server = acp
    provider.requests.clear()  # Session construction probes model metadata before prompting.
    script = provider.script = _GatedReplies()
    if cancel_at == "delivery":
        send = conn.session_update

        async def cancel_during_delivery(session_id, update):
            await send(session_id, update)
            # A cancel can arrive with the last usage update, after core cleanup.
            if update.session_update == "usage_update":
                await server.cancel(session_id)
                conn.session_update = send

        conn.session_update = cancel_during_delivery
    first = asyncio.create_task(server.prompt(prompt=_text("first request"), session_id=sid))
    second = None
    try:
        assert await asyncio.to_thread(script.started.wait, 10)
        if cancel_at in ("model", "overlap"):
            await server.cancel(sid)
        if cancel_at == "overlap":
            second = asyncio.create_task(server.prompt(prompt=_text("follow-up request"), session_id=sid))
        script.release.set()
        first_result = await asyncio.wait_for(first, 20)
        if cancel_at == "idle":
            await server.cancel(sid)
        if second is None:
            second = asyncio.create_task(server.prompt(prompt=_text("follow-up request"), session_id=sid))
        second_result = await asyncio.wait_for(second, 20)
    finally:
        script.release.set()
        await asyncio.wait_for(first, 20)
        if second is not None:
            await asyncio.wait_for(second, 20)

    if cancel_at in ("model", "overlap"):
        assert first_result.stop_reason == "cancelled"
    assert second_result.stop_reason == "end_turn"
    assert len(provider.requests) == 2
    users = [m["content"] for m in provider.requests[-1]["messages"] if m["role"] == "user"]
    assert "follow-up request" in str(users[-1])
    assert rows()[-1] == ("assistant", "follow-up reply")
    assert "not processed" not in str(conn.updates)

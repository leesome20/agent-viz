"""Unit tests for agent_viz.tracer – Tracer event emission and queue behaviour."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from agent_viz.event_model import (
    ErrorEvent,
    LLMResponseEvent,
    NodeEndEvent,
    NodeStartEvent,
    ToolCallEvent,
)
from agent_viz.tracer import Tracer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tracer(**kwargs) -> Tracer:
    return Tracer(session_name="unit-test", **kwargs)


def _drain_queue(queue: asyncio.Queue) -> list[Any]:
    """Drain all currently available items from the queue without blocking."""
    items = []
    while not queue.empty():
        try:
            items.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return items


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestTracerConstruction:
    def test_session_id_auto_generated(self) -> None:
        t = _make_tracer()
        assert len(t.session_id) == 36

    def test_explicit_session_id(self) -> None:
        t = Tracer(session_id="my-session")
        assert t.session_id == "my-session"

    def test_not_closed_initially(self) -> None:
        t = _make_tracer()
        assert not t.is_closed

    def test_repr(self) -> None:
        t = _make_tracer()
        r = repr(t)
        assert t.session_id in r
        assert "Tracer" in r


# ---------------------------------------------------------------------------
# Queue management
# ---------------------------------------------------------------------------

class TestQueueManagement:
    def test_get_queue_returns_queue(self) -> None:
        t = _make_tracer()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            q = t.get_queue()
            assert isinstance(q, asyncio.Queue)
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_get_queue_returns_same_instance(self) -> None:
        t = _make_tracer()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            q1 = t.get_queue()
            q2 = t.get_queue()
            assert q1 is q2
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_set_queue(self) -> None:
        t = _make_tracer()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            custom_q: asyncio.Queue = asyncio.Queue()
            t.set_queue(custom_q)
            assert t.get_queue() is custom_q
        finally:
            loop.close()
            asyncio.set_event_loop(None)


# ---------------------------------------------------------------------------
# Synchronous event emission
# ---------------------------------------------------------------------------

class TestSyncEmission:
    def test_node_start_returns_node_id(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="Root")
        assert isinstance(nid, str)
        assert len(nid) == 36

    def test_node_start_explicit_id(self) -> None:
        t = _make_tracer()
        nid = t.node_start("my-node", label="Root")
        assert nid == "my-node"

    def test_node_start_persisted_to_session(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="Root")
        assert nid in t.session.node_ids

    def test_node_start_event_type(self) -> None:
        t = _make_tracer()
        t.node_start("n1", label="R")
        events = t.session.events
        assert events[0].event_type == "node_start"
        assert isinstance(events[0], NodeStartEvent)

    def test_node_end_persisted(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="R")
        t.node_end(nid)
        events = t.session.events
        assert events[-1].event_type == "node_end"
        assert isinstance(events[-1], NodeEndEvent)

    def test_node_end_status_error(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="R")
        t.node_end(nid, status="error")
        ev = t.session.events[-1]
        assert isinstance(ev, NodeEndEvent)
        assert ev.status == "error"

    def test_tool_call_persisted(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="R")
        t.tool_call(nid, tool_name="search", inputs={"q": "test"}, outputs={"r": 1})
        ev = t.session.events[-1]
        assert isinstance(ev, ToolCallEvent)
        assert ev.tool_name == "search"
        assert ev.inputs == {"q": "test"}
        assert ev.outputs == {"r": 1}

    def test_llm_response_persisted(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="R")
        t.llm_response(
            nid,
            model="gpt-4o",
            response="Hello!",
            prompt="Say hi",
            prompt_tokens=5,
            completion_tokens=3,
        )
        ev = t.session.events[-1]
        assert isinstance(ev, LLMResponseEvent)
        assert ev.model == "gpt-4o"
        assert ev.response == "Hello!"
        assert ev.prompt_tokens == 5

    def test_error_persisted(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="R")
        t.error(nid, error_type="ValueError", message="bad input")
        ev = t.session.events[-1]
        assert isinstance(ev, ErrorEvent)
        assert ev.error_type == "ValueError"

    def test_error_from_exception(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="R")
        exc = RuntimeError("something broke")
        t.error_from_exception(nid, exc)
        ev = t.session.events[-1]
        assert isinstance(ev, ErrorEvent)
        assert ev.error_type == "RuntimeError"
        assert "something broke" in ev.message

    def test_metadata_attached(self) -> None:
        t = _make_tracer()
        nid = t.node_start("n1", label="R", metadata={"env": "test"})
        ev = t.session.events[0]
        assert ev.metadata == {"env": "test"}

    def test_parent_node_id_sets_edge(self) -> None:
        t = _make_tracer()
        root = t.node_start("root", label="Root")
        t.node_start("child", label="Child", parent_node_id=root)
        assert len(t.session.edges) == 1
        assert t.session.edges[0].source_node_id == "root"
        assert t.session.edges[0].target_node_id == "child"


# ---------------------------------------------------------------------------
# Queue population (sync path)
# ---------------------------------------------------------------------------

class TestSyncQueuePopulation:
    def test_events_enqueued(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            t = _make_tracer()
            nid = t.node_start(label="Root")
            t.node_end(nid)
            items = _drain_queue(t.get_queue())
            assert len(items) == 2
            assert items[0].event_type == "node_start"
            assert items[1].event_type == "node_end"
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_queue_full_does_not_raise(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            t = Tracer(session_name="test", queue_maxsize=1)
            t.node_start(label="Root")       # fills queue
            t.node_start(label="Overflow")   # would overflow – should not raise
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_session_still_updated_when_no_loop(self) -> None:
        # Force no event loop
        asyncio.set_event_loop(None)
        t = _make_tracer()
        nid = t.node_start(label="Root")
        t.node_end(nid)
        # Events should still be in session even without a loop
        assert len(t.session) == 2


# ---------------------------------------------------------------------------
# Async event emission
# ---------------------------------------------------------------------------

class TestAsyncEmission:
    @pytest.mark.asyncio
    async def test_anode_start_returns_node_id(self) -> None:
        t = _make_tracer()
        nid = await t.anode_start(label="Root")
        assert isinstance(nid, str)
        assert len(nid) == 36

    @pytest.mark.asyncio
    async def test_anode_start_explicit_id(self) -> None:
        t = _make_tracer()
        nid = await t.anode_start("async-node", label="Root")
        assert nid == "async-node"

    @pytest.mark.asyncio
    async def test_anode_end_persisted(self) -> None:
        t = _make_tracer()
        nid = await t.anode_start(label="R")
        await t.anode_end(nid)
        assert len(t.session) == 2
        assert t.session.events[-1].event_type == "node_end"

    @pytest.mark.asyncio
    async def test_atool_call_persisted(self) -> None:
        t = _make_tracer()
        nid = await t.anode_start(label="R")
        await t.atool_call(nid, tool_name="calc", inputs={"x": 1})
        ev = t.session.events[-1]
        assert isinstance(ev, ToolCallEvent)
        assert ev.tool_name == "calc"

    @pytest.mark.asyncio
    async def test_allm_response_persisted(self) -> None:
        t = _make_tracer()
        nid = await t.anode_start(label="R")
        await t.allm_response(nid, model="claude-3", response="Ok")
        ev = t.session.events[-1]
        assert isinstance(ev, LLMResponseEvent)
        assert ev.model == "claude-3"

    @pytest.mark.asyncio
    async def test_aerror_persisted(self) -> None:
        t = _make_tracer()
        nid = await t.anode_start(label="R")
        await t.aerror(nid, error_type="IOError", message="disk full")
        ev = t.session.events[-1]
        assert isinstance(ev, ErrorEvent)
        assert ev.error_type == "IOError"

    @pytest.mark.asyncio
    async def test_async_events_enqueued(self) -> None:
        t = _make_tracer()
        nid = await t.anode_start(label="Root")
        await t.anode_end(nid)
        items = _drain_queue(t.get_queue())
        assert len(items) == 2


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------

class TestSpanContextManager:
    def test_span_emits_start_and_end(self) -> None:
        t = _make_tracer()
        with t.span(label="Step") as nid:
            assert isinstance(nid, str)
        events = t.session.events
        assert events[0].event_type == "node_start"
        assert events[-1].event_type == "node_end"
        assert isinstance(events[-1], NodeEndEvent)
        assert events[-1].status == "success"

    def test_span_emits_error_on_exception(self) -> None:
        t = _make_tracer()
        with pytest.raises(ValueError):
            with t.span(label="Step"):
                raise ValueError("oops")
        event_types = [e.event_type for e in t.session.events]
        assert "node_start" in event_types
        assert "error" in event_types
        assert "node_end" in event_types
        end_events = [e for e in t.session.events if e.event_type == "node_end"]
        assert end_events[-1].status == "error"

    def test_span_with_parent(self) -> None:
        t = _make_tracer()
        root = t.node_start("root", label="Root")
        with t.span("child", label="Child", parent_node_id=root):
            pass
        assert len(t.session.edges) == 1

    def test_span_auto_end_false(self) -> None:
        t = _make_tracer()
        with t.span(label="Step", auto_end=False) as nid:
            pass
        event_types = [e.event_type for e in t.session.events]
        assert "node_start" in event_types
        assert "node_end" not in event_types
        # Manually end it
        t.node_end(nid)
        event_types = [e.event_type for e in t.session.events]
        assert "node_end" in event_types

    @pytest.mark.asyncio
    async def test_aspan_emits_start_and_end(self) -> None:
        t = _make_tracer()
        async with t.aspan(label="AsyncStep") as nid:
            assert isinstance(nid, str)
        events = t.session.events
        assert events[0].event_type == "node_start"
        assert events[-1].event_type == "node_end"
        assert events[-1].status == "success"

    @pytest.mark.asyncio
    async def test_aspan_emits_error_on_exception(self) -> None:
        t = _make_tracer()
        with pytest.raises(RuntimeError):
            async with t.aspan(label="AsyncStep"):
                raise RuntimeError("async oops")
        event_types = [e.event_type for e in t.session.events]
        assert "error" in event_types
        end_events = [e for e in t.session.events if e.event_type == "node_end"]
        assert end_events[-1].status == "error"


# ---------------------------------------------------------------------------
# Lifecycle: close
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_close_sets_is_closed(self) -> None:
        t = _make_tracer()
        t.node_start(label="R")
        t.close()
        assert t.is_closed

    def test_close_returns_json(self) -> None:
        import json
        t = _make_tracer()
        t.node_start(label="R")
        json_str = t.close()
        parsed = json.loads(json_str)
        assert parsed["session_id"] == t.session_id
        assert len(parsed["events"]) == 1

    def test_close_idempotent(self) -> None:
        t = _make_tracer()
        t.close()
        ended_at = t.session.ended_at
        t.close()
        assert t.session.ended_at == ended_at

    def test_session_store_accessible_after_close(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="R")
        t.node_end(nid)
        t.close()
        assert len(t.session) == 2


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestTracerThreadSafety:
    def test_concurrent_node_starts(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            t = _make_tracer()
            errors: list[Exception] = []

            def _work(i: int) -> None:
                try:
                    t.node_start(f"node-{i}", label=f"Node {i}")
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [threading.Thread(target=_work, args=(i,)) for i in range(30)]
            for thr in threads:
                thr.start()
            for thr in threads:
                thr.join()

            assert errors == []
            assert len(t.session) == 30
        finally:
            loop.close()
            asyncio.set_event_loop(None)

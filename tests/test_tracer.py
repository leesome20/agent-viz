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

    def test_session_name_stored(self) -> None:
        t = Tracer(session_name="my run")
        assert t.session.session_name == "my run"

    def test_repr(self) -> None:
        t = _make_tracer()
        r = repr(t)
        assert t.session_id in r
        assert "Tracer" in r

    def test_session_store_accessible(self) -> None:
        t = _make_tracer()
        from agent_viz.session import SessionStore
        assert isinstance(t.session, SessionStore)


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

    def test_queue_maxsize_respected(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            t = Tracer(session_name="test", queue_maxsize=5)
            q = t.get_queue()
            assert q.maxsize == 5
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_unbounded_queue_by_default(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            t = _make_tracer()
            q = t.get_queue()
            assert q.maxsize == 0
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

    def test_node_start_label_stored(self) -> None:
        t = _make_tracer()
        t.node_start("n1", label="My Label")
        ev = t.session.events[0]
        assert isinstance(ev, NodeStartEvent)
        assert ev.label == "My Label"

    def test_node_start_node_type_stored(self) -> None:
        t = _make_tracer()
        t.node_start("n1", label="R", node_type="llm")
        ev = t.session.events[0]
        assert isinstance(ev, NodeStartEvent)
        assert ev.node_type == "llm"

    def test_node_start_default_node_type(self) -> None:
        t = _make_tracer()
        t.node_start("n1", label="R")
        ev = t.session.events[0]
        assert isinstance(ev, NodeStartEvent)
        assert ev.node_type == "default"

    def test_node_start_session_id_correct(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="R")
        ev = t.session.events[0]
        assert ev.session_id == t.session_id

    def test_node_end_persisted(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="R")
        t.node_end(nid)
        events = t.session.events
        assert events[-1].event_type == "node_end"
        assert isinstance(events[-1], NodeEndEvent)

    def test_node_end_default_status_success(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="R")
        t.node_end(nid)
        ev = t.session.events[-1]
        assert isinstance(ev, NodeEndEvent)
        assert ev.status == "success"

    def test_node_end_status_error(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="R")
        t.node_end(nid, status="error")
        ev = t.session.events[-1]
        assert isinstance(ev, NodeEndEvent)
        assert ev.status == "error"

    def test_node_end_status_skipped(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="R")
        t.node_end(nid, status="skipped")
        ev = t.session.events[-1]
        assert isinstance(ev, NodeEndEvent)
        assert ev.status == "skipped"

    def test_node_end_output_stored(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="R")
        t.node_end(nid, output={"result": 42})
        ev = t.session.events[-1]
        assert isinstance(ev, NodeEndEvent)
        assert ev.output == {"result": 42}

    def test_tool_call_persisted(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="R")
        t.tool_call(nid, tool_name="search", inputs={"q": "test"}, outputs={"r": 1})
        ev = t.session.events[-1]
        assert isinstance(ev, ToolCallEvent)
        assert ev.tool_name == "search"
        assert ev.inputs == {"q": "test"}
        assert ev.outputs == {"r": 1}

    def test_tool_call_duration_stored(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="R")
        t.tool_call(nid, tool_name="t", duration_ms=50.5)
        ev = t.session.events[-1]
        assert isinstance(ev, ToolCallEvent)
        assert ev.duration_ms == 50.5

    def test_tool_call_empty_inputs_default(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="R")
        t.tool_call(nid, tool_name="t")
        ev = t.session.events[-1]
        assert isinstance(ev, ToolCallEvent)
        assert ev.inputs == {}
        assert ev.outputs is None

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
        assert ev.completion_tokens == 3

    def test_llm_response_prompt_stored(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="R")
        t.llm_response(nid, model="m", response="r", prompt="test prompt")
        ev = t.session.events[-1]
        assert isinstance(ev, LLMResponseEvent)
        assert ev.prompt == "test prompt"

    def test_llm_response_message_list_prompt(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="R")
        messages = [{"role": "user", "content": "hello"}]
        t.llm_response(nid, model="m", response="r", prompt=messages)
        ev = t.session.events[-1]
        assert isinstance(ev, LLMResponseEvent)
        assert ev.prompt == messages

    def test_llm_response_duration_stored(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="R")
        t.llm_response(nid, model="m", response="r", duration_ms=200.0)
        ev = t.session.events[-1]
        assert isinstance(ev, LLMResponseEvent)
        assert ev.duration_ms == 200.0

    def test_error_persisted(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="R")
        t.error(nid, error_type="ValueError", message="bad input")
        ev = t.session.events[-1]
        assert isinstance(ev, ErrorEvent)
        assert ev.error_type == "ValueError"
        assert ev.message == "bad input"

    def test_error_traceback_stored(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="R")
        t.error(nid, error_type="ValueError", message="bad", traceback="tb here")
        ev = t.session.events[-1]
        assert isinstance(ev, ErrorEvent)
        assert ev.traceback == "tb here"

    def test_error_from_exception(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="R")
        exc = RuntimeError("something broke")
        try:
            raise exc
        except RuntimeError:
            t.error_from_exception(nid, exc)
        ev = t.session.events[-1]
        assert isinstance(ev, ErrorEvent)
        assert ev.error_type == "RuntimeError"
        assert "something broke" in ev.message

    def test_error_from_exception_captures_traceback(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="R")
        try:
            raise ValueError("oops")
        except ValueError as exc:
            t.error_from_exception(nid, exc)
        ev = t.session.events[-1]
        assert isinstance(ev, ErrorEvent)
        assert ev.traceback is not None
        assert len(ev.traceback) > 0

    def test_metadata_attached_to_node_start(self) -> None:
        t = _make_tracer()
        nid = t.node_start("n1", label="R", metadata={"env": "test"})
        ev = t.session.events[0]
        assert ev.metadata == {"env": "test"}

    def test_metadata_attached_to_tool_call(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="R")
        t.tool_call(nid, tool_name="t", metadata={"version": "2"})
        ev = t.session.events[-1]
        assert ev.metadata == {"version": "2"}

    def test_metadata_default_empty_dict(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="R")
        ev = t.session.events[0]
        assert ev.metadata == {}

    def test_parent_node_id_sets_edge(self) -> None:
        t = _make_tracer()
        root = t.node_start("root", label="Root")
        t.node_start("child", label="Child", parent_node_id=root)
        assert len(t.session.edges) == 1
        assert t.session.edges[0].source_node_id == "root"
        assert t.session.edges[0].target_node_id == "child"

    def test_multiple_children_create_multiple_edges(self) -> None:
        t = _make_tracer()
        root = t.node_start("root", label="Root")
        t.node_start("c1", label="C1", parent_node_id=root)
        t.node_start("c2", label="C2", parent_node_id=root)
        assert len(t.session.edges) == 2

    def test_total_event_count_increases(self) -> None:
        t = _make_tracer()
        assert len(t.session) == 0
        t.node_start(label="R")
        assert len(t.session) == 1
        nid = t.node_start(label="R2")
        t.node_end(nid)
        assert len(t.session) == 3

    def test_node_ids_registered(self) -> None:
        t = _make_tracer()
        n1 = t.node_start("n1", label="A")
        n2 = t.node_start("n2", label="B")
        assert "n1" in t.session.node_ids
        assert "n2" in t.session.node_ids

    def test_node_id_unique_per_call(self) -> None:
        t = _make_tracer()
        ids = {t.node_start(label="N") for _ in range(20)}
        assert len(ids) == 20


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

    def test_tool_call_enqueued(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            t = _make_tracer()
            nid = t.node_start(label="R")
            t.tool_call(nid, tool_name="search")
            items = _drain_queue(t.get_queue())
            assert any(i.event_type == "tool_call" for i in items)
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_llm_response_enqueued(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            t = _make_tracer()
            nid = t.node_start(label="R")
            t.llm_response(nid, model="gpt-4o", response="hi")
            items = _drain_queue(t.get_queue())
            assert any(i.event_type == "llm_response" for i in items)
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_error_enqueued(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            t = _make_tracer()
            nid = t.node_start(label="R")
            t.error(nid, error_type="E", message="m")
            items = _drain_queue(t.get_queue())
            assert any(i.event_type == "error" for i in items)
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

    def test_queue_contains_event_objects_not_dicts(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            t = _make_tracer()
            t.node_start(label="R")
            items = _drain_queue(t.get_queue())
            assert len(items) == 1
            assert isinstance(items[0], NodeStartEvent)
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_enqueued_event_matches_session_event(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            t = _make_tracer()
            nid = t.node_start("explicit", label="Test")
            items = _drain_queue(t.get_queue())
            assert len(items) == 1
            assert items[0].node_id == "explicit"
            session_ev = t.session.events[0]
            assert items[0].event_id == session_ev.event_id
        finally:
            loop.close()
            asyncio.set_event_loop(None)


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
    async def test_anode_start_persisted(self) -> None:
        t = _make_tracer()
        nid = await t.anode_start(label="R")
        assert nid in t.session.node_ids
        assert t.session.events[0].event_type == "node_start"

    @pytest.mark.asyncio
    async def test_anode_end_persisted(self) -> None:
        t = _make_tracer()
        nid = await t.anode_start(label="R")
        await t.anode_end(nid)
        assert len(t.session) == 2
        assert t.session.events[-1].event_type == "node_end"

    @pytest.mark.asyncio
    async def test_anode_end_status(self) -> None:
        t = _make_tracer()
        nid = await t.anode_start(label="R")
        await t.anode_end(nid, status="error")
        ev = t.session.events[-1]
        assert isinstance(ev, NodeEndEvent)
        assert ev.status == "error"

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

    @pytest.mark.asyncio
    async def test_async_event_queue_put_awaited(self) -> None:
        """Using a finite maxsize queue with async puts should not drop events."""
        t = Tracer(session_name="test", queue_maxsize=10)
        nid = await t.anode_start(label="R")
        await t.atool_call(nid, tool_name="t")
        await t.allm_response(nid, model="m", response="r")
        await t.anode_end(nid)
        items = _drain_queue(t.get_queue())
        assert len(items) == 4

    @pytest.mark.asyncio
    async def test_async_parent_node_id_edge_inference(self) -> None:
        t = _make_tracer()
        root = await t.anode_start("root", label="Root")
        await t.anode_start("child", label="Child", parent_node_id=root)
        assert len(t.session.edges) == 1


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

    def test_span_yields_node_id(self) -> None:
        t = _make_tracer()
        with t.span("explicit-id", label="Step") as nid:
            assert nid == "explicit-id"

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

    def test_span_error_message_captured(self) -> None:
        t = _make_tracer()
        with pytest.raises(ValueError):
            with t.span(label="Step"):
                raise ValueError("specific error message")
        error_events = [e for e in t.session.events if e.event_type == "error"]
        assert len(error_events) == 1
        assert isinstance(error_events[0], ErrorEvent)
        assert "specific error message" in error_events[0].message

    def test_span_with_parent(self) -> None:
        t = _make_tracer()
        root = t.node_start("root", label="Root")
        with t.span("child", label="Child", parent_node_id=root):
            pass
        assert len(t.session.edges) == 1

    def test_span_node_type_passed(self) -> None:
        t = _make_tracer()
        with t.span(label="LLM Step", node_type="llm") as nid:
            pass
        ev = t.session.events[0]
        assert isinstance(ev, NodeStartEvent)
        assert ev.node_type == "llm"

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

    def test_span_auto_end_false_on_exception(self) -> None:
        """With auto_end=False, no node_end is emitted even on exception."""
        t = _make_tracer()
        with pytest.raises(RuntimeError):
            with t.span(label="Step", auto_end=False):
                raise RuntimeError("fail")
        event_types = [e.event_type for e in t.session.events]
        # error is emitted, but not node_end
        assert "error" in event_types
        assert "node_end" not in event_types

    def test_span_metadata_passed(self) -> None:
        t = _make_tracer()
        with t.span(label="S", metadata={"key": "val"}):
            pass
        ev = t.session.events[0]
        assert ev.metadata == {"key": "val"}

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

    @pytest.mark.asyncio
    async def test_aspan_auto_end_false(self) -> None:
        t = _make_tracer()
        async with t.aspan(label="Step", auto_end=False) as nid:
            pass
        event_types = [e.event_type for e in t.session.events]
        assert "node_start" in event_types
        assert "node_end" not in event_types

    @pytest.mark.asyncio
    async def test_aspan_yields_node_id(self) -> None:
        t = _make_tracer()
        async with t.aspan("async-explicit", label="S") as nid:
            assert nid == "async-explicit"


# ---------------------------------------------------------------------------
# Event ordering and sequence
# ---------------------------------------------------------------------------

class TestEventOrdering:
    def test_events_in_insertion_order(self) -> None:
        t = _make_tracer()
        root = t.node_start("root", label="Root")
        t.tool_call(root, tool_name="search")
        t.llm_response(root, model="gpt-4o", response="ok")
        t.node_end(root)
        types = [e.event_type for e in t.session.events]
        assert types == ["node_start", "tool_call", "llm_response", "node_end"]

    def test_session_id_on_all_events(self) -> None:
        t = Tracer(session_id="fixed-id")
        root = t.node_start(label="R")
        t.tool_call(root, tool_name="t")
        t.llm_response(root, model="m", response="r")
        t.error(root, error_type="E", message="m")
        t.node_end(root)
        for ev in t.session.events:
            assert ev.session_id == "fixed-id"

    def test_node_id_on_all_events(self) -> None:
        t = _make_tracer()
        root = t.node_start("my-node", label="R")
        t.tool_call("my-node", tool_name="t")
        t.llm_response("my-node", model="m", response="r")
        t.node_end("my-node")
        for ev in t.session.events:
            assert ev.node_id == "my-node"

    def test_timestamps_non_decreasing(self) -> None:
        import time as time_mod
        t = _make_tracer()
        root = t.node_start(label="R")
        time_mod.sleep(0.01)
        t.tool_call(root, tool_name="t")
        time_mod.sleep(0.01)
        t.node_end(root)
        events = t.session.events
        for i in range(1, len(events)):
            assert events[i].timestamp >= events[i - 1].timestamp


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

    def test_close_json_contains_all_events(self) -> None:
        import json
        t = _make_tracer()
        root = t.node_start(label="R")
        t.tool_call(root, tool_name="t")
        t.node_end(root)
        json_str = t.close()
        parsed = json.loads(json_str)
        assert len(parsed["events"]) == 3

    def test_close_sets_session_ended_at(self) -> None:
        t = _make_tracer()
        t.close()
        assert t.session.ended_at is not None

    def test_close_idempotent(self) -> None:
        t = _make_tracer()
        t.close()
        ended_at = t.session.ended_at
        t.close()
        assert t.session.ended_at == ended_at

    def test_close_idempotent_is_closed(self) -> None:
        t = _make_tracer()
        t.close()
        t.close()  # second call
        assert t.is_closed

    def test_session_store_accessible_after_close(self) -> None:
        t = _make_tracer()
        nid = t.node_start(label="R")
        t.node_end(nid)
        t.close()
        assert len(t.session) == 2

    def test_events_still_emitted_after_close(self) -> None:
        """Emission methods should still work after close (session_ended_at won't change)."""
        t = _make_tracer()
        t.close()
        # Should not raise
        t.node_start(label="Post-close")
        assert len(t.session) == 1


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

    def test_concurrent_mixed_events(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            t = _make_tracer()
            root = t.node_start("root", label="Root")
            errors: list[Exception] = []

            def _emit_tool(i: int) -> None:
                try:
                    t.tool_call("root", tool_name=f"tool-{i}")
                except Exception as exc:
                    errors.append(exc)

            def _emit_llm(i: int) -> None:
                try:
                    t.llm_response("root", model="m", response=f"resp-{i}")
                except Exception as exc:
                    errors.append(exc)

            threads = (
                [threading.Thread(target=_emit_tool, args=(i,)) for i in range(15)]
                + [threading.Thread(target=_emit_llm, args=(i,)) for i in range(15)]
            )
            for thr in threads:
                thr.start()
            for thr in threads:
                thr.join()

            assert errors == []
            # 1 node_start + 30 other events
            assert len(t.session) == 31
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_concurrent_queue_get_set(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            t = _make_tracer()
            errors: list[Exception] = []

            def _get_queue() -> None:
                try:
                    t.get_queue()
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=_get_queue) for _ in range(20)]
            for thr in threads:
                thr.start()
            for thr in threads:
                thr.join()

            assert errors == []
        finally:
            loop.close()
            asyncio.set_event_loop(None)

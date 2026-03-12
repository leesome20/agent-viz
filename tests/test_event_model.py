"""Unit tests for agent_viz.event_model – Pydantic event models and SessionReplay."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from agent_viz.event_model import (
    AnyEvent,
    ErrorEvent,
    LLMResponseEvent,
    NodeEndEvent,
    NodeStartEvent,
    SessionReplay,
    ToolCallEvent,
    _new_event_id,
    _utcnow,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SESSION_ID = "test-session-001"


def _base_kwargs(**extra) -> dict:
    return {"session_id": SESSION_ID, "node_id": "node-1", **extra}


# ---------------------------------------------------------------------------
# _utcnow / _new_event_id
# ---------------------------------------------------------------------------

def test_utcnow_returns_utc_datetime() -> None:
    dt = _utcnow()
    assert dt.tzinfo is not None
    assert dt.tzinfo == timezone.utc


def test_new_event_id_is_unique() -> None:
    ids = {_new_event_id() for _ in range(100)}
    assert len(ids) == 100


# ---------------------------------------------------------------------------
# NodeStartEvent
# ---------------------------------------------------------------------------

class TestNodeStartEvent:
    def test_defaults(self) -> None:
        ev = NodeStartEvent(**_base_kwargs(label="Start"))
        assert ev.event_type == "node_start"
        assert ev.node_type == "default"
        assert ev.parent_node_id is None
        assert ev.metadata == {}

    def test_custom_fields(self) -> None:
        ev = NodeStartEvent(
            **_base_kwargs(label="Router", node_type="router", parent_node_id="root")
        )
        assert ev.label == "Router"
        assert ev.node_type == "router"
        assert ev.parent_node_id == "root"

    def test_event_type_literal_enforced(self) -> None:
        with pytest.raises(ValidationError):
            NodeStartEvent(**_base_kwargs(label="x", event_type="node_end"))


# ---------------------------------------------------------------------------
# NodeEndEvent
# ---------------------------------------------------------------------------

class TestNodeEndEvent:
    def test_defaults(self) -> None:
        ev = NodeEndEvent(**_base_kwargs())
        assert ev.event_type == "node_end"
        assert ev.status == "success"
        assert ev.output is None

    def test_error_status(self) -> None:
        ev = NodeEndEvent(**_base_kwargs(status="error", output="boom"))
        assert ev.status == "error"
        assert ev.output == "boom"

    def test_invalid_status(self) -> None:
        with pytest.raises(ValidationError):
            NodeEndEvent(**_base_kwargs(status="pending"))


# ---------------------------------------------------------------------------
# ToolCallEvent
# ---------------------------------------------------------------------------

class TestToolCallEvent:
    def test_defaults(self) -> None:
        ev = ToolCallEvent(**_base_kwargs(tool_name="search"))
        assert ev.event_type == "tool_call"
        assert ev.inputs == {}
        assert ev.outputs is None
        assert ev.duration_ms is None

    def test_with_io(self) -> None:
        ev = ToolCallEvent(
            **_base_kwargs(
                tool_name="calculator",
                inputs={"expr": "2+2"},
                outputs={"result": 4},
                duration_ms=12.5,
            )
        )
        assert ev.inputs == {"expr": "2+2"}
        assert ev.outputs == {"result": 4}
        assert ev.duration_ms == 12.5


# ---------------------------------------------------------------------------
# LLMResponseEvent
# ---------------------------------------------------------------------------

class TestLLMResponseEvent:
    def test_minimal(self) -> None:
        ev = LLMResponseEvent(**_base_kwargs(model="gpt-4o", response="Hello"))
        assert ev.event_type == "llm_response"
        assert ev.model == "gpt-4o"
        assert ev.response == "Hello"
        assert ev.prompt is None
        assert ev.prompt_tokens is None
        assert ev.completion_tokens is None

    def test_full(self) -> None:
        ev = LLMResponseEvent(
            **_base_kwargs(
                model="gpt-4o",
                response="Sure!",
                prompt="Tell me a joke",
                prompt_tokens=10,
                completion_tokens=20,
                duration_ms=350.0,
            )
        )
        assert ev.prompt == "Tell me a joke"
        assert ev.prompt_tokens == 10
        assert ev.completion_tokens == 20
        assert ev.duration_ms == 350.0

    def test_prompt_as_message_list(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        ev = LLMResponseEvent(
            **_base_kwargs(model="gpt-4o", response="hey", prompt=messages)
        )
        assert ev.prompt == messages


# ---------------------------------------------------------------------------
# ErrorEvent
# ---------------------------------------------------------------------------

class TestErrorEvent:
    def test_minimal(self) -> None:
        ev = ErrorEvent(
            **_base_kwargs(error_type="ValueError", message="bad input")
        )
        assert ev.event_type == "error"
        assert ev.traceback is None

    def test_with_traceback(self) -> None:
        ev = ErrorEvent(
            **_base_kwargs(
                error_type="RuntimeError",
                message="oops",
                traceback="Traceback (most recent call last):\n  ...",
            )
        )
        assert "Traceback" in ev.traceback


# ---------------------------------------------------------------------------
# AnyEvent discriminated union
# ---------------------------------------------------------------------------

class TestAnyEvent:
    def test_deserialise_node_start(self) -> None:
        raw = NodeStartEvent(**_base_kwargs(label="s")).model_dump()
        from pydantic import TypeAdapter
        ta = TypeAdapter(AnyEvent)
        ev = ta.validate_python(raw)
        assert isinstance(ev, NodeStartEvent)

    def test_deserialise_tool_call(self) -> None:
        raw = ToolCallEvent(**_base_kwargs(tool_name="t")).model_dump()
        from pydantic import TypeAdapter
        ta = TypeAdapter(AnyEvent)
        ev = ta.validate_python(raw)
        assert isinstance(ev, ToolCallEvent)

    def test_unknown_event_type_raises(self) -> None:
        from pydantic import TypeAdapter
        ta = TypeAdapter(AnyEvent)
        with pytest.raises(ValidationError):
            ta.validate_python({"event_type": "unknown", "session_id": "x", "node_id": "y"})


# ---------------------------------------------------------------------------
# SessionReplay
# ---------------------------------------------------------------------------

class TestSessionReplay:
    def _make_replay(self, events: list | None = None) -> SessionReplay:
        return SessionReplay(
            session_id=SESSION_ID,
            session_name="test run",
            events=events or [],
        )

    def test_empty_replay(self) -> None:
        replay = self._make_replay()
        assert replay.schema_version == 1
        assert replay.events == []
        assert replay.ended_at is None

    def test_round_trip_json(self) -> None:
        ev = NodeStartEvent(**_base_kwargs(label="root"))
        replay = self._make_replay(events=[ev])
        raw_json = replay.to_json()
        loaded = SessionReplay.from_json(raw_json)
        assert loaded.session_id == SESSION_ID
        assert len(loaded.events) == 1
        assert isinstance(loaded.events[0], NodeStartEvent)

    def test_compact_json(self) -> None:
        replay = self._make_replay()
        compact = replay.to_json(indent=None)
        assert "\n" not in compact

    def test_session_id_mismatch_raises(self) -> None:
        ev = NodeStartEvent(
            session_id="other-session",
            node_id="n",
            label="x",
        )
        with pytest.raises(ValidationError, match="does not match"):
            SessionReplay(session_id=SESSION_ID, events=[ev])

    def test_from_json_invalid_raises(self) -> None:
        with pytest.raises((ValidationError, Exception)):
            SessionReplay.from_json("{\"not\": \"valid\"}")

    def test_multiple_event_types_round_trip(self) -> None:
        events = [
            NodeStartEvent(**_base_kwargs(label="start")),
            ToolCallEvent(**_base_kwargs(tool_name="search")),
            LLMResponseEvent(**_base_kwargs(model="gpt-4o", response="ok")),
            ErrorEvent(**_base_kwargs(error_type="ValueError", message="err")),
            NodeEndEvent(**_base_kwargs()),
        ]
        replay = self._make_replay(events=events)
        loaded = SessionReplay.from_json(replay.to_json())
        types = [type(e).__name__ for e in loaded.events]
        assert types == [
            "NodeStartEvent",
            "ToolCallEvent",
            "LLMResponseEvent",
            "ErrorEvent",
            "NodeEndEvent",
        ]

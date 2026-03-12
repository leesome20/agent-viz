"""Unit tests for agent_viz.session – SessionStore and GraphEdge."""

from __future__ import annotations

import threading
import time
from datetime import timezone

import pytest
from pydantic import ValidationError

from agent_viz.event_model import (
    ErrorEvent,
    LLMResponseEvent,
    NodeEndEvent,
    NodeStartEvent,
    SessionReplay,
    ToolCallEvent,
)
from agent_viz.session import GraphEdge, SessionStore


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SESSION_ID = "test-session-store"


def _make_store(**kwargs) -> SessionStore:
    return SessionStore(session_id=SESSION_ID, **kwargs)


def _node_start(node_id: str, *, parent_node_id: str | None = None, label: str = "test") -> NodeStartEvent:
    return NodeStartEvent(
        session_id=SESSION_ID,
        node_id=node_id,
        label=label,
        parent_node_id=parent_node_id,
    )


def _node_end(node_id: str, status: str = "success") -> NodeEndEvent:
    return NodeEndEvent(
        session_id=SESSION_ID,
        node_id=node_id,
        status=status,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# GraphEdge
# ---------------------------------------------------------------------------

class TestGraphEdge:
    def test_basic_construction(self) -> None:
        edge = GraphEdge("a", "b", label="child")
        assert edge.source_node_id == "a"
        assert edge.target_node_id == "b"
        assert edge.label == "child"

    def test_equality(self) -> None:
        e1 = GraphEdge("a", "b")
        e2 = GraphEdge("a", "b", label="different label")
        assert e1 == e2

    def test_inequality(self) -> None:
        e1 = GraphEdge("a", "b")
        e2 = GraphEdge("b", "a")
        assert e1 != e2

    def test_hash_consistency(self) -> None:
        e1 = GraphEdge("a", "b")
        e2 = GraphEdge("a", "b")
        assert hash(e1) == hash(e2)
        edge_set = {e1, e2}
        assert len(edge_set) == 1

    def test_to_dict(self) -> None:
        edge = GraphEdge("x", "y", label="lbl")
        d = edge.to_dict()
        assert d == {"source_node_id": "x", "target_node_id": "y", "label": "lbl"}

    def test_repr(self) -> None:
        edge = GraphEdge("a", "b")
        assert "a" in repr(edge) and "b" in repr(edge)


# ---------------------------------------------------------------------------
# SessionStore construction
# ---------------------------------------------------------------------------

class TestSessionStoreConstruction:
    def test_explicit_session_id(self) -> None:
        store = _make_store()
        assert store.session_id == SESSION_ID

    def test_auto_session_id(self) -> None:
        store = SessionStore()
        assert len(store.session_id) == 36  # UUID4

    def test_session_name(self) -> None:
        store = SessionStore(session_name="My Run")
        assert store.session_name == "My Run"

    def test_initial_state(self) -> None:
        store = _make_store()
        assert len(store) == 0
        assert store.node_ids == set()
        assert store.edges == []
        assert not store.is_closed
        assert store.ended_at is None

    def test_started_at_is_utc(self) -> None:
        store = _make_store()
        assert store.started_at.tzinfo is not None
        assert store.started_at.tzinfo == timezone.utc

    def test_repr(self) -> None:
        store = _make_store(session_name="test")
        r = repr(store)
        assert SESSION_ID in r
        assert "test" in r


# ---------------------------------------------------------------------------
# add_event
# ---------------------------------------------------------------------------

class TestAddEvent:
    def test_add_single_event(self) -> None:
        store = _make_store()
        ev = _node_start("n1")
        store.add_event(ev)
        assert len(store) == 1
        assert "n1" in store.node_ids

    def test_add_multiple_events(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        store.add_event(_node_start("n2"))
        store.add_event(_node_end("n1"))
        assert len(store) == 3
        assert store.node_ids == {"n1", "n2"}

    def test_duplicate_event_id_ignored(self) -> None:
        store = _make_store()
        ev = _node_start("n1")
        store.add_event(ev)
        store.add_event(ev)  # same instance / same event_id
        assert len(store) == 1

    def test_session_id_mismatch_raises(self) -> None:
        store = _make_store()
        ev = NodeStartEvent(
            session_id="other-session",
            node_id="n",
            label="x",
        )
        with pytest.raises(ValueError, match="does not match"):
            store.add_event(ev)

    def test_events_property_returns_copy(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        ev_list = store.events
        ev_list.clear()  # mutate the copy
        assert len(store) == 1  # internal list unchanged


# ---------------------------------------------------------------------------
# Edge inference
# ---------------------------------------------------------------------------

class TestEdgeInference:
    def test_no_edge_for_root_node(self) -> None:
        store = _make_store()
        store.add_event(_node_start("root", parent_node_id=None))
        assert store.edges == []

    def test_edge_inferred_from_parent(self) -> None:
        store = _make_store()
        store.add_event(_node_start("root"))
        store.add_event(_node_start("child", parent_node_id="root", label="Child"))
        edges = store.edges
        assert len(edges) == 1
        assert edges[0].source_node_id == "root"
        assert edges[0].target_node_id == "child"

    def test_duplicate_edge_not_added(self) -> None:
        store = _make_store()
        store.add_event(_node_start("root"))
        # Both events reference the same parent-child pair
        store.add_event(_node_start("child", parent_node_id="root"))
        # Additional event for the same node still referencing the same parent
        store.add_event(
            LLMResponseEvent(
                session_id=SESSION_ID,
                node_id="child",
                parent_node_id="root",
                model="gpt-4o",
                response="hi",
            )
        )
        assert len(store.edges) == 1

    def test_multiple_children_of_same_parent(self) -> None:
        store = _make_store()
        store.add_event(_node_start("root"))
        store.add_event(_node_start("c1", parent_node_id="root"))
        store.add_event(_node_start("c2", parent_node_id="root"))
        assert len(store.edges) == 2

    def test_edge_label_from_node_start(self) -> None:
        store = _make_store()
        store.add_event(_node_start("root"))
        store.add_event(
            NodeStartEvent(
                session_id=SESSION_ID,
                node_id="child",
                parent_node_id="root",
                label="My Child",
            )
        )
        assert store.edges[0].label == "My Child"


# ---------------------------------------------------------------------------
# node helpers
# ---------------------------------------------------------------------------

class TestNodeHelpers:
    def test_node_event_count(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        store.add_event(
            ToolCallEvent(
                session_id=SESSION_ID,
                node_id="n1",
                tool_name="search",
            )
        )
        store.add_event(_node_end("n1"))
        assert store.node_event_count("n1") == 3
        assert store.node_event_count("n2") == 0

    def test_get_node_events(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        store.add_event(_node_start("n2"))
        store.add_event(_node_end("n1"))
        n1_events = store.get_node_events("n1")
        assert len(n1_events) == 2
        assert all(e.node_id == "n1" for e in n1_events)


# ---------------------------------------------------------------------------
# close / lifecycle
# ---------------------------------------------------------------------------

class TestClose:
    def test_close_sets_ended_at(self) -> None:
        store = _make_store()
        assert store.ended_at is None
        store.close()
        assert store.ended_at is not None
        assert store.ended_at.tzinfo == timezone.utc

    def test_close_idempotent(self) -> None:
        store = _make_store()
        store.close()
        first_ended_at = store.ended_at
        time.sleep(0.01)
        store.close()
        assert store.ended_at == first_ended_at

    def test_is_closed(self) -> None:
        store = _make_store()
        assert not store.is_closed
        store.close()
        assert store.is_closed


# ---------------------------------------------------------------------------
# build_summary
# ---------------------------------------------------------------------------

class TestBuildSummary:
    def test_empty_summary(self) -> None:
        store = _make_store()
        s = store.build_summary()
        assert s["total_events"] == 0
        assert s["total_nodes"] == 0
        assert s["total_edges"] == 0
        assert s["duration_seconds"] is None

    def test_summary_counts(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        store.add_event(_node_start("n2", parent_node_id="n1"))
        store.add_event(
            ToolCallEvent(session_id=SESSION_ID, node_id="n1", tool_name="t")
        )
        store.add_event(_node_end("n1"))
        store.add_event(_node_end("n2"))
        s = store.build_summary()
        assert s["total_events"] == 5
        assert s["total_nodes"] == 2
        assert s["total_edges"] == 1
        assert s["event_type_counts"]["node_start"] == 2
        assert s["event_type_counts"]["tool_call"] == 1
        assert s["event_type_counts"]["node_end"] == 2

    def test_summary_duration_after_close(self) -> None:
        store = _make_store()
        time.sleep(0.02)
        store.close()
        s = store.build_summary()
        assert s["duration_seconds"] is not None
        assert s["duration_seconds"] >= 0.0


# ---------------------------------------------------------------------------
# Serialisation: to_replay / to_replay_json / from_replay
# ---------------------------------------------------------------------------

class TestSerialisation:
    def _populated_store(self) -> SessionStore:
        store = _make_store(session_name="test run")
        store.add_event(_node_start("root", label="Root"))
        store.add_event(_node_start("child", parent_node_id="root", label="Child"))
        store.add_event(
            LLMResponseEvent(
                session_id=SESSION_ID,
                node_id="child",
                model="gpt-4o",
                response="Done",
            )
        )
        store.add_event(_node_end("child"))
        store.add_event(_node_end("root"))
        store.close()
        return store

    def test_to_replay_returns_session_replay(self) -> None:
        store = self._populated_store()
        replay = store.to_replay()
        assert isinstance(replay, SessionReplay)
        assert replay.session_id == SESSION_ID
        assert replay.session_name == "test run"
        assert len(replay.events) == 5
        assert replay.ended_at is not None

    def test_to_replay_json_round_trip(self) -> None:
        store = self._populated_store()
        json_str = store.to_replay_json()
        assert isinstance(json_str, str)
        loaded = SessionReplay.from_json(json_str)
        assert loaded.session_id == SESSION_ID
        assert len(loaded.events) == 5

    def test_to_replay_json_compact(self) -> None:
        store = _make_store()
        json_str = store.to_replay_json(indent=None)
        assert "\n" not in json_str

    def test_from_replay_restores_events(self) -> None:
        store = self._populated_store()
        replay = store.to_replay()
        restored = SessionStore.from_replay(replay)
        assert len(restored) == len(store)
        assert restored.session_id == store.session_id
        assert restored.session_name == store.session_name
        assert restored.is_closed

    def test_from_replay_restores_edges(self) -> None:
        store = self._populated_store()
        replay = store.to_replay()
        restored = SessionStore.from_replay(replay)
        assert len(restored.edges) == len(store.edges)

    def test_from_replay_open_session(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        # Do NOT close – ended_at remains None
        replay = store.to_replay()
        restored = SessionStore.from_replay(replay)
        assert not restored.is_closed

    def test_snapshot_before_close(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        replay = store.to_replay()
        assert replay.ended_at is None
        assert len(replay.events) == 1


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_add_event(self) -> None:
        store = _make_store()
        errors: list[Exception] = []

        def _add(i: int) -> None:
            try:
                ev = NodeStartEvent(
                    session_id=SESSION_ID,
                    node_id=f"node-{i}",
                    label=f"Node {i}",
                )
                store.add_event(ev)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_add, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(store) == 50
        assert len(store.node_ids) == 50

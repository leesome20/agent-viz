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


def _node_start(
    node_id: str,
    *,
    parent_node_id: str | None = None,
    label: str = "test",
    node_type: str = "default",
) -> NodeStartEvent:
    return NodeStartEvent(
        session_id=SESSION_ID,
        node_id=node_id,
        label=label,
        parent_node_id=parent_node_id,
        node_type=node_type,
    )


def _node_end(node_id: str, status: str = "success") -> NodeEndEvent:
    return NodeEndEvent(
        session_id=SESSION_ID,
        node_id=node_id,
        status=status,  # type: ignore[arg-type]
    )


def _tool_call(node_id: str, tool_name: str = "t") -> ToolCallEvent:
    return ToolCallEvent(
        session_id=SESSION_ID,
        node_id=node_id,
        tool_name=tool_name,
    )


def _llm_response(node_id: str) -> LLMResponseEvent:
    return LLMResponseEvent(
        session_id=SESSION_ID,
        node_id=node_id,
        model="gpt-4o",
        response="ok",
    )


def _error_event(node_id: str) -> ErrorEvent:
    return ErrorEvent(
        session_id=SESSION_ID,
        node_id=node_id,
        error_type="ValueError",
        message="oops",
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

    def test_default_label_empty(self) -> None:
        edge = GraphEdge("a", "b")
        assert edge.label == ""

    def test_equality_same_endpoints(self) -> None:
        e1 = GraphEdge("a", "b")
        e2 = GraphEdge("a", "b", label="different label")
        assert e1 == e2

    def test_inequality_reversed_endpoints(self) -> None:
        e1 = GraphEdge("a", "b")
        e2 = GraphEdge("b", "a")
        assert e1 != e2

    def test_inequality_different_source(self) -> None:
        e1 = GraphEdge("a", "c")
        e2 = GraphEdge("b", "c")
        assert e1 != e2

    def test_inequality_different_target(self) -> None:
        e1 = GraphEdge("a", "b")
        e2 = GraphEdge("a", "c")
        assert e1 != e2

    def test_equality_with_non_edge_returns_not_implemented(self) -> None:
        edge = GraphEdge("a", "b")
        result = edge.__eq__("not an edge")
        assert result is NotImplemented

    def test_hash_consistency(self) -> None:
        e1 = GraphEdge("a", "b")
        e2 = GraphEdge("a", "b")
        assert hash(e1) == hash(e2)

    def test_hash_in_set(self) -> None:
        e1 = GraphEdge("a", "b")
        e2 = GraphEdge("a", "b", label="different")
        edge_set = {e1, e2}
        assert len(edge_set) == 1

    def test_hash_different_edges(self) -> None:
        e1 = GraphEdge("a", "b")
        e2 = GraphEdge("b", "a")
        assert hash(e1) != hash(e2)

    def test_to_dict(self) -> None:
        edge = GraphEdge("x", "y", label="lbl")
        d = edge.to_dict()
        assert d == {"source_node_id": "x", "target_node_id": "y", "label": "lbl"}

    def test_to_dict_keys_present(self) -> None:
        edge = GraphEdge("a", "b")
        d = edge.to_dict()
        assert "source_node_id" in d
        assert "target_node_id" in d
        assert "label" in d

    def test_repr_contains_node_ids(self) -> None:
        edge = GraphEdge("src", "dst")
        r = repr(edge)
        assert "src" in r
        assert "dst" in r

    def test_repr_is_string(self) -> None:
        edge = GraphEdge("a", "b")
        assert isinstance(repr(edge), str)


# ---------------------------------------------------------------------------
# SessionStore construction
# ---------------------------------------------------------------------------

class TestSessionStoreConstruction:
    def test_explicit_session_id(self) -> None:
        store = _make_store()
        assert store.session_id == SESSION_ID

    def test_auto_session_id_is_uuid(self) -> None:
        store = SessionStore()
        assert len(store.session_id) == 36
        # UUID4 format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
        parts = store.session_id.split("-")
        assert len(parts) == 5

    def test_auto_session_ids_are_unique(self) -> None:
        ids = {SessionStore().session_id for _ in range(20)}
        assert len(ids) == 20

    def test_session_name_default_empty(self) -> None:
        store = SessionStore()
        assert store.session_name == ""

    def test_session_name_stored(self) -> None:
        store = SessionStore(session_name="My Run")
        assert store.session_name == "My Run"

    def test_initial_events_empty(self) -> None:
        store = _make_store()
        assert len(store) == 0
        assert store.events == []

    def test_initial_node_ids_empty(self) -> None:
        store = _make_store()
        assert store.node_ids == set()

    def test_initial_edges_empty(self) -> None:
        store = _make_store()
        assert store.edges == []

    def test_not_closed_initially(self) -> None:
        store = _make_store()
        assert not store.is_closed
        assert store.ended_at is None

    def test_started_at_is_utc(self) -> None:
        store = _make_store()
        assert store.started_at.tzinfo is not None
        assert store.started_at.tzinfo == timezone.utc

    def test_repr_contains_session_id(self) -> None:
        store = _make_store(session_name="test")
        r = repr(store)
        assert SESSION_ID in r

    def test_repr_contains_session_name(self) -> None:
        store = _make_store(session_name="test")
        r = repr(store)
        assert "test" in r

    def test_repr_is_string(self) -> None:
        store = _make_store()
        assert isinstance(repr(store), str)


# ---------------------------------------------------------------------------
# add_event
# ---------------------------------------------------------------------------

class TestAddEvent:
    def test_add_single_node_start(self) -> None:
        store = _make_store()
        ev = _node_start("n1")
        store.add_event(ev)
        assert len(store) == 1
        assert "n1" in store.node_ids

    def test_add_node_end(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        store.add_event(_node_end("n1"))
        assert len(store) == 2

    def test_add_tool_call(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        store.add_event(_tool_call("n1", "search"))
        assert len(store) == 2

    def test_add_llm_response(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        store.add_event(_llm_response("n1"))
        assert len(store) == 2

    def test_add_error_event(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        store.add_event(_error_event("n1"))
        assert len(store) == 2

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

    def test_duplicate_event_id_different_instance(self) -> None:
        """Two different instances with the same event_id: second is ignored."""
        store = _make_store()
        ev1 = _node_start("n1")
        # Create a second event with the same event_id by modelling a copy
        ev2 = NodeStartEvent(
            event_id=ev1.event_id,
            session_id=SESSION_ID,
            node_id="n1",
            label="duplicate",
        )
        store.add_event(ev1)
        store.add_event(ev2)
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

    def test_events_in_insertion_order(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        store.add_event(_tool_call("n1"))
        store.add_event(_node_end("n1"))
        types = [e.event_type for e in store.events]
        assert types == ["node_start", "tool_call", "node_end"]

    def test_node_id_registered_on_first_event(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        store.add_event(_tool_call("n1"))
        assert "n1" in store.node_ids
        # Still only one unique node id
        assert len(store.node_ids) == 1

    def test_multiple_node_ids_registered(self) -> None:
        store = _make_store()
        for i in range(5):
            store.add_event(_node_start(f"n{i}"))
        assert len(store.node_ids) == 5


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

    def test_edge_label_from_node_start(self) -> None:
        store = _make_store()
        store.add_event(_node_start("root"))
        store.add_event(_node_start("child", parent_node_id="root", label="My Child"))
        assert store.edges[0].label == "My Child"

    def test_edge_label_empty_for_non_start_event(self) -> None:
        store = _make_store()
        store.add_event(_node_start("root"))
        # First event for child has parent but is NOT a NodeStartEvent
        ev = LLMResponseEvent(
            session_id=SESSION_ID,
            node_id="child",
            parent_node_id="root",
            model="gpt-4o",
            response="hi",
        )
        store.add_event(ev)
        assert len(store.edges) == 1
        assert store.edges[0].label == ""

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

    def test_chain_of_nodes(self) -> None:
        store = _make_store()
        store.add_event(_node_start("a"))
        store.add_event(_node_start("b", parent_node_id="a"))
        store.add_event(_node_start("c", parent_node_id="b"))
        assert len(store.edges) == 2
        sources = {e.source_node_id for e in store.edges}
        targets = {e.target_node_id for e in store.edges}
        assert sources == {"a", "b"}
        assert targets == {"b", "c"}

    def test_edges_property_returns_list(self) -> None:
        store = _make_store()
        store.add_event(_node_start("root"))
        store.add_event(_node_start("child", parent_node_id="root"))
        assert isinstance(store.edges, list)

    def test_edges_property_returns_copy(self) -> None:
        store = _make_store()
        store.add_event(_node_start("root"))
        store.add_event(_node_start("child", parent_node_id="root"))
        edges1 = store.edges
        edges1.clear()
        assert len(store.edges) == 1

    def test_non_start_event_with_new_parent_creates_edge(self) -> None:
        store = _make_store()
        store.add_event(_node_start("root"))
        # A tool call from a new child node with a parent
        ev = ToolCallEvent(
            session_id=SESSION_ID,
            node_id="worker",
            parent_node_id="root",
            tool_name="compute",
        )
        store.add_event(ev)
        assert len(store.edges) == 1
        assert store.edges[0].source_node_id == "root"
        assert store.edges[0].target_node_id == "worker"


# ---------------------------------------------------------------------------
# node helpers
# ---------------------------------------------------------------------------

class TestNodeHelpers:
    def test_node_event_count_single_node(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        store.add_event(_tool_call("n1", "search"))
        store.add_event(_node_end("n1"))
        assert store.node_event_count("n1") == 3

    def test_node_event_count_unknown_node(self) -> None:
        store = _make_store()
        assert store.node_event_count("nonexistent") == 0

    def test_node_event_count_multiple_nodes(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        store.add_event(_node_start("n2"))
        store.add_event(_node_end("n2"))
        assert store.node_event_count("n1") == 1
        assert store.node_event_count("n2") == 2

    def test_get_node_events_returns_correct_events(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        store.add_event(_node_start("n2"))
        store.add_event(_node_end("n1"))
        n1_events = store.get_node_events("n1")
        assert len(n1_events) == 2
        assert all(e.node_id == "n1" for e in n1_events)

    def test_get_node_events_empty_for_unknown(self) -> None:
        store = _make_store()
        assert store.get_node_events("unknown") == []

    def test_get_node_events_returns_list(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        result = store.get_node_events("n1")
        assert isinstance(result, list)

    def test_get_node_events_in_order(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        store.add_event(_tool_call("n1"))
        store.add_event(_llm_response("n1"))
        store.add_event(_node_end("n1"))
        events = store.get_node_events("n1")
        types = [e.event_type for e in events]
        assert types == ["node_start", "tool_call", "llm_response", "node_end"]


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

    def test_close_sets_is_closed(self) -> None:
        store = _make_store()
        assert not store.is_closed
        store.close()
        assert store.is_closed

    def test_close_idempotent_ended_at(self) -> None:
        store = _make_store()
        store.close()
        first_ended_at = store.ended_at
        time.sleep(0.01)
        store.close()
        assert store.ended_at == first_ended_at

    def test_close_idempotent_is_closed(self) -> None:
        store = _make_store()
        store.close()
        store.close()  # second call
        assert store.is_closed

    def test_events_still_addable_after_close(self) -> None:
        """Close only marks ended_at; events can still be added."""
        store = _make_store()
        store.close()
        store.add_event(_node_start("n1"))
        assert len(store) == 1


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
        assert s["event_type_counts"] == {}

    def test_summary_total_events(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        store.add_event(_node_end("n1"))
        s = store.build_summary()
        assert s["total_events"] == 2

    def test_summary_total_nodes(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        store.add_event(_node_start("n2"))
        s = store.build_summary()
        assert s["total_nodes"] == 2

    def test_summary_total_edges(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        store.add_event(_node_start("n2", parent_node_id="n1"))
        s = store.build_summary()
        assert s["total_edges"] == 1

    def test_summary_event_type_counts(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        store.add_event(_node_start("n2", parent_node_id="n1"))
        store.add_event(_tool_call("n1"))
        store.add_event(_node_end("n1"))
        store.add_event(_node_end("n2"))
        s = store.build_summary()
        assert s["event_type_counts"]["node_start"] == 2
        assert s["event_type_counts"]["tool_call"] == 1
        assert s["event_type_counts"]["node_end"] == 2

    def test_summary_duration_none_before_close(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        s = store.build_summary()
        assert s["duration_seconds"] is None

    def test_summary_duration_after_close(self) -> None:
        store = _make_store()
        time.sleep(0.02)
        store.close()
        s = store.build_summary()
        assert s["duration_seconds"] is not None
        assert s["duration_seconds"] >= 0.0

    def test_summary_all_event_types(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        store.add_event(_tool_call("n1"))
        store.add_event(_llm_response("n1"))
        store.add_event(_error_event("n1"))
        store.add_event(_node_end("n1"))
        s = store.build_summary()
        counts = s["event_type_counts"]
        assert counts["node_start"] == 1
        assert counts["tool_call"] == 1
        assert counts["llm_response"] == 1
        assert counts["error"] == 1
        assert counts["node_end"] == 1


# ---------------------------------------------------------------------------
# Serialisation: to_replay / to_replay_json / from_replay
# ---------------------------------------------------------------------------

class TestSerialisation:
    def _populated_store(self) -> SessionStore:
        store = _make_store(session_name="test run")
        store.add_event(_node_start("root", label="Root"))
        store.add_event(_node_start("child", parent_node_id="root", label="Child"))
        store.add_event(_llm_response("child"))
        store.add_event(_node_end("child"))
        store.add_event(_node_end("root"))
        store.close()
        return store

    def test_to_replay_returns_session_replay(self) -> None:
        store = self._populated_store()
        replay = store.to_replay()
        assert isinstance(replay, SessionReplay)

    def test_to_replay_session_id(self) -> None:
        store = self._populated_store()
        replay = store.to_replay()
        assert replay.session_id == SESSION_ID

    def test_to_replay_session_name(self) -> None:
        store = self._populated_store()
        replay = store.to_replay()
        assert replay.session_name == "test run"

    def test_to_replay_event_count(self) -> None:
        store = self._populated_store()
        replay = store.to_replay()
        assert len(replay.events) == 5

    def test_to_replay_ended_at_set(self) -> None:
        store = self._populated_store()
        replay = store.to_replay()
        assert replay.ended_at is not None

    def test_to_replay_json_round_trip(self) -> None:
        store = self._populated_store()
        json_str = store.to_replay_json()
        assert isinstance(json_str, str)
        loaded = SessionReplay.from_json(json_str)
        assert loaded.session_id == SESSION_ID
        assert len(loaded.events) == 5

    def test_to_replay_json_event_types_preserved(self) -> None:
        store = self._populated_store()
        loaded = SessionReplay.from_json(store.to_replay_json())
        types = [e.event_type for e in loaded.events]
        assert types == ["node_start", "node_start", "llm_response", "node_end", "node_end"]

    def test_to_replay_json_compact(self) -> None:
        store = _make_store()
        json_str = store.to_replay_json(indent=None)
        assert "\n" not in json_str

    def test_to_replay_json_indented(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        json_str = store.to_replay_json(indent=2)
        assert "\n" in json_str

    def test_to_replay_includes_summary(self) -> None:
        store = self._populated_store()
        replay = store.to_replay()
        assert "total_events" in replay.summary
        assert replay.summary["total_events"] == 5

    def test_from_replay_restores_events(self) -> None:
        store = self._populated_store()
        replay = store.to_replay()
        restored = SessionStore.from_replay(replay)
        assert len(restored) == len(store)

    def test_from_replay_restores_session_id(self) -> None:
        store = self._populated_store()
        replay = store.to_replay()
        restored = SessionStore.from_replay(replay)
        assert restored.session_id == store.session_id

    def test_from_replay_restores_session_name(self) -> None:
        store = self._populated_store()
        replay = store.to_replay()
        restored = SessionStore.from_replay(replay)
        assert restored.session_name == store.session_name

    def test_from_replay_restores_closed_state(self) -> None:
        store = self._populated_store()
        replay = store.to_replay()
        restored = SessionStore.from_replay(replay)
        assert restored.is_closed

    def test_from_replay_restores_edges(self) -> None:
        store = self._populated_store()
        replay = store.to_replay()
        restored = SessionStore.from_replay(replay)
        assert len(restored.edges) == len(store.edges)

    def test_from_replay_restores_node_ids(self) -> None:
        store = self._populated_store()
        replay = store.to_replay()
        restored = SessionStore.from_replay(replay)
        assert restored.node_ids == store.node_ids

    def test_from_replay_open_session(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        # Do NOT close
        replay = store.to_replay()
        restored = SessionStore.from_replay(replay)
        assert not restored.is_closed

    def test_snapshot_before_close_ended_at_none(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        replay = store.to_replay()
        assert replay.ended_at is None

    def test_snapshot_event_count_before_close(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        replay = store.to_replay()
        assert len(replay.events) == 1

    def test_all_event_types_round_trip(self) -> None:
        store = _make_store()
        store.add_event(_node_start("n1"))
        store.add_event(_tool_call("n1"))
        store.add_event(_llm_response("n1"))
        store.add_event(_error_event("n1"))
        store.add_event(_node_end("n1"))
        store.close()
        loaded = SessionReplay.from_json(store.to_replay_json())
        types = [type(e).__name__ for e in loaded.events]
        assert types == [
            "NodeStartEvent",
            "ToolCallEvent",
            "LLMResponseEvent",
            "ErrorEvent",
            "NodeEndEvent",
        ]

    def test_event_ids_preserved_in_round_trip(self) -> None:
        store = _make_store()
        events_added = [
            _node_start("n1"),
            _tool_call("n1"),
            _node_end("n1"),
        ]
        for ev in events_added:
            store.add_event(ev)
        loaded = SessionReplay.from_json(store.to_replay_json())
        for orig, restored in zip(events_added, loaded.events):
            assert orig.event_id == restored.event_id


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

    def test_concurrent_read_during_write(self) -> None:
        store = _make_store()
        errors: list[Exception] = []

        def _write(i: int) -> None:
            try:
                ev = NodeStartEvent(
                    session_id=SESSION_ID,
                    node_id=f"node-{i}",
                    label=f"Node {i}",
                )
                store.add_event(ev)
            except Exception as exc:
                errors.append(exc)

        def _read() -> None:
            try:
                _ = store.events
                _ = store.node_ids
                _ = store.edges
                _ = len(store)
            except Exception as exc:
                errors.append(exc)

        threads = (
            [threading.Thread(target=_write, args=(i,)) for i in range(25)]
            + [threading.Thread(target=_read) for _ in range(25)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []

    def test_concurrent_close(self) -> None:
        store = _make_store()
        errors: list[Exception] = []

        def _close() -> None:
            try:
                store.close()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_close) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert store.is_closed

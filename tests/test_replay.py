"""Unit tests for agent_viz.replay – ReplayLoader, validate_replay_file, and CLI."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent_viz.event_model import (
    LLMResponseEvent,
    NodeEndEvent,
    NodeStartEvent,
    SessionReplay,
    ToolCallEvent,
)
from agent_viz.replay import (
    ReplayLoader,
    _build_parser,
    main,
    validate_replay_file,
)
from agent_viz.session import SessionStore


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SESSION_ID = "replay-test-session"


def _make_session_replay(
    n_events: int = 4,
    session_name: str = "test run",
    closed: bool = True,
) -> SessionReplay:
    """Build a minimal but complete SessionReplay for testing."""
    store = SessionStore(session_id=SESSION_ID, session_name=session_name)
    root_id = "root-node"
    store.add_event(
        NodeStartEvent(
            session_id=SESSION_ID,
            node_id=root_id,
            label="Root",
        )
    )
    if n_events >= 2:
        store.add_event(
            ToolCallEvent(
                session_id=SESSION_ID,
                node_id=root_id,
                tool_name="search",
                inputs={"q": "test"},
            )
        )
    if n_events >= 3:
        store.add_event(
            LLMResponseEvent(
                session_id=SESSION_ID,
                node_id=root_id,
                model="gpt-4o",
                response="Hello!",
            )
        )
    if n_events >= 4:
        store.add_event(
            NodeEndEvent(
                session_id=SESSION_ID,
                node_id=root_id,
                status="success",
            )
        )
    if closed:
        store.close()
    return store.to_replay()


def _write_replay_file(replay: SessionReplay, suffix: str = ".json") -> Path:
    """Write a SessionReplay to a temp file and return its path."""
    fd, path_str = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    path = Path(path_str)
    path.write_text(replay.to_json(), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# validate_replay_file
# ---------------------------------------------------------------------------

class TestValidateReplayFile:
    def test_valid_file(self) -> None:
        replay = _make_session_replay()
        path = _write_replay_file(replay)
        try:
            loaded = validate_replay_file(path)
            assert loaded.session_id == SESSION_ID
            assert len(loaded.events) == 4
        finally:
            path.unlink(missing_ok=True)

    def test_valid_file_string_path(self) -> None:
        replay = _make_session_replay()
        path = _write_replay_file(replay)
        try:
            loaded = validate_replay_file(str(path))
            assert loaded.session_id == SESSION_ID
        finally:
            path.unlink(missing_ok=True)

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            validate_replay_file("/nonexistent/path/session.json")

    def test_invalid_json(self) -> None:
        fd, path_str = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        path = Path(path_str)
        try:
            path.write_text("{invalid json", encoding="utf-8")
            with pytest.raises(ValueError, match="Replay validation failed"):
                validate_replay_file(path)
        finally:
            path.unlink(missing_ok=True)

    def test_wrong_schema(self) -> None:
        fd, path_str = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        path = Path(path_str)
        try:
            path.write_text(json.dumps({"not": "a replay"}), encoding="utf-8")
            with pytest.raises(ValueError, match="Replay validation failed"):
                validate_replay_file(path)
        finally:
            path.unlink(missing_ok=True)

    def test_returns_session_replay_instance(self) -> None:
        replay = _make_session_replay()
        path = _write_replay_file(replay)
        try:
            loaded = validate_replay_file(path)
            assert isinstance(loaded, SessionReplay)
        finally:
            path.unlink(missing_ok=True)

    def test_events_count_preserved(self) -> None:
        for n in [1, 2, 3, 4]:
            replay = _make_session_replay(n_events=n)
            path = _write_replay_file(replay)
            try:
                loaded = validate_replay_file(path)
                assert len(loaded.events) == n
            finally:
                path.unlink(missing_ok=True)

    def test_empty_events_file(self) -> None:
        store = SessionStore(session_id=SESSION_ID)
        store.close()
        path = _write_replay_file(store.to_replay())
        try:
            loaded = validate_replay_file(path)
            assert len(loaded.events) == 0
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# ReplayLoader construction
# ---------------------------------------------------------------------------

class TestReplayLoaderConstruction:
    def test_basic_construction(self) -> None:
        replay = _make_session_replay()
        loader = ReplayLoader(replay)
        assert loader.replay is replay
        assert loader.event_count == 4
        assert loader.session_id == SESSION_ID
        assert loader.session_name == "test run"

    def test_tracer_has_correct_session_id(self) -> None:
        replay = _make_session_replay()
        loader = ReplayLoader(replay)
        assert loader.tracer.session_id == SESSION_ID

    def test_tracer_starts_empty(self) -> None:
        """The tracer's session store must be empty – events added during replay only."""
        replay = _make_session_replay(n_events=4)
        loader = ReplayLoader(replay)
        assert len(loader.tracer.session) == 0

    def test_tracer_session_name_matches(self) -> None:
        replay = _make_session_replay(session_name="my session")
        loader = ReplayLoader(replay)
        assert loader.tracer.session.session_name == "my session"

    def test_speed_default(self) -> None:
        replay = _make_session_replay()
        loader = ReplayLoader(replay)
        assert loader._speed == 1.0

    def test_speed_stored(self) -> None:
        replay = _make_session_replay()
        loader = ReplayLoader(replay, speed=2.0)
        assert loader._speed == 2.0

    def test_speed_clamped_to_zero(self) -> None:
        replay = _make_session_replay()
        loader = ReplayLoader(replay, speed=-1.0)
        assert loader._speed == 0.0

    def test_speed_zero_stored(self) -> None:
        replay = _make_session_replay()
        loader = ReplayLoader(replay, speed=0.0)
        assert loader._speed == 0.0

    def test_inter_event_delay_default_none(self) -> None:
        replay = _make_session_replay()
        loader = ReplayLoader(replay)
        assert loader._inter_event_delay is None

    def test_inter_event_delay_stored(self) -> None:
        replay = _make_session_replay()
        loader = ReplayLoader(replay, inter_event_delay=0.1)
        assert loader._inter_event_delay == 0.1

    def test_inter_event_delay_zero(self) -> None:
        replay = _make_session_replay()
        loader = ReplayLoader(replay, inter_event_delay=0.0)
        assert loader._inter_event_delay == 0.0

    def test_repr(self) -> None:
        replay = _make_session_replay()
        loader = ReplayLoader(replay)
        r = repr(loader)
        assert "ReplayLoader" in r
        assert SESSION_ID in r

    def test_repr_includes_event_count(self) -> None:
        replay = _make_session_replay(n_events=4)
        loader = ReplayLoader(replay)
        assert "4" in repr(loader)

    def test_event_count_property(self) -> None:
        for n in [1, 2, 3, 4]:
            replay = _make_session_replay(n_events=n)
            loader = ReplayLoader(replay)
            assert loader.event_count == n


# ---------------------------------------------------------------------------
# ReplayLoader.from_file
# ---------------------------------------------------------------------------

class TestReplayLoaderFromFile:
    def test_from_file_valid(self) -> None:
        replay = _make_session_replay()
        path = _write_replay_file(replay)
        try:
            loader = ReplayLoader.from_file(path)
            assert loader.event_count == 4
        finally:
            path.unlink(missing_ok=True)

    def test_from_file_string_path(self) -> None:
        replay = _make_session_replay()
        path = _write_replay_file(replay)
        try:
            loader = ReplayLoader.from_file(str(path))
            assert loader.event_count == 4
        finally:
            path.unlink(missing_ok=True)

    def test_from_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            ReplayLoader.from_file("/no/such/file.json")

    def test_from_file_invalid_json(self) -> None:
        fd, path_str = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        path = Path(path_str)
        try:
            path.write_text("not json at all", encoding="utf-8")
            with pytest.raises(ValueError, match="Invalid replay file"):
                ReplayLoader.from_file(path)
        finally:
            path.unlink(missing_ok=True)

    def test_from_file_passes_speed(self) -> None:
        replay = _make_session_replay()
        path = _write_replay_file(replay)
        try:
            loader = ReplayLoader.from_file(path, speed=3.0)
            assert loader._speed == 3.0
        finally:
            path.unlink(missing_ok=True)

    def test_from_file_passes_inter_event_delay(self) -> None:
        replay = _make_session_replay()
        path = _write_replay_file(replay)
        try:
            loader = ReplayLoader.from_file(path, inter_event_delay=0.05)
            assert loader._inter_event_delay == 0.05
        finally:
            path.unlink(missing_ok=True)

    def test_from_file_passes_both_kwargs(self) -> None:
        replay = _make_session_replay()
        path = _write_replay_file(replay)
        try:
            loader = ReplayLoader.from_file(path, speed=3.0, inter_event_delay=0.05)
            assert loader._speed == 3.0
            assert loader._inter_event_delay == 0.05
        finally:
            path.unlink(missing_ok=True)

    def test_from_file_session_id_preserved(self) -> None:
        replay = _make_session_replay()
        path = _write_replay_file(replay)
        try:
            loader = ReplayLoader.from_file(path)
            assert loader.session_id == SESSION_ID
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# ReplayLoader.from_json_string
# ---------------------------------------------------------------------------

class TestReplayLoaderFromJsonString:
    def test_valid_json_string(self) -> None:
        replay = _make_session_replay()
        raw = replay.to_json()
        loader = ReplayLoader.from_json_string(raw)
        assert loader.event_count == 4

    def test_valid_json_bytes(self) -> None:
        replay = _make_session_replay()
        raw = replay.to_json().encode()
        loader = ReplayLoader.from_json_string(raw)
        assert loader.event_count == 4

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid replay JSON"):
            ReplayLoader.from_json_string("{bad json")

    def test_empty_events(self) -> None:
        store = SessionStore(session_id=SESSION_ID)
        store.close()
        raw = store.to_replay_json()
        loader = ReplayLoader.from_json_string(raw)
        assert loader.event_count == 0

    def test_session_id_from_json_string(self) -> None:
        replay = _make_session_replay()
        loader = ReplayLoader.from_json_string(replay.to_json())
        assert loader.session_id == SESSION_ID

    def test_kwargs_forwarded(self) -> None:
        replay = _make_session_replay()
        loader = ReplayLoader.from_json_string(replay.to_json(), speed=5.0)
        assert loader._speed == 5.0


# ---------------------------------------------------------------------------
# _compute_delays
# ---------------------------------------------------------------------------

class TestComputeDelays:
    def test_empty_events_returns_empty(self) -> None:
        store = SessionStore(session_id=SESSION_ID)
        replay = store.to_replay()
        loader = ReplayLoader(replay)
        assert loader._compute_delays() == []

    def test_single_event_returns_zero(self) -> None:
        store = SessionStore(session_id=SESSION_ID)
        store.add_event(NodeStartEvent(session_id=SESSION_ID, node_id="n", label="x"))
        replay = store.to_replay()
        loader = ReplayLoader(replay)
        delays = loader._compute_delays()
        assert delays == [0.0]

    def test_fixed_delay_override(self) -> None:
        replay = _make_session_replay(n_events=4)
        loader = ReplayLoader(replay, inter_event_delay=0.5)
        delays = loader._compute_delays()
        assert len(delays) == 4
        assert delays[0] == 0.0
        assert all(d == 0.5 for d in delays[1:])

    def test_fixed_delay_zero(self) -> None:
        replay = _make_session_replay(n_events=4)
        loader = ReplayLoader(replay, inter_event_delay=0.0)
        delays = loader._compute_delays()
        assert all(d == 0.0 for d in delays)

    def test_instant_replay_all_zeros(self) -> None:
        replay = _make_session_replay(n_events=4)
        loader = ReplayLoader(replay, speed=0.0)
        delays = loader._compute_delays()
        assert all(d == 0.0 for d in delays)

    def test_delays_length_matches_events(self) -> None:
        replay = _make_session_replay(n_events=4)
        loader = ReplayLoader(replay, speed=1.0)
        delays = loader._compute_delays()
        assert len(delays) == 4

    def test_delays_first_element_always_zero(self) -> None:
        replay = _make_session_replay(n_events=4)
        for speed in [0.5, 1.0, 2.0, 5.0]:
            loader = ReplayLoader(replay, speed=speed)
            delays = loader._compute_delays()
            assert delays[0] == 0.0, f"First delay should be 0 at speed={speed}"

    def test_delays_non_negative(self) -> None:
        replay = _make_session_replay(n_events=4)
        loader = ReplayLoader(replay, speed=1.0)
        delays = loader._compute_delays()
        assert all(d >= 0.0 for d in delays)

    def test_speed_2x_halves_delays(self) -> None:
        replay = _make_session_replay(n_events=4)
        loader_1x = ReplayLoader(replay, speed=1.0)
        loader_2x = ReplayLoader(replay, speed=2.0)
        delays_1x = loader_1x._compute_delays()
        delays_2x = loader_2x._compute_delays()
        # Each 2x delay should be roughly half the 1x delay
        for d1, d2 in zip(delays_1x[1:], delays_2x[1:]):
            assert abs(d2 - d1 / 2.0) < 1e-9

    def test_delays_capped_at_10s(self) -> None:
        """Pathological timestamps (e.g. gaps of hours) must be capped."""
        from datetime import timedelta
        store = SessionStore(session_id=SESSION_ID)
        ev1 = NodeStartEvent(session_id=SESSION_ID, node_id="n1", label="a")
        ev2 = NodeStartEvent(
            session_id=SESSION_ID,
            node_id="n2",
            label="b",
            timestamp=ev1.timestamp + timedelta(hours=2),
        )
        store.add_event(ev1)
        store.add_event(ev2)
        loader = ReplayLoader(store.to_replay(), speed=1.0)
        delays = loader._compute_delays()
        assert delays[1] <= 10.0

    def test_delays_with_speed_half(self) -> None:
        replay = _make_session_replay(n_events=4)
        loader_1x = ReplayLoader(replay, speed=1.0)
        loader_half = ReplayLoader(replay, speed=0.5)
        delays_1x = loader_1x._compute_delays()
        delays_half = loader_half._compute_delays()
        # 0.5x speed means delays double
        for d1, d_half in zip(delays_1x[1:], delays_half[1:]):
            expected = min(d1 * 2.0, 10.0)
            assert abs(d_half - expected) < 1e-9

    def test_inter_event_delay_ignores_speed(self) -> None:
        """When inter_event_delay is set, speed should be irrelevant."""
        replay = _make_session_replay(n_events=4)
        loader_s1 = ReplayLoader(replay, speed=1.0, inter_event_delay=0.25)
        loader_s5 = ReplayLoader(replay, speed=5.0, inter_event_delay=0.25)
        assert loader_s1._compute_delays() == loader_s5._compute_delays()


# ---------------------------------------------------------------------------
# _emit_event (queue population)
# ---------------------------------------------------------------------------

class TestEmitEvent:
    def test_event_placed_on_queue(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            replay = _make_session_replay(n_events=1)
            loader = ReplayLoader(replay)
            event = replay.events[0]
            loader._emit_event(event)
            q = loader.tracer.get_queue()
            assert not q.empty()
            retrieved = q.get_nowait()
            assert retrieved.event_id == event.event_id
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_session_store_not_polluted_by_emit(self) -> None:
        """_emit_event must NOT add events to the Tracer's SessionStore."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            replay = _make_session_replay(n_events=4)
            loader = ReplayLoader(replay)
            for ev in replay.events:
                loader._emit_event(ev)
            # Session store should remain empty
            assert len(loader.tracer.session) == 0
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_queue_full_does_not_raise(self) -> None:
        from agent_viz.tracer import Tracer  # noqa: PLC0415
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            replay = _make_session_replay(n_events=4)
            loader = ReplayLoader(replay)
            # Replace tracer with one that has a tiny queue
            small_tracer = Tracer(session_id=SESSION_ID, queue_maxsize=1)
            loader.tracer = small_tracer
            # Fill the queue
            loader._emit_event(replay.events[0])
            # Overflow – must not raise
            loader._emit_event(replay.events[1])
            loader._emit_event(replay.events[2])
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_no_event_loop_does_not_raise(self) -> None:
        asyncio.set_event_loop(None)
        replay = _make_session_replay(n_events=1)
        loader = ReplayLoader(replay)
        event = replay.events[0]
        # Should swallow the RuntimeError silently
        loader._emit_event(event)

    def test_emitted_event_type_preserved(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            replay = _make_session_replay(n_events=4)
            loader = ReplayLoader(replay)
            for ev in replay.events:
                loader._emit_event(ev)
            q = loader.tracer.get_queue()
            emitted = []
            while not q.empty():
                emitted.append(q.get_nowait())
            types = [e.event_type for e in emitted]
            assert types == ["node_start", "tool_call", "llm_response", "node_end"]
        finally:
            loop.close()
            asyncio.set_event_loop(None)


# ---------------------------------------------------------------------------
# Ordering: events emitted in correct sequence
# ---------------------------------------------------------------------------

class TestEventOrdering:
    def test_all_events_emitted_in_order(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            replay = _make_session_replay(n_events=4)
            loader = ReplayLoader(replay, inter_event_delay=0.0)

            emitted: list[Any] = []
            original_emit = loader._emit_event

            def _capture(ev):
                emitted.append(ev)
                original_emit(ev)

            loader._emit_event = _capture  # type: ignore[method-assign]

            for event in replay.events:
                loader._emit_event(event)

            assert len(emitted) == 4
            for original, captured in zip(replay.events, emitted):
                assert original.event_id == captured.event_id
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_event_types_in_order(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            replay = _make_session_replay(n_events=4)
            loader = ReplayLoader(replay, inter_event_delay=0.0)

            emitted_types: list[str] = []
            original_emit = loader._emit_event

            def _capture(ev):
                emitted_types.append(ev.event_type)
                original_emit(ev)

            loader._emit_event = _capture  # type: ignore[method-assign]

            for ev in replay.events:
                loader._emit_event(ev)

            assert emitted_types == [
                "node_start",
                "tool_call",
                "llm_response",
                "node_end",
            ]
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_single_event_ordering(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            replay = _make_session_replay(n_events=1)
            loader = ReplayLoader(replay)
            loader._emit_event(replay.events[0])
            q = loader.tracer.get_queue()
            item = q.get_nowait()
            assert item.event_type == "node_start"
        finally:
            loop.close()
            asyncio.set_event_loop(None)


# ---------------------------------------------------------------------------
# JSON round-trip: write file → load via from_file → emit all events
# ---------------------------------------------------------------------------

class TestJsonRoundTrip:
    def test_write_and_reload(self) -> None:
        original = _make_session_replay(n_events=4)
        path = _write_replay_file(original)
        try:
            loader = ReplayLoader.from_file(path)
            assert loader.event_count == 4
            for orig_ev, loaded_ev in zip(original.events, loader.replay.events):
                assert orig_ev.event_id == loaded_ev.event_id
                assert orig_ev.event_type == loaded_ev.event_type
        finally:
            path.unlink(missing_ok=True)

    def test_session_metadata_preserved(self) -> None:
        original = _make_session_replay(session_name="My Session")
        path = _write_replay_file(original)
        try:
            loader = ReplayLoader.from_file(path)
            assert loader.session_id == SESSION_ID
            assert loader.session_name == "My Session"
        finally:
            path.unlink(missing_ok=True)

    def test_empty_session_round_trip(self) -> None:
        store = SessionStore(session_id=SESSION_ID, session_name="empty")
        store.close()
        path = _write_replay_file(store.to_replay())
        try:
            loader = ReplayLoader.from_file(path)
            assert loader.event_count == 0
        finally:
            path.unlink(missing_ok=True)

    def test_event_payloads_preserved(self) -> None:
        original = _make_session_replay(n_events=4)
        path = _write_replay_file(original)
        try:
            loader = ReplayLoader.from_file(path)
            # Check tool call inputs
            tool_ev_orig = next(e for e in original.events if e.event_type == "tool_call")
            tool_ev_loaded = next(e for e in loader.replay.events if e.event_type == "tool_call")
            assert isinstance(tool_ev_loaded, ToolCallEvent)
            assert tool_ev_loaded.inputs == tool_ev_orig.inputs
            assert tool_ev_loaded.tool_name == tool_ev_orig.tool_name
        finally:
            path.unlink(missing_ok=True)

    def test_all_event_types_preserved(self) -> None:
        original = _make_session_replay(n_events=4)
        path = _write_replay_file(original)
        try:
            loader = ReplayLoader.from_file(path)
            orig_types = [e.event_type for e in original.events]
            loaded_types = [e.event_type for e in loader.replay.events]
            assert orig_types == loaded_types
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

class TestBuildParser:
    def test_parser_returns_parser(self) -> None:
        import argparse
        parser = _build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_default_host_and_port(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["session.json"])
        assert args.host == "127.0.0.1"
        assert args.port == 8765

    def test_custom_host(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["session.json", "--host", "0.0.0.0"])
        assert args.host == "0.0.0.0"

    def test_custom_port(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["session.json", "--port", "9000"])
        assert args.port == 9000

    def test_custom_host_and_port(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["session.json", "--host", "0.0.0.0", "--port", "9000"])
        assert args.host == "0.0.0.0"
        assert args.port == 9000

    def test_speed_default(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["session.json"])
        assert args.speed == 1.0

    def test_speed_custom(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["session.json", "--speed", "2.5"])
        assert args.speed == 2.5

    def test_delay_default_none(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["session.json"])
        assert args.delay is None

    def test_delay_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["session.json", "--delay", "0.3"])
        assert args.delay == pytest.approx(0.3)

    def test_instant_default_false(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["session.json"])
        assert args.instant is False

    def test_instant_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["session.json", "--instant"])
        assert args.instant is True

    def test_open_browser_default_false(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["session.json"])
        assert args.open_browser is False

    def test_open_browser_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["session.json", "--open-browser"])
        assert args.open_browser is True

    def test_no_wait_default_false(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["session.json"])
        assert args.no_wait is False

    def test_no_wait_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["session.json", "--no-wait"])
        assert args.no_wait is True

    def test_validate_only_default_false(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["session.json"])
        assert args.validate_only is False

    def test_validate_only_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["session.json", "--validate-only"])
        assert args.validate_only is True

    def test_log_level_default(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["session.json"])
        assert args.log_level == "warning"

    def test_log_level_choices(self) -> None:
        parser = _build_parser()
        for level in ["debug", "info", "warning", "error", "critical"]:
            args = parser.parse_args(["session.json", "--log-level", level])
            assert args.log_level == level

    def test_file_argument_stored(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["my_session.json"])
        assert args.file == "my_session.json"

    def test_client_timeout_default(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["session.json"])
        assert args.client_timeout == 30.0

    def test_client_timeout_custom(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["session.json", "--client-timeout", "60"])
        assert args.client_timeout == 60.0


# ---------------------------------------------------------------------------
# CLI main() function
# ---------------------------------------------------------------------------

class TestMain:
    def test_missing_file_returns_exit_2(self) -> None:
        rc = main(["/absolutely/does/not/exist.json"])
        assert rc == 2

    def test_invalid_json_returns_exit_1(self) -> None:
        fd, path_str = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        path = Path(path_str)
        try:
            path.write_text("{broken", encoding="utf-8")
            rc = main([str(path)])
            assert rc == 1
        finally:
            path.unlink(missing_ok=True)

    def test_validate_only_returns_0_for_valid_file(self, capsys) -> None:
        replay = _make_session_replay(n_events=4)
        path = _write_replay_file(replay)
        try:
            rc = main([str(path), "--validate-only"])
            assert rc == 0
            captured = capsys.readouterr()
            assert "valid" in captured.out.lower()
            assert SESSION_ID in captured.out
        finally:
            path.unlink(missing_ok=True)

    def test_validate_only_prints_event_count(self, capsys) -> None:
        replay = _make_session_replay(n_events=4)
        path = _write_replay_file(replay)
        try:
            main([str(path), "--validate-only"])
            captured = capsys.readouterr()
            assert "4" in captured.out
        finally:
            path.unlink(missing_ok=True)

    def test_validate_only_prints_session_name(self, capsys) -> None:
        replay = _make_session_replay(session_name="named session")
        path = _write_replay_file(replay)
        try:
            main([str(path), "--validate-only"])
            captured = capsys.readouterr()
            assert "named session" in captured.out
        finally:
            path.unlink(missing_ok=True)

    def test_validate_only_invalid_file_returns_1(self) -> None:
        fd, path_str = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        path = Path(path_str)
        try:
            path.write_text(json.dumps({"schema_version": "bad"}), encoding="utf-8")
            rc = main([str(path), "--validate-only"])
            assert rc == 1
        finally:
            path.unlink(missing_ok=True)

    def test_instant_flag_sets_inter_event_delay_zero(self) -> None:
        """When --instant is given, ReplayLoader should be created with inter_event_delay=0."""
        replay = _make_session_replay(n_events=2)
        path = _write_replay_file(replay)
        try:
            captured_loaders: list[ReplayLoader] = []

            original_init = ReplayLoader.__init__

            def _mock_init(self, replay, **kwargs):
                original_init(self, replay, **kwargs)
                captured_loaders.append(self)

            with patch.object(ReplayLoader, "__init__", _mock_init):
                with patch.object(ReplayLoader, "run_replay", return_value=None):
                    main([str(path), "--instant", "--no-wait"])

            assert len(captured_loaders) == 1
            assert captured_loaders[0]._inter_event_delay == 0.0
        finally:
            path.unlink(missing_ok=True)

    def test_delay_flag_sets_inter_event_delay(self) -> None:
        replay = _make_session_replay(n_events=2)
        path = _write_replay_file(replay)
        try:
            captured_loaders: list[ReplayLoader] = []

            original_init = ReplayLoader.__init__

            def _mock_init(self, replay, **kwargs):
                original_init(self, replay, **kwargs)
                captured_loaders.append(self)

            with patch.object(ReplayLoader, "__init__", _mock_init):
                with patch.object(ReplayLoader, "run_replay", return_value=None):
                    main([str(path), "--delay", "0.25", "--no-wait"])

            assert len(captured_loaders) == 1
            assert captured_loaders[0]._inter_event_delay == pytest.approx(0.25)
        finally:
            path.unlink(missing_ok=True)

    def test_run_replay_called_on_valid_file(self) -> None:
        replay = _make_session_replay(n_events=2)
        path = _write_replay_file(replay)
        try:
            with patch.object(ReplayLoader, "run_replay", return_value=None) as mock_run:
                rc = main([str(path), "--no-wait", "--instant"])
            assert rc == 0
            mock_run.assert_called_once()
        finally:
            path.unlink(missing_ok=True)

    def test_run_replay_receives_correct_kwargs(self) -> None:
        replay = _make_session_replay(n_events=2)
        path = _write_replay_file(replay)
        try:
            with patch.object(ReplayLoader, "run_replay", return_value=None) as mock_run:
                main([
                    str(path),
                    "--host", "0.0.0.0",
                    "--port", "9999",
                    "--log-level", "info",
                    "--no-wait",
                    "--instant",
                ])
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["host"] == "0.0.0.0"
            assert call_kwargs["port"] == 9999
            assert call_kwargs["log_level"] == "info"
            assert call_kwargs["wait_for_client"] is False
        finally:
            path.unlink(missing_ok=True)

    def test_keyboard_interrupt_returns_0(self) -> None:
        replay = _make_session_replay(n_events=2)
        path = _write_replay_file(replay)
        try:
            with patch.object(ReplayLoader, "run_replay", side_effect=KeyboardInterrupt):
                rc = main([str(path), "--no-wait", "--instant"])
            assert rc == 0
        finally:
            path.unlink(missing_ok=True)

    def test_unexpected_exception_returns_1(self) -> None:
        replay = _make_session_replay(n_events=2)
        path = _write_replay_file(replay)
        try:
            with patch.object(ReplayLoader, "run_replay", side_effect=RuntimeError("boom")):
                rc = main([str(path), "--no-wait", "--instant"])
            assert rc == 1
        finally:
            path.unlink(missing_ok=True)

    def test_speed_passed_to_loader(self) -> None:
        replay = _make_session_replay(n_events=2)
        path = _write_replay_file(replay)
        try:
            captured_loaders: list[ReplayLoader] = []
            original_init = ReplayLoader.__init__

            def _mock_init(self, replay, **kwargs):
                original_init(self, replay, **kwargs)
                captured_loaders.append(self)

            with patch.object(ReplayLoader, "__init__", _mock_init):
                with patch.object(ReplayLoader, "run_replay", return_value=None):
                    main([str(path), "--speed", "3.0", "--no-wait"])

            assert len(captured_loaders) == 1
            assert captured_loaders[0]._speed == 3.0
        finally:
            path.unlink(missing_ok=True)

    def test_no_wait_flag_sets_wait_for_client_false(self) -> None:
        replay = _make_session_replay(n_events=2)
        path = _write_replay_file(replay)
        try:
            with patch.object(ReplayLoader, "run_replay", return_value=None) as mock_run:
                main([str(path), "--no-wait", "--instant"])
            assert mock_run.call_args.kwargs["wait_for_client"] is False
        finally:
            path.unlink(missing_ok=True)

    def test_open_browser_kwarg_passed(self) -> None:
        replay = _make_session_replay(n_events=2)
        path = _write_replay_file(replay)
        try:
            with patch.object(ReplayLoader, "run_replay", return_value=None) as mock_run:
                main([str(path), "--open-browser", "--no-wait", "--instant"])
            assert mock_run.call_args.kwargs["open_browser"] is True
        finally:
            path.unlink(missing_ok=True)

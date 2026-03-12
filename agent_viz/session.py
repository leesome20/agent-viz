"""SessionStore for accumulating agent events in memory.

The SessionStore is the central in-memory accumulation layer for an agent_viz
session.  It receives events from the Tracer, maintains an ordered log, tracks
unique node IDs, infers graph edges from parent-child relationships, and can
serialise the entire session to a portable ``SessionReplay`` JSON file.

Typical usage::

    store = SessionStore(session_id="abc", session_name="my run")
    store.add_event(some_event)
    replay_json = store.to_replay_json()
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from agent_viz.event_model import (
    AnyEvent,
    NodeEndEvent,
    NodeStartEvent,
    SessionReplay,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC datetime with timezone info."""
    return datetime.now(tz=timezone.utc)


def _new_session_id() -> str:
    """Generate a new unique session identifier."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Edge model
# ---------------------------------------------------------------------------

class GraphEdge:
    """Represents a directed edge between two nodes in the agent graph.

    Attributes:
        source_node_id: The parent / source node identifier.
        target_node_id: The child / target node identifier.
        label: Optional human-readable label for the edge.
    """

    __slots__ = ("source_node_id", "target_node_id", "label")

    def __init__(
        self,
        source_node_id: str,
        target_node_id: str,
        label: str = "",
    ) -> None:
        """Initialise a GraphEdge.

        Args:
            source_node_id: Identifier of the source (parent) node.
            target_node_id: Identifier of the target (child) node.
            label: Optional edge label displayed in the UI.
        """
        self.source_node_id = source_node_id
        self.target_node_id = target_node_id
        self.label = label

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphEdge):
            return NotImplemented
        return (
            self.source_node_id == other.source_node_id
            and self.target_node_id == other.target_node_id
        )

    def __hash__(self) -> int:
        return hash((self.source_node_id, self.target_node_id))

    def __repr__(self) -> str:
        return (
            f"GraphEdge({self.source_node_id!r} -> {self.target_node_id!r}"
            f", label={self.label!r})"
        )

    def to_dict(self) -> dict[str, str]:
        """Serialise the edge to a plain dictionary.

        Returns:
            A dict with keys ``source_node_id``, ``target_node_id``, and
            ``label``.
        """
        return {
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "label": self.label,
        }


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------

class SessionStore:
    """In-memory store for a single agent visualisation session.

    The store is thread-safe: a reentrant lock guards all mutations so that
    the Tracer can call :meth:`add_event` from background threads while the
    WebSocket server reads :attr:`events` concurrently.

    Attributes:
        session_id: Unique identifier for this session.
        session_name: Optional human-readable name.
        started_at: UTC datetime when the store was created.
        ended_at: UTC datetime when :meth:`close` was called; ``None`` until
            then.
    """

    def __init__(
        self,
        *,
        session_id: str | None = None,
        session_name: str = "",
    ) -> None:
        """Initialise a new SessionStore.

        Args:
            session_id: Optional explicit session identifier.  A UUID4 is
                generated automatically when ``None``.
            session_name: Optional human-readable label for the session.
        """
        self.session_id: str = session_id or _new_session_id()
        self.session_name: str = session_name
        self.started_at: datetime = _utcnow()
        self.ended_at: datetime | None = None

        self._events: list[AnyEvent] = []
        self._node_ids: dict[str, int] = {}   # node_id -> first-seen event index
        self._edges: dict[tuple[str, str], GraphEdge] = {}
        self._lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def events(self) -> list[AnyEvent]:
        """A shallow copy of the accumulated events list.

        Returns a copy to prevent external mutation of the internal list.

        Returns:
            List of all events in chronological insertion order.
        """
        with self._lock:
            return list(self._events)

    @property
    def node_ids(self) -> set[str]:
        """Set of all unique node identifiers seen so far.

        Returns:
            Frozenset-like set of node ID strings.
        """
        with self._lock:
            return set(self._node_ids.keys())

    @property
    def edges(self) -> list[GraphEdge]:
        """List of all inferred graph edges.

        Returns:
            List of :class:`GraphEdge` objects in insertion order.
        """
        with self._lock:
            return list(self._edges.values())

    @property
    def is_closed(self) -> bool:
        """``True`` if :meth:`close` has been called on this store."""
        return self.ended_at is not None

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_event(self, event: AnyEvent) -> None:
        """Append an event to the store and update internal indices.

        The method is idempotent with respect to duplicate ``event_id``
        values: duplicate events are silently ignored to guard against
        accidental double-emission.

        Side effects:
        - Registers the event's ``node_id`` in the node ID registry.
        - Infers a graph edge when ``parent_node_id`` is set and no edge
          between that parent and the current node has been recorded yet.

        Args:
            event: A validated event instance (any concrete subclass of
                :class:`~agent_viz.event_model.BaseEvent`).

        Raises:
            ValueError: If the event's ``session_id`` does not match this
                store's ``session_id``.
        """
        if event.session_id != self.session_id:
            raise ValueError(
                f"Event session_id {event.session_id!r} does not match "
                f"store session_id {self.session_id!r}."
            )

        with self._lock:
            # Deduplicate by event_id
            existing_ids = {e.event_id for e in self._events}
            if event.event_id in existing_ids:
                return

            idx = len(self._events)
            self._events.append(event)

            # Register node
            if event.node_id not in self._node_ids:
                self._node_ids[event.node_id] = idx

            # Infer edge from parent relationship
            if event.parent_node_id is not None:
                edge_key = (event.parent_node_id, event.node_id)
                if edge_key not in self._edges:
                    label = ""
                    if isinstance(event, NodeStartEvent):
                        label = event.label
                    self._edges[edge_key] = GraphEdge(
                        source_node_id=event.parent_node_id,
                        target_node_id=event.node_id,
                        label=label,
                    )

    def close(self) -> None:
        """Mark the session as ended, recording the current UTC time.

        Calling this method more than once is a no-op; only the first
        call sets :attr:`ended_at`.
        """
        if self.ended_at is None:
            self.ended_at = _utcnow()

    def node_event_count(self, node_id: str) -> int:
        """Return the total number of events associated with a given node.

        Args:
            node_id: The node identifier to count events for.

        Returns:
            Integer count of events whose ``node_id`` matches.
        """
        with self._lock:
            return sum(1 for e in self._events if e.node_id == node_id)

    def get_node_events(self, node_id: str) -> list[AnyEvent]:
        """Return all events associated with a specific node.

        Args:
            node_id: The node identifier to filter by.

        Returns:
            List of events in insertion order whose ``node_id`` matches.
        """
        with self._lock:
            return [e for e in self._events if e.node_id == node_id]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def build_summary(self) -> dict[str, Any]:
        """Compute aggregate statistics for the session summary block.

        Returns:
            Dictionary containing counts for each event type, total node
            count, edge count, and session duration in seconds (or ``None``
            if the session has not been closed yet).
        """
        with self._lock:
            type_counts: dict[str, int] = {}
            for event in self._events:
                type_counts[event.event_type] = (
                    type_counts.get(event.event_type, 0) + 1
                )

            duration_s: float | None = None
            if self.ended_at is not None:
                duration_s = (
                    self.ended_at - self.started_at
                ).total_seconds()

            return {
                "total_events": len(self._events),
                "total_nodes": len(self._node_ids),
                "total_edges": len(self._edges),
                "event_type_counts": type_counts,
                "duration_seconds": duration_s,
            }

    def to_replay(self) -> SessionReplay:
        """Construct a :class:`~agent_viz.event_model.SessionReplay` snapshot.

        The snapshot captures the store's state at the time of the call.  It
        is safe to call before :meth:`close` has been called.

        Returns:
            A fully validated ``SessionReplay`` instance.
        """
        with self._lock:
            return SessionReplay(
                session_id=self.session_id,
                session_name=self.session_name,
                started_at=self.started_at,
                ended_at=self.ended_at,
                events=list(self._events),
                summary=self.build_summary(),
            )

    def to_replay_json(self, *, indent: int | None = 2) -> str:
        """Serialise the session to a JSON string.

        Args:
            indent: JSON indentation level; ``None`` for compact output.

        Returns:
            UTF-8 JSON string representing the complete session replay.
        """
        return self.to_replay().to_json(indent=indent)

    @classmethod
    def from_replay(cls, replay: SessionReplay) -> "SessionStore":
        """Reconstruct a SessionStore from a :class:`~agent_viz.event_model.SessionReplay`.

        This is useful in the replay CLI to restore full session state from a
        JSON file without re-running the agent.

        Args:
            replay: A validated ``SessionReplay`` instance.

        Returns:
            A ``SessionStore`` whose events, node IDs, and edges mirror the
            replay contents.  The store will be in closed state if
            ``replay.ended_at`` is not ``None``.
        """
        store = cls(
            session_id=replay.session_id,
            session_name=replay.session_name,
        )
        store.started_at = replay.started_at
        for event in replay.events:
            store.add_event(event)
        if replay.ended_at is not None:
            store.ended_at = replay.ended_at
        return store

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of accumulated events."""
        with self._lock:
            return len(self._events)

    def __repr__(self) -> str:
        return (
            f"SessionStore(session_id={self.session_id!r}, "
            f"session_name={self.session_name!r}, "
            f"events={len(self)}, "
            f"closed={self.is_closed})"
        )

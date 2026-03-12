"""Tracer – the primary SDK entry-point for agent instrumentation.

The Tracer class provides a simple, synchronous-friendly API that agent code
calls to emit typed events (node_start, node_end, tool_call, llm_response,
error) into an async queue.  The queue is consumed by the WebSocket server
and relayed to connected browser clients in real time.

Basic usage (synchronous context)::

    from agent_viz import Tracer

    tracer = Tracer(session_name="my run")

    node_id = tracer.node_start("step_1", label="Fetch data")
    tracer.tool_call(node_id, tool_name="http_get",
                     inputs={"url": "https://example.com"},
                     outputs={"status": 200})
    tracer.node_end(node_id)

Context-manager usage::

    with tracer.span("step_2", label="LLM call", parent_node_id=node_id) as sid:
        tracer.llm_response(sid, model="gpt-4o", response="Hello!")

Async usage::

    await tracer.anode_start("step_1", label="Async step")
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import traceback as tb
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator

from agent_viz.event_model import (
    ErrorEvent,
    LLMResponseEvent,
    NodeEndEvent,
    NodeStartEvent,
    ToolCallEvent,
)
from agent_viz.session import SessionStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _new_node_id() -> str:
    """Generate a new unique node identifier."""
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    """Return the current UTC datetime with timezone info."""
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------

class Tracer:
    """Emit structured agent events into an async queue and a SessionStore.

    The Tracer is the primary interface for developers instrumenting their
    agent code.  It is deliberately framework-agnostic: no monkey-patching,
    no framework hooks – just method calls.

    Thread safety
    ~~~~~~~~~~~~~
    All public methods are thread-safe.  Internally the Tracer uses
    :meth:`asyncio.Queue.put_nowait` when an event loop is running, falling
    back to a thread-safe deque drain when called from a non-async context.
    A background drainer task keeps the async queue flushed when the server
    is active.

    Attributes:
        session: The underlying :class:`~agent_viz.session.SessionStore`
            accumulating all events for this session.
    """

    def __init__(
        self,
        *,
        session_id: str | None = None,
        session_name: str = "",
        queue_maxsize: int = 0,
    ) -> None:
        """Initialise a new Tracer.

        Args:
            session_id: Optional explicit session identifier.  A UUID4 is
                generated automatically when ``None``.
            session_name: Optional human-readable label surfaced in the UI.
            queue_maxsize: Maximum number of events buffered in the async
                queue before backpressure kicks in.  ``0`` means unbounded.
        """
        self.session: SessionStore = SessionStore(
            session_id=session_id,
            session_name=session_name,
        )
        self._queue_maxsize = queue_maxsize
        # The async queue is created lazily (or explicitly) because event
        # loops may not exist at construction time.
        self._queue: asyncio.Queue | None = None
        self._lock = threading.Lock()
        self._closed = False

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def get_queue(self) -> asyncio.Queue:
        """Return the async event queue, creating it if necessary.

        The queue is created the first time this method is called.  Callers
        that need a queue must ensure an event loop is running before the
        first call, or call :meth:`set_event_loop` explicitly.

        Returns:
            The :class:`asyncio.Queue` used to stream events.

        Raises:
            RuntimeError: If no running event loop exists and the queue has
                not been pre-created.
        """
        with self._lock:
            if self._queue is None:
                self._queue = asyncio.Queue(maxsize=self._queue_maxsize)
            return self._queue

    def set_queue(self, queue: asyncio.Queue) -> None:
        """Replace the internal async queue with an externally created one.

        This is useful when the WebSocket server creates the queue on its
        event loop and injects it into the Tracer.

        Args:
            queue: The replacement :class:`asyncio.Queue` instance.
        """
        with self._lock:
            self._queue = queue

    def _enqueue(self, event: Any) -> None:
        """Put an event onto the async queue without blocking.

        Attempts to use ``put_nowait`` when a loop is running.  If the queue
        is full, the event is dropped and a warning is logged rather than
        raising to avoid disrupting the agent under observation.

        Args:
            event: A validated event instance to enqueue.
        """
        try:
            queue = self.get_queue()
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(
                    "agent_viz Tracer: event queue is full; dropping event "
                    "%s (%s). Consider increasing queue_maxsize.",
                    event.event_id,
                    event.event_type,
                )
        except RuntimeError:
            # No running event loop – store-only mode, event still persisted.
            pass

    def _emit(self, event: Any) -> Any:
        """Persist an event to the SessionStore and enqueue it.

        This is the single internal path through which all events flow.

        Args:
            event: A validated event instance.

        Returns:
            The same event, allowing callers to chain.
        """
        self.session.add_event(event)
        self._enqueue(event)
        return event

    # ------------------------------------------------------------------
    # Synchronous event-emission API
    # ------------------------------------------------------------------

    def node_start(
        self,
        node_id: str | None = None,
        *,
        label: str = "",
        node_type: str = "default",
        parent_node_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Emit a ``node_start`` event and return the node identifier.

        Args:
            node_id: Optional explicit node identifier.  A UUID4 is generated
                when ``None``.
            label: Human-readable label for the node displayed in the UI.
            node_type: Categorical tag for colour-coding
                (e.g. ``"llm"``, ``"tool"``, ``"router"``).
            parent_node_id: Identifier of the parent node, if any.
            metadata: Optional arbitrary key-value pairs.

        Returns:
            The ``node_id`` string for use in subsequent event calls.
        """
        node_id = node_id or _new_node_id()
        event = NodeStartEvent(
            session_id=self.session.session_id,
            node_id=node_id,
            parent_node_id=parent_node_id,
            label=label or node_id,
            node_type=node_type,
            metadata=metadata or {},
        )
        self._emit(event)
        return node_id

    def node_end(
        self,
        node_id: str,
        *,
        status: str = "success",
        output: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit a ``node_end`` event for the given node.

        Args:
            node_id: Identifier of the node that has finished.
            status: Terminal status; one of ``"success"``, ``"error"``,
                or ``"skipped"``.
            output: Optional freeform output produced by the node.
            metadata: Optional arbitrary key-value pairs.
        """
        event = NodeEndEvent(
            session_id=self.session.session_id,
            node_id=node_id,
            status=status,  # type: ignore[arg-type]
            output=output,
            metadata=metadata or {},
        )
        self._emit(event)

    def tool_call(
        self,
        node_id: str,
        *,
        tool_name: str,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit a ``tool_call`` event.

        Args:
            node_id: Identifier of the node that invoked the tool.
            tool_name: Name of the tool or function.
            inputs: Arguments passed to the tool.
            outputs: Results returned by the tool; ``None`` if pending.
            duration_ms: Wall-clock duration of the call in milliseconds.
            metadata: Optional arbitrary key-value pairs.
        """
        event = ToolCallEvent(
            session_id=self.session.session_id,
            node_id=node_id,
            tool_name=tool_name,
            inputs=inputs or {},
            outputs=outputs,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
        self._emit(event)

    def llm_response(
        self,
        node_id: str,
        *,
        model: str,
        response: str,
        prompt: str | list[dict[str, Any]] | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit an ``llm_response`` event.

        Args:
            node_id: Identifier of the node that made the LLM call.
            model: Model identifier string (e.g. ``"gpt-4o"``).
            response: Textual response from the model.
            prompt: The prompt or messages sent to the model.
            prompt_tokens: Token count for the prompt.
            completion_tokens: Token count for the completion.
            duration_ms: Wall-clock duration of the inference call in ms.
            metadata: Optional arbitrary key-value pairs.
        """
        event = LLMResponseEvent(
            session_id=self.session.session_id,
            node_id=node_id,
            model=model,
            response=response,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
        self._emit(event)

    def error(
        self,
        node_id: str,
        *,
        error_type: str,
        message: str,
        traceback: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit an ``error`` event.

        Args:
            node_id: Identifier of the node where the error occurred.
            error_type: Exception class name or short error category.
            message: Human-readable error description.
            traceback: Optional full traceback string.
            metadata: Optional arbitrary key-value pairs.
        """
        event = ErrorEvent(
            session_id=self.session.session_id,
            node_id=node_id,
            error_type=error_type,
            message=message,
            traceback=traceback,
            metadata=metadata or {},
        )
        self._emit(event)

    def error_from_exception(
        self,
        node_id: str,
        exc: BaseException,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Convenience wrapper that extracts fields from a live exception.

        Args:
            node_id: Identifier of the node where the exception was raised.
            exc: The caught exception instance.
            metadata: Optional arbitrary key-value pairs.
        """
        self.error(
            node_id,
            error_type=type(exc).__name__,
            message=str(exc),
            traceback=tb.format_exc(),
            metadata=metadata or {},
        )

    # ------------------------------------------------------------------
    # Async event-emission API
    # ------------------------------------------------------------------

    async def anode_start(
        self,
        node_id: str | None = None,
        *,
        label: str = "",
        node_type: str = "default",
        parent_node_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Async variant of :meth:`node_start`.

        Uses ``await queue.put(event)`` for proper backpressure when the
        queue has a finite ``maxsize``.

        Args:
            node_id: Optional explicit node identifier.
            label: Human-readable label for the node.
            node_type: Categorical tag for colour-coding.
            parent_node_id: Identifier of the parent node.
            metadata: Optional arbitrary key-value pairs.

        Returns:
            The ``node_id`` string.
        """
        node_id = node_id or _new_node_id()
        event = NodeStartEvent(
            session_id=self.session.session_id,
            node_id=node_id,
            parent_node_id=parent_node_id,
            label=label or node_id,
            node_type=node_type,
            metadata=metadata or {},
        )
        self.session.add_event(event)
        await self.get_queue().put(event)
        return node_id

    async def anode_end(
        self,
        node_id: str,
        *,
        status: str = "success",
        output: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Async variant of :meth:`node_end`.

        Args:
            node_id: Identifier of the node that has finished.
            status: Terminal status.
            output: Optional freeform output.
            metadata: Optional arbitrary key-value pairs.
        """
        event = NodeEndEvent(
            session_id=self.session.session_id,
            node_id=node_id,
            status=status,  # type: ignore[arg-type]
            output=output,
            metadata=metadata or {},
        )
        self.session.add_event(event)
        await self.get_queue().put(event)

    async def atool_call(
        self,
        node_id: str,
        *,
        tool_name: str,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Async variant of :meth:`tool_call`.

        Args:
            node_id: Identifier of the node that invoked the tool.
            tool_name: Name of the tool or function.
            inputs: Arguments passed to the tool.
            outputs: Results returned by the tool.
            duration_ms: Wall-clock duration in milliseconds.
            metadata: Optional arbitrary key-value pairs.
        """
        event = ToolCallEvent(
            session_id=self.session.session_id,
            node_id=node_id,
            tool_name=tool_name,
            inputs=inputs or {},
            outputs=outputs,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
        self.session.add_event(event)
        await self.get_queue().put(event)

    async def allm_response(
        self,
        node_id: str,
        *,
        model: str,
        response: str,
        prompt: str | list[dict[str, Any]] | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Async variant of :meth:`llm_response`.

        Args:
            node_id: Identifier of the node that made the LLM call.
            model: Model identifier string.
            response: Textual response from the model.
            prompt: The prompt or messages sent to the model.
            prompt_tokens: Token count for the prompt.
            completion_tokens: Token count for the completion.
            duration_ms: Wall-clock duration in milliseconds.
            metadata: Optional arbitrary key-value pairs.
        """
        event = LLMResponseEvent(
            session_id=self.session.session_id,
            node_id=node_id,
            model=model,
            response=response,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
        self.session.add_event(event)
        await self.get_queue().put(event)

    async def aerror(
        self,
        node_id: str,
        *,
        error_type: str,
        message: str,
        traceback: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Async variant of :meth:`error`.

        Args:
            node_id: Identifier of the node where the error occurred.
            error_type: Exception class name.
            message: Human-readable error description.
            traceback: Optional full traceback string.
            metadata: Optional arbitrary key-value pairs.
        """
        event = ErrorEvent(
            session_id=self.session.session_id,
            node_id=node_id,
            error_type=error_type,
            message=message,
            traceback=traceback,
            metadata=metadata or {},
        )
        self.session.add_event(event)
        await self.get_queue().put(event)

    # ------------------------------------------------------------------
    # Context-manager helpers
    # ------------------------------------------------------------------

    @contextmanager
    def span(
        self,
        node_id: str | None = None,
        *,
        label: str = "",
        node_type: str = "default",
        parent_node_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        auto_end: bool = True,
    ) -> Generator[str, None, None]:
        """Synchronous context manager that wraps a node with start/end events.

        Automatically emits ``node_start`` on enter and ``node_end`` on exit.
        If the body raises an exception, an ``error`` event is emitted and the
        node is ended with ``status='error'``.

        Args:
            node_id: Optional explicit node identifier.
            label: Human-readable label.
            node_type: Categorical tag.
            parent_node_id: Parent node identifier.
            metadata: Optional arbitrary key-value pairs.
            auto_end: If ``True`` (default), emit ``node_end`` automatically
                on context exit.  Set to ``False`` to suppress this and
                emit it manually.

        Yields:
            The ``node_id`` string.
        """
        nid = self.node_start(
            node_id,
            label=label,
            node_type=node_type,
            parent_node_id=parent_node_id,
            metadata=metadata,
        )
        try:
            yield nid
        except Exception as exc:
            self.error_from_exception(nid, exc)
            if auto_end:
                self.node_end(nid, status="error")
            raise
        else:
            if auto_end:
                self.node_end(nid, status="success")

    @contextlib.asynccontextmanager
    async def aspan(
        self,
        node_id: str | None = None,
        *,
        label: str = "",
        node_type: str = "default",
        parent_node_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        auto_end: bool = True,
    ):
        """Async context manager that wraps a node with start/end events.

        Behaves identically to :meth:`span` but uses the async event-emission
        methods for proper backpressure support.

        Args:
            node_id: Optional explicit node identifier.
            label: Human-readable label.
            node_type: Categorical tag.
            parent_node_id: Parent node identifier.
            metadata: Optional arbitrary key-value pairs.
            auto_end: If ``True`` (default), emit ``node_end`` on exit.

        Yields:
            The ``node_id`` string.
        """
        nid = await self.anode_start(
            node_id,
            label=label,
            node_type=node_type,
            parent_node_id=parent_node_id,
            metadata=metadata,
        )
        try:
            yield nid
        except Exception as exc:
            await self.aerror(
                nid,
                error_type=type(exc).__name__,
                message=str(exc),
                traceback=tb.format_exc(),
            )
            if auto_end:
                await self.anode_end(nid, status="error")
            raise
        else:
            if auto_end:
                await self.anode_end(nid, status="success")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> str:
        """Close the tracer and finalise the session.

        Marks the underlying SessionStore as ended and sets the closed flag.
        Subsequent calls to event-emission methods will still succeed but the
        session's ``ended_at`` timestamp will not change.

        Returns:
            The JSON string of the complete session replay, suitable for
            writing to a file.
        """
        with self._lock:
            if not self._closed:
                self.session.close()
                self._closed = True
        return self.session.to_replay_json()

    @property
    def session_id(self) -> str:
        """Shortcut accessor for the underlying session identifier."""
        return self.session.session_id

    @property
    def is_closed(self) -> bool:
        """``True`` if :meth:`close` has been called."""
        return self._closed

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Tracer(session_id={self.session_id!r}, "
            f"events={len(self.session)}, "
            f"closed={self.is_closed})"
        )

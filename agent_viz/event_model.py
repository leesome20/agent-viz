"""Pydantic models defining the canonical structure for all agent events.

All events emitted by the Tracer, stored in the SessionStore, and streamed
over WebSockets conform to the models defined here.  The module also
provides the top-level ``SessionReplay`` model used for JSON export and
import of complete agent sessions.

Event hierarchy
---------------
- ``BaseEvent``       – common fields shared by every event
- ``NodeStartEvent``  – marks the beginning of an agent node / step
- ``NodeEndEvent``    – marks the completion of an agent node / step
- ``ToolCallEvent``   – records a single tool / function invocation
- ``LLMResponseEvent``– records an LLM inference call and its output
- ``ErrorEvent``      – captures exceptions or failure states
- ``AnyEvent``        – type alias for the discriminated union of all events

Session replay
--------------
- ``SessionReplay``   – envelope wrapping a list of events for export/import
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC datetime with timezone info."""
    return datetime.now(tz=timezone.utc)


def _new_event_id() -> str:
    """Generate a new unique event identifier."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Base event
# ---------------------------------------------------------------------------

class BaseEvent(BaseModel):
    """Fields common to every agent event.

    Attributes:
        event_id: Unique identifier for this event instance.
        event_type: Discriminator string identifying the concrete event kind.
        session_id: Identifier of the session this event belongs to.
        node_id: Identifier of the agent node that produced this event.
        parent_node_id: Optional identifier of the parent node, enabling
            tree / graph edge inference.
        timestamp: UTC datetime when the event was created.
        metadata: Arbitrary key-value pairs for extensibility.
    """

    event_id: str = Field(
        default_factory=_new_event_id,
        description="Unique identifier for this event.",
    )
    event_type: str = Field(
        description="Discriminator string for the concrete event type."
    )
    session_id: str = Field(
        description="Identifier of the session that owns this event."
    )
    node_id: str = Field(
        description="Identifier of the agent node that produced this event."
    )
    parent_node_id: str | None = Field(
        default=None,
        description="Parent node identifier for edge inference; None for root nodes.",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC datetime when the event was created.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional data attached to this event.",
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Concrete event types
# ---------------------------------------------------------------------------

class NodeStartEvent(BaseEvent):
    """Signals that an agent node / workflow step has begun.

    Attributes:
        event_type: Always ``"node_start"``.
        label: Human-readable label for the node shown in the UI.
        node_type: Optional categorical tag (e.g. ``"llm"``, ``"tool"``,
            ``"router"``).  Used for colour-coding in the graph.
    """

    event_type: Literal["node_start"] = "node_start"
    label: str = Field(
        description="Human-readable label displayed on the graph node."
    )
    node_type: str = Field(
        default="default",
        description="Categorical tag used for colour-coding in the graph UI.",
    )


class NodeEndEvent(BaseEvent):
    """Signals that an agent node / workflow step has finished.

    Attributes:
        event_type: Always ``"node_end"``.
        status: Outcome of the node; one of ``"success"``, ``"error"``,
            or ``"skipped"``.
        output: Optional freeform output produced by the node.
    """

    event_type: Literal["node_end"] = "node_end"
    status: Literal["success", "error", "skipped"] = Field(
        default="success",
        description="Terminal status of the agent node.",
    )
    output: Any = Field(
        default=None,
        description="Optional output value produced by the node.",
    )


class ToolCallEvent(BaseEvent):
    """Records a single tool or external function invocation.

    Attributes:
        event_type: Always ``"tool_call"``.
        tool_name: Name of the tool that was called.
        inputs: Mapping of argument names to values passed to the tool.
        outputs: Mapping of result names to values returned by the tool.
            May be ``None`` if the call has not yet completed.
        duration_ms: Wall-clock duration of the tool call in milliseconds.
    """

    event_type: Literal["tool_call"] = "tool_call"
    tool_name: str = Field(
        description="Name of the tool or function that was invoked."
    )
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments passed to the tool.",
    )
    outputs: dict[str, Any] | None = Field(
        default=None,
        description="Results returned by the tool; None if call is pending.",
    )
    duration_ms: float | None = Field(
        default=None,
        description="Wall-clock duration of the tool call in milliseconds.",
    )


class LLMResponseEvent(BaseEvent):
    """Records an LLM inference call and its response.

    Attributes:
        event_type: Always ``"llm_response"``.
        model: Model identifier string (e.g. ``"gpt-4o"``).
        prompt: The prompt or messages sent to the model.  May be a string
            or a list of message dicts depending on the calling convention.
        response: The model's textual response.
        prompt_tokens: Number of tokens in the prompt, if known.
        completion_tokens: Number of tokens in the completion, if known.
        duration_ms: Wall-clock duration of the inference call in milliseconds.
    """

    event_type: Literal["llm_response"] = "llm_response"
    model: str = Field(
        description="Model identifier string used for the inference call."
    )
    prompt: str | list[dict[str, Any]] | None = Field(
        default=None,
        description="Prompt or messages sent to the model.",
    )
    response: str = Field(
        description="Textual response produced by the model."
    )
    prompt_tokens: int | None = Field(
        default=None,
        description="Token count for the prompt, if reported by the API.",
    )
    completion_tokens: int | None = Field(
        default=None,
        description="Token count for the completion, if reported by the API.",
    )
    duration_ms: float | None = Field(
        default=None,
        description="Wall-clock duration of the inference call in milliseconds.",
    )


class ErrorEvent(BaseEvent):
    """Captures an exception or failure state within an agent node.

    Attributes:
        event_type: Always ``"error"``.
        error_type: The exception class name (e.g. ``"ValueError"``).
        message: Human-readable error description.
        traceback: Optional full traceback string for debugging.
    """

    event_type: Literal["error"] = "error"
    error_type: str = Field(
        description="Exception class name or short error category."
    )
    message: str = Field(
        description="Human-readable description of the error."
    )
    traceback: str | None = Field(
        default=None,
        description="Full traceback string for post-mortem debugging.",
    )


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------

#: Type alias for the discriminated union of all concrete event types.
#: Use this when deserialising events from JSON so Pydantic selects the
#: correct model automatically via the ``event_type`` discriminator field.
AnyEvent = Annotated[
    Union[
        NodeStartEvent,
        NodeEndEvent,
        ToolCallEvent,
        LLMResponseEvent,
        ErrorEvent,
    ],
    Field(discriminator="event_type"),
]


# ---------------------------------------------------------------------------
# Session replay envelope
# ---------------------------------------------------------------------------

class SessionReplay(BaseModel):
    """Portable JSON envelope for a complete agent session.

    A ``SessionReplay`` file can be produced by the SessionStore after an
    agent run and later loaded by the replay CLI to re-stream events to
    the browser UI at configurable speed.

    Attributes:
        schema_version: Integer version of the replay file format.  Bumped
            on backwards-incompatible changes.
        session_id: Unique identifier for the captured session.
        session_name: Optional human-readable name for the session.
        started_at: UTC datetime when the session was created.
        ended_at: UTC datetime when the session was closed; ``None`` if the
            session is still active.
        events: Ordered list of all events captured during the session.
        summary: Optional free-form summary metadata (e.g. total tokens,
            number of tool calls).
    """

    schema_version: int = Field(
        default=1,
        description="Replay file format version; bump on breaking changes.",
    )
    session_id: str = Field(
        description="Unique identifier for the captured agent session."
    )
    session_name: str = Field(
        default="",
        description="Optional human-readable label for the session.",
    )
    started_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC datetime when the session began.",
    )
    ended_at: datetime | None = Field(
        default=None,
        description="UTC datetime when the session ended; None if still active.",
    )
    events: list[AnyEvent] = Field(
        default_factory=list,
        description="Chronologically ordered list of all captured events.",
    )
    summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional aggregate statistics or annotations for the session.",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_event_session_ids(self) -> "SessionReplay":
        """Ensure every event's session_id matches the envelope session_id.

        Raises:
            ValueError: If any event carries a mismatched session_id.
        """
        for event in self.events:
            if event.session_id != self.session_id:
                raise ValueError(
                    f"Event {event.event_id!r} has session_id "
                    f"{event.session_id!r} which does not match the "
                    f"replay envelope session_id {self.session_id!r}."
                )
        return self

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise the session replay to a JSON string.

        Args:
            indent: JSON indentation level.  Pass ``None`` for compact output.

        Returns:
            A UTF-8 JSON string representing the full session replay.
        """
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, raw: str | bytes) -> "SessionReplay":
        """Deserialise a session replay from a JSON string or bytes.

        Args:
            raw: A JSON string or bytes object produced by :meth:`to_json`.

        Returns:
            A fully validated ``SessionReplay`` instance.

        Raises:
            pydantic.ValidationError: If the JSON does not conform to the schema.
            json.JSONDecodeError: If *raw* is not valid JSON.
        """
        return cls.model_validate_json(raw)

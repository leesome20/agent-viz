"""agent_viz - Real-time visualization of AI agent workflows.

Public SDK surface exposing the Tracer class, server startup helper,
and all event type constants for convenient agent instrumentation.

Typical usage::

    from agent_viz import Tracer, start_server

    tracer = Tracer(session_name="my_agent_run")
    start_server(tracer, host="127.0.0.1", port=8765)

    with tracer.span("root", label="Agent Start") as node_id:
        tracer.llm_response(node_id, model="gpt-4", response="Hello!")
        tracer.tool_call(node_id, tool_name="search", inputs={"q": "foo"})

Event type constants are provided as module-level string attributes so
callers can reference them symbolically instead of hard-coding strings::

    from agent_viz import EVENT_NODE_START, EVENT_TOOL_CALL

All heavy dependencies (FastAPI, uvicorn) are loaded lazily – importing
``agent_viz`` is cheap even if you only need the Tracer.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Package version
# ---------------------------------------------------------------------------

#: Package version string following PEP 440.
__version__: str = "0.1.0"

# ---------------------------------------------------------------------------
# Event type string constants
# ---------------------------------------------------------------------------

#: Event type discriminator for ``NodeStartEvent``.
EVENT_NODE_START: str = "node_start"

#: Event type discriminator for ``NodeEndEvent``.
EVENT_NODE_END: str = "node_end"

#: Event type discriminator for ``ToolCallEvent``.
EVENT_TOOL_CALL: str = "tool_call"

#: Event type discriminator for ``LLMResponseEvent``.
EVENT_LLM_RESPONSE: str = "llm_response"

#: Event type discriminator for ``ErrorEvent``.
EVENT_ERROR: str = "error"

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
# The heavy modules (FastAPI, asyncio server, pydantic models) are only loaded
# when the symbols are actually accessed.  This keeps the import footprint
# minimal for users who only need the Tracer or the event constants.


def __getattr__(name: str) -> object:
    """Lazy attribute loader for heavy SDK symbols.

    Supports deferred imports of ``Tracer``, ``SessionStore``, and
    ``start_server`` so that importing ``agent_viz`` does not force FastAPI
    and uvicorn to load unless the caller actually uses those symbols.

    Pydantic event model classes are also available for convenience::

        from agent_viz import NodeStartEvent, ToolCallEvent

    Args:
        name: The attribute name being accessed.

    Returns:
        The resolved attribute value.

    Raises:
        AttributeError: If *name* is not a known public symbol.
    """
    # ------------------------------------------------------------------ #
    # Core SDK classes                                                     #
    # ------------------------------------------------------------------ #
    if name == "Tracer":
        from agent_viz.tracer import Tracer  # noqa: PLC0415
        return Tracer

    if name == "start_server":
        from agent_viz.server import start_server  # noqa: PLC0415
        return start_server

    if name == "SessionStore":
        from agent_viz.session import SessionStore  # noqa: PLC0415
        return SessionStore

    # ------------------------------------------------------------------ #
    # Replay utilities                                                     #
    # ------------------------------------------------------------------ #
    if name == "ReplayLoader":
        from agent_viz.replay import ReplayLoader  # noqa: PLC0415
        return ReplayLoader

    if name == "validate_replay_file":
        from agent_viz.replay import validate_replay_file  # noqa: PLC0415
        return validate_replay_file

    # ------------------------------------------------------------------ #
    # Event model classes                                                  #
    # ------------------------------------------------------------------ #
    if name == "NodeStartEvent":
        from agent_viz.event_model import NodeStartEvent  # noqa: PLC0415
        return NodeStartEvent

    if name == "NodeEndEvent":
        from agent_viz.event_model import NodeEndEvent  # noqa: PLC0415
        return NodeEndEvent

    if name == "ToolCallEvent":
        from agent_viz.event_model import ToolCallEvent  # noqa: PLC0415
        return ToolCallEvent

    if name == "LLMResponseEvent":
        from agent_viz.event_model import LLMResponseEvent  # noqa: PLC0415
        return LLMResponseEvent

    if name == "ErrorEvent":
        from agent_viz.event_model import ErrorEvent  # noqa: PLC0415
        return ErrorEvent

    if name == "SessionReplay":
        from agent_viz.event_model import SessionReplay  # noqa: PLC0415
        return SessionReplay

    if name == "AnyEvent":
        from agent_viz.event_model import AnyEvent  # noqa: PLC0415
        return AnyEvent

    # ------------------------------------------------------------------ #
    # Server factory (advanced usage)                                      #
    # ------------------------------------------------------------------ #
    if name == "create_app":
        from agent_viz.server import create_app  # noqa: PLC0415
        return create_app

    raise AttributeError(f"module 'agent_viz' has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__: list[str] = [
    # Version
    "__version__",
    # Core SDK (lazily imported)
    "Tracer",
    "SessionStore",
    "start_server",
    "create_app",
    # Replay utilities (lazily imported)
    "ReplayLoader",
    "validate_replay_file",
    # Event model classes (lazily imported)
    "NodeStartEvent",
    "NodeEndEvent",
    "ToolCallEvent",
    "LLMResponseEvent",
    "ErrorEvent",
    "SessionReplay",
    "AnyEvent",
    # Event type string constants
    "EVENT_NODE_START",
    "EVENT_NODE_END",
    "EVENT_TOOL_CALL",
    "EVENT_LLM_RESPONSE",
    "EVENT_ERROR",
]

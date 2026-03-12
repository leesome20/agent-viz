"""agent_viz - Real-time visualization of AI agent workflows.

Public SDK surface exposing the Tracer class, server startup helper,
and all event type constants for convenient agent instrumentation.

Typical usage::

    from agent_viz import Tracer, start_server

    tracer = Tracer(session_name="my_agent_run")
    start_server(tracer, host="127.0.0.1", port=8765)

    with tracer.node_start("root", label="Agent Start") as node_id:
        tracer.llm_response(node_id, model="gpt-4", response="Hello!")
        tracer.tool_call(node_id, tool_name="search", inputs={"q": "foo"})
        tracer.node_end(node_id)
"""

from __future__ import annotations

# Version of the package
__version__ = "0.1.0"

# Event type string constants for external consumers
EVENT_NODE_START: str = "node_start"
EVENT_NODE_END: str = "node_end"
EVENT_TOOL_CALL: str = "tool_call"
EVENT_LLM_RESPONSE: str = "llm_response"
EVENT_ERROR: str = "error"

# Lazy imports: the heavy modules (FastAPI, asyncio server) are only loaded
# when the symbols are actually accessed, keeping the import footprint minimal
# for users who only need the Tracer.

def __getattr__(name: str) -> object:
    """Lazy attribute loader for heavy SDK symbols.

    Supports deferred imports of ``Tracer`` and ``start_server`` so that
    importing ``agent_viz`` does not force FastAPI / uvicorn to load unless
    the caller actually uses those symbols.

    Args:
        name: The attribute name being accessed.

    Returns:
        The resolved attribute value.

    Raises:
        AttributeError: If *name* is not a known public symbol.
    """
    if name == "Tracer":
        from agent_viz.tracer import Tracer  # noqa: PLC0415
        return Tracer
    if name == "start_server":
        from agent_viz.server import start_server  # noqa: PLC0415
        return start_server
    if name == "SessionStore":
        from agent_viz.session import SessionStore  # noqa: PLC0415
        return SessionStore
    raise AttributeError(f"module 'agent_viz' has no attribute {name!r}")


__all__ = [
    # Version
    "__version__",
    # Public classes (lazily imported)
    "Tracer",
    "SessionStore",
    "start_server",
    # Event type constants
    "EVENT_NODE_START",
    "EVENT_NODE_END",
    "EVENT_TOOL_CALL",
    "EVENT_LLM_RESPONSE",
    "EVENT_ERROR",
]

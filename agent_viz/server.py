"""FastAPI WebSocket server for agent_viz.

This module provides the FastAPI application that:
- Serves the static single-page UI (index.html, app.js, style.css)
- Exposes a WebSocket endpoint at ``/ws`` for real-time event streaming
- Manages a set of connected browser clients and broadcasts events to all
- Provides a REST endpoint to export the current session as JSON
- Exposes a ``start_server`` helper that runs uvicorn in a background thread

Typical usage::

    from agent_viz import Tracer, start_server

    tracer = Tracer(session_name="my run")
    start_server(tracer, host="127.0.0.1", port=8765)

    # Now instrument your agent – events stream to the browser live
    node_id = tracer.node_start(label="step 1")
    tracer.node_end(node_id)
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agent_viz.tracer import Tracer

logger = logging.getLogger(__name__)

# Path to the bundled static assets
_STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Manages active WebSocket connections and broadcasts events.

    Thread-safe: connections are added/removed under an asyncio lock so
    concurrent broadcast calls do not race with connect/disconnect.
    """

    def __init__(self) -> None:
        """Initialise an empty connection manager."""
        self._connections: list[WebSocket] = []
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        """Return the asyncio lock, creating it lazily on first access.

        The lock must be created inside a running event loop.

        Returns:
            The :class:`asyncio.Lock` instance.
        """
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection and register it.

        Args:
            websocket: The incoming WebSocket connection to accept.
        """
        await websocket.accept()
        async with self._get_lock():
            self._connections.append(websocket)
        logger.info("WebSocket client connected. Total: %d", len(self._connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection from the registry.

        Args:
            websocket: The WebSocket connection to remove.
        """
        async with self._get_lock():
            try:
                self._connections.remove(websocket)
            except ValueError:
                pass
        logger.info("WebSocket client disconnected. Total: %d", len(self._connections))

    async def broadcast(self, message: str) -> None:
        """Send a text message to all currently connected clients.

        Dead connections are silently removed during the broadcast.

        Args:
            message: UTF-8 text (typically JSON) to send to every client.
        """
        async with self._get_lock():
            live: list[WebSocket] = []
            for ws in list(self._connections):
                try:
                    await ws.send_text(message)
                    live.append(ws)
                except Exception:  # noqa: BLE001
                    logger.debug("Removing dead WebSocket connection during broadcast.")
            self._connections = live

    @property
    def client_count(self) -> int:
        """Number of currently connected WebSocket clients."""
        return len(self._connections)


# ---------------------------------------------------------------------------
# Event broadcaster task
# ---------------------------------------------------------------------------

async def _broadcast_loop(
    tracer: Tracer,
    manager: ConnectionManager,
) -> None:
    """Continuously drain the tracer's event queue and broadcast to clients.

    Runs until cancelled (e.g. when the server shuts down).

    Args:
        tracer: The :class:`~agent_viz.tracer.Tracer` whose queue to drain.
        manager: The :class:`ConnectionManager` to broadcast through.
    """
    queue = tracer.get_queue()
    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=1.0)
            payload = event.model_dump_json()
            await manager.broadcast(payload)
            queue.task_done()
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            logger.debug("Broadcast loop cancelled.")
            break
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error in broadcast loop: %s", exc)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(tracer: Tracer) -> FastAPI:
    """Create and return the configured FastAPI application.

    The app mounts static files, registers the WebSocket endpoint, and wires
    up an on-startup task that begins draining the tracer's event queue.

    Args:
        tracer: The :class:`~agent_viz.tracer.Tracer` instance whose events
            should be streamed to connected browser clients.

    Returns:
        A fully configured :class:`fastapi.FastAPI` application instance.
    """
    app = FastAPI(
        title="agent_viz",
        description="Real-time visualization of AI agent workflows.",
        version="0.1.0",
    )

    manager = ConnectionManager()
    _broadcast_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @app.on_event("startup")
    async def _startup() -> None:
        nonlocal _broadcast_task
        _broadcast_task = asyncio.create_task(
            _broadcast_loop(tracer, manager),
            name="agent_viz_broadcast",
        )
        logger.info("agent_viz broadcast loop started.")

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        if _broadcast_task is not None:
            _broadcast_task.cancel()
            try:
                await _broadcast_task
            except asyncio.CancelledError:
                pass
        logger.info("agent_viz broadcast loop stopped.")

    # ------------------------------------------------------------------
    # Static files
    # ------------------------------------------------------------------

    if _STATIC_DIR.exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(_STATIC_DIR)),
            name="static",
        )

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def _root() -> FileResponse:
        """Serve the single-page UI."""
        index_path = _STATIC_DIR / "index.html"
        return FileResponse(str(index_path), media_type="text/html")

    @app.get("/health", tags=["meta"])
    async def _health() -> dict[str, Any]:
        """Return server health and session metadata.

        Returns:
            Dictionary with status, session information, and client count.
        """
        return {
            "status": "ok",
            "session_id": tracer.session_id,
            "session_name": tracer.session.session_name,
            "event_count": len(tracer.session),
            "client_count": manager.client_count,
        }

    @app.get("/export", tags=["session"])
    async def _export() -> JSONResponse:
        """Export the current session as a JSON replay file.

        Returns:
            A JSON response containing the full session replay.
        """
        replay_json = tracer.session.to_replay_json(indent=2)
        return JSONResponse(
            content=json.loads(replay_json),
            headers={
                "Content-Disposition": (
                    f'attachment; filename="session_{tracer.session_id[:8]}.json"'
                )
            },
        )

    @app.get("/session", tags=["session"])
    async def _session_info() -> JSONResponse:
        """Return high-level session summary metadata.

        Returns:
            JSON object with session identity and aggregate statistics.
        """
        summary = tracer.session.build_summary()
        return JSONResponse(
            content={
                "session_id": tracer.session_id,
                "session_name": tracer.session.session_name,
                "started_at": tracer.session.started_at.isoformat(),
                "ended_at": (
                    tracer.session.ended_at.isoformat()
                    if tracer.session.ended_at
                    else None
                ),
                "summary": summary,
            }
        )

    @app.websocket("/ws")
    async def _websocket_endpoint(websocket: WebSocket) -> None:
        """WebSocket endpoint for real-time event streaming.

        On connection, the server replays all events accumulated so far so
        the client can reconstruct the full graph from any connection point.
        Subsequent events are pushed as they arrive via the broadcast loop.

        Args:
            websocket: The incoming WebSocket connection.
        """
        await manager.connect(websocket)
        try:
            # Replay historical events so a late-joining client sees the full graph
            historical = tracer.session.events
            if historical:
                batch_payload = json.dumps(
                    {
                        "type": "batch",
                        "events": [json.loads(e.model_dump_json()) for e in historical],
                    }
                )
                try:
                    await websocket.send_text(batch_payload)
                except Exception:  # noqa: BLE001
                    logger.debug("Failed to send historical batch to new client.")
                    await manager.disconnect(websocket)
                    return

            # Keep the connection alive until the client disconnects
            while True:
                try:
                    # We just need to detect disconnect; ignore any client messages
                    await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                except asyncio.TimeoutError:
                    # Send a ping to keep the connection alive
                    try:
                        await websocket.send_text(json.dumps({"type": "ping"}))
                    except Exception:  # noqa: BLE001
                        break
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("WebSocket error: %s", exc)
        finally:
            await manager.disconnect(websocket)

    return app


# ---------------------------------------------------------------------------
# Server startup helper
# ---------------------------------------------------------------------------

def start_server(
    tracer: Tracer,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    log_level: str = "warning",
    open_browser: bool = False,
    daemon: bool = True,
) -> threading.Thread:
    """Start the agent_viz web server in a background thread.

    The server begins accepting WebSocket connections immediately.  Events
    emitted by *tracer* are streamed to all connected clients in real time.

    Args:
        tracer: The :class:`~agent_viz.tracer.Tracer` instance to visualise.
        host: Hostname or IP address to bind the server to.
        port: TCP port to listen on.  Defaults to ``8765``.
        log_level: Uvicorn log level string (e.g. ``"info"``, ``"warning"``).
        open_browser: If ``True``, attempt to open the UI in the default
            browser after the server starts.
        daemon: If ``True`` (default), the server thread is a daemon thread
            and will not prevent the process from exiting.

    Returns:
        The :class:`threading.Thread` running the server.  The caller does
        not need to join it unless they want to wait for shutdown.
    """
    app = create_app(tracer)
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=log_level,
        loop="asyncio",
    )
    server = uvicorn.Server(config)

    def _run() -> None:
        asyncio.run(server.serve())

    thread = threading.Thread(target=_run, name="agent_viz_server", daemon=daemon)
    thread.start()

    # Give the server a moment to bind before returning
    import time
    deadline = time.monotonic() + 5.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)

    url = f"http://{host}:{port}"
    logger.info("agent_viz server running at %s", url)
    print(f"[agent_viz] Dashboard: {url}")

    if open_browser:
        import webbrowser
        webbrowser.open(url)

    return thread

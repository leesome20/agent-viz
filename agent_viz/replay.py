"""Replay loader and CLI for agent_viz.

This module provides the ``ReplayLoader`` class that reads a JSON session
replay file produced by the SessionStore and re-streams all events to
connected browser clients at a configurable playback speed.

It is also directly invokable from the command line via the
``agent-viz-replay`` entry point defined in ``pyproject.toml``::

    agent-viz-replay session_replay.json --speed 1.0 --port 8765

or equivalently::

    python -m agent_viz.replay session_replay.json

The replay loader:
- Validates the JSON file against the ``SessionReplay`` schema
- Instantiates a ``Tracer`` pre-populated with the session metadata
- Starts the FastAPI / WebSocket server so a browser can connect
- Re-emits every event from the file into the Tracer's queue at the
  requested inter-event delay
- Optionally opens the browser automatically
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from agent_viz.event_model import AnyEvent, SessionReplay
from agent_viz.session import SessionStore
from agent_viz.tracer import Tracer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ReplayLoader
# ---------------------------------------------------------------------------

class ReplayLoader:
    """Load a JSON session replay file and re-stream its events.

    The loader validates the file, populates a :class:`~agent_viz.tracer.Tracer`
    with the historical session state, starts the WebSocket server, waits for
    at least one browser client to connect (optional), then emits each event
    into the live queue with the requested inter-event delay so the browser
    graph animates in real time.

    Attributes:
        replay: The validated :class:`~agent_viz.event_model.SessionReplay`
            instance loaded from the file.
        tracer: The :class:`~agent_viz.tracer.Tracer` used for re-emission.
    """

    def __init__(
        self,
        replay: SessionReplay,
        *,
        speed: float = 1.0,
        inter_event_delay: float | None = None,
    ) -> None:
        """Initialise the ReplayLoader.

        Args:
            replay: A validated ``SessionReplay`` instance.
            speed: Playback speed multiplier.  ``1.0`` means real-time
                (honours the original inter-event timestamps).  ``2.0``
                plays back at double speed; ``0.0`` means instant (no
                delay between events).
            inter_event_delay: Override for a fixed delay in *seconds*
                between each emitted event.  When provided, ``speed`` is
                ignored.  Pass ``0.0`` for instant replay.
        """
        self.replay: SessionReplay = replay
        self._speed: float = max(0.0, speed)
        self._inter_event_delay: float | None = inter_event_delay

        # Build a Tracer pre-seeded with the replay's session identity but
        # with an *empty* event store so the server starts clean and the
        # browser sees events arriving one by one.
        self.tracer: Tracer = Tracer(
            session_id=replay.session_id,
            session_name=replay.session_name,
        )

    # ------------------------------------------------------------------
    # Class-level factory
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: Any) -> "ReplayLoader":
        """Create a ``ReplayLoader`` from a JSON file path.

        Args:
            path: File system path to a ``SessionReplay`` JSON file.
            **kwargs: Additional keyword arguments forwarded to
                :meth:`__init__` (e.g. ``speed``, ``inter_event_delay``).

        Returns:
            A fully initialised ``ReplayLoader``.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not valid JSON or fails schema
                validation.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Replay file not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a regular file: {file_path}")

        try:
            raw = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"Could not read replay file: {exc}") from exc

        try:
            replay = SessionReplay.from_json(raw)
        except Exception as exc:
            raise ValueError(
                f"Invalid replay file {file_path!r}: {exc}"
            ) from exc

        return cls(replay, **kwargs)

    @classmethod
    def from_json_string(cls, raw: str | bytes, **kwargs: Any) -> "ReplayLoader":
        """Create a ``ReplayLoader`` from a raw JSON string.

        Args:
            raw: JSON string or bytes conforming to the
                :class:`~agent_viz.event_model.SessionReplay` schema.
            **kwargs: Additional keyword arguments forwarded to
                :meth:`__init__`.

        Returns:
            A fully initialised ``ReplayLoader``.

        Raises:
            ValueError: If the JSON is invalid or fails schema validation.
        """
        try:
            replay = SessionReplay.from_json(raw)
        except Exception as exc:
            raise ValueError(f"Invalid replay JSON: {exc}") from exc
        return cls(replay, **kwargs)

    # ------------------------------------------------------------------
    # Delay computation
    # ------------------------------------------------------------------

    def _compute_delays(self) -> list[float]:
        """Compute per-event wait durations (in seconds) for replay.

        When ``inter_event_delay`` is set, every event uses that fixed
        delay.  Otherwise the original timestamps are honoured and scaled
        by ``1 / speed``.

        Returns:
            List of floats with length ``len(self.replay.events)`` where
            ``delays[i]`` is the number of seconds to wait *before*
            emitting ``events[i]``.  The first event always has delay 0.
        """
        events = self.replay.events
        if not events:
            return []

        # Fixed delay override
        if self._inter_event_delay is not None:
            fixed = max(0.0, self._inter_event_delay)
            return [0.0] + [fixed] * (len(events) - 1)

        # Instant replay
        if self._speed == 0.0:
            return [0.0] * len(events)

        # Timestamp-derived delays scaled by playback speed
        delays: list[float] = [0.0]
        for i in range(1, len(events)):
            prev_ts = events[i - 1].timestamp
            curr_ts = events[i].timestamp
            delta = (curr_ts - prev_ts).total_seconds()
            # Guard against negative or extreme deltas
            scaled = max(0.0, delta) / self._speed
            # Cap individual delay to 10 s to avoid very long pauses
            delays.append(min(scaled, 10.0))

        return delays

    # ------------------------------------------------------------------
    # Synchronous replay runner
    # ------------------------------------------------------------------

    def run_replay(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        open_browser: bool = False,
        log_level: str = "warning",
        wait_for_client: bool = True,
        client_wait_timeout: float = 30.0,
    ) -> None:
        """Start the server and replay all events synchronously.

        This is the primary entry-point for the CLI.  It:

        1. Starts the FastAPI / WebSocket server in a background thread.
        2. Optionally waits for a browser client to connect.
        3. Emits each event into the Tracer queue with the computed delay.
        4. Blocks until all events have been emitted.

        After all events are replayed the server continues running so the
        user can inspect the final graph.  Press Ctrl-C to exit.

        Args:
            host: Hostname or IP to bind the server.
            port: TCP port to listen on.
            open_browser: If ``True``, open the default browser after the
                server starts.
            log_level: Uvicorn log level string.
            wait_for_client: If ``True`` (default), pause before replaying
                until at least one browser client has connected.
            client_wait_timeout: Maximum seconds to wait for a client when
                ``wait_for_client`` is ``True``.  After the timeout, replay
                proceeds regardless.
        """
        from agent_viz.server import create_app, start_server  # noqa: PLC0415

        n_events = len(self.replay.events)
        print(
            f"[agent_viz replay] Session: {self.replay.session_name or self.replay.session_id}"
        )
        print(f"[agent_viz replay] Events to replay: {n_events}")
        print(f"[agent_viz replay] Speed: {self._speed}x")

        # Start server
        thread = start_server(
            self.tracer,
            host=host,
            port=port,
            log_level=log_level,
            open_browser=open_browser,
            daemon=True,
        )

        if n_events == 0:
            print("[agent_viz replay] No events to replay.  Server running – Ctrl-C to stop.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n[agent_viz replay] Stopped.")
            return

        if wait_for_client:
            self._wait_for_client(host, port, timeout=client_wait_timeout)

        delays = self._compute_delays()

        print("[agent_viz replay] Starting replay…")
        for idx, (event, delay) in enumerate(zip(self.replay.events, delays), start=1):
            if delay > 0:
                time.sleep(delay)
            self._emit_event(event)
            if idx % 10 == 0 or idx == n_events:
                print(f"[agent_viz replay] {idx}/{n_events} events emitted.", end="\r")

        print(f"\n[agent_viz replay] Replay complete ({n_events} events).")
        print("[agent_viz replay] Server still running – Ctrl-C to stop.")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[agent_viz replay] Stopped.")

    # ------------------------------------------------------------------
    # Async replay runner
    # ------------------------------------------------------------------

    async def run_replay_async(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        open_browser: bool = False,
        log_level: str = "warning",
    ) -> None:
        """Async version of the replay runner.

        Emits events using ``asyncio.sleep`` for accurate delays without
        blocking the event loop.  Useful when embedding replay into an
        existing async application.

        Args:
            host: Hostname or IP to bind the server.
            port: TCP port to listen on.
            open_browser: If ``True``, open the browser.
            log_level: Uvicorn log level string.
        """
        from agent_viz.server import start_server  # noqa: PLC0415

        start_server(
            self.tracer,
            host=host,
            port=port,
            log_level=log_level,
            open_browser=open_browser,
            daemon=True,
        )

        delays = self._compute_delays()
        n_events = len(self.replay.events)

        for idx, (event, delay) in enumerate(zip(self.replay.events, delays), start=1):
            if delay > 0:
                await asyncio.sleep(delay)
            self._emit_event(event)
            if idx % 10 == 0 or idx == n_events:
                logger.debug("Replay: %d/%d events emitted.", idx, n_events)

        logger.info("Async replay complete: %d events.", n_events)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit_event(self, event: AnyEvent) -> None:
        """Push a single event directly onto the Tracer's async queue.

        We bypass the normal Tracer emission methods and push directly to
        the queue so that the SessionStore is not double-populated (the
        replay events are not re-added to the store).

        Args:
            event: The event instance to re-emit.
        """
        try:
            queue = self.tracer.get_queue()
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(
                    "[replay] Event queue full; dropping event %s (%s).",
                    event.event_id,
                    event.event_type,
                )
        except RuntimeError:
            # No running event loop – cannot enqueue.
            logger.warning(
                "[replay] No running event loop; cannot enqueue event %s.",
                event.event_id,
            )

    def _wait_for_client(
        self,
        host: str,
        port: int,
        *,
        timeout: float = 30.0,
    ) -> None:
        """Poll the /health endpoint until a browser client connects.

        Uses the ``client_count`` field from the ``/health`` REST endpoint.
        Falls back to a simple timeout if the endpoint is unavailable.

        Args:
            host: Server hostname.
            port: Server port.
            timeout: Maximum seconds to wait.
        """
        import urllib.error  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415

        url = f"http://{host}:{port}/health"
        deadline = time.monotonic() + timeout
        print(
            f"[agent_viz replay] Waiting for browser client at http://{host}:{port} "
            f"(timeout: {timeout:.0f}s)…"
        )

        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310
                    data = json.loads(resp.read())
                    if data.get("client_count", 0) > 0:
                        print("[agent_viz replay] Browser client connected.")
                        return
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)

        print(
            "[agent_viz replay] No client connected within timeout; proceeding with replay."
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def event_count(self) -> int:
        """Total number of events in the loaded replay."""
        return len(self.replay.events)

    @property
    def session_id(self) -> str:
        """Session identifier from the loaded replay."""
        return self.replay.session_id

    @property
    def session_name(self) -> str:
        """Human-readable session name from the loaded replay."""
        return self.replay.session_name

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ReplayLoader(session_id={self.session_id!r}, "
            f"events={self.event_count}, "
            f"speed={self._speed}x)"
        )


# ---------------------------------------------------------------------------
# Static validation helper
# ---------------------------------------------------------------------------

def validate_replay_file(path: str | Path) -> SessionReplay:
    """Load and validate a replay JSON file without starting a server.

    Useful for pre-flight checks in tests and scripts.

    Args:
        path: File system path to the JSON replay file.

    Returns:
        A validated :class:`~agent_viz.event_model.SessionReplay` instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file fails validation.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Replay file not found: {file_path}")

    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not read file: {exc}") from exc

    try:
        return SessionReplay.from_json(raw)
    except Exception as exc:
        raise ValueError(f"Replay validation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="agent-viz-replay",
        description=(
            "Load a JSON session replay file and re-stream events to a "
            "connected browser via the agent_viz WebSocket server."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  agent-viz-replay session.json\n"
            "  agent-viz-replay session.json --speed 2.0 --port 9000\n"
            "  agent-viz-replay session.json --delay 0.5 --open-browser\n"
            "  agent-viz-replay session.json --instant\n"
        ),
    )

    parser.add_argument(
        "file",
        metavar="FILE",
        help="Path to the JSON session replay file.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        metavar="HOST",
        help="Server bind address (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        metavar="PORT",
        help="Server TCP port (default: 8765).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        metavar="MULTIPLIER",
        help=(
            "Playback speed multiplier. 1.0 = real-time, 2.0 = double speed, "
            "0.0 = instant (default: 1.0). Ignored when --delay or --instant is set."
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "Fixed delay in seconds between each emitted event. "
            "Overrides --speed when set."
        ),
    )
    parser.add_argument(
        "--instant",
        action="store_true",
        default=False,
        help="Emit all events instantly with no delay. Equivalent to --delay 0.",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        default=False,
        help="Open the default browser after the server starts.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        default=False,
        help="Do not wait for a browser client before starting replay.",
    )
    parser.add_argument(
        "--client-timeout",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="Maximum seconds to wait for a client connection (default: 30).",
    )
    parser.add_argument(
        "--log-level",
        default="warning",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Uvicorn log level (default: warning).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        default=False,
        help="Validate the replay file and print a summary without starting the server.",
    )

    return parser


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """CLI entry-point for the ``agent-viz-replay`` command.

    Parses command-line arguments, loads and validates the replay file, and
    starts the WebSocket server to replay events to connected browser clients.

    Args:
        argv: Optional list of argument strings.  Uses ``sys.argv[1:]`` when
            ``None``.

    Returns:
        Integer exit code (``0`` for success, non-zero for failure).
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    parser = _build_parser()
    args = parser.parse_args(argv)

    # --- Load and validate file ------------------------------------------
    try:
        replay = validate_replay_file(args.file)
    except FileNotFoundError as exc:
        print(f"[agent-viz-replay] Error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"[agent-viz-replay] Validation error: {exc}", file=sys.stderr)
        return 1

    # --- Validate-only mode ----------------------------------------------
    if args.validate_only:
        print(f"File:          {args.file}")
        print(f"Session ID:    {replay.session_id}")
        print(f"Session name:  {replay.session_name or '(unnamed)'}")
        print(f"Schema version:{replay.schema_version}")
        print(f"Events:        {len(replay.events)}")
        print(f"Started at:    {replay.started_at.isoformat()}")
        print(f"Ended at:      {replay.ended_at.isoformat() if replay.ended_at else 'n/a'}")
        if replay.summary:
            print("Summary:")
            for k, v in replay.summary.items():
                print(f"  {k}: {v}")
        print("[agent-viz-replay] File is valid.")
        return 0

    # --- Determine delay strategy -----------------------------------------
    inter_event_delay: float | None = None
    speed = args.speed

    if args.instant:
        inter_event_delay = 0.0
    elif args.delay is not None:
        inter_event_delay = max(0.0, args.delay)

    # --- Build loader and run ---------------------------------------------
    loader = ReplayLoader(
        replay,
        speed=speed,
        inter_event_delay=inter_event_delay,
    )

    try:
        loader.run_replay(
            host=args.host,
            port=args.port,
            open_browser=args.open_browser,
            log_level=args.log_level,
            wait_for_client=not args.no_wait,
            client_wait_timeout=args.client_timeout,
        )
    except KeyboardInterrupt:
        print("\n[agent-viz-replay] Interrupted by user.")
    except Exception as exc:  # noqa: BLE001
        print(f"[agent-viz-replay] Fatal error: {exc}", file=sys.stderr)
        logger.exception("Fatal error during replay.")
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

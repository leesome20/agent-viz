# agent_viz

> Real-time visualization of AI agent workflows — watch your agent think, step by step.

agent_viz is a developer tool that renders AI agent decision trees as live, interactive graphs directly in your browser. Instrument your existing Python agent with a single `Tracer` object, then watch tool calls, LLM decisions, and node transitions appear in real time over WebSockets. When a session ends, export it as a portable JSON file for sharing, post-mortems, or offline replay.

---

## Quick Start

**Install**

```bash
pip install agent_viz
```

**Instrument your agent and launch the visualizer**

```python
from agent_viz import Tracer, start_server

tracer = Tracer(session_name="my_agent_run")
start_server(tracer, host="127.0.0.1", port=8765)

# Open http://127.0.0.1:8765 in your browser, then run your agent:
with tracer.span("root", label="Agent Start") as node_id:
    tracer.llm_response(node_id, model="gpt-4", response="Analyzing query...")
    tracer.tool_call(node_id, tool_name="web_search", inputs={"q": "latest AI papers"})
```

Open **http://127.0.0.1:8765** — the graph builds live as events arrive.

**Run the demo**

```bash
python examples/demo_agent.py
```

---

## Features

- **Zero-friction SDK** — instrument any Python agent with a single `Tracer` object and one-line event calls (`node_start`, `tool_call`, `llm_response`, `node_end`, `error`). No monkey-patching, no framework hooks.
- **Live WebSocket graph** — the browser UI auto-updates the decision tree in real time with color-coded node types, edge labels, and a sidebar event log.
- **Session export & replay** — save any agent run as a portable JSON file and replay it in the visualizer at adjustable speed for post-mortems and team sharing.
- **Structured event model** — all events carry timestamps, parent node references, metadata payloads, and status fields, enabling rich graph layout and filtering.
- **Framework-agnostic** — works with LangChain, AutoGen, custom loops, or any Python agent via a pure async queue. No guardrails, no behavior changes.

---

## Usage Examples

### Basic instrumentation

```python
from agent_viz import Tracer, start_server

tracer = Tracer(session_name="research_agent")
start_server(tracer, port=8765)

# Context manager — node_end is called automatically
with tracer.span("step_1", label="Fetch Data") as node_id:
    tracer.tool_call(
        node_id,
        tool_name="http_get",
        inputs={"url": "https://api.example.com/data"},
        outputs={"status": 200, "items": 42},
    )
    tracer.llm_response(
        node_id,
        model="gpt-4o",
        prompt="Summarize the results.",
        response="Found 42 relevant items.",
    )
```

### Manual event calls

```python
node_id = tracer.node_start("plan", label="Planning Step", metadata={"strategy": "react"})

try:
    result = my_tool.run()
    tracer.tool_call(node_id, tool_name="my_tool", outputs=result)
    tracer.node_end(node_id, status="success")
except Exception as exc:
    tracer.error(node_id, message=str(exc))
    tracer.node_end(node_id, status="error")
```

### Replay a saved session

```bash
# From the CLI
agent-viz-replay demo_session.json --speed 1.5 --port 8765

# Or via Python module
python -m agent_viz.replay demo_session.json --speed 0.5
```

### Export the current session via REST

```bash
curl http://127.0.0.1:8765/export > session_replay.json
```

---

## Project Structure

```
agent_viz/
├── __init__.py          # Public SDK: Tracer, start_server, event type constants
├── tracer.py            # Tracer class — emits typed events into the async queue
├── server.py            # FastAPI app: WebSocket endpoint, static UI, REST export
├── session.py           # SessionStore — accumulates events, infers edges, serializes
├── event_model.py       # Pydantic models for all events and SessionReplay schema
├── replay.py            # ReplayLoader — re-streams JSON session files to the browser
└── static/
    ├── index.html       # Single-page UI: live graph + export/replay controls
    ├── app.js           # WebSocket client, Vis.js graph builder, replay loader
    └── style.css        # Minimal dark-theme stylesheet

examples/
└── demo_agent.py        # Self-contained demo simulating a tool-calling agent

tests/
├── test_tracer.py       # Tracer event emission and queue behavior
├── test_session.py      # SessionStore accumulation, edge inference, serialization
├── test_replay.py       # Replay loading, validation, and event re-emission
└── test_event_model.py  # Pydantic event model and SessionReplay validation

pyproject.toml           # Project metadata, dependencies, and build configuration
```

---

## Configuration

### `start_server()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tracer` | `Tracer` | required | The `Tracer` instance to stream events from |
| `host` | `str` | `"127.0.0.1"` | Host address to bind the server to |
| `port` | `int` | `8765` | Port to listen on |
| `open_browser` | `bool` | `True` | Auto-open the UI in your default browser |

### `Tracer()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_name` | `str` | `"session"` | Human-readable label for the session |
| `session_id` | `str \| None` | auto-generated UUID | Override the session identifier |

### `agent-viz-replay` CLI

```
usage: agent-viz-replay <replay_file.json> [--speed FLOAT] [--port INT] [--host STR]

  --speed FLOAT   Playback speed multiplier (default: 1.0, faster = higher value)
  --port  INT     Server port (default: 8765)
  --host  STR     Server host (default: 127.0.0.1)
```

### Event types (module-level constants)

```python
from agent_viz import (
    EVENT_NODE_START,    # "node_start"
    EVENT_NODE_END,      # "node_end"
    EVENT_TOOL_CALL,     # "tool_call"
    EVENT_LLM_RESPONSE,  # "llm_response"
    EVENT_ERROR,         # "error"
)
```

---

## License

MIT © agent_viz contributors

---

*Built with [Jitter](https://github.com/jitter-ai) — an AI agent that ships code daily.*

"""Self-contained demo agent for agent_viz.

This script simulates a simple tool-calling agent that exercises all
Tracer event types (node_start, node_end, tool_call, llm_response, error)
and produces a live visualization in the browser.

Usage::

    python examples/demo_agent.py

The script will:
1. Start the agent_viz WebSocket server on http://127.0.0.1:8765
2. Run a simulated multi-step agent with tool calls and LLM responses
3. Export the full session replay to ``demo_session.json`` on completion
4. Keep the server alive so you can inspect the final graph

Press Ctrl-C to stop.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure the package root is importable when run directly from the repo
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent_viz import (
    EVENT_ERROR,
    EVENT_LLM_RESPONSE,
    EVENT_NODE_END,
    EVENT_NODE_START,
    EVENT_TOOL_CALL,
    Tracer,
    start_server,
)

# ---------------------------------------------------------------------------
# Simulated external tools
# ---------------------------------------------------------------------------


def tool_web_search(query: str) -> dict[str, Any]:
    """Simulate a web search tool call.

    Args:
        query: The search query string.

    Returns:
        A dictionary containing simulated search results.
    """
    time.sleep(random.uniform(0.1, 0.3))  # simulate network latency
    results = [
        {"title": f"Result {i} for '{query}'", "url": f"https://example.com/{i}"}
        for i in range(1, 4)
    ]
    return {"query": query, "results": results, "total": len(results)}


def tool_calculator(expression: str) -> dict[str, Any]:
    """Simulate a safe calculator tool.

    Args:
        expression: A simple arithmetic expression string.

    Returns:
        A dictionary containing the evaluated result.

    Raises:
        ValueError: If the expression cannot be safely evaluated.
    """
    time.sleep(random.uniform(0.05, 0.15))
    # Only allow simple digit arithmetic for safety in demo
    allowed_chars = set("0123456789 +-*/().")
    if not all(c in allowed_chars for c in expression):
        raise ValueError(f"Unsafe expression: {expression!r}")
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as exc:
        raise ValueError(f"Could not evaluate {expression!r}: {exc}") from exc


def tool_summarise(text: str, max_words: int = 30) -> dict[str, Any]:
    """Simulate a text summarisation tool.

    Args:
        text: The text to summarise.
        max_words: Maximum words in the summary.

    Returns:
        A dictionary containing the summary.
    """
    time.sleep(random.uniform(0.05, 0.2))
    words = text.split()
    summary = " ".join(words[:max_words]) + ("…" if len(words) > max_words else "")
    return {"summary": summary, "original_words": len(words), "summary_words": len(words[:max_words])}


# ---------------------------------------------------------------------------
# Simulated LLM calls
# ---------------------------------------------------------------------------

_LLM_RESPONSES: dict[str, str] = {
    "plan": (
        "I'll break this task into three steps: (1) search for relevant information, "
        "(2) perform any necessary calculations, and (3) summarise the findings."
    ),
    "analyse": (
        "Based on the search results, the key findings are: the topic is well-documented "
        "with multiple authoritative sources. I should cross-reference the numerical data."
    ),
    "calculate_intent": (
        "I need to calculate the growth rate. I'll use the calculator tool with the "
        "values extracted from the search results."
    ),
    "synthesise": (
        "Combining the search results and calculations: the overall trend shows a positive "
        "trajectory with approximately 15% growth. The summarised data confirms this."
    ),
    "final_answer": (
        "The answer to your question is: Based on my research and analysis, the topic "
        "shows strong growth patterns with reliable data from multiple sources. "
        "The calculated growth rate is 15.3% and the sources are credible."
    ),
}


def simulated_llm(
    prompt: str,
    model: str = "demo-gpt-4o",
    response_key: str = "plan",
) -> tuple[str, int, int]:
    """Simulate an LLM API call.

    Args:
        prompt: The prompt to send to the model.
        model: The model identifier.
        response_key: Key into the canned response dictionary.

    Returns:
        A tuple of (response_text, prompt_tokens, completion_tokens).
    """
    time.sleep(random.uniform(0.2, 0.5))  # simulate inference latency
    response = _LLM_RESPONSES.get(response_key, "I have completed the task.")
    prompt_tokens = len(prompt.split()) + random.randint(10, 50)
    completion_tokens = len(response.split()) + random.randint(5, 20)
    return response, prompt_tokens, completion_tokens


# ---------------------------------------------------------------------------
# Agent workflow steps
# ---------------------------------------------------------------------------


def step_plan(tracer: Tracer, root_node_id: str) -> str:
    """Agent planning step: ask the LLM to devise a plan.

    Args:
        tracer: Active Tracer instance.
        root_node_id: Parent node identifier.

    Returns:
        The node ID of the planning step.
    """
    with tracer.span(
        label="Plan",
        node_type="llm",
        parent_node_id=root_node_id,
        metadata={"step": 1},
    ) as node_id:
        prompt = "You are a research agent. Plan how to answer: 'What is the growth rate of AI adoption?'"
        t0 = time.monotonic()
        response, p_tok, c_tok = simulated_llm(prompt, response_key="plan")
        duration_ms = (time.monotonic() - t0) * 1000

        tracer.llm_response(
            node_id,
            model="demo-gpt-4o",
            response=response,
            prompt=prompt,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            duration_ms=round(duration_ms, 2),
            metadata={"temperature": 0.7},
        )
    return node_id


def step_search(tracer: Tracer, plan_node_id: str) -> str:
    """Agent search step: invoke the web search tool.

    Args:
        tracer: Active Tracer instance.
        plan_node_id: Parent node identifier (the planning step).

    Returns:
        The node ID of the search step.
    """
    with tracer.span(
        label="Web Search",
        node_type="tool",
        parent_node_id=plan_node_id,
        metadata={"step": 2},
    ) as node_id:
        query = "AI adoption growth rate 2024"
        t0 = time.monotonic()
        results = tool_web_search(query)
        duration_ms = (time.monotonic() - t0) * 1000

        tracer.tool_call(
            node_id,
            tool_name="web_search",
            inputs={"query": query},
            outputs=results,
            duration_ms=round(duration_ms, 2),
            metadata={"engine": "demo-search"},
        )

        # LLM analyses the search results
        prompt = f"Analyse these search results: {json.dumps(results['results'][:2])}"
        t1 = time.monotonic()
        analysis, p_tok, c_tok = simulated_llm(prompt, response_key="analyse")
        dur2 = (time.monotonic() - t1) * 1000

        tracer.llm_response(
            node_id,
            model="demo-gpt-4o",
            response=analysis,
            prompt=prompt,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            duration_ms=round(dur2, 2),
        )
    return node_id


def step_calculate(tracer: Tracer, search_node_id: str) -> str:
    """Agent calculation step: use the calculator tool.

    Args:
        tracer: Active Tracer instance.
        search_node_id: Parent node identifier (the search step).

    Returns:
        The node ID of the calculation step.
    """
    with tracer.span(
        label="Calculate Growth Rate",
        node_type="tool",
        parent_node_id=search_node_id,
        metadata={"step": 3},
    ) as node_id:
        # LLM decides what to calculate
        prompt = "Based on the search results, what calculation should I perform?"
        t0 = time.monotonic()
        intent, p_tok, c_tok = simulated_llm(prompt, response_key="calculate_intent")
        dur0 = (time.monotonic() - t0) * 1000

        tracer.llm_response(
            node_id,
            model="demo-gpt-4o",
            response=intent,
            prompt=prompt,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            duration_ms=round(dur0, 2),
        )

        # Run the calculator
        expression = "(234 - 203) / 203 * 100"
        t1 = time.monotonic()
        calc_result = tool_calculator(expression)
        dur1 = (time.monotonic() - t1) * 1000

        tracer.tool_call(
            node_id,
            tool_name="calculator",
            inputs={"expression": expression},
            outputs=calc_result,
            duration_ms=round(dur1, 2),
            metadata={"unit": "percent"},
        )
    return node_id


def step_summarise(tracer: Tracer, calc_node_id: str) -> str:
    """Agent summarisation step: call the summarise tool.

    Args:
        tracer: Active Tracer instance.
        calc_node_id: Parent node identifier (the calculation step).

    Returns:
        The node ID of the summarisation step.
    """
    with tracer.span(
        label="Summarise Findings",
        node_type="tool",
        parent_node_id=calc_node_id,
        metadata={"step": 4},
    ) as node_id:
        text = (
            "AI adoption has grown significantly. Multiple data points confirm a "
            "positive trajectory. Market research from 2024 indicates strong uptake "
            "across enterprise and consumer segments with an estimated 15.3% YoY growth. "
            "Sources include industry reports, analyst estimates, and survey data."
        )
        t0 = time.monotonic()
        summary_result = tool_summarise(text, max_words=25)
        dur = (time.monotonic() - t0) * 1000

        tracer.tool_call(
            node_id,
            tool_name="summarise",
            inputs={"text": text[:80] + "…", "max_words": 25},
            outputs=summary_result,
            duration_ms=round(dur, 2),
        )

        # LLM synthesises everything
        prompt = f"Synthesise the findings: search + calculation + summary = {summary_result['summary']}"
        t1 = time.monotonic()
        synthesis, p_tok, c_tok = simulated_llm(prompt, response_key="synthesise")
        dur2 = (time.monotonic() - t1) * 1000

        tracer.llm_response(
            node_id,
            model="demo-gpt-4o",
            response=synthesis,
            prompt=prompt,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            duration_ms=round(dur2, 2),
        )
    return node_id


def step_error_demo(tracer: Tracer, parent_node_id: str) -> str:
    """Demonstrate error event capture with a deliberately failing tool call.

    Args:
        tracer: Active Tracer instance.
        parent_node_id: Parent node identifier.

    Returns:
        The node ID of the error demonstration step.
    """
    node_id = tracer.node_start(
        label="Risky Tool Call",
        node_type="tool",
        parent_node_id=parent_node_id,
        metadata={"step": "5-error-demo", "expected": "error"},
    )
    try:
        # Intentionally trigger a ValueError in the calculator
        bad_expression = "import os; os.system('echo hacked')"
        t0 = time.monotonic()
        result = tool_calculator(bad_expression)
        dur = (time.monotonic() - t0) * 1000
        tracer.tool_call(
            node_id,
            tool_name="calculator",
            inputs={"expression": bad_expression},
            outputs=result,
            duration_ms=round(dur, 2),
        )
        tracer.node_end(node_id, status="success")
    except ValueError as exc:
        tracer.error_from_exception(node_id, exc, metadata={"expression": bad_expression})
        tracer.node_end(node_id, status="error", output=str(exc))
    return node_id


def step_final_answer(tracer: Tracer, parent_node_id: str) -> str:
    """Agent final answer step: LLM produces the answer to the user.

    Args:
        tracer: Active Tracer instance.
        parent_node_id: Parent node identifier.

    Returns:
        The node ID of the final answer step.
    """
    with tracer.span(
        label="Final Answer",
        node_type="llm",
        parent_node_id=parent_node_id,
        metadata={"step": 6, "final": True},
    ) as node_id:
        prompt = (
            "Given all the research and analysis, provide a concise final answer to: "
            "'What is the growth rate of AI adoption?'"
        )
        t0 = time.monotonic()
        answer, p_tok, c_tok = simulated_llm(prompt, response_key="final_answer")
        duration_ms = (time.monotonic() - t0) * 1000

        tracer.llm_response(
            node_id,
            model="demo-gpt-4o",
            response=answer,
            prompt=prompt,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            duration_ms=round(duration_ms, 2),
            metadata={"finish_reason": "stop"},
        )
    return node_id


# ---------------------------------------------------------------------------
# Main agent runner
# ---------------------------------------------------------------------------


def run_demo_agent(tracer: Tracer) -> None:
    """Execute the full demo agent workflow.

    Runs a six-step simulated agent that exercises every Tracer event type:
    - node_start / node_end (via span context manager)
    - llm_response (planning, analysis, synthesis, final answer)
    - tool_call (web search, calculator, summarise)
    - error (deliberately failed calculator call)

    Args:
        tracer: An active :class:`~agent_viz.Tracer` instance connected to
            the running WebSocket server.
    """
    print("[demo_agent] Starting agent workflow…")

    # Root orchestrator node
    root_id = tracer.node_start(
        label="Agent: AI Adoption Research",
        node_type="default",
        metadata={"user_query": "What is the growth rate of AI adoption?", "version": "1.0"},
    )

    try:
        print("[demo_agent] Step 1/6: Planning…")
        plan_id = step_plan(tracer, root_id)
        time.sleep(0.1)

        print("[demo_agent] Step 2/6: Web search…")
        search_id = step_search(tracer, plan_id)
        time.sleep(0.1)

        print("[demo_agent] Step 3/6: Calculating growth rate…")
        calc_id = step_calculate(tracer, search_id)
        time.sleep(0.1)

        print("[demo_agent] Step 4/6: Summarising findings…")
        summ_id = step_summarise(tracer, calc_id)
        time.sleep(0.1)

        print("[demo_agent] Step 5/6: Demonstrating error handling…")
        _error_id = step_error_demo(tracer, summ_id)
        time.sleep(0.1)

        print("[demo_agent] Step 6/6: Generating final answer…")
        _final_id = step_final_answer(tracer, summ_id)
        time.sleep(0.1)

        tracer.node_end(
            root_id,
            status="success",
            output="Agent run completed successfully.",
        )
        print("[demo_agent] Agent workflow completed successfully.")

    except Exception as exc:
        tracer.error_from_exception(root_id, exc)
        tracer.node_end(root_id, status="error")
        print(f"[demo_agent] Agent workflow failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the agent_viz server and run the demo agent.

    Opens the dashboard URL in the console, runs the simulated agent, exports
    the session replay to ``demo_session.json``, then keeps the server alive
    for interactive inspection.
    """
    host = "127.0.0.1"
    port = 8765
    output_file = Path("demo_session.json")

    # ------------------------------------------------------------------ #
    # 1. Create Tracer                                                     #
    # ------------------------------------------------------------------ #
    tracer = Tracer(session_name="Demo Agent – AI Adoption Research")
    print(f"[demo_agent] Session ID: {tracer.session_id}")
    print(f"[demo_agent] Event constants available: {EVENT_NODE_START}, {EVENT_NODE_END}, "
          f"{EVENT_TOOL_CALL}, {EVENT_LLM_RESPONSE}, {EVENT_ERROR}")

    # ------------------------------------------------------------------ #
    # 2. Start the WebSocket server                                        #
    # ------------------------------------------------------------------ #
    print(f"[demo_agent] Starting server at http://{host}:{port} …")
    _server_thread = start_server(
        tracer,
        host=host,
        port=port,
        log_level="warning",
        open_browser=False,
        daemon=True,
    )
    print(f"[demo_agent] Dashboard: http://{host}:{port}")
    print("[demo_agent] Open the URL above in your browser, then watch the graph build live.")
    print("[demo_agent] Starting agent in 2 seconds…")
    time.sleep(2.0)

    # ------------------------------------------------------------------ #
    # 3. Run the demo agent                                               #
    # ------------------------------------------------------------------ #
    try:
        run_demo_agent(tracer)
    except Exception as exc:  # noqa: BLE001
        print(f"[demo_agent] Agent raised an unexpected error: {exc}")

    # ------------------------------------------------------------------ #
    # 4. Export the session replay                                         #
    # ------------------------------------------------------------------ #
    replay_json = tracer.close()
    output_file.write_text(replay_json, encoding="utf-8")
    print(f"[demo_agent] Session replay saved to: {output_file.resolve()}")

    n_events = len(tracer.session)
    n_nodes = len(tracer.session.node_ids)
    n_edges = len(tracer.session.edges)
    print(
        f"[demo_agent] Summary: {n_events} events, {n_nodes} nodes, {n_edges} edges."
    )
    summary = tracer.session.build_summary()
    print("[demo_agent] Event type breakdown:")
    for etype, count in sorted(summary.get("event_type_counts", {}).items()):
        print(f"  {etype}: {count}")

    # ------------------------------------------------------------------ #
    # 5. Keep server alive for inspection                                  #
    # ------------------------------------------------------------------ #
    print("[demo_agent] Server still running – inspect the graph at "
          f"http://{host}:{port}")
    print("[demo_agent] Press Ctrl-C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[demo_agent] Shutting down. Goodbye!")


if __name__ == "__main__":
    main()

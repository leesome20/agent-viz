/**
 * agent_viz – client-side application
 *
 * Responsibilities:
 * - Open and maintain a WebSocket connection to the server
 * - Handle incoming events (single events and historical batches)
 * - Incrementally build and update a Vis.js network graph
 * - Maintain a sidebar event log
 * - Implement JSON export of the current session
 * - Implement JSON replay loading with configurable playback speed
 */

/* =========================================================================
   Constants & configuration
   ========================================================================= */

/** Node background colours keyed by node_type. */
const NODE_COLORS = {
  default:  { background: "#4a9eff", border: "#2979d9", highlight: { background: "#74b6ff", border: "#2979d9" } },
  llm:      { background: "#a78bfa", border: "#7c3aed", highlight: { background: "#c4b5fd", border: "#7c3aed" } },
  tool:     { background: "#34d399", border: "#059669", highlight: { background: "#6ee7b7", border: "#059669" } },
  router:   { background: "#fbbf24", border: "#d97706", highlight: { background: "#fcd34d", border: "#d97706" } },
  error:    { background: "#f87171", border: "#dc2626", highlight: { background: "#fca5a5", border: "#dc2626" } },
};

/** Status overlay colour for node borders. */
const STATUS_BORDER = {
  success: "#10b981",
  error:   "#ef4444",
  skipped: "#6b7280",
};

/** Maximum number of entries kept in the sidebar event log. */
const MAX_LOG_ENTRIES = 300;

/** WebSocket reconnect delay in milliseconds. */
const WS_RECONNECT_DELAY_MS = 3000;

/* =========================================================================
   State
   ========================================================================= */

/** @type {vis.DataSet<object>} */
let visNodes;
/** @type {vis.DataSet<object>} */
let visEdges;
/** @type {vis.Network|null} */
let network = null;

/** All raw events accumulated in this browser session. */
const allEvents = [];

/** Map from node_id → accumulated node metadata for the detail panel. */
const nodeMetaMap = {};

/** Pending replay data while modal is open. */
let pendingReplayEvents = null;

/** Replay interval handle. */
let replayInterval = null;

/** WebSocket instance. */
let ws = null;

/* =========================================================================
   Vis.js initialisation
   ========================================================================= */

function initGraph() {
  visNodes = new vis.DataSet();
  visEdges = new vis.DataSet();

  const container = document.getElementById("graph-canvas");
  const data = { nodes: visNodes, edges: visEdges };
  const options = {
    layout: {
      hierarchical: {
        enabled: true,
        direction: "UD",
        sortMethod: "directed",
        levelSeparation: 100,
        nodeSpacing: 160,
        treeSpacing: 200,
      },
    },
    physics: { enabled: false },
    edges: {
      arrows: { to: { enabled: true, scaleFactor: 0.7 } },
      color: { color: "#4b5563", highlight: "#9ca3af" },
      font: { color: "#9ca3af", size: 11, face: "monospace" },
      smooth: { type: "cubicBezier", forceDirection: "vertical", roundness: 0.4 },
      width: 1.5,
    },
    nodes: {
      shape: "box",
      margin: { top: 8, right: 12, bottom: 8, left: 12 },
      font: { color: "#f9fafb", size: 13, face: "monospace" },
      borderWidth: 2,
      shadow: { enabled: true, color: "rgba(0,0,0,0.4)", size: 6, x: 2, y: 2 },
    },
    interaction: {
      hover: true,
      tooltipDelay: 200,
      navigationButtons: false,
      keyboard: false,
    },
    autoResize: true,
  };

  network = new vis.Network(container, data, options);

  network.on("click", (params) => {
    if (params.nodes.length > 0) {
      showNodeDetail(params.nodes[0]);
    }
  });

  network.on("hoverNode", () => {
    container.style.cursor = "pointer";
  });

  network.on("blurNode", () => {
    container.style.cursor = "default";
  });
}

/* =========================================================================
   Event processing
   ========================================================================= */

/**
 * Process a single incoming event object (already parsed from JSON).
 * Updates the graph, event log, and internal state.
 *
 * @param {object} event – parsed event object
 */
function processEvent(event) {
  allEvents.push(event);
  appendEventLog(event);
  updateStats();

  // Accumulate node metadata for the detail panel
  if (!nodeMetaMap[event.node_id]) {
    nodeMetaMap[event.node_id] = { node_id: event.node_id, events: [] };
  }
  nodeMetaMap[event.node_id].events.push(event);

  switch (event.event_type) {
    case "node_start":
      handleNodeStart(event);
      break;
    case "node_end":
      handleNodeEnd(event);
      break;
    case "tool_call":
      handleToolCall(event);
      break;
    case "llm_response":
      handleLLMResponse(event);
      break;
    case "error":
      handleError(event);
      break;
    default:
      break;
  }
}

/** @param {object} event */
function handleNodeStart(event) {
  const colorSet = NODE_COLORS[event.node_type] || NODE_COLORS.default;
  const nodeData = {
    id: event.node_id,
    label: event.label || event.node_id,
    color: { ...colorSet },
    title: buildNodeTooltip(event),
    _nodeType: event.node_type || "default",
  };

  if (visNodes.get(event.node_id)) {
    visNodes.update(nodeData);
  } else {
    visNodes.add(nodeData);
  }

  // Add edge if parent exists
  if (event.parent_node_id) {
    const edgeId = `${event.parent_node_id}__${event.node_id}`;
    if (!visEdges.get(edgeId)) {
      visEdges.add({
        id: edgeId,
        from: event.parent_node_id,
        to: event.node_id,
        label: event.label || "",
      });
    }
  }
}

/** @param {object} event */
function handleNodeEnd(event) {
  const existing = visNodes.get(event.node_id);
  if (existing) {
    const borderColor = STATUS_BORDER[event.status] || STATUS_BORDER.success;
    visNodes.update({
      id: event.node_id,
      color: {
        ...existing.color,
        border: borderColor,
        highlight: { ...existing.color.highlight, border: borderColor },
      },
      title: buildNodeTooltip(event),
    });
  }
}

/** @param {object} event */
function handleToolCall(event) {
  // Annotate the node with a tool call badge if the node exists
  const existing = visNodes.get(event.node_id);
  if (existing) {
    const currentLabel = existing.label || event.node_id;
    const badge = `\n🔧 ${event.tool_name}`;
    if (!currentLabel.includes(badge.trim())) {
      visNodes.update({
        id: event.node_id,
        label: currentLabel + badge,
      });
    }
  }
}

/** @param {object} event */
function handleLLMResponse(event) {
  const existing = visNodes.get(event.node_id);
  if (existing) {
    const currentLabel = existing.label || event.node_id;
    const badge = `\n🤖 ${event.model}`;
    if (!currentLabel.includes(badge.trim())) {
      visNodes.update({
        id: event.node_id,
        label: currentLabel + badge,
      });
    }
  }
}

/** @param {object} event */
function handleError(event) {
  const existing = visNodes.get(event.node_id);
  if (existing) {
    visNodes.update({
      id: event.node_id,
      color: {
        ...NODE_COLORS.error,
      },
    });
  } else {
    // Create a node for the error if it doesn't exist yet
    visNodes.add({
      id: event.node_id,
      label: `⚠ ${event.error_type}`,
      color: { ...NODE_COLORS.error },
      title: `Error: ${event.message}`,
      _nodeType: "error",
    });
  }
}

/**
 * Build an HTML tooltip string for a node event.
 *
 * @param {object} event
 * @returns {string}
 */
function buildNodeTooltip(event) {
  const lines = [
    `<strong>${event.label || event.node_id}</strong>`,
    `<code>id: ${event.node_id}</code>`,
  ];
  if (event.node_type) lines.push(`type: ${event.node_type}`);
  if (event.status)    lines.push(`status: ${event.status}`);
  if (event.timestamp) lines.push(`at: ${new Date(event.timestamp).toLocaleTimeString()}`);
  return lines.join("<br/>");
}

/* =========================================================================
   Event log sidebar
   ========================================================================= */

const EVENT_ICONS = {
  node_start:   "▶",
  node_end:     "■",
  tool_call:    "🔧",
  llm_response: "🤖",
  error:        "⚠",
};

/** @param {object} event */
function appendEventLog(event) {
  const log = document.getElementById("event-log");
  if (!log) return;

  // Prune old entries
  while (log.children.length >= MAX_LOG_ENTRIES) {
    log.removeChild(log.firstChild);
  }

  const row = document.createElement("div");
  row.className = `log-row log-${event.event_type}`;
  row.dataset.nodeId = event.node_id;

  const icon  = EVENT_ICONS[event.event_type] || "•";
  const time  = new Date(event.timestamp).toLocaleTimeString([], { hour12: false });
  const label = getEventLabel(event);

  row.innerHTML =
    `<span class="log-icon">${icon}</span>` +
    `<span class="log-label">${escapeHtml(label)}</span>` +
    `<span class="log-time">${time}</span>`;

  row.addEventListener("click", () => showNodeDetail(event.node_id));
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
}

/**
 * Extract a short human-readable label for an event row.
 *
 * @param {object} event
 * @returns {string}
 */
function getEventLabel(event) {
  switch (event.event_type) {
    case "node_start":   return event.label || event.node_id;
    case "node_end":     return `${event.node_id.slice(0, 8)}… [${event.status}]`;
    case "tool_call":    return `tool: ${event.tool_name}`;
    case "llm_response": return `llm: ${event.model} – ${(event.response || "").slice(0, 40)}`;
    case "error":        return `error: ${event.error_type} – ${event.message}`;
    default:             return event.event_type;
  }
}

/* =========================================================================
   Node detail panel
   ========================================================================= */

/** @param {string} nodeId */
function showNodeDetail(nodeId) {
  const card = document.getElementById("detail-card");
  const title = document.getElementById("detail-title");
  const body  = document.getElementById("detail-body");
  if (!card || !title || !body) return;

  const meta = nodeMetaMap[nodeId];
  if (!meta) return;

  title.textContent = `Node: ${nodeId.slice(0, 16)}…`;
  body.innerHTML = "";

  meta.events.forEach((ev) => {
    const section = document.createElement("div");
    section.className = "detail-event";
    section.innerHTML =
      `<div class="detail-event-type log-${ev.event_type}">${EVENT_ICONS[ev.event_type] || "•"} ${ev.event_type}</div>` +
      `<pre class="detail-json">${escapeHtml(JSON.stringify(ev, null, 2))}</pre>`;
    body.appendChild(section);
  });

  card.style.display = "flex";

  // Highlight the node in the graph
  if (network) {
    network.selectNodes([nodeId]);
  }
}

/* =========================================================================
   Stats bar
   ========================================================================= */

function updateStats() {
  const elEvents = document.getElementById("stat-events");
  const elNodes  = document.getElementById("stat-nodes");
  const elEdges  = document.getElementById("stat-edges");
  if (elEvents) elEvents.textContent = allEvents.length;
  if (elNodes)  elNodes.textContent  = visNodes ? visNodes.length : 0;
  if (elEdges)  elEdges.textContent  = visEdges ? visEdges.length : 0;
}

/* =========================================================================
   WebSocket connection
   ========================================================================= */

function connectWebSocket() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const url = `${protocol}://${location.host}/ws`;
  const statusEl = document.getElementById("ws-status");

  ws = new WebSocket(url);

  ws.onopen = () => {
    if (statusEl) {
      statusEl.textContent = "●";
      statusEl.classList.remove("ws-disconnected");
      statusEl.classList.add("ws-connected");
    }
    setSessionBadge("connected");
    console.log("[agent_viz] WebSocket connected.");
  };

  ws.onmessage = (evt) => {
    let msg;
    try {
      msg = JSON.parse(evt.data);
    } catch (e) {
      console.warn("[agent_viz] Failed to parse WS message:", evt.data);
      return;
    }

    if (msg.type === "ping") {
      // Server keepalive – nothing to do
      return;
    }

    if (msg.type === "batch") {
      // Historical replay batch on initial connect
      const events = msg.events || [];
      events.forEach(processEvent);
      setTimeout(() => network && network.fit(), 200);
      return;
    }

    // Single live event (the server sends raw event JSON without a "type" wrapper)
    if (msg.event_type) {
      processEvent(msg);
    }
  };

  ws.onerror = (err) => {
    console.error("[agent_viz] WebSocket error:", err);
  };

  ws.onclose = () => {
    if (statusEl) {
      statusEl.textContent = "●";
      statusEl.classList.remove("ws-connected");
      statusEl.classList.add("ws-disconnected");
    }
    setSessionBadge("disconnected – reconnecting…");
    console.log(`[agent_viz] WebSocket closed. Reconnecting in ${WS_RECONNECT_DELAY_MS}ms…`);
    setTimeout(connectWebSocket, WS_RECONNECT_DELAY_MS);
  };
}

/* =========================================================================
   Session badge
   ========================================================================= */

function setSessionBadge(text) {
  const el = document.getElementById("session-badge");
  if (el) el.textContent = text;
}

/**
 * Fetch session info from the REST API and update the badge.
 */
async function fetchSessionInfo() {
  try {
    const resp = await fetch("/session");
    if (!resp.ok) return;
    const data = await resp.json();
    const name = data.session_name || data.session_id || "";
    setSessionBadge(name);
  } catch (_) {
    // Non-fatal
  }
}

/* =========================================================================
   Export
   ========================================================================= */

function exportSession() {
  // Build a minimal SessionReplay-compatible object from local state
  const payload = {
    schema_version: 1,
    session_id: "browser-export",
    session_name: "browser export",
    started_at: allEvents.length > 0 ? allEvents[0].timestamp : new Date().toISOString(),
    ended_at: allEvents.length > 0 ? allEvents[allEvents.length - 1].timestamp : null,
    events: allEvents,
    summary: {
      total_events: allEvents.length,
      total_nodes: visNodes ? visNodes.length : 0,
      total_edges: visEdges ? visEdges.length : 0,
    },
  };

  // Try the server export endpoint first for a canonical replay file
  fetch("/export")
    .then((r) => (r.ok ? r.blob() : Promise.reject()))
    .then((blob) => {
      triggerDownload(blob, `session_export_${Date.now()}.json`);
    })
    .catch(() => {
      // Fallback: use local state
      const blob = new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json",
      });
      triggerDownload(blob, `session_export_${Date.now()}.json`);
    });
}

/**
 * Trigger a file download in the browser.
 *
 * @param {Blob} blob
 * @param {string} filename
 */
function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a   = document.createElement("a");
  a.href     = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, 100);
}

/* =========================================================================
   Replay loading
   ========================================================================= */

/** @param {File} file */
function loadReplayFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    let data;
    try {
      data = JSON.parse(e.target.result);
    } catch (err) {
      alert("Invalid JSON file: " + err.message);
      return;
    }

    if (!Array.isArray(data.events)) {
      alert("Invalid replay file: missing 'events' array.");
      return;
    }

    pendingReplayEvents = data.events;
    document.getElementById("replay-event-count").textContent = pendingReplayEvents.length;
    document.getElementById("replay-modal").style.display = "flex";
  };
  reader.readAsText(file);
}

function startReplay() {
  if (!pendingReplayEvents || pendingReplayEvents.length === 0) return;

  // Clear current graph
  clearGraph();

  const events   = pendingReplayEvents.slice();
  const speedMs  = parseInt(document.getElementById("replay-speed").value, 10);
  pendingReplayEvents = null;
  closeReplayModal();

  if (speedMs === 0) {
    // Instant replay
    events.forEach(processEvent);
    setTimeout(() => network && network.fit(), 100);
    return;
  }

  let idx = 0;
  replayInterval = setInterval(() => {
    if (idx >= events.length) {
      clearInterval(replayInterval);
      replayInterval = null;
      setTimeout(() => network && network.fit(), 100);
      return;
    }
    processEvent(events[idx++]);
  }, speedMs);
}

function closeReplayModal() {
  document.getElementById("replay-modal").style.display = "none";
  pendingReplayEvents = null;
}

/* =========================================================================
   Graph utilities
   ========================================================================= */

function clearGraph() {
  if (replayInterval) {
    clearInterval(replayInterval);
    replayInterval = null;
  }
  if (visNodes) visNodes.clear();
  if (visEdges) visEdges.clear();
  allEvents.length = 0;
  Object.keys(nodeMetaMap).forEach((k) => delete nodeMetaMap[k]);
  const log = document.getElementById("event-log");
  if (log) log.innerHTML = "";
  const detailCard = document.getElementById("detail-card");
  if (detailCard) detailCard.style.display = "none";
  updateStats();
}

/* =========================================================================
   Utility
   ========================================================================= */

/**
 * Escape HTML special characters to prevent XSS in innerHTML.
 *
 * @param {string} str
 * @returns {string}
 */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/* =========================================================================
   Wiring: DOM event listeners
   ========================================================================= */

document.addEventListener("DOMContentLoaded", () => {
  // Initialise Vis.js graph
  initGraph();

  // Connect WebSocket
  connectWebSocket();

  // Fetch and display session name
  fetchSessionInfo();

  // Toolbar buttons
  document.getElementById("btn-fit").addEventListener("click", () => {
    if (network) network.fit();
  });

  document.getElementById("btn-clear").addEventListener("click", () => {
    if (confirm("Clear the current graph and event log?")) {
      clearGraph();
    }
  });

  document.getElementById("btn-export").addEventListener("click", exportSession);

  // Replay file input
  document.getElementById("input-replay").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
      loadReplayFile(file);
      e.target.value = ""; // reset so same file can be loaded again
    }
  });

  // Replay modal
  document.getElementById("btn-replay-cancel").addEventListener("click", closeReplayModal);
  document.getElementById("btn-replay-start").addEventListener("click", startReplay);

  // Close detail panel
  document.getElementById("btn-close-detail").addEventListener("click", () => {
    document.getElementById("detail-card").style.display = "none";
    if (network) network.unselectAll();
  });

  // Clear event log
  document.getElementById("btn-clear-log").addEventListener("click", () => {
    const log = document.getElementById("event-log");
    if (log) log.innerHTML = "";
  });

  // Close modal on overlay click
  document.getElementById("replay-modal").addEventListener("click", (e) => {
    if (e.target === document.getElementById("replay-modal")) {
      closeReplayModal();
    }
  });
});

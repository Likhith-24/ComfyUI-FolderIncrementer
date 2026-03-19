import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * Parameter Memory — Tracks every widget change across ALL nodes.
 *
 * Features:
 *   1. Records every parameter change with timestamp (stored in node.properties)
 *   2. Hover any widget → tooltip shows: current value, previous value, default value
 *   3. Right-click any node → "Show Parameter History" or "Reset to Defaults"
 *   4. Before each queue, takes a full snapshot → can diff what changed between runs
 *   5. Sends snapshots to Python backend for persistent DB storage
 *   6. Execution-level change tracking: see what you tweaked before pressing Queue
 *
 * Data structure (per-node in node.properties._paramMemory):
 *   {
 *     defaults:  { widgetName: defaultValue, ... },
 *     history:   [ { ts, changes: { widgetName: { from, to } } }, ... ],
 *     snapshots: [ { ts, run_id, values: { widgetName: value, ... } }, ... ],
 *   }
 */

// ── Constants ────────────────────────────────────────────────────────
const MAX_HISTORY    = 100;   // max change entries per node
const MAX_SNAPSHOTS  = 50;    // max execution snapshots per node
const TOOLTIP_BG     = "#1e1e2edd";
const TOOLTIP_FG     = "#cdd6f4";
const TOOLTIP_ACCENT = "#89b4fa";
const TOOLTIP_WARN   = "#f9e2af";
const TOOLTIP_DIM    = "#6c7086";
const TOOLTIP_RADIUS = 6;
const TOOLTIP_PAD    = 8;
const TOOLTIP_MAX_W  = 280;

let _runCounter = 0;
let _preQueueSnapshots = new Map(); // nodeId → { values }

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Helpers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function _now() {
  return new Date().toISOString().replace("T", " ").slice(0, 19);
}

function _getMemory(node) {
  if (!node.properties) node.properties = {};
  if (!node.properties._paramMemory) {
    node.properties._paramMemory = {
      defaults:  {},
      history:   [],
      snapshots: [],
    };
  }
  return node.properties._paramMemory;
}

function _captureValues(node) {
  const vals = {};
  if (node.widgets) {
    for (const w of node.widgets) {
      if (w.name && w.type !== "canvas" && !w.name.startsWith("_")) {
        vals[w.name] = w.value;
      }
    }
  }
  return vals;
}

function _captureDefaults(node) {
  const defs = {};
  if (node.widgets) {
    for (const w of node.widgets) {
      if (w.name && w.type !== "canvas" && !w.name.startsWith("_")) {
        // Try multiple sources for the default value
        const d = w.options?.default ?? w.options?.defaultVal ?? w.value;
        defs[w.name] = d;
      }
    }
  }
  return defs;
}

function _recordChange(node, widgetName, oldVal, newVal) {
  const mem = _getMemory(node);
  const entry = {
    ts:   _now(),
    name: widgetName,
    from: oldVal,
    to:   newVal,
  };
  mem.history.push(entry);
  if (mem.history.length > MAX_HISTORY) {
    mem.history = mem.history.slice(-MAX_HISTORY);
  }
}

function _takeSnapshot(node, runId) {
  const mem  = _getMemory(node);
  const vals = _captureValues(node);
  const snap = {
    ts:     _now(),
    run_id: runId,
    values: vals,
  };
  mem.snapshots.push(snap);
  if (mem.snapshots.length > MAX_SNAPSHOTS) {
    mem.snapshots = mem.snapshots.slice(-MAX_SNAPSHOTS);
  }
  return snap;
}

/**
 * Given a widget, return a short summary of its change history.
 */
function _widgetSummary(node, widgetName) {
  const mem = _getMemory(node);
  const lines = [];

  // Default value
  const defVal = mem.defaults[widgetName];
  if (defVal !== undefined) {
    lines.push(`Default: ${_formatVal(defVal)}`);
  }

  // Current value
  const w = node.widgets?.find(w => w.name === widgetName);
  if (w) {
    lines.push(`Current: ${_formatVal(w.value)}`);
  }

  // Changes for this widget (last 5)
  const changes = mem.history.filter(h => h.name === widgetName);
  if (changes.length > 0) {
    lines.push("─ Recent changes ─");
    const recent = changes.slice(-5);
    for (const c of recent) {
      lines.push(`  ${c.ts.slice(11)}: ${_formatVal(c.from)} → ${_formatVal(c.to)}`);
    }
    if (changes.length > 5) {
      lines.push(`  ... and ${changes.length - 5} more`);
    }
  }

  // Snapshot comparison (last 2 runs)
  const snaps = mem.snapshots;
  if (snaps.length >= 2) {
    const prev = snaps[snaps.length - 2];
    const curr = snaps[snaps.length - 1];
    const prevVal = prev.values[widgetName];
    const currVal = curr.values[widgetName];
    if (prevVal !== undefined && currVal !== undefined && prevVal !== currVal) {
      lines.push(`─ Last run diff ─`);
      lines.push(`  Run ${prev.run_id}: ${_formatVal(prevVal)}`);
      lines.push(`  Run ${curr.run_id}: ${_formatVal(currVal)}`);
    }
  }

  return lines;
}

function _formatVal(v) {
  if (v === undefined) return "—";
  if (v === null) return "null";
  if (typeof v === "string") {
    return v.length > 30 ? `"${v.slice(0, 27)}..."` : `"${v}"`;
  }
  if (typeof v === "number") return String(v);
  if (typeof v === "boolean") return String(v);
  return String(v);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Tooltip rendering
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function _drawTooltip(ctx, lines, x, y) {
  if (!lines || lines.length === 0) return;

  const fontSize = 11;
  ctx.font = `${fontSize}px Inter, system-ui, monospace`;

  // Measure
  let maxW = 0;
  for (const line of lines) {
    const m = ctx.measureText(line);
    maxW = Math.max(maxW, m.width);
  }
  maxW = Math.min(maxW, TOOLTIP_MAX_W);
  const totalH = lines.length * (fontSize + 4) + TOOLTIP_PAD * 2;
  const totalW = maxW + TOOLTIP_PAD * 2;

  // Position (above the cursor, clamped to canvas)
  let tx = x - totalW / 2;
  let ty = y - totalH - 10;

  // Background
  ctx.save();
  ctx.fillStyle = TOOLTIP_BG;
  _roundRect(ctx, tx, ty, totalW, totalH, TOOLTIP_RADIUS);
  ctx.fill();

  // Border
  ctx.strokeStyle = "#45475a";
  ctx.lineWidth = 1;
  _roundRect(ctx, tx, ty, totalW, totalH, TOOLTIP_RADIUS);
  ctx.stroke();

  // Text
  let lineY = ty + TOOLTIP_PAD + fontSize;
  for (const line of lines) {
    if (line.startsWith("Default:")) {
      ctx.fillStyle = TOOLTIP_DIM;
    } else if (line.startsWith("Current:")) {
      ctx.fillStyle = TOOLTIP_ACCENT;
    } else if (line.includes("→")) {
      ctx.fillStyle = TOOLTIP_WARN;
    } else if (line.startsWith("─")) {
      ctx.fillStyle = TOOLTIP_DIM;
    } else {
      ctx.fillStyle = TOOLTIP_FG;
    }
    ctx.fillText(line, tx + TOOLTIP_PAD, lineY);
    lineY += fontSize + 4;
  }
  ctx.restore();
}

function _roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Main extension
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app.registerExtension({
  name: "MEC.ParameterMemory",

  // ── Setup: hook into queue lifecycle ─────────────────────────────
  setup() {
    // Before queue → take a pre-execution snapshot of all nodes
    const origQueuePrompt = api.queuePrompt;
    if (origQueuePrompt) {
      api.queuePrompt = async function (...args) {
        _runCounter++;
        _preQueueSnapshots.clear();

        const allNodes = app.graph?._nodes || app.graph?.nodes || [];
        for (const node of allNodes) {
          if (!node.widgets || node.widgets.length === 0) continue;
          const vals = _captureValues(node);
          _preQueueSnapshots.set(node.id, { values: vals });
          _takeSnapshot(node, _runCounter);
        }

        // Send snapshot to backend for DB persistence
        try {
          const payload = {};
          for (const [nid, snap] of _preQueueSnapshots) {
            const node = app.graph?.getNodeById(nid);
            payload[nid] = {
              title:  node?.title || node?.comfyClass || `Node_${nid}`,
              class:  node?.comfyClass || "unknown",
              run_id: _runCounter,
              ts:     _now(),
              values: snap.values,
            };
          }
          // Fire-and-forget to the backend endpoint
          fetch("/mec/param_history", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          }).catch(() => {}); // Silently ignore if backend endpoint not available
        } catch (_) { /* ignore */ }

        return origQueuePrompt.apply(this, args);
      };
    }

    // After execution → detect what changed
    api.addEventListener("executed", (event) => {
      const nodeId = event?.detail?.node;
      if (!nodeId) return;

      const node = app.graph?.getNodeById(Number(nodeId));
      if (!node) return;

      const preSnap = _preQueueSnapshots.get(node.id);
      if (!preSnap) return;

      // Compare pre-execution values with current values
      const currentVals = _captureValues(node);
      const mem = _getMemory(node);
      for (const [name, preVal] of Object.entries(preSnap.values)) {
        const curVal = currentVals[name];
        if (curVal !== undefined && preVal !== curVal) {
          _recordChange(node, name, preVal, curVal);
        }
      }
    });
  },

  // ── Hook every node creation ─────────────────────────────────────
  nodeCreated(node) {
    if (!node.widgets || node.widgets.length === 0) return;

    // Capture defaults on first creation
    setTimeout(() => {
      const mem = _getMemory(node);
      if (Object.keys(mem.defaults).length === 0) {
        mem.defaults = _captureDefaults(node);
      }
    }, 100);

    // ── Intercept widget value changes ───────────────────────────
    for (const w of node.widgets) {
      if (!w.name || w.type === "canvas" || w.name.startsWith("_")) continue;

      const origCb = w.callback;
      const wName  = w.name;

      // Track the last-known value for change detection
      let lastKnownValue = w.value;

      w.callback = function (value) {
        if (value !== lastKnownValue) {
          _recordChange(node, wName, lastKnownValue, value);
          lastKnownValue = value;
        }
        if (origCb) return origCb.call(this, value);
      };

      // Also intercept set via property descriptor for combo/number widgets
      // that may update .value directly without callback
      const descriptor = Object.getOwnPropertyDescriptor(w, "value");
      if (descriptor && descriptor.configurable !== false) {
        let _val = w.value;
        try {
          Object.defineProperty(w, "value", {
            get() { return _val; },
            set(v) {
              if (v !== _val) {
                _recordChange(node, wName, _val, v);
                lastKnownValue = v;
              }
              _val = v;
            },
            configurable: true,
            enumerable: true,
          });
        } catch (_) {
          // Some widgets may not allow property redefinition — that's OK
        }
      }
    }

    // ── Hover tooltip: draw on widget hover ─────────────────────
    const origDrawFg = node.onDrawForeground;
    node.onDrawForeground = function (ctx) {
      origDrawFg?.apply(this, arguments);

      // Check if mouse is over a widget area
      const graphCanvas = app.canvas;
      if (!graphCanvas) return;

      const mouse = graphCanvas.graph_mouse;
      if (!mouse) return;

      // Convert graph coords to node-local coords
      const localX = mouse[0] - this.pos[0];
      const localY = mouse[1] - this.pos[1];

      // Check each widget's Y position
      if (!this.widgets || localX < 0 || localX > this.size[0]) return;

      // LiteGraph positions widgets below title bar
      const titleH = LiteGraph.NODE_TITLE_HEIGHT || 30;
      let widgetY = titleH;

      for (const w of this.widgets) {
        if (!w.name || w.type === "canvas" || w.name.startsWith("_")) continue;

        const wH = LiteGraph.NODE_WIDGET_HEIGHT || 20;
        const wBottom = widgetY + wH;

        // Alt key held → show parameter memory tooltip
        if (localY >= widgetY && localY < wBottom && graphCanvas.keys_alt) {
          const lines = _widgetSummary(this, w.name);
          if (lines.length > 0) {
            // Draw tooltip at widget position (in node-local coords)
            _drawTooltip(ctx, lines, this.size[0] / 2, widgetY - 5);
          }
          break;
        }
        widgetY = wBottom;
      }
    };

    // ── Context menu entries ─────────────────────────────────────
    const origGetMenu = node.getExtraMenuOptions;
    node.getExtraMenuOptions = function (canvas, options) {
      origGetMenu?.apply(this, arguments);

      options.push(null); // separator

      // Show Parameter History
      options.push({
        content: "📋 Show Parameter History",
        callback: () => {
          _showHistoryDialog(this);
        },
      });

      // Show Changes Since Last Run
      options.push({
        content: "🔍 Show Changes Since Last Run",
        callback: () => {
          _showChangesDialog(this);
        },
      });

      // Reset to Defaults
      options.push({
        content: "↩ Reset to Defaults",
        callback: () => {
          _resetToDefaults(this);
        },
      });

      // Clear History
      options.push({
        content: "🗑 Clear Parameter History",
        callback: () => {
          const mem = _getMemory(this);
          mem.history = [];
          mem.snapshots = [];
          alert("Parameter history cleared.");
        },
      });
    };
  },
});

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Dialogs
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function _showHistoryDialog(node) {
  const mem   = _getMemory(node);
  const title = node.title || node.comfyClass || `Node ${node.id}`;

  let html = `<div style="font-family:monospace;font-size:12px;max-height:500px;overflow-y:auto;padding:12px;background:#1e1e2e;color:#cdd6f4;border-radius:8px;">`;
  html += `<h3 style="color:#89b4fa;margin:0 0 8px 0;">${_escapeHtml(title)} — Parameter History</h3>`;

  // Defaults
  html += `<div style="color:#6c7086;margin-bottom:8px;">`;
  html += `<strong>Defaults:</strong><br/>`;
  for (const [k, v] of Object.entries(mem.defaults)) {
    html += `  ${_escapeHtml(k)}: ${_escapeHtml(_formatVal(v))}<br/>`;
  }
  html += `</div>`;

  // Current
  html += `<div style="color:#a6e3a1;margin-bottom:8px;">`;
  html += `<strong>Current Values:</strong><br/>`;
  const curVals = _captureValues(node);
  for (const [k, v] of Object.entries(curVals)) {
    const def = mem.defaults[k];
    const isChanged = def !== undefined && def !== v;
    const style = isChanged ? "color:#f9e2af;" : "";
    html += `  <span style="${style}">${_escapeHtml(k)}: ${_escapeHtml(_formatVal(v))}`;
    if (isChanged) html += ` (default: ${_escapeHtml(_formatVal(def))})`;
    html += `</span><br/>`;
  }
  html += `</div>`;

  // Change log
  if (mem.history.length > 0) {
    html += `<div style="margin-bottom:8px;">`;
    html += `<strong style="color:#89b4fa;">Change Log (${mem.history.length}):</strong><br/>`;
    const recent = mem.history.slice(-30);
    for (const h of recent) {
      html += `<span style="color:#6c7086;">${_escapeHtml(h.ts.slice(11))}</span> `;
      html += `<span style="color:#cba6f7;">${_escapeHtml(h.name)}</span>: `;
      html += `${_escapeHtml(_formatVal(h.from))} → `;
      html += `<span style="color:#f9e2af;">${_escapeHtml(_formatVal(h.to))}</span><br/>`;
    }
    if (mem.history.length > 30) {
      html += `<span style="color:#6c7086;">... and ${mem.history.length - 30} more</span><br/>`;
    }
    html += `</div>`;
  }

  // Snapshots
  if (mem.snapshots.length > 0) {
    html += `<div>`;
    html += `<strong style="color:#89b4fa;">Execution Snapshots (${mem.snapshots.length}):</strong><br/>`;
    const recent = mem.snapshots.slice(-10);
    for (const s of recent) {
      html += `<span style="color:#6c7086;">${_escapeHtml(s.ts)} — Run #${s.run_id}</span><br/>`;
      for (const [k, v] of Object.entries(s.values)) {
        html += `  ${_escapeHtml(k)}: ${_escapeHtml(_formatVal(v))}<br/>`;
      }
      html += `<br/>`;
    }
    html += `</div>`;
  }

  html += `</div>`;
  _showModal(html, "Parameter History");
}

function _showChangesDialog(node) {
  const mem   = _getMemory(node);
  const title = node.title || node.comfyClass || `Node ${node.id}`;

  const snaps = mem.snapshots;
  if (snaps.length < 2) {
    alert("Not enough execution snapshots to compare. Run the workflow at least twice.");
    return;
  }

  const prev = snaps[snaps.length - 2];
  const curr = snaps[snaps.length - 1];

  let html = `<div style="font-family:monospace;font-size:12px;max-height:400px;overflow-y:auto;padding:12px;background:#1e1e2e;color:#cdd6f4;border-radius:8px;">`;
  html += `<h3 style="color:#89b4fa;margin:0 0 8px 0;">${_escapeHtml(title)} — Changes Between Runs</h3>`;
  html += `<div style="color:#6c7086;margin-bottom:8px;">Run #${prev.run_id} (${_escapeHtml(prev.ts)}) → Run #${curr.run_id} (${_escapeHtml(curr.ts)})</div>`;

  let hasChanges = false;
  for (const [name, currVal] of Object.entries(curr.values)) {
    const prevVal = prev.values[name];
    if (prevVal !== currVal) {
      hasChanges = true;
      html += `<span style="color:#cba6f7;">${_escapeHtml(name)}</span>: `;
      html += `<span style="color:#f38ba8;">${_escapeHtml(_formatVal(prevVal))}</span> → `;
      html += `<span style="color:#a6e3a1;">${_escapeHtml(_formatVal(currVal))}</span><br/>`;
    }
  }

  if (!hasChanges) {
    html += `<span style="color:#a6adc8;">No parameters changed between these runs.</span>`;
  }

  html += `</div>`;
  _showModal(html, "Run Diff");
}

function _resetToDefaults(node) {
  const mem = _getMemory(node);
  if (!mem.defaults || Object.keys(mem.defaults).length === 0) {
    alert("No defaults recorded for this node.");
    return;
  }

  let changed = 0;
  for (const w of node.widgets || []) {
    if (!w.name || w.name.startsWith("_")) continue;
    const def = mem.defaults[w.name];
    if (def !== undefined && w.value !== def) {
      _recordChange(node, w.name, w.value, def);
      w.value = def;
      w.callback?.(def);
      changed++;
    }
  }

  if (changed > 0) {
    node.setDirtyCanvas?.(true, true);
    alert(`Reset ${changed} parameter(s) to defaults.`);
  } else {
    alert("All parameters are already at their default values.");
  }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Modal utility
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function _showModal(html, title) {
  // Remove existing modal
  const existing = document.getElementById("mec-param-modal");
  if (existing) existing.remove();

  const overlay = document.createElement("div");
  overlay.id = "mec-param-modal";
  overlay.style.cssText = `
    position:fixed; top:0; left:0; width:100vw; height:100vh;
    background:rgba(0,0,0,0.6); z-index:99999;
    display:flex; align-items:center; justify-content:center;
  `;
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) overlay.remove();
  });

  const dialog = document.createElement("div");
  dialog.style.cssText = `
    background:#181825; border:1px solid #313244; border-radius:12px;
    padding:0; min-width:400px; max-width:700px; max-height:80vh;
    overflow:hidden; box-shadow:0 8px 32px rgba(0,0,0,0.5);
  `;

  // Title bar
  const titleBar = document.createElement("div");
  titleBar.style.cssText = `
    padding:10px 16px; background:#11111b; border-bottom:1px solid #313244;
    display:flex; justify-content:space-between; align-items:center;
  `;
  titleBar.innerHTML = `
    <span style="color:#89b4fa;font-weight:bold;font-size:14px;">${_escapeHtml(title)}</span>
    <button id="mec-modal-close" style="background:none;border:none;color:#6c7086;cursor:pointer;font-size:18px;padding:2px 6px;">✕</button>
  `;
  dialog.appendChild(titleBar);

  // Content
  const content = document.createElement("div");
  content.style.cssText = `padding:0; max-height:calc(80vh - 50px); overflow-y:auto;`;
  content.innerHTML = html;
  dialog.appendChild(content);

  overlay.appendChild(dialog);
  document.body.appendChild(overlay);

  document.getElementById("mec-modal-close")?.addEventListener("click", () => {
    overlay.remove();
  });

  // ESC to close
  const escHandler = (e) => {
    if (e.key === "Escape") {
      overlay.remove();
      document.removeEventListener("keydown", escHandler);
    }
  };
  document.addEventListener("keydown", escHandler);
}

function _escapeHtml(s) {
  if (typeof s !== "string") s = String(s ?? "");
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * Parameter Memory v2 — Compact parameter tracking + named presets.
 *
 * Redesign goals (inspired by ComfyUI GitHub #12677, #12942, #13017):
 *   - Zero workflow JSON bloat (all data in-memory Map + server-side SQLite)
 *   - Snapshots as diffs from defaults (not full value clones)
 *   - Per-widget change history (grouped, not flat)
 *   - Named presets: save / load / manage parameter sets per node type
 *   - Compact backend payloads (delta-only)
 *
 * In-memory data per node (_memoryStore Map):
 *   {
 *     defaults:  { widget: value, ... },
 *     changes:   { widget: [ [epoch, from, to], ... ], ... },
 *     diffs:     [ [run_id, epoch, { widget: value, ... }], ... ],
 *     presets:   { "name": { widget: value, ... }, ... },
 *   }
 */

// ── Tunables ─────────────────────────────────────────────────────────
const MAX_CHANGES_PER_WIDGET = 8;
const MAX_DIFFS              = 15;
const PRESET_STORAGE_KEY     = "MEC.ParamPresets";

// ── Tooltip styling ──────────────────────────────────────────────────
const TT = {
  bg: "#1e1e2edd", fg: "#cdd6f4", accent: "#89b4fa", warn: "#f9e2af",
  dim: "#6c7086", radius: 6, pad: 8, maxW: 280,
};

let _runCounter = 0;
let _preQueueSnapshots = new Map();

// In-memory only — never serialized into workflow JSON.
const _memoryStore = new Map();

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Helpers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function _epoch() { return Math.floor(Date.now() / 1000); }

function _ts(epoch) {
  const d = new Date(epoch * 1000);
  return String(d.getHours()).padStart(2, "0") + ":"
       + String(d.getMinutes()).padStart(2, "0") + ":"
       + String(d.getSeconds()).padStart(2, "0");
}

function _fullTs(epoch) {
  return new Date(epoch * 1000).toISOString().replace("T", " ").slice(0, 19);
}

function _getMem(node) {
  const id = node.id;
  if (!_memoryStore.has(id)) {
    _memoryStore.set(id, { defaults: {}, changes: {}, diffs: [], presets: {} });
  }
  return _memoryStore.get(id);
}

function _captureValues(node) {
  const v = {};
  if (node.widgets) {
    for (const w of node.widgets)
      if (w.name && w.type !== "canvas" && !w.name.startsWith("_"))
        v[w.name] = w.value;
  }
  return v;
}

function _captureDefaults(node) {
  const d = {};
  if (node.widgets) {
    for (const w of node.widgets)
      if (w.name && w.type !== "canvas" && !w.name.startsWith("_"))
        d[w.name] = w.options?.default ?? w.options?.defaultVal ?? w.value;
  }
  return d;
}

/** Record a single widget change into per-widget ring buffer. */
function _recordChange(node, widgetName, oldVal, newVal) {
  const mem = _getMem(node);
  if (!mem.changes[widgetName]) mem.changes[widgetName] = [];
  const ring = mem.changes[widgetName];
  ring.push([_epoch(), oldVal, newVal]);
  if (ring.length > MAX_CHANGES_PER_WIDGET)
    mem.changes[widgetName] = ring.slice(-MAX_CHANGES_PER_WIDGET);
}

/** Take a diff-only snapshot (stores only params that differ from defaults). */
function _takeDiffSnapshot(node, runId) {
  const mem  = _getMem(node);
  const vals = _captureValues(node);
  const diff = {};
  for (const [k, v] of Object.entries(vals))
    if (mem.defaults[k] === undefined || mem.defaults[k] !== v) diff[k] = v;
  mem.diffs.push([runId, _epoch(), diff]);
  if (mem.diffs.length > MAX_DIFFS)
    mem.diffs = mem.diffs.slice(-MAX_DIFFS);
  return diff;
}

/** Build delta payload for backend — only changed-from-default values. */
function _buildDeltaPayload(node, runId) {
  const mem  = _getMem(node);
  const vals = _captureValues(node);
  const delta = {};
  let any = false;
  for (const [k, v] of Object.entries(vals)) {
    if (mem.defaults[k] === undefined || mem.defaults[k] !== v) {
      delta[k] = v;
      any = true;
    }
  }
  if (!any) return null;
  return {
    title:  node.title || node.comfyClass || `Node_${node.id}`,
    class:  node.comfyClass || "unknown",
    run_id: runId,
    ts:     _fullTs(_epoch()),
    delta,
  };
}

function _fv(v) {
  if (v === undefined) return "\u2014";
  if (v === null) return "null";
  if (typeof v === "string")
    return v.length > 28 ? `"${v.slice(0, 25)}\u2026"` : `"${v}"`;
  return String(v);
}

// ── Preset persistence (localStorage, small footprint) ───────────────

function _loadAllPresets() {
  try { return JSON.parse(localStorage.getItem(PRESET_STORAGE_KEY) || "{}"); }
  catch { return {}; }
}

function _loadPresets(nodeClass) {
  return _loadAllPresets()[nodeClass] || {};
}

function _savePresets(nodeClass, presets) {
  try {
    const all = _loadAllPresets();
    if (Object.keys(presets).length === 0) delete all[nodeClass];
    else all[nodeClass] = presets;
    localStorage.setItem(PRESET_STORAGE_KEY, JSON.stringify(all));
  } catch { /* quota — presets are small, unlikely */ }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Tooltip
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function _widgetSummary(node, wName) {
  const mem = _getMem(node);
  const lines = [];
  const def = mem.defaults[wName];
  const w = node.widgets?.find(x => x.name === wName);
  const cur = w?.value;

  if (def !== undefined) lines.push(`\u2699 Default: ${_fv(def)}`);
  if (cur !== undefined) {
    const marker = (def !== undefined && def !== cur) ? "\u25B8" : "\u2022";
    lines.push(`${marker} Current: ${_fv(cur)}`);
  }

  const ring = mem.changes[wName];
  if (ring && ring.length > 0) {
    lines.push("\u2500\u2500\u2500\u2500\u2500");
    for (const [ep, from, to] of ring.slice(-4))
      lines.push(`${_ts(ep)}  ${_fv(from)} \u2192 ${_fv(to)}`);
    if (ring.length > 4) lines.push(`\u2026 +${ring.length - 4} more`);
  }

  if (mem.diffs.length >= 2) {
    const prev = mem.diffs[mem.diffs.length - 2];
    const curr = mem.diffs[mem.diffs.length - 1];
    const pv = prev[2][wName] ?? def;
    const cv = curr[2][wName] ?? def;
    if (pv !== cv) {
      lines.push("\u2500\u2500\u2500\u2500\u2500");
      lines.push(`Run #${prev[0]}: ${_fv(pv)}`);
      lines.push(`Run #${curr[0]}: ${_fv(cv)}`);
    }
  }

  return lines;
}

function _drawTooltip(ctx, lines, x, y) {
  if (!lines.length) return;
  const fs = 11;
  ctx.font = `${fs}px Inter, system-ui, monospace`;
  let maxW = 0;
  for (const l of lines) maxW = Math.max(maxW, ctx.measureText(l).width);
  maxW = Math.min(maxW, TT.maxW);
  const h = lines.length * (fs + 4) + TT.pad * 2;
  const w = maxW + TT.pad * 2;
  const tx = x - w / 2, ty = y - h - 10;

  ctx.save();
  ctx.fillStyle = TT.bg;
  _rr(ctx, tx, ty, w, h, TT.radius); ctx.fill();
  ctx.strokeStyle = "#45475a"; ctx.lineWidth = 1;
  _rr(ctx, tx, ty, w, h, TT.radius); ctx.stroke();

  let ly = ty + TT.pad + fs;
  for (const l of lines) {
    ctx.fillStyle = l.startsWith("\u2699") ? TT.dim
      : l.startsWith("\u25B8") ? TT.warn
      : l.startsWith("\u2022") ? TT.accent
      : l.includes("\u2192") ? TT.warn
      : l.startsWith("\u2500") ? TT.dim
      : l.startsWith("Run") ? TT.accent
      : TT.fg;
    ctx.fillText(l, tx + TT.pad, ly);
    ly += fs + 4;
  }
  ctx.restore();
}

function _rr(ctx, x, y, w, h, r) {
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
//  Main Extension
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app.registerExtension({
  name: "MEC.ParameterMemory",

  setup() {
    const origQP = api.queuePrompt;
    if (origQP) {
      api.queuePrompt = async function (...args) {
        _runCounter++;
        _preQueueSnapshots.clear();

        const allNodes = app.graph?._nodes || app.graph?.nodes || [];
        const payload = {};

        for (const node of allNodes) {
          if (!node.widgets || node.widgets.length === 0) continue;
          const vals = _captureValues(node);
          _preQueueSnapshots.set(node.id, vals);
          _takeDiffSnapshot(node, _runCounter);

          const delta = _buildDeltaPayload(node, _runCounter);
          if (delta) payload[node.id] = delta;
        }

        if (Object.keys(payload).length > 0) {
          fetch("/mec/param_history", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          }).catch(() => {});
        }

        return origQP.apply(this, args);
      };
    }

    api.addEventListener("executed", (event) => {
      const nodeId = event?.detail?.node;
      if (!nodeId) return;
      const node = app.graph?.getNodeById(Number(nodeId));
      if (!node) return;
      const pre = _preQueueSnapshots.get(node.id);
      if (!pre) return;
      const cur = _captureValues(node);
      for (const [name, preVal] of Object.entries(pre))
        if (cur[name] !== undefined && preVal !== cur[name])
          _recordChange(node, name, preVal, cur[name]);
    });
  },

  nodeCreated(node) {
    if (!node.widgets || node.widgets.length === 0) return;

    // Migrate legacy _paramMemory from v1
    if (node.properties?._paramMemory) {
      const legacy = node.properties._paramMemory;
      const mem = _getMem(node);
      if (legacy.defaults && Object.keys(legacy.defaults).length > 0)
        Object.assign(mem.defaults, legacy.defaults);
      delete node.properties._paramMemory;
    }

    // Capture defaults + load presets
    setTimeout(() => {
      const mem = _getMem(node);
      if (Object.keys(mem.defaults).length === 0)
        mem.defaults = _captureDefaults(node);
      const cls = node.comfyClass || node.type;
      if (cls) mem.presets = _loadPresets(cls);
    }, 100);

    // Widget change interception
    for (const w of node.widgets) {
      if (!w.name || w.type === "canvas" || w.name.startsWith("_")) continue;
      const origCb = w.callback;
      const wName = w.name;
      let lastVal = w.value;

      w.callback = function (value) {
        if (value !== lastVal) {
          _recordChange(node, wName, lastVal, value);
          lastVal = value;
        }
        if (origCb) return origCb.call(this, value);
      };

      const desc = Object.getOwnPropertyDescriptor(w, "value");
      if (desc && desc.configurable !== false) {
        let _v = w.value;
        try {
          Object.defineProperty(w, "value", {
            get() { return _v; },
            set(v) {
              if (v !== _v) {
                _recordChange(node, wName, _v, v);
                lastVal = v;
              }
              _v = v;
            },
            configurable: true,
            enumerable: true,
          });
        } catch { /* some widgets resist redefinition */ }
      }
    }

    // Hover tooltip (Alt + hover)
    const origFg = node.onDrawForeground;
    node.onDrawForeground = function (ctx) {
      origFg?.apply(this, arguments);
      const gc = app.canvas;
      if (!gc?.graph_mouse) return;
      const m = gc.graph_mouse;
      const lx = m[0] - this.pos[0], ly = m[1] - this.pos[1];
      if (!this.widgets || lx < 0 || lx > this.size[0]) return;
      const tH = LiteGraph.NODE_TITLE_HEIGHT || 30;
      let wy = tH;
      for (const w of this.widgets) {
        if (!w.name || w.type === "canvas" || w.name.startsWith("_")) continue;
        const wH = LiteGraph.NODE_WIDGET_HEIGHT || 20;
        if (ly >= wy && ly < wy + wH && gc.keys_alt) {
          const lines = _widgetSummary(this, w.name);
          if (lines.length) _drawTooltip(ctx, lines, this.size[0] / 2, wy - 5);
          break;
        }
        wy += wH;
      }
    };

    // Context menu
    const origMenu = node.getExtraMenuOptions;
    node.getExtraMenuOptions = function (canvas, options) {
      origMenu?.apply(this, arguments);
      const self = this;
      options.push(null);

      options.push({
        content: "\uD83D\uDCCB Parameter History",
        callback: () => _showHistoryDialog(self),
      });
      options.push({
        content: "\uD83D\uDD0D Changes Since Last Run",
        callback: () => _showDiffDialog(self),
      });

      // Presets submenu
      const mem = _getMem(self);
      const presetNames = Object.keys(mem.presets);
      const presetItems = [];

      presetItems.push({
        content: "\uD83D\uDCBE Save Current as Preset\u2026",
        callback: () => _savePresetDialog(self),
      });

      if (presetNames.length > 0) {
        presetItems.push(null);
        for (const name of presetNames) {
          presetItems.push({
            content: `\u25B8 ${name}`,
            callback: () => _applyPreset(self, name),
          });
        }
        presetItems.push(null);
        presetItems.push({
          content: "\uD83D\uDDD1 Manage Presets\u2026",
          callback: () => _managePresetsDialog(self),
        });
      }

      options.push({
        content: "\u26A1 Presets",
        submenu: { options: presetItems },
      });

      options.push({
        content: "\u21A9 Reset to Defaults",
        callback: () => _resetToDefaults(self),
      });
      options.push({
        content: "\uD83D\uDDD1 Clear History",
        callback: () => {
          const m = _getMem(self);
          m.changes = {};
          m.diffs = [];
        },
      });
    };
  },
});

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  History Dialog
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function _showHistoryDialog(node) {
  const mem   = _getMem(node);
  const title = node.title || node.comfyClass || `Node ${node.id}`;
  const cur   = _captureValues(node);
  const S     = _esc;

  let html = `<div style="font-family:'JetBrains Mono',Consolas,monospace;font-size:12px;padding:14px;background:#1e1e2e;color:#cdd6f4;">`;

  // ── Widget values table ──
  html += `<div style="margin-bottom:12px;">`;
  html += `<div style="color:#89b4fa;font-weight:bold;margin-bottom:6px;">Widget Values</div>`;
  html += `<table style="width:100%;border-collapse:collapse;font-size:11px;">`;
  html += `<tr style="color:#6c7086;border-bottom:1px solid #313244;">
    <td style="padding:3px 6px;">Widget</td>
    <td style="padding:3px 6px;">Current</td>
    <td style="padding:3px 6px;">Default</td>
    <td style="padding:3px 6px;width:20px;"></td></tr>`;

  for (const [k, v] of Object.entries(cur)) {
    const def = mem.defaults[k];
    const changed = def !== undefined && def !== v;
    const icon = changed
      ? `<span style="color:#f9e2af;">~</span>`
      : `<span style="color:#a6e3a1;">\u2022</span>`;
    const style = changed ? "color:#f9e2af;" : "";
    html += `<tr style="border-bottom:1px solid #1e1e2e88;">
      <td style="padding:2px 6px;color:#cba6f7;">${S(k)}</td>
      <td style="padding:2px 6px;${style}">${S(_fv(v))}</td>
      <td style="padding:2px 6px;color:#6c7086;">${S(_fv(def))}</td>
      <td style="padding:2px 6px;">${icon}</td></tr>`;
  }
  html += `</table></div>`;

  // ── Per-widget change log ──
  const changedWidgets = Object.keys(mem.changes).filter(k => mem.changes[k].length > 0);
  if (changedWidgets.length > 0) {
    html += `<div style="margin-bottom:12px;">`;
    html += `<div style="color:#89b4fa;font-weight:bold;margin-bottom:6px;">Change History</div>`;
    for (const wn of changedWidgets) {
      const ring = mem.changes[wn];
      html += `<div style="margin-bottom:6px;">`;
      html += `<span style="color:#cba6f7;">${S(wn)}</span> <span style="color:#6c7086;">(${ring.length})</span><br/>`;
      for (const [ep, from, to] of ring.slice(-5))
        html += `<span style="color:#6c7086;margin-left:8px;">${_ts(ep)}</span> ${S(_fv(from))} <span style="color:#f9e2af;">\u2192</span> ${S(_fv(to))}<br/>`;
      if (ring.length > 5)
        html += `<span style="color:#6c7086;margin-left:8px;">\u2026 +${ring.length - 5} more</span><br/>`;
      html += `</div>`;
    }
    html += `</div>`;
  }

  // ── Diff snapshots ──
  if (mem.diffs.length > 0) {
    html += `<div>`;
    html += `<div style="color:#89b4fa;font-weight:bold;margin-bottom:6px;">Run Snapshots <span style="color:#6c7086;font-weight:normal;">(diffs from defaults)</span></div>`;
    for (const [rid, ep, diff] of mem.diffs.slice(-6)) {
      const keys = Object.keys(diff);
      if (keys.length === 0) {
        html += `<span style="color:#6c7086;">Run #${rid} (${_fullTs(ep)}) \u2014 all defaults</span><br/>`;
      } else {
        html += `<span style="color:#a6adc8;">Run #${rid}</span> <span style="color:#6c7086;">${_fullTs(ep)}</span><br/>`;
        for (const k of keys)
          html += `<span style="margin-left:8px;color:#cba6f7;">${S(k)}</span>: <span style="color:#f9e2af;">${S(_fv(diff[k]))}</span><br/>`;
      }
    }
    html += `</div>`;
  }

  html += `</div>`;
  _showModal(html, `${S(title)} \u2014 History`);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Diff Dialog
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function _showDiffDialog(node) {
  const mem   = _getMem(node);
  const title = node.title || node.comfyClass || `Node ${node.id}`;
  const S     = _esc;

  if (mem.diffs.length < 2) {
    _showModal(
      `<div style="font-family:monospace;padding:20px;color:#a6adc8;background:#1e1e2e;">Run the workflow at least twice to compare.</div>`,
      "Run Diff"
    );
    return;
  }

  const [ridA, , diffA] = mem.diffs[mem.diffs.length - 2];
  const [ridB, , diffB] = mem.diffs[mem.diffs.length - 1];

  let html = `<div style="font-family:'JetBrains Mono',Consolas,monospace;font-size:12px;padding:14px;background:#1e1e2e;color:#cdd6f4;">`;
  html += `<div style="color:#6c7086;margin-bottom:8px;">Run #${ridA} \u2192 Run #${ridB}</div>`;

  const allKeys = new Set([...Object.keys(diffA), ...Object.keys(diffB)]);
  let hasChanges = false;

  html += `<table style="width:100%;border-collapse:collapse;font-size:11px;">`;
  html += `<tr style="color:#6c7086;border-bottom:1px solid #313244;">
    <td style="padding:3px 6px;">Widget</td>
    <td style="padding:3px 6px;">Run #${ridA}</td>
    <td style="padding:3px 6px;">Run #${ridB}</td></tr>`;

  for (const k of allKeys) {
    const va = diffA[k] ?? mem.defaults[k];
    const vb = diffB[k] ?? mem.defaults[k];
    if (va !== vb) {
      hasChanges = true;
      html += `<tr style="border-bottom:1px solid #1e1e2e88;">
        <td style="padding:2px 6px;color:#cba6f7;">${S(k)}</td>
        <td style="padding:2px 6px;color:#f38ba8;">${S(_fv(va))}</td>
        <td style="padding:2px 6px;color:#a6e3a1;">${S(_fv(vb))}</td></tr>`;
    }
  }
  html += `</table>`;

  if (!hasChanges)
    html += `<div style="color:#a6adc8;margin-top:8px;">No parameters changed between these runs.</div>`;

  html += `</div>`;
  _showModal(html, `${S(title)} \u2014 Run Diff`);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Presets
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function _savePresetDialog(node) {
  const name = prompt("Preset name:", "");
  if (!name || !name.trim()) return;
  const clean = name.trim().slice(0, 64);
  const mem = _getMem(node);
  mem.presets[clean] = _captureValues(node);
  const cls = node.comfyClass || node.type;
  if (cls) _savePresets(cls, mem.presets);
}

function _applyPreset(node, name) {
  const mem = _getMem(node);
  const preset = mem.presets[name];
  if (!preset) return;
  let applied = 0;
  for (const w of node.widgets || []) {
    if (!w.name || w.name.startsWith("_")) continue;
    const pv = preset[w.name];
    if (pv !== undefined && w.value !== pv) {
      _recordChange(node, w.name, w.value, pv);
      w.value = pv;
      w.callback?.(pv);
      applied++;
    }
  }
  if (applied > 0) {
    node.setDirtyCanvas?.(true, true);
    const orig = node.bgcolor;
    node.bgcolor = "#313244";
    setTimeout(() => { node.bgcolor = orig; node.setDirtyCanvas?.(true, true); }, 300);
  }
}

function _managePresetsDialog(node) {
  const mem = _getMem(node);
  const cls = node.comfyClass || node.type;
  const S = _esc;
  const names = Object.keys(mem.presets);

  let html = `<div style="font-family:'JetBrains Mono',Consolas,monospace;font-size:12px;padding:14px;background:#1e1e2e;color:#cdd6f4;">`;

  if (names.length === 0) {
    html += `<div style="color:#a6adc8;">No presets saved for this node type.</div>`;
  } else {
    for (const pname of names) {
      const preset = mem.presets[pname];
      const n = Object.keys(preset).length;
      html += `<div style="display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid #313244;">`;
      html += `<div><span style="color:#89b4fa;font-weight:bold;">${S(pname)}</span> <span style="color:#6c7086;">(${n} params)</span></div>`;
      html += `<button data-preset="${S(pname)}" class="mec-preset-del" style="background:#f38ba8;color:#1e1e2e;border:none;border-radius:4px;padding:2px 8px;cursor:pointer;font-size:11px;">Delete</button>`;
      html += `</div>`;
      html += `<div style="margin:4px 0 8px 12px;color:#6c7086;font-size:10px;">`;
      const entries = Object.entries(preset).slice(0, 8);
      html += entries.map(([k, v]) => `${S(k)}=${S(_fv(v))}`).join(", ");
      if (n > 8) html += ` \u2026 +${n - 8}`;
      html += `</div>`;
    }
  }
  html += `</div>`;
  _showModal(html, "Manage Presets");

  setTimeout(() => {
    document.querySelectorAll(".mec-preset-del").forEach(btn => {
      btn.addEventListener("click", () => {
        const pn = btn.dataset.preset;
        delete mem.presets[pn];
        if (cls) _savePresets(cls, mem.presets);
        btn.closest("div[style*='flex']").nextElementSibling?.remove();
        btn.closest("div[style*='flex']").remove();
      });
    });
  }, 50);
}

function _resetToDefaults(node) {
  const mem = _getMem(node);
  if (!mem.defaults || Object.keys(mem.defaults).length === 0) return;
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
  if (changed > 0) node.setDirtyCanvas?.(true, true);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Modal
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function _showModal(html, title) {
  const old = document.getElementById("mec-param-modal");
  if (old) old.remove();

  const overlay = document.createElement("div");
  overlay.id = "mec-param-modal";
  overlay.style.cssText = "position:fixed;top:0;left:0;width:100vw;height:100vh;background:rgba(0,0,0,0.6);z-index:99999;display:flex;align-items:center;justify-content:center;";
  overlay.addEventListener("click", e => { if (e.target === overlay) overlay.remove(); });

  const dlg = document.createElement("div");
  dlg.style.cssText = "background:#181825;border:1px solid #313244;border-radius:12px;min-width:380px;max-width:680px;max-height:80vh;overflow:hidden;box-shadow:0 8px 32px rgba(0,0,0,0.5);";

  const bar = document.createElement("div");
  bar.style.cssText = "padding:10px 16px;background:#11111b;border-bottom:1px solid #313244;display:flex;justify-content:space-between;align-items:center;";
  bar.innerHTML = `<span style="color:#89b4fa;font-weight:bold;font-size:13px;">${_esc(title)}</span>
    <button id="mec-modal-x" style="background:none;border:none;color:#6c7086;cursor:pointer;font-size:16px;padding:2px 6px;">\u2715</button>`;
  dlg.appendChild(bar);

  const body = document.createElement("div");
  body.style.cssText = "max-height:calc(80vh - 46px);overflow-y:auto;";
  body.innerHTML = html;
  dlg.appendChild(body);

  overlay.appendChild(dlg);
  document.body.appendChild(overlay);

  document.getElementById("mec-modal-x")?.addEventListener("click", () => overlay.remove());
  const esc = e => {
    if (e.key === "Escape") { overlay.remove(); document.removeEventListener("keydown", esc); }
  };
  document.addEventListener("keydown", esc);
}

function _esc(s) {
  if (typeof s !== "string") s = String(s ?? "");
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

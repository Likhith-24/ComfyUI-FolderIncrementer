/**
 * Universal Reroute ("Dot") Node — MEC dynamic rerouter for ComfyUI
 *
 * Virtual node: strips itself from the backend prompt so it never
 * causes "Required input is missing" errors.
 *
 * Features:
 *   - Drop onto ANY connection → auto-adapts slot types
 *   - Compact circle with web-strand accents (lightweight canvas only)
 *   - Bundle drop: intercept nearby wires on placement
 *   - Right-click → "Remove Reroute (reconnect)" to dissolve
 *   - Double-click to toggle type label
 *   - Zero GPU cost — pure Canvas2D rendering
 */

import { app } from "../../scripts/app.js";

const NODE_TYPE   = "UniversalRerouteMEC";
const NODE_WIDTH  = 40;
const NODE_HEIGHT = 30;
const DOT_RADIUS  = 9;
const HIT_RADIUS  = 100;

// ── Type → color (matches ComfyUI link palette) ─────────────────────
const TYPE_COLORS = {
  IMAGE:        "#64b5f6",
  LATENT:       "#ff6e9c",
  MASK:         "#81c784",
  MODEL:        "#b39ddb",
  CLIP:         "#ffd54f",
  VAE:          "#4dd0e1",
  CONDITIONING: "#ffa726",
  INT:          "#a1c4fd",
  FLOAT:        "#a1c4fd",
  STRING:       "#c5e1a5",
  BOOLEAN:      "#ce93d8",
  COMBO:        "#90a4ae",
  BBOX:         "#ef9a9a",
  SAM_MODEL:    "#b39ddb",
  CONTROL_NET:  "#4db6ac",
  SEC_MODEL:    "#e57373",
  SAM2MODEL:    "#b39ddb",
  "*":          "#888",
};

function typeColor(t) { return TYPE_COLORS[t] || TYPE_COLORS["*"]; }

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app.registerExtension({
  name: "MEC.UniversalReroute",

  // ── Backend node hooks ─────────────────────────────────────────────
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;
    nodeType.prototype.onConnectInput  = () => true;
    nodeType.prototype.onConnectOutput = () => true;
  },

  // ── Per-instance setup ─────────────────────────────────────────────
  nodeCreated(node) {
    if (node.comfyClass !== NODE_TYPE) return;

    node.setSize([NODE_WIDTH, NODE_HEIGHT]);
    node.color   = "#1a1a2e";
    node.bgcolor = "#1a1a2e";
    node.shape   = LiteGraph.BOX_SHAPE;
    node.serialize_widgets = false;
    node.isVirtualNode = true;

    if (!node.properties) node.properties = {};
    node.properties.showLabel = false;

    // ── Slot type adaptation ─────────────────────────────────────────
    const origCC = node.onConnectionsChange;
    node.onConnectionsChange = function (side, _idx, connected, linkInfo) {
      origCC?.apply(this, arguments);
      if (!linkInfo) return;
      const g = this.graph || app.graph;
      if (!g) return;
      const link = g.links?.[linkInfo.id ?? linkInfo];
      if (!link) return;

      let resolved = null;
      if (side === LiteGraph.INPUT && connected) {
        const src = g.getNodeById(link.origin_id);
        resolved = src?.outputs?.[link.origin_slot]?.type || "*";
      } else if (side === LiteGraph.OUTPUT && connected) {
        const tgt = g.getNodeById(link.target_id);
        resolved = tgt?.inputs?.[link.target_slot]?.type || "*";
      }

      if (resolved && resolved !== "*") {
        if (this.inputs?.[0])  { this.inputs[0].type = resolved;  this.inputs[0].name = ""; }
        if (this.outputs?.[0]) { this.outputs[0].type = resolved; this.outputs[0].name = ""; }
      }

      if (!connected) {
        const hasIn  = this.inputs?.[0]?.link != null;
        const hasOut = this.outputs?.[0]?.links?.length > 0;
        if (!hasIn && !hasOut) {
          if (this.inputs?.[0])  { this.inputs[0].type = "*"; this.inputs[0].name = ""; }
          if (this.outputs?.[0]) { this.outputs[0].type = "*"; this.outputs[0].name = ""; }
        }
      }
      this.setDirtyCanvas?.(true, true);
    };

    // ── Draw: compact circle with web-strand accents ─────────────────
    node.onDrawForeground = function (ctx) {
      const t = this.inputs?.[0]?.type || this.outputs?.[0]?.type || "*";
      const c = typeColor(t);
      const cx = this.size[0] / 2;
      const cy = this.size[1] / 2;
      const r  = DOT_RADIUS;

      // Outer ring
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = "#16213e";
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = c;
      ctx.stroke();

      // Inner dot
      ctx.beginPath();
      ctx.arc(cx, cy, r * 0.35, 0, Math.PI * 2);
      ctx.fillStyle = c;
      ctx.fill();

      // Web strands — 6 thin lines radiating from center (very lightweight)
      ctx.save();
      ctx.globalAlpha = 0.25;
      ctx.strokeStyle = c;
      ctx.lineWidth = 0.7;
      for (let i = 0; i < 6; i++) {
        const angle = (Math.PI * 2 * i) / 6;
        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(angle) * r * 0.4, cy + Math.sin(angle) * r * 0.4);
        ctx.lineTo(cx + Math.cos(angle) * r * 0.9, cy + Math.sin(angle) * r * 0.9);
        ctx.stroke();
      }
      ctx.restore();

      // Type label
      if (this.properties.showLabel && t !== "*") {
        ctx.font = "8px Inter, system-ui, sans-serif";
        ctx.fillStyle = "#bac2de";
        ctx.textAlign = "center";
        ctx.fillText(t, cx, cy + r + 11);
      }
    };

    // ── Double-click → toggle label ──────────────────────────────────
    const origDbl = node.onDblClick;
    node.onDblClick = function () {
      origDbl?.apply(this, arguments);
      this.properties.showLabel = !this.properties.showLabel;
      this.setDirtyCanvas?.(true, true);
    };

    // ── Context menu ─────────────────────────────────────────────────
    const origMenu = node.getExtraMenuOptions;
    node.getExtraMenuOptions = function (_canvas, options) {
      origMenu?.apply(this, arguments);
      options.unshift(
        {
          content: "Remove Reroute (reconnect)",
          callback: () => dissolveReroute(this),
        },
        {
          content: this.properties.showLabel ? "Hide Type Label" : "Show Type Label",
          callback: () => {
            this.properties.showLabel = !this.properties.showLabel;
            this.setDirtyCanvas?.(true, true);
          },
        },
      );
    };
  },

  // ── Canvas-level setup ─────────────────────────────────────────────
  setup() {
    // Strip from prompt before execution
    const origGTP = app.graphToPrompt?.bind(app);
    if (origGTP) {
      app.graphToPrompt = async function () {
        const p = await origGTP();
        if (p?.output) {
          for (const k of Object.keys(p.output)) {
            if (p.output[k]?.class_type === NODE_TYPE) delete p.output[k];
          }
        }
        return p;
      };
    }

    // Bundle-drop on move
    const origMoved = app.canvas?.onNodeMoved?.bind(app.canvas);
    if (app.canvas) {
      app.canvas.onNodeMoved = function (node) {
        origMoved?.(node);
        if (node?.comfyClass === NODE_TYPE && !node._mecWired) tryWireOnDrop(node);
      };
    }

    // Right-click canvas → "Insert Reroute (MEC)"
    const origCanvasMenu = LGraphCanvas.prototype.getCanvasMenuOptions;
    if (origCanvasMenu) {
      LGraphCanvas.prototype.getCanvasMenuOptions = function () {
        const opts = origCanvasMenu.apply(this, arguments);
        opts.push(null, {
          content: "Insert Reroute (MEC)",
          callback: () => insertRerouteAtMouse(this),
        });
        return opts;
      };
    }
  },
});

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Dissolve — reconnect source → targets, remove the dot
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function dissolveReroute(node) {
  const g = node.graph || app.graph;
  if (!g) return;

  const inLink = node.inputs?.[0]?.link;
  let srcId = null, srcSlot = null;
  if (inLink != null) {
    const lk = g.links?.[inLink];
    if (lk) { srcId = lk.origin_id; srcSlot = lk.origin_slot; }
  }

  const targets = [];
  for (const lid of (node.outputs?.[0]?.links || [])) {
    const lk = g.links?.[lid];
    if (lk) targets.push({ id: lk.target_id, slot: lk.target_slot });
  }

  node.disconnectInput(0);
  node.disconnectOutput(0);

  if (srcId != null) {
    const src = g.getNodeById(srcId);
    if (src) {
      for (const t of targets) {
        const tgt = g.getNodeById(t.id);
        if (tgt) forceConnect(g, src, srcSlot, tgt, t.slot);
      }
    }
  }
  g.remove(node);
  g.setDirtyCanvas?.(true, true);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Wire-on-drop — insert into nearby link when node is placed
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function tryWireOnDrop(node) {
  const g = app.graph;
  if (!g) return;
  node._mecWired = true;

  const gx = node.pos[0] + NODE_WIDTH / 2;
  const gy = node.pos[1] + NODE_HEIGHT / 2;
  const hits = linksNearPoint(g, gx, gy, node.id);
  if (!hits.length) return;

  const h = hits[0];
  const srcNode = g.getNodeById(h.link.origin_id);
  const tgtNode = g.getNodeById(h.link.target_id);
  if (!srcNode || !tgtNode) return;

  const srcType = srcNode.outputs?.[h.link.origin_slot]?.type || "*";
  if (node.inputs?.[0])  node.inputs[0].type = srcType;
  if (node.outputs?.[0]) node.outputs[0].type = srcType;

  g.removeLink(h.link.id);
  forceConnect(g, srcNode, h.link.origin_slot, node, 0);
  forceConnect(g, node, 0, tgtNode, h.link.target_slot);
  g.setDirtyCanvas(true, true);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Insert Reroute at mouse position
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function insertRerouteAtMouse(canvas) {
  const g = canvas.graph || app.graph;
  if (!g) return;

  const pos = canvas.canvas_mouse || canvas.last_mouse_position;
  if (!pos) return;
  const gp = canvas.convertEventToCanvasOffset({ clientX: pos[0], clientY: pos[1] });
  const gx = gp[0] || pos[0], gy = gp[1] || pos[1];

  const near = linksNearPoint(g, gx, gy, -1);
  const rr = LiteGraph.createNode(NODE_TYPE);
  if (!rr) return;
  rr.pos = [gx - NODE_WIDTH / 2, gy - NODE_HEIGHT / 2];
  g.add(rr);

  if (near.length) {
    const h = near[0];
    const sn = g.getNodeById(h.link.origin_id);
    const tn = g.getNodeById(h.link.target_id);
    if (sn && tn) {
      const sType = sn.outputs?.[h.link.origin_slot]?.type || "*";
      if (rr.inputs?.[0])  rr.inputs[0].type = sType;
      if (rr.outputs?.[0]) rr.outputs[0].type = sType;
      g.removeLink(h.link.id);
      forceConnect(g, sn, h.link.origin_slot, rr, 0);
      forceConnect(g, rr, 0, tn, h.link.target_slot);
    }
  }
  g.setDirtyCanvas?.(true, true);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Geometry helpers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function forceConnect(graph, srcNode, srcSlot, dstNode, dstSlot) {
  const sOut = srcNode.outputs?.[srcSlot];
  const dIn  = dstNode.inputs?.[dstSlot];
  if (!sOut || !dIn) return;
  const oldS = sOut.type, oldD = dIn.type;
  sOut.type = "*"; dIn.type = "*";
  srcNode.connect(srcSlot, dstNode, dstSlot);
  sOut.type = oldS; dIn.type = oldD;
}

function linksNearPoint(graph, gx, gy, excludeId) {
  const hits = [];
  for (const lid in graph.links) {
    const lk = graph.links[lid];
    if (!lk) continue;
    if (lk.origin_id === excludeId || lk.target_id === excludeId) continue;
    const sn = graph.getNodeById(lk.origin_id);
    const tn = graph.getNodeById(lk.target_id);
    if (!sn || !tn) continue;
    const sp = sn.getConnectionPos(false, lk.origin_slot);
    const tp = tn.getConnectionPos(true,  lk.target_slot);
    if (!sp || !tp) continue;
    const d = ptSegDist(gx, gy, sp[0], sp[1], tp[0], tp[1]);
    if (d <= HIT_RADIUS) hits.push({ link: lk, dist: d });
  }
  hits.sort((a, b) => a.dist - b.dist);
  return hits;
}

function ptSegDist(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  const lenSq = dx * dx + dy * dy;
  if (lenSq === 0) return Math.hypot(px - ax, py - ay);
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lenSq));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

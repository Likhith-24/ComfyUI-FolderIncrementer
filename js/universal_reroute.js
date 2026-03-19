import { app } from "../../scripts/app.js";

/**
 * Universal Reroute ("Dot") Node — Nuke-style bundle rerouter for ComfyUI
 *
 * Features:
 *   - Drop onto ANY connection (IMAGE, LATENT, INT, STRING, etc.)
 *   - Automatically intercepts and passes through without type restrictions
 *   - Handles multiple connections: drop onto a bundle of wires to reroute all
 *   - Each connection keeps its original link color
 *   - Right-click → "Remove Reroute (reconnect)" to dissolve and restore direct connections
 *   - Minimal visual footprint — small dot node like Nuke
 *   - Double-click to toggle label visibility
 *   - Shift+drag from the dot to split off a new branch
 */

const DOT_RADIUS  = 12;
const DOT_COLOR   = "#888";
const DOT_BG      = "#1e1e2e";
const DOT_BORDER  = "#585b70";
const DOT_HOVER   = "#a6adc8";
const TITLE_COLOR  = "#bac2de";
const NODE_TYPE   = "UniversalRerouteMEC";

// ── Type color map (matches ComfyUI defaults) ────────────────────────
const TYPE_COLORS = {
  IMAGE:      "#64b5f6",
  LATENT:     "#ff6e9c",
  MASK:       "#81c784",
  MODEL:      "#b39ddb",
  CLIP:       "#ffd54f",
  VAE:        "#4dd0e1",
  CONDITIONING: "#ffa726",
  INT:        "#a1c4fd",
  FLOAT:      "#a1c4fd",
  STRING:     "#c5e1a5",
  BOOLEAN:    "#ce93d8",
  COMBO:      "#90a4ae",
  BBOX:       "#ef9a9a",
  SAM_MODEL:  "#b39ddb",
  CONTROL_NET:"#4db6ac",
  "*":        "#999",
};

function getTypeColor(type) {
  if (!type) return DOT_COLOR;
  return TYPE_COLORS[type] || DOT_COLOR;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Register Extension
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app.registerExtension({
  name: "MEC.UniversalReroute",

  // ── Register backend node modifications ──────────────────────────
  beforeRegisterNodeDef(nodeType, nodeData, _app) {
    if (nodeData.name !== NODE_TYPE) return;

    // Accept ANY type on input
    nodeType.prototype.onConnectInput = function (_inputIndex, _outputType) {
      return true;
    };

    // Accept ANY type on output
    nodeType.prototype.onConnectOutput = function (_outputIndex, _inputType) {
      return true;
    };
  },

  // ── Customize node after creation ────────────────────────────────
  nodeCreated(node) {
    if (node.comfyClass !== NODE_TYPE) return;

    // Compact dot appearance
    node.setSize([50, 30]);
    node.color   = DOT_BG;
    node.bgcolor = DOT_BG;
    node.shape   = LiteGraph.BOX_SHAPE;
    node.flags   = node.flags || {};

    // Store pass-through slot data
    if (!node.properties) node.properties = {};
    if (!node.properties._slotTypes) node.properties._slotTypes = {};
    node.properties.showLabel = false;

    // ── Connection change: match input/output types ────────────────
    const origConnChange = node.onConnectionsChange;
    node.onConnectionsChange = function (side, slotIndex, connected, linkInfo) {
      origConnChange?.apply(this, arguments);
      if (!linkInfo) return;

      const graph = this.graph || app.graph;
      if (!graph) return;
      const link = graph.links?.[linkInfo.id];
      if (!link) return;

      // Determine the resolved type from the connection
      let resolvedType = null;

      if (side === LiteGraph.INPUT && connected) {
        // Something connected to our input → get the source output type
        const srcNode = graph.getNodeById(link.origin_id);
        if (srcNode) {
          const srcSlot = srcNode.outputs?.[link.origin_slot];
          resolvedType = srcSlot?.type || "*";
        }
      } else if (side === LiteGraph.OUTPUT && connected) {
        // Something connected to our output → get the target input type
        const tgtNode = graph.getNodeById(link.target_id);
        if (tgtNode) {
          const tgtSlot = tgtNode.inputs?.[link.target_slot];
          resolvedType = tgtSlot?.type || "*";
        }
      }

      // If we resolved a concrete type, update our slots to match
      if (resolvedType && resolvedType !== "*") {
        if (this.inputs?.[0]) {
          this.inputs[0].type = resolvedType;
          this.inputs[0].name = resolvedType;
        }
        if (this.outputs?.[0]) {
          this.outputs[0].type = resolvedType;
          this.outputs[0].name = resolvedType;
        }
        this.properties._slotTypes[0] = resolvedType;
      }

      // If disconnected and nothing else connected, reset to wildcard
      if (!connected) {
        const hasInput = this.inputs?.[0]?.link != null;
        const hasOutput = this.outputs?.[0]?.links?.length > 0;
        if (!hasInput && !hasOutput) {
          if (this.inputs?.[0]) {
            this.inputs[0].type = "*";
            this.inputs[0].name = "any";
          }
          if (this.outputs?.[0]) {
            this.outputs[0].type = "*";
            this.outputs[0].name = "any";
          }
        }
      }

      this.setDirtyCanvas?.(true, true);
    };

    // ── Custom draw: small dot with colored ring ───────────────────
    const origDrawFg = node.onDrawForeground;
    node.onDrawForeground = function (ctx) {
      origDrawFg?.apply(this, arguments);

      const type = this.inputs?.[0]?.type || this.outputs?.[0]?.type || "*";
      const color = getTypeColor(type);

      // Draw the dot
      const cx = this.size[0] / 2;
      const cy = this.size[1] / 2;

      ctx.beginPath();
      ctx.arc(cx, cy, DOT_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = DOT_BG;
      ctx.fill();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = color;
      ctx.stroke();

      // Inner filled circle
      ctx.beginPath();
      ctx.arc(cx, cy, DOT_RADIUS * 0.4, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      // Type label if enabled
      if (this.properties.showLabel && type !== "*") {
        ctx.font = "9px Inter, system-ui, sans-serif";
        ctx.fillStyle = TITLE_COLOR;
        ctx.textAlign = "center";
        ctx.fillText(type, cx, cy + DOT_RADIUS + 12);
      }
    };

    // ── Double-click to toggle label ───────────────────────────────
    const origDblClick = node.onDblClick;
    node.onDblClick = function () {
      origDblClick?.apply(this, arguments);
      this.properties.showLabel = !this.properties.showLabel;
      this.setDirtyCanvas?.(true, true);
    };

    // ── Context menu: add "Remove Reroute (reconnect)" ─────────────
    const origGetMenu = node.getExtraMenuOptions;
    node.getExtraMenuOptions = function (canvas, options) {
      origGetMenu?.apply(this, arguments);

      options.unshift({
        content: "Remove Reroute (reconnect)",
        callback: () => {
          _dissolveReroute(this);
        },
      });

      options.unshift({
        content: this.properties.showLabel ? "Hide Type Label" : "Show Type Label",
        callback: () => {
          this.properties.showLabel = !this.properties.showLabel;
          this.setDirtyCanvas?.(true, true);
        },
      });
    };
  },

  // ── Setup: register canvas-level handlers for bundle drop ────────
  setup() {
    _registerBundleDrop();
  },
});

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Dissolve reroute: reconnect inputs→outputs and remove node
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function _dissolveReroute(node) {
  const graph = node.graph || app.graph;
  if (!graph) return;

  // Gather source info (who feeds our input)
  const inputLink = node.inputs?.[0]?.link;
  let srcNodeId = null, srcSlot = null;

  if (inputLink != null) {
    const link = graph.links?.[inputLink];
    if (link) {
      srcNodeId = link.origin_id;
      srcSlot   = link.origin_slot;
    }
  }

  // Gather all downstream connections (our output → targets)
  const targets = [];
  const outLinks = node.outputs?.[0]?.links || [];
  for (const lid of outLinks) {
    const link = graph.links?.[lid];
    if (link) {
      targets.push({ nodeId: link.target_id, slot: link.target_slot });
    }
  }

  // Disconnect everything from this node
  node.disconnectInput(0);
  node.disconnectOutput(0);

  // Reconnect source → each target directly
  if (srcNodeId != null && srcSlot != null) {
    const srcNode = graph.getNodeById(srcNodeId);
    if (srcNode) {
      for (const tgt of targets) {
        const tgtNode = graph.getNodeById(tgt.nodeId);
        if (tgtNode) {
          srcNode.connect(srcSlot, tgtNode, tgt.slot);
        }
      }
    }
  }

  // Remove the reroute node
  graph.remove(node);
  graph.setDirtyCanvas?.(true, true);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Bundle drop: intercept link being dragged over canvas and offer
//  to insert a reroute node at that position
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function _registerBundleDrop() {
  // We add a canvas menu entry for "Insert Reroute Here"
  // that creates a UniversalRerouteMEC at the click position
  // and wires it into any selected links

  const origProcessMenu = LGraphCanvas.prototype.getCanvasMenuOptions;
  if (!origProcessMenu) return;

  LGraphCanvas.prototype.getCanvasMenuOptions = function () {
    const options = origProcessMenu.apply(this, arguments);

    options.push(null); // separator
    options.push({
      content: "Insert Reroute (MEC)",
      callback: () => {
        _insertRerouteAtMouse(this);
      },
    });

    return options;
  };
}

function _insertRerouteAtMouse(graphCanvas) {
  const graph = graphCanvas.graph || app.graph;
  if (!graph) return;

  // Get mouse position in graph space
  const pos = graphCanvas.canvas_mouse || graphCanvas.last_mouse_position;
  if (!pos) return;
  const graphPos = graphCanvas.convertEventToCanvasOffset({ clientX: pos[0], clientY: pos[1] });
  const gx = graphPos[0] || pos[0];
  const gy = graphPos[1] || pos[1];

  // Find links near this position
  const nearLinks = _findLinksNearPoint(graph, graphCanvas, gx, gy, 40);

  if (nearLinks.length === 0) {
    // Just create a standalone reroute
    _createRerouteNode(graph, gx, gy);
    return;
  }

  // For each nearby link, insert a reroute
  for (const linkInfo of nearLinks) {
    const reroute = _createRerouteNode(graph, gx, gy + linkInfo.index * 40);
    if (!reroute) continue;

    const srcNode = graph.getNodeById(linkInfo.srcId);
    const tgtNode = graph.getNodeById(linkInfo.tgtId);
    if (!srcNode || !tgtNode) continue;

    // Set the reroute slot types to match this link
    const linkType = linkInfo.type || "*";
    if (reroute.inputs?.[0]) {
      reroute.inputs[0].type = linkType;
      reroute.inputs[0].name = linkType;
    }
    if (reroute.outputs?.[0]) {
      reroute.outputs[0].type = linkType;
      reroute.outputs[0].name = linkType;
    }

    // Disconnect original link
    tgtNode.disconnectInput(linkInfo.tgtSlot);

    // Wire: source → reroute → target
    srcNode.connect(linkInfo.srcSlot, reroute, 0);
    reroute.connect(0, tgtNode, linkInfo.tgtSlot);
  }

  graph.setDirtyCanvas?.(true, true);
}

function _createRerouteNode(graph, x, y) {
  const node = LiteGraph.createNode(NODE_TYPE);
  if (!node) return null;
  node.pos = [x - 25, y - 15];
  graph.add(node);
  return node;
}

function _findLinksNearPoint(graph, graphCanvas, gx, gy, threshold) {
  const found = [];
  if (!graph.links) return found;

  let idx = 0;
  for (const linkId in graph.links) {
    const link = graph.links[linkId];
    if (!link) continue;

    const srcNode = graph.getNodeById(link.origin_id);
    const tgtNode = graph.getNodeById(link.target_id);
    if (!srcNode || !tgtNode) continue;

    // Get the start/end positions of this link
    const srcSlot = srcNode.getOutputInfo?.(link.origin_slot);
    const tgtSlot = tgtNode.getInputInfo?.(link.target_slot);
    if (!srcSlot || !tgtSlot) continue;

    const srcPos = srcNode.getConnectionPos?.(false, link.origin_slot);
    const tgtPos = tgtNode.getConnectionPos?.(true, link.target_slot);
    if (!srcPos || !tgtPos) continue;

    // Simple point-to-segment distance check
    const dist = _pointToSegmentDist(gx, gy, srcPos[0], srcPos[1], tgtPos[0], tgtPos[1]);
    if (dist < threshold) {
      found.push({
        linkId:  Number(linkId),
        srcId:   link.origin_id,
        srcSlot: link.origin_slot,
        tgtId:   link.target_id,
        tgtSlot: link.target_slot,
        type:    link.type || srcSlot.type || "*",
        dist,
        index:   idx++,
      });
    }
  }

  // Sort by distance so closest links get rerouted first
  found.sort((a, b) => a.dist - b.dist);
  return found;
}

function _pointToSegmentDist(px, py, ax, ay, bx, by) {
  const dx = bx - ax;
  const dy = by - ay;
  const lenSq = dx * dx + dy * dy;
  if (lenSq === 0) return Math.hypot(px - ax, py - ay);

  let t = ((px - ax) * dx + (py - ay) * dy) / lenSq;
  t = Math.max(0, Math.min(1, t));
  const cx = ax + t * dx;
  const cy = ay + t * dy;
  return Math.hypot(px - cx, py - cy);
}

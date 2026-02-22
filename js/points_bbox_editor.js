import { app } from "../../scripts/app.js";

/**
 * MaskEditControl – Unified Points & BBox Editor Widget
 *
 * Interaction model (NO mode switching):
 *   Left-click             → add positive point (label=1)
 *   Right-click            → add negative point (label=0)
 *   CTRL + Left-drag       → draw positive bounding box (green)
 *   CTRL + Right-drag      → draw negative bounding box (red)
 *   Shift + click on point → delete that point
 *   Shift + click on bbox  → delete that bbox
 *   Delete / Backspace     → delete hovered element
 *   Middle-drag            → pan canvas
 *   Scroll                 → adjust point radius
 *   CTRL + Scroll          → zoom canvas
 *   CTRL + Z               → undo
 *   CTRL + Shift + Z / Y   → redo
 *   CTRL + C               → clear all
 *   R                      → reset view (zoom & pan)
 */

// ── Visual constants ─────────────────────────────────────────────────
const POINT_COLORS    = { positive: "#22d65a", negative: "#ff4466" };
const BBOX_COLORS     = { positive: "#22d65a", negative: "#ff4466" };
const CROSSHAIR_COLOR = "#ffffffaa";
const GRID_COLOR      = "#ffffff08";
const CANVAS_BG       = "#181825";
const TOOLBAR_BG      = "#11111bdd";
const TOOLBAR_H       = 32;
const BTN_COLORS = {
    default:  { bg: "#313244", fg: "#cdd6f4", hover: "#45475a" },
    danger:   { bg: "#45171e", fg: "#ff6688", hover: "#5c1f28" },
    accent:   { bg: "#1e3a2f", fg: "#22d65a", hover: "#2a5040" },
};

// ── Rounded rect helper ──────────────────────────────────────────────
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
class PointsBBoxEditor {
    constructor(node) {
        this.node = node;
        this.points = [];
        this.bboxes = [];      // [[x1,y1,x2,y2,label], ...]
        this.currentRadius = 3.0;
        this.zoom = 1.0;
        this.panX = 0;
        this.panY = 0;
        this.hoveredPoint = -1;
        this.hoveredBbox  = -1;

        // Drag state
        this.isPanning  = false;
        this.panStart   = null;
        this.isBboxDrag = false;
        this.bboxDragLabel = 1;
        this.dragStart  = null;
        this._dragEnd   = null;

        // Reference image
        this._refImage  = null;
        this._refLoaded = false;
        this._lastRefUrl = null;
        this._containerEl = null;
        this._canvasW   = 512;
        this._canvasH   = 512;

        // Undo/redo
        this._undoStack = [];
        this._redoStack = [];

        // Toolbar state
        this._toolbarButtons = [];
        this._hoveredButton  = null;

        // Mouse pos for status bar
        this._mouseCanvasX = 0;
        this._mouseCanvasY = 0;
    }

    // ── Coordinate transforms ────────────────────────────────────────
    screenToCanvas(sx, sy) {
        return { x: (sx - this.panX) / this.zoom, y: (sy - this.panY) / this.zoom };
    }
    canvasToScreen(cx, cy) {
        return { x: cx * this.zoom + this.panX, y: cy * this.zoom + this.panY };
    }

    // ── Hit testing ──────────────────────────────────────────────────
    findPointAt(cx, cy) {
        const threshold = Math.max(8, 10 / this.zoom);
        for (let i = this.points.length - 1; i >= 0; i--) {
            const p = this.points[i];
            const dx = p.x - cx, dy = p.y - cy;
            const r  = p.radius || this.currentRadius;
            if (dx * dx + dy * dy <= (Math.max(r, threshold)) ** 2) return i;
        }
        return -1;
    }
    findBboxAt(cx, cy) {
        for (let i = this.bboxes.length - 1; i >= 0; i--) {
            const b = this.bboxes[i];
            if (cx >= b[0] && cx <= b[2] && cy >= b[1] && cy <= b[3]) return i;
        }
        return -1;
    }

    // ── Undo / Redo ──────────────────────────────────────────────────
    saveState() {
        this._undoStack.push(JSON.stringify({ points: this.points, bboxes: this.bboxes }));
        if (this._undoStack.length > 50) this._undoStack.shift();
        this._redoStack = [];
    }
    undo() {
        if (!this._undoStack.length) return;
        this._redoStack.push(JSON.stringify({ points: this.points, bboxes: this.bboxes }));
        const prev = JSON.parse(this._undoStack.pop());
        this.points = prev.points;
        this.bboxes = prev.bboxes;
        this.updateWidgets();
    }
    redo() {
        if (!this._redoStack.length) return;
        this._undoStack.push(JSON.stringify({ points: this.points, bboxes: this.bboxes }));
        const next = JSON.parse(this._redoStack.pop());
        this.points = next.points;
        this.bboxes = next.bboxes;
        this.updateWidgets();
    }

    // ── Widget sync ──────────────────────────────────────────────────
    updateWidgets() {
        const data = JSON.stringify({
            points: this.points,
            bboxes: this.bboxes.map(b => b.slice(0, 4)),
        });
        const w = this.node.widgets?.find(w => w.name === "editor_data");
        if (w) { w.value = data; w.callback?.(data); }
    }

    // ── Load reference image ─────────────────────────────────────────
    loadRefImage(imageUrl) {
        if (!imageUrl || imageUrl === this._lastRefUrl) return;
        this._lastRefUrl = imageUrl;
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => {
            this._refImage  = img;
            this._refLoaded = true;
            this._canvasW   = img.naturalWidth;
            this._canvasH   = img.naturalHeight;
            const wW = this.node.widgets?.find(w => w.name === "width");
            const hW = this.node.widgets?.find(w => w.name === "height");
            if (wW) wW.value = img.naturalWidth;
            if (hW) hW.value = img.naturalHeight;
            if (this._containerEl) {
                const r = this._containerEl.getBoundingClientRect();
                if (r.width > 0 && r.height > 0) this.fitView(r.width, r.height - TOOLBAR_H);
            }
            this.node._mecRender?.();
        };
        img.onerror = () => console.warn("[MEC] Failed to load ref image:", imageUrl);
        img.src = imageUrl;
    }

    fitView(containerW, containerH) {
        if (this._canvasW <= 0 || this._canvasH <= 0) return;
        this.zoom = Math.min(containerW / this._canvasW, containerH / this._canvasH) * 0.92;
        this.panX = (containerW - this._canvasW * this.zoom) / 2;
        this.panY = (containerH - this._canvasH * this.zoom) / 2;
    }

    // ── Main draw ────────────────────────────────────────────────────
    draw(ctx, x, y, w, h) {
        const toolbarY = y;
        const canvasY  = y + TOOLBAR_H;
        const canvasH  = h - TOOLBAR_H;

        // === Canvas area ===
        ctx.save();
        ctx.beginPath();
        ctx.rect(x, canvasY, w, canvasH);
        ctx.clip();

        ctx.fillStyle = CANVAS_BG;
        ctx.fillRect(x, canvasY, w, canvasH);

        ctx.save();
        ctx.translate(x + this.panX, canvasY + this.panY);
        ctx.scale(this.zoom, this.zoom);

        // Reference image or grid
        if (this._refLoaded && this._refImage) {
            ctx.drawImage(this._refImage, 0, 0, this._canvasW, this._canvasH);
            ctx.fillStyle = "#00000010";
            ctx.fillRect(0, 0, this._canvasW, this._canvasH);
        } else {
            ctx.strokeStyle = GRID_COLOR;
            ctx.lineWidth = 1 / this.zoom;
            const step = 64;
            for (let gx = 0; gx <= this._canvasW; gx += step) {
                ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, this._canvasH); ctx.stroke();
            }
            for (let gy = 0; gy <= this._canvasH; gy += step) {
                ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(this._canvasW, gy); ctx.stroke();
            }
            ctx.strokeStyle = "#585b7040";
            ctx.lineWidth = 1.5 / this.zoom;
            ctx.strokeRect(0, 0, this._canvasW, this._canvasH);
        }

        // ── Bboxes ───────────────────────────────────────────────────
        for (let i = 0; i < this.bboxes.length; i++) {
            const b = this.bboxes[i];
            const isPos = (b[4] ?? 1) === 1;
            const color = isPos ? BBOX_COLORS.positive : BBOX_COLORS.negative;
            const hov = (i === this.hoveredBbox);
            const bw = b[2] - b[0], bh = b[3] - b[1];

            ctx.fillStyle = color + (hov ? "30" : "15");
            ctx.fillRect(b[0], b[1], bw, bh);
            ctx.strokeStyle = color + (hov ? "dd" : "88");
            ctx.lineWidth = (hov ? 2.5 : 1.5) / this.zoom;
            ctx.setLineDash(isPos ? [] : [6 / this.zoom, 4 / this.zoom]);
            ctx.strokeRect(b[0], b[1], bw, bh);
            ctx.setLineDash([]);

            // Corner markers
            const cm = Math.min(12, Math.min(bw, bh) / 3) / this.zoom;
            ctx.strokeStyle = color + "cc";
            ctx.lineWidth = 2 / this.zoom;
            for (const [cx, cy, dx, dy] of [
                [b[0], b[1], 1, 1], [b[2], b[1], -1, 1],
                [b[0], b[3], 1, -1], [b[2], b[3], -1, -1]
            ]) {
                ctx.beginPath();
                ctx.moveTo(cx + dx * cm, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy + dy * cm);
                ctx.stroke();
            }

            // Label badge
            const label = isPos ? `+B${i}` : `-B${i}`;
            const fs = Math.max(9, 11 / this.zoom);
            ctx.font = `bold ${fs}px Inter, system-ui, sans-serif`;
            const tw = ctx.measureText(label).width;
            const badgeH = fs + 4 / this.zoom;
            ctx.fillStyle = color + "cc";
            _roundRect(ctx, b[0], b[1] - badgeH - 2 / this.zoom, tw + 8 / this.zoom, badgeH, 3 / this.zoom);
            ctx.fill();
            ctx.fillStyle = "#000000cc";
            ctx.fillText(label, b[0] + 4 / this.zoom, b[1] - 4 / this.zoom);
        }

        // ── In-progress bbox drag ────────────────────────────────────
        if (this.isBboxDrag && this.dragStart && this._dragEnd) {
            const bx1 = Math.min(this.dragStart.x, this._dragEnd.x);
            const by1 = Math.min(this.dragStart.y, this._dragEnd.y);
            const bx2 = Math.max(this.dragStart.x, this._dragEnd.x);
            const by2 = Math.max(this.dragStart.y, this._dragEnd.y);
            const color = this.bboxDragLabel === 1 ? BBOX_COLORS.positive : BBOX_COLORS.negative;

            ctx.fillStyle = color + "20";
            ctx.fillRect(bx1, by1, bx2 - bx1, by2 - by1);
            ctx.strokeStyle = color + "cc";
            ctx.lineWidth = 2 / this.zoom;
            ctx.setLineDash([5 / this.zoom, 3 / this.zoom]);
            ctx.strokeRect(bx1, by1, bx2 - bx1, by2 - by1);
            ctx.setLineDash([]);

            const dimText = `${Math.abs(bx2 - bx1).toFixed(0)}\u00d7${Math.abs(by2 - by1).toFixed(0)}`;
            const fs2 = Math.max(9, 10 / this.zoom);
            ctx.font = `${fs2}px Inter, system-ui, sans-serif`;
            const dtw = ctx.measureText(dimText).width;
            ctx.fillStyle = "#000000bb";
            _roundRect(ctx, bx1, by2 + 4 / this.zoom, dtw + 8 / this.zoom, fs2 + 5 / this.zoom, 3 / this.zoom);
            ctx.fill();
            ctx.fillStyle = "#ffffffee";
            ctx.fillText(dimText, bx1 + 4 / this.zoom, by2 + fs2 + 2 / this.zoom);
        }

        // ── Points ───────────────────────────────────────────────────
        for (let i = 0; i < this.points.length; i++) {
            const p = this.points[i];
            const r = p.radius || this.currentRadius;
            const hov = (i === this.hoveredPoint);
            const color = p.label === 1 ? POINT_COLORS.positive : POINT_COLORS.negative;

            // Outer glow ring
            ctx.beginPath();
            ctx.arc(p.x, p.y, r + 2 / this.zoom, 0, Math.PI * 2);
            ctx.strokeStyle = color + (hov ? "77" : "33");
            ctx.lineWidth = 3 / this.zoom;
            ctx.stroke();

            // Main radius circle
            ctx.beginPath();
            ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
            ctx.fillStyle = color + (hov ? "44" : "22");
            ctx.fill();
            ctx.strokeStyle = color + (hov ? "ee" : "aa");
            ctx.lineWidth = 1.5 / this.zoom;
            ctx.stroke();

            // Center dot
            const dotR = Math.max(2.5, 3.5 / this.zoom);
            ctx.beginPath();
            ctx.arc(p.x, p.y, dotR, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
            ctx.beginPath();
            ctx.arc(p.x, p.y, dotR + 1 / this.zoom, 0, Math.PI * 2);
            ctx.strokeStyle = "#ffffffaa";
            ctx.lineWidth = 0.8 / this.zoom;
            ctx.stroke();

            // Crosshair on hover
            if (hov) {
                ctx.strokeStyle = CROSSHAIR_COLOR;
                ctx.lineWidth = 0.8 / this.zoom;
                ctx.setLineDash([3 / this.zoom, 3 / this.zoom]);
                const ext = r + 8 / this.zoom;
                ctx.beginPath();
                ctx.moveTo(p.x - ext, p.y); ctx.lineTo(p.x + ext, p.y);
                ctx.moveTo(p.x, p.y - ext); ctx.lineTo(p.x, p.y + ext);
                ctx.stroke();
                ctx.setLineDash([]);
            }

            // Index badge
            const sign = p.label === 1 ? "+" : "\u2212";
            const idxText = `${sign}${i}`;
            const fs3 = Math.max(8, 9 / this.zoom);
            ctx.font = `bold ${fs3}px Inter, system-ui, sans-serif`;
            const idxW = ctx.measureText(idxText).width;
            const badgeX = p.x + r + 4 / this.zoom;
            const badgeY = p.y - fs3 / 2 - 2 / this.zoom;
            ctx.fillStyle = "#000000bb";
            _roundRect(ctx, badgeX - 2 / this.zoom, badgeY, idxW + 5 / this.zoom, fs3 + 3 / this.zoom, 2 / this.zoom);
            ctx.fill();
            ctx.fillStyle = color;
            ctx.fillText(idxText, badgeX, badgeY + fs3);

            // Tooltip on hover
            if (hov) {
                const text = `(${p.x.toFixed(1)}, ${p.y.toFixed(1)}) r=${r.toFixed(1)}`;
                const fs4 = Math.max(9, 10 / this.zoom);
                ctx.font = `${fs4}px Inter, system-ui, sans-serif`;
                const ttw = ctx.measureText(text).width;
                const tx = p.x - ttw / 2;
                const ty = p.y - r - 20 / this.zoom;
                ctx.fillStyle = "#181825ee";
                _roundRect(ctx, tx - 6 / this.zoom, ty - 1 / this.zoom, ttw + 12 / this.zoom, fs4 + 6 / this.zoom, 4 / this.zoom);
                ctx.fill();
                ctx.strokeStyle = color + "66";
                ctx.lineWidth = 1 / this.zoom;
                _roundRect(ctx, tx - 6 / this.zoom, ty - 1 / this.zoom, ttw + 12 / this.zoom, fs4 + 6 / this.zoom, 4 / this.zoom);
                ctx.stroke();
                ctx.fillStyle = "#cdd6f4";
                ctx.fillText(text, tx, ty + fs4);
            }
        }

        ctx.restore(); // zoom/pan
        ctx.restore(); // clip

        // === Toolbar ===
        this._drawToolbar(ctx, x, toolbarY, w);

        // === Status bar ===
        ctx.save();
        ctx.fillStyle = "#11111bcc";
        ctx.fillRect(x, y + h - 20, w, 20);
        ctx.fillStyle = "#a6adc8";
        ctx.font = "10px Inter, system-ui, sans-serif";
        const statusParts = [
            `${this._canvasW}\u00d7${this._canvasH}`,
            `cursor: ${this._mouseCanvasX.toFixed(0)},${this._mouseCanvasY.toFixed(0)}`,
            `zoom: ${(this.zoom * 100).toFixed(0)}%`,
        ];
        ctx.fillText(statusParts.join("  \u2502  "), x + 8, y + h - 6);
        const helpText = "L=+pt  R=\u2212pt  Ctrl+drag=bbox  Shift=del  Scroll=radius";
        const helpW = ctx.measureText(helpText).width;
        ctx.fillStyle = "#585b70";
        ctx.fillText(helpText, x + w - helpW - 8, y + h - 6);
        ctx.restore();
    }

    // ── Toolbar ──────────────────────────────────────────────────────
    _drawToolbar(ctx, x, y, w) {
        ctx.save();
        ctx.fillStyle = TOOLBAR_BG;
        ctx.fillRect(x, y, w, TOOLBAR_H);
        ctx.fillStyle = "#313244";
        ctx.fillRect(x, y + TOOLBAR_H - 1, w, 1);

        this._toolbarButtons = [];
        let bx = x + 6;
        const by = y + 4;
        const bh = TOOLBAR_H - 8;

        const posCount = this.points.filter(p => p.label === 1).length;
        const negCount = this.points.filter(p => p.label === 0).length;

        bx = this._drawPill(ctx, bx, by, bh, `${posCount}`, POINT_COLORS.positive, "+pts");
        bx = this._drawPill(ctx, bx + 4, by, bh, `${negCount}`, POINT_COLORS.negative, "\u2212pts");
        bx = this._drawPill(ctx, bx + 4, by, bh, `${this.bboxes.length}`, "#89b4fa", "bbox");
        bx += 8;
        ctx.fillStyle = "#45475a"; ctx.fillRect(bx, by + 2, 1, bh - 4); bx += 8;

        bx = this._drawPill(ctx, bx, by, bh, `r:${this.currentRadius.toFixed(1)}`, "#cdd6f4", null);
        bx += 8;
        ctx.fillStyle = "#45475a"; ctx.fillRect(bx, by + 2, 1, bh - 4); bx += 8;

        bx = this._drawButton(ctx, bx, by, bh, "\u2715 Pts", "clearPoints", BTN_COLORS.danger);
        bx += 4;
        bx = this._drawButton(ctx, bx, by, bh, "\u2715 BBox", "clearBboxes", BTN_COLORS.danger);
        bx += 4;
        bx = this._drawButton(ctx, bx, by, bh, "\u2715 All", "clearAll", BTN_COLORS.danger);
        bx += 8;
        ctx.fillStyle = "#45475a"; ctx.fillRect(bx, by + 2, 1, bh - 4); bx += 8;

        bx = this._drawButton(ctx, bx, by, bh, "\u21b6 Undo", "undo", BTN_COLORS.default);
        bx += 4;
        bx = this._drawButton(ctx, bx, by, bh, "Redo \u21b7", "redo", BTN_COLORS.default);
        bx += 4;
        bx = this._drawButton(ctx, bx, by, bh, "\u25a3 Fit", "fitView", BTN_COLORS.accent);

        ctx.restore();
    }

    _drawPill(ctx, x, y, h, text, color, subLabel) {
        ctx.font = "bold 10px Inter, system-ui, sans-serif";
        const tw = ctx.measureText(text).width;
        let subW = 0;
        if (subLabel) {
            ctx.font = "9px Inter, system-ui, sans-serif";
            subW = ctx.measureText(subLabel).width + 4;
        }
        const pw = tw + subW + 12;
        ctx.fillStyle = color + "22";
        _roundRect(ctx, x, y, pw, h, 4); ctx.fill();
        ctx.strokeStyle = color + "44"; ctx.lineWidth = 1;
        _roundRect(ctx, x, y, pw, h, 4); ctx.stroke();
        ctx.font = "bold 10px Inter, system-ui, sans-serif";
        ctx.fillStyle = color;
        ctx.fillText(text, x + 5, y + h / 2 + 4);
        if (subLabel) {
            ctx.fillStyle = color + "88";
            ctx.font = "9px Inter, system-ui, sans-serif";
            ctx.fillText(subLabel, x + tw + 7, y + h / 2 + 3);
        }
        return x + pw;
    }

    _drawButton(ctx, x, y, h, label, action, colors) {
        ctx.font = "bold 10px Inter, system-ui, sans-serif";
        const tw = ctx.measureText(label).width;
        const bw = tw + 14;
        const isHov = this._hoveredButton === action;
        ctx.fillStyle = isHov ? colors.hover : colors.bg;
        _roundRect(ctx, x, y, bw, h, 4); ctx.fill();
        ctx.fillStyle = colors.fg;
        ctx.fillText(label, x + 7, y + h / 2 + 4);
        this._toolbarButtons.push({ x, y, w: bw, h, action });
        return x + bw;
    }

    handleToolbarClick(action, renderFn) {
        switch (action) {
            case "clearPoints":
                this.saveState();
                this.points = []; this.hoveredPoint = -1;
                this.updateWidgets(); break;
            case "clearBboxes":
                this.saveState();
                this.bboxes = []; this.hoveredBbox = -1;
                this.updateWidgets(); break;
            case "clearAll":
                this.saveState();
                this.points = []; this.bboxes = [];
                this.hoveredPoint = -1; this.hoveredBbox = -1;
                this.updateWidgets(); break;
            case "undo": this.undo(); break;
            case "redo": this.redo(); break;
            case "fitView":
                if (this._containerEl) {
                    const r = this._containerEl.getBoundingClientRect();
                    if (r.width > 0 && r.height > 0) this.fitView(r.width, r.height - TOOLBAR_H);
                }
                break;
        }
        renderFn?.();
    }

    findToolbarButton(sx, sy) {
        if (!this._toolbarButtons) return null;
        for (const btn of this._toolbarButtons) {
            if (sx >= btn.x && sx <= btn.x + btn.w && sy >= btn.y && sy <= btn.y + btn.h) return btn.action;
        }
        return null;
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Register with ComfyUI
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app.registerExtension({
    name: "MaskEditControl.PointsBBoxEditor",

    async nodeCreated(node) {
        if (!["PointsMaskEditor", "SAMMaskGeneratorMEC"].includes(node.comfyClass)) return;

        const editor = new PointsBBoxEditor(node);
        editor.saveState();

        // ── Sync canvas dimensions from widget values ────────────────
        function syncCanvasSize() {
            const wW = node.widgets?.find(w => w.name === "width");
            const hW = node.widgets?.find(w => w.name === "height");
            if (wW && hW) { editor._canvasW = wW.value; editor._canvasH = hW.value; }
        }
        syncCanvasSize();
        for (const wName of ["width", "height"]) {
            const wid = node.widgets?.find(w => w.name === wName);
            if (wid) {
                const origCb = wid.callback;
                wid.callback = function(v) {
                    origCb?.call(this, v);
                    syncCanvasSize();
                    if (typeof render === "function") render();
                };
            }
        }

        // ── Create DOM widget ────────────────────────────────────────
        const container = document.createElement("div");
        container.style.cssText = "width:100%;height:380px;position:relative;overflow:hidden;cursor:crosshair;border-radius:6px;border:1px solid #313244;box-sizing:border-box;background:#181825;";
        editor._containerEl = container;

        const canvas = document.createElement("canvas");
        canvas.style.cssText = "width:100%;height:100%;display:block;";
        container.appendChild(canvas);

        node.addDOMWidget("points_editor_canvas", "canvas", container, { serialize: false });
        const ctx = canvas.getContext("2d");

        // ── Resize ───────────────────────────────────────────────────
        function resize() {
            const rect = container.getBoundingClientRect();
            if (rect.width === 0 || rect.height === 0) return;
            const dpr = window.devicePixelRatio || 1;
            canvas.width  = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            render();
        }
        function render() {
            const rect = container.getBoundingClientRect();
            ctx.clearRect(0, 0, rect.width, rect.height);
            editor.draw(ctx, 0, 0, rect.width, rect.height);
        }
        const ro = new ResizeObserver(() => resize());
        ro.observe(container);
        setTimeout(resize, 150);

        // ── Try to load reference image ──────────────────────────────
        function tryLoadRefImage() {
            if (!node.inputs) return;
            for (const inp of node.inputs) {
                if (inp.name === "reference_image" && inp.link != null) {
                    const linkInfo = app.graph.links[inp.link];
                    if (!linkInfo) continue;
                    const srcNode = app.graph.getNodeById(linkInfo.origin_id);
                    if (!srcNode) continue;

                    if (srcNode.imgs?.length > 0) { editor.loadRefImage(srcNode.imgs[0].src); return; }

                    const imgW = srcNode.widgets?.find(w => w.name === "image");
                    if (imgW?.value) {
                        const parts = imgW.value.split("/");
                        const subfolder = parts.length > 1 ? parts.slice(0, -1).join("/") : "";
                        const filename = parts[parts.length - 1];
                        editor.loadRefImage(`/view?filename=${encodeURIComponent(filename)}&subfolder=${encodeURIComponent(subfolder)}&type=input`);
                        return;
                    }

                    if (srcNode.inputs) {
                        for (const upInp of srcNode.inputs) {
                            if (upInp.link != null) {
                                const upLink = app.graph.links[upInp.link];
                                if (!upLink) continue;
                                const upNode = app.graph.getNodeById(upLink.origin_id);
                                if (!upNode) continue;
                                if (upNode.imgs?.length > 0) { editor.loadRefImage(upNode.imgs[0].src); return; }
                                const upImgW = upNode.widgets?.find(w => w.name === "image");
                                if (upImgW?.value) {
                                    const p2 = upImgW.value.split("/");
                                    const sf = p2.length > 1 ? p2.slice(0, -1).join("/") : "";
                                    const fn = p2[p2.length - 1];
                                    editor.loadRefImage(`/view?filename=${encodeURIComponent(fn)}&subfolder=${encodeURIComponent(sf)}&type=input`);
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        }

        const origConnChange = node.onConnectionsChange;
        node.onConnectionsChange = function(type, index, connected, link_info) {
            origConnChange?.apply(this, arguments);
            if (connected) { setTimeout(() => { tryLoadRefImage(); render(); }, 200); }
            else { editor._refLoaded = false; editor._refImage = null; editor._lastRefUrl = null; syncCanvasSize(); render(); }
        };
        const _imgInterval = setInterval(() => {
            if (!editor._refLoaded) tryLoadRefImage(); else clearInterval(_imgInterval);
        }, 2000);
        const origExecuted = node.onExecuted;
        node.onExecuted = function(output) {
            origExecuted?.apply(this, arguments);
            tryLoadRefImage(); render();
        };

        // ── POINTER DOWN ─────────────────────────────────────────────
        canvas.addEventListener("pointerdown", (e) => {
            e.preventDefault(); e.stopPropagation();
            canvas.focus();
            canvas.setPointerCapture(e.pointerId);

            const rect = canvas.getBoundingClientRect();
            const sx = e.clientX - rect.left;
            const sy = e.clientY - rect.top;

            // Toolbar click?
            if (sy < TOOLBAR_H) {
                const action = editor.findToolbarButton(sx, sy);
                if (action) editor.handleToolbarClick(action, render);
                return;
            }

            const canvasSy = sy - TOOLBAR_H;
            const c = editor.screenToCanvas(sx, canvasSy);

            if (e.button === 1) {
                editor.isPanning = true;
                editor.panStart = { x: e.clientX - editor.panX, y: e.clientY - editor.panY };
                canvas.style.cursor = "grab"; return;
            }
            if (e.ctrlKey && !e.shiftKey && (e.button === 0 || e.button === 2)) {
                editor.isBboxDrag = true;
                editor.bboxDragLabel = e.button === 0 ? 1 : 0;
                editor.dragStart = { x: c.x, y: c.y };
                editor._dragEnd  = { x: c.x, y: c.y };
                return;
            }
            if (e.button === 0 || e.button === 2) {
                if (e.shiftKey) {
                    const idx = editor.findPointAt(c.x, c.y);
                    if (idx >= 0) {
                        editor.saveState(); editor.points.splice(idx, 1);
                        editor.hoveredPoint = -1; editor.updateWidgets(); render(); return;
                    }
                    const bidx = editor.findBboxAt(c.x, c.y);
                    if (bidx >= 0) {
                        editor.saveState(); editor.bboxes.splice(bidx, 1);
                        editor.hoveredBbox = -1; editor.updateWidgets(); render(); return;
                    }
                    return;
                }
                editor.saveState();
                editor.points.push({
                    x: parseFloat(c.x.toFixed(2)),
                    y: parseFloat(c.y.toFixed(2)),
                    label: e.button === 0 ? 1 : 0,
                    radius: editor.currentRadius,
                });
                editor.updateWidgets(); render();
            }
        });

        // ── POINTER MOVE ─────────────────────────────────────────────
        canvas.addEventListener("pointermove", (e) => {
            const rect = canvas.getBoundingClientRect();
            const sx = e.clientX - rect.left;
            const sy = e.clientY - rect.top;

            if (sy < TOOLBAR_H) {
                const oldHov = editor._hoveredButton;
                editor._hoveredButton = editor.findToolbarButton(sx, sy);
                canvas.style.cursor = editor._hoveredButton ? "pointer" : "crosshair";
                if (oldHov !== editor._hoveredButton) render();
                return;
            }
            if (editor._hoveredButton) { editor._hoveredButton = null; render(); }
            canvas.style.cursor = editor.isPanning ? "grab" : "crosshair";

            const canvasSy = sy - TOOLBAR_H;
            const c = editor.screenToCanvas(sx, canvasSy);
            editor._mouseCanvasX = c.x;
            editor._mouseCanvasY = c.y;

            if (editor.isPanning) {
                editor.panX = e.clientX - editor.panStart.x;
                editor.panY = e.clientY - editor.panStart.y;
                render(); return;
            }
            if (editor.isBboxDrag) { editor._dragEnd = { x: c.x, y: c.y }; render(); return; }

            const oldP = editor.hoveredPoint, oldB = editor.hoveredBbox;
            editor.hoveredPoint = editor.findPointAt(c.x, c.y);
            editor.hoveredBbox  = editor.hoveredPoint < 0 ? editor.findBboxAt(c.x, c.y) : -1;
            if (oldP !== editor.hoveredPoint || oldB !== editor.hoveredBbox) render();
        });

        // ── POINTER UP ───────────────────────────────────────────────
        canvas.addEventListener("pointerup", (e) => {
            try { canvas.releasePointerCapture(e.pointerId); } catch(_) {}
            if (editor.isPanning) { editor.isPanning = false; canvas.style.cursor = "crosshair"; return; }
            if (editor.isBboxDrag && editor.dragStart && editor._dragEnd) {
                const x1 = Math.min(editor.dragStart.x, editor._dragEnd.x);
                const y1 = Math.min(editor.dragStart.y, editor._dragEnd.y);
                const x2 = Math.max(editor.dragStart.x, editor._dragEnd.x);
                const y2 = Math.max(editor.dragStart.y, editor._dragEnd.y);
                if (x2 - x1 > 3 && y2 - y1 > 3) {
                    editor.saveState();
                    editor.bboxes.push([Math.round(x1), Math.round(y1), Math.round(x2), Math.round(y2), editor.bboxDragLabel]);
                    editor.updateWidgets();
                }
                editor.isBboxDrag = false; editor.dragStart = null; editor._dragEnd = null;
                render();
            }
        });

        // ── SCROLL ───────────────────────────────────────────────────
        canvas.addEventListener("wheel", (e) => {
            e.preventDefault();
            const rect = canvas.getBoundingClientRect();
            if (e.clientY - rect.top < TOOLBAR_H) return;
            if (e.ctrlKey) {
                const factor = e.deltaY < 0 ? 1.12 : 0.89;
                const mx = e.clientX - rect.left;
                const my = e.clientY - rect.top - TOOLBAR_H;
                editor.panX = mx - (mx - editor.panX) * factor;
                editor.panY = my - (my - editor.panY) * factor;
                editor.zoom *= factor;
                editor.zoom = Math.max(0.1, Math.min(editor.zoom, 20));
            } else {
                editor.currentRadius += e.deltaY < 0 ? 0.5 : -0.5;
                editor.currentRadius = Math.max(0.5, Math.min(editor.currentRadius, 256));
            }
            render();
        }, { passive: false });

        canvas.addEventListener("contextmenu", (e) => e.preventDefault());

        // ── Keyboard ─────────────────────────────────────────────────
        canvas.tabIndex = 0;
        canvas.style.outline = "none";
        canvas.addEventListener("keydown", (e) => {
            if (e.key === "z" && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                if (e.shiftKey) editor.redo(); else editor.undo();
                render();
            } else if (e.key === "y" && (e.ctrlKey || e.metaKey)) {
                e.preventDefault(); editor.redo(); render();
            } else if (e.key === "c" && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                editor.saveState(); editor.points = []; editor.bboxes = [];
                editor.hoveredPoint = -1; editor.hoveredBbox = -1;
                editor.updateWidgets(); render();
            } else if (e.key === "Delete" || e.key === "Backspace") {
                e.preventDefault();
                if (editor.hoveredPoint >= 0) {
                    editor.saveState(); editor.points.splice(editor.hoveredPoint, 1);
                    editor.hoveredPoint = -1; editor.updateWidgets(); render();
                } else if (editor.hoveredBbox >= 0) {
                    editor.saveState(); editor.bboxes.splice(editor.hoveredBbox, 1);
                    editor.hoveredBbox = -1; editor.updateWidgets(); render();
                }
            } else if (e.key === "r" || e.key === "R") {
                if (editor._containerEl) {
                    const r = editor._containerEl.getBoundingClientRect();
                    if (r.width > 0 && r.height > 0) editor.fitView(r.width, r.height - TOOLBAR_H);
                    else { editor.zoom = 1; editor.panX = 0; editor.panY = 0; }
                }
                render();
            }
        });

        node._mecEditor = editor;
        node._mecRender = render;
    },

    // ── Serialization hooks ──────────────────────────────────────────
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!["PointsMaskEditor", "SAMMaskGeneratorMEC"].includes(nodeData.name)) return;
        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            origOnConfigure?.apply(this, arguments);
            if (this._mecEditor && info._mecState) {
                this._mecEditor.points = info._mecState.points || [];
                this._mecEditor.bboxes = info._mecState.bboxes || [];
                this._mecEditor.updateWidgets();
                this._mecRender?.();
            }
        };
        const origSerialize = nodeType.prototype.serialize;
        nodeType.prototype.serialize = function() {
            const data = origSerialize?.apply(this, arguments) || {};
            if (this._mecEditor) {
                data._mecState = { points: this._mecEditor.points, bboxes: this._mecEditor.bboxes };
            }
            return data;
        };
    },
});

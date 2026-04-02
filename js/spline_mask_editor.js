import { app } from "../../scripts/app.js";

/**
 * MaskEditControl – Spline Mask Editor Widget
 *
 * Interactive canvas for drawing closed/open spline shapes.
 * Serializes control points to the `spline_data` hidden widget
 * so the Python node can rasterize them into a mask.
 *
 * Interaction:
 *   Left-click             → add control point to active shape
 *   Shift + Left-click     → delete hovered point
 *   Click & drag point     → move control point
 *   Middle-drag / Alt+drag → pan canvas
 *   Scroll                 → zoom (centered on cursor)
 *   N                      → new shape
 *   Z                      → undo last point add/move
 *   Delete / Backspace     → delete hovered point
 *   Escape                 → deselect active shape
 *   C                      → toggle closed/open for active shape
 */

// ── Visual constants ─────────────────────────────────────────────────
const CANVAS_BG        = "#181825";
const GRID_COLOR       = "#ffffff08";
const POINT_RADIUS     = 5;
const POINT_COLOR      = "#22d65a";
const POINT_HOVER      = "#80ffb0";
const POINT_SELECTED   = "#ffdd44";
const CURVE_COLOR      = "#4488ffcc";
const CURVE_WIDTH      = 2;
const FILL_COLOR       = "#4488ff22";
const TOOLBAR_BG       = "#1e1e2eee";
const TOOLBAR_H        = 32;
const BTN_COLORS       = {
    default: { bg: "#45475a", fg: "#cdd6f4", hover: "#585b70" },
    accent:  { bg: "#2a6040", fg: "#80ffb0", hover: "#3a7850" },
    danger:  { bg: "#6c2030", fg: "#ffb0c0", hover: "#8c2840" },
};
const HANDLE_COLOR     = "#ff8844aa";
const HANDLE_LINE      = "#ff884466";
const HANDLE_RADIUS    = 4;

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

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SplineEditor {
    constructor(node) {
        this.node = node;
        // shapes: [{points: [{x,y},...], closed: bool, type: string, handles: [{cp1x,cp1y,cp2x,cp2y},...]}]
        this.shapes = [{ points: [], closed: true, type: "catmull_rom", handles: [] }];
        this.activeShapeIdx = 0;
        this.zoom = 1.0;
        this.panX = 0;
        this.panY = 0;
        this.hoveredPoint = -1;
        this.selectedPoint = -1;

        // Drag state
        this.isDragging = false;
        this.dragPointIdx = -1;
        this.dragOriginal = null;
        this.isPanning = false;
        this.panStart = null;

        // Handle drag
        this.isDraggingHandle = false;
        this.dragHandleType = null; // "cp1" or "cp2"
        this.dragHandleIdx = -1;

        // Reference image
        this._refImage = null;
        this._refLoaded = false;
        this._lastRefUrl = null;
        this._canvasW = 512;
        this._canvasH = 512;
        this._hasAutoFitted = false;

        // Undo
        this._undoStack = [];

        // Toolbar
        this._toolbarButtons = [];
        this._hoveredButton = null;

        // Mouse position
        this._mouseX = 0;
        this._mouseY = 0;
    }

    get activeShape() {
        return this.shapes[this.activeShapeIdx] || null;
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
        const shape = this.activeShape;
        if (!shape) return -1;
        const threshold = Math.max(8, 10 / this.zoom);
        for (let i = shape.points.length - 1; i >= 0; i--) {
            const p = shape.points[i];
            const dx = p.x - cx, dy = p.y - cy;
            if (dx * dx + dy * dy <= threshold * threshold) return i;
        }
        return -1;
    }

    findHandleAt(cx, cy) {
        const shape = this.activeShape;
        if (!shape || shape.type !== "bezier") return null;
        const threshold = Math.max(6, 8 / this.zoom);
        for (let i = shape.handles.length - 1; i >= 0; i--) {
            const h = shape.handles[i];
            if (!h) continue;
            for (const type of ["cp1", "cp2"]) {
                const hx = h[type + "x"], hy = h[type + "y"];
                if (hx == null || hy == null) continue;
                const dx = hx - cx, dy = hy - cy;
                if (dx * dx + dy * dy <= threshold * threshold) {
                    return { idx: i, type };
                }
            }
        }
        return null;
    }

    // ── Undo ─────────────────────────────────────────────────────────
    _pushUndo() {
        this._undoStack.push(JSON.stringify(this.shapes));
        if (this._undoStack.length > 50) this._undoStack.shift();
    }
    undo() {
        if (this._undoStack.length === 0) return;
        this.shapes = JSON.parse(this._undoStack.pop());
        if (this.activeShapeIdx >= this.shapes.length) {
            this.activeShapeIdx = Math.max(0, this.shapes.length - 1);
        }
        this._serialize();
    }

    // ── Serialize to widget ──────────────────────────────────────────
    _serialize() {
        const w = this.node.widgets?.find(w => w.name === "spline_data");
        if (w) {
            w.value = JSON.stringify(this.shapes.map(s => ({
                points: s.points,
                closed: s.closed,
                type: s.type,
                handles: s.handles || [],
            })));
        }
    }

    _deserialize() {
        const w = this.node.widgets?.find(w => w.name === "spline_data");
        if (!w || !w.value) return;
        try {
            const data = JSON.parse(w.value);
            if (Array.isArray(data) && data.length > 0) {
                this.shapes = data.map(s => ({
                    points: s.points || [],
                    closed: s.closed !== false,
                    type: s.type || "catmull_rom",
                    handles: s.handles || [],
                }));
                this.activeShapeIdx = Math.min(this.activeShapeIdx, this.shapes.length - 1);
            }
        } catch (e) { /* ignore parse errors */ }
    }

    // ── Add point ────────────────────────────────────────────────────
    addPoint(cx, cy) {
        const shape = this.activeShape;
        if (!shape) return;
        this._pushUndo();
        shape.points.push({ x: cx, y: cy });
        // For bezier, add default handle (co-located with point = straight segment)
        if (shape.type === "bezier") {
            shape.handles.push({ cp1x: cx, cp1y: cy, cp2x: cx, cp2y: cy });
        }
        this._serialize();
    }

    deletePoint(idx) {
        const shape = this.activeShape;
        if (!shape || idx < 0 || idx >= shape.points.length) return;
        this._pushUndo();
        shape.points.splice(idx, 1);
        if (shape.handles.length > idx) shape.handles.splice(idx, 1);
        this.selectedPoint = -1;
        this.hoveredPoint = -1;
        this._serialize();
    }

    newShape() {
        this._pushUndo();
        const type = this.activeShape?.type || "catmull_rom";
        this.shapes.push({ points: [], closed: true, type, handles: [] });
        this.activeShapeIdx = this.shapes.length - 1;
        this._serialize();
    }

    toggleClosed() {
        const shape = this.activeShape;
        if (!shape) return;
        this._pushUndo();
        shape.closed = !shape.closed;
        this._serialize();
    }

    // ── Catmull-Rom sampling for preview ─────────────────────────────
    _sampleCatmullRom(points, closed) {
        const n = points.length;
        if (n < 2) return points.map(p => [p.x, p.y]);
        if (n === 2) {
            const result = [];
            for (let i = 0; i <= 20; i++) {
                const t = i / 20;
                result.push([
                    points[0].x + t * (points[1].x - points[0].x),
                    points[0].y + t * (points[1].y - points[0].y),
                ]);
            }
            return result;
        }

        const ext = [];
        if (closed) {
            ext.push(points[n - 1]);
            for (const p of points) ext.push(p);
            ext.push(points[0]);
            ext.push(points[1]);
        } else {
            ext.push({
                x: 2 * points[0].x - points[1].x,
                y: 2 * points[0].y - points[1].y,
            });
            for (const p of points) ext.push(p);
            ext.push({
                x: 2 * points[n - 1].x - points[n - 2].x,
                y: 2 * points[n - 1].y - points[n - 2].y,
            });
        }

        const segs = closed ? n : n - 1;
        const samplesPerSeg = 20;
        const result = [];

        for (let seg = 0; seg < segs; seg++) {
            const p0 = ext[seg], p1 = ext[seg + 1], p2 = ext[seg + 2], p3 = ext[seg + 3];
            const dist = (a, b) => Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2) + 1e-8;
            const d01 = Math.sqrt(dist(p0, p1));
            const d12 = Math.sqrt(dist(p1, p2));
            const d23 = Math.sqrt(dist(p2, p3));
            const t0 = 0, t1 = t0 + d01, t2 = t1 + d12, t3 = t2 + d23;

            for (let i = 0; i < samplesPerSeg; i++) {
                const t = t1 + (t2 - t1) * (i / samplesPerSeg);
                const lerp = (pa, pb, ta, tb) => {
                    const w = (t - ta) / Math.max(tb - ta, 1e-10);
                    return { x: pa.x + w * (pb.x - pa.x), y: pa.y + w * (pb.y - pa.y) };
                };
                const a1 = lerp(p0, p1, t0, t1);
                const a2 = lerp(p1, p2, t1, t2);
                const a3 = lerp(p2, p3, t2, t3);
                const b1 = lerp(a1, a2, t0, t2);
                const b2 = lerp(a2, a3, t1, t3);
                const c = lerp(b1, b2, t1, t2);
                result.push([c.x, c.y]);
            }
        }
        if (!closed && points.length > 0) {
            result.push([points[n - 1].x, points[n - 1].y]);
        }
        return result;
    }

    _sampleBezier(points, handles) {
        const n = points.length;
        if (n < 2) return points.map(p => [p.x, p.y]);
        const result = [];
        for (let seg = 0; seg < n - 1; seg++) {
            const p0 = points[seg], p1 = points[seg + 1];
            const h0 = handles[seg] || {};
            const h1 = handles[seg + 1] || {};
            const cp1x = h0.cp2x ?? p0.x, cp1y = h0.cp2y ?? p0.y;
            const cp2x = h1.cp1x ?? p1.x, cp2y = h1.cp1y ?? p1.y;
            for (let i = 0; i <= 20; i++) {
                const t = i / 20;
                const omt = 1 - t;
                result.push([
                    omt ** 3 * p0.x + 3 * omt ** 2 * t * cp1x + 3 * omt * t ** 2 * cp2x + t ** 3 * p1.x,
                    omt ** 3 * p0.y + 3 * omt ** 2 * t * cp1y + 3 * omt * t ** 2 * cp2y + t ** 3 * p1.y,
                ]);
            }
        }
        return result;
    }

    _sampleShape(shape) {
        if (!shape || shape.points.length < 2) return [];
        if (shape.type === "bezier" && shape.handles?.length > 0) {
            return this._sampleBezier(shape.points, shape.handles);
        }
        if (shape.type === "polyline") {
            const pts = shape.points.map(p => [p.x, p.y]);
            if (shape.closed && pts.length >= 3) pts.push(pts[0]);
            return pts;
        }
        return this._sampleCatmullRom(shape.points, shape.closed);
    }

    // ── Auto-fit view ────────────────────────────────────────────────
    autoFit(widgetW, widgetH) {
        const drawH = widgetH - TOOLBAR_H;
        if (drawH <= 0) return;
        const scaleX = widgetW / this._canvasW;
        const scaleY = drawH / this._canvasH;
        this.zoom = Math.min(scaleX, scaleY) * 0.95;
        this.panX = (widgetW - this._canvasW * this.zoom) / 2;
        this.panY = TOOLBAR_H + (drawH - this._canvasH * this.zoom) / 2;
    }

    // ── Reference image loading ──────────────────────────────────────
    _updateRefImage() {
        const imgWidget = this.node.widgets?.find(w => w.name === "reference_image_url");
        const url = imgWidget?.value;
        if (!url || url === this._lastRefUrl) return;
        this._lastRefUrl = url;
        this._refLoaded = false;
        const img = new Image();
        img.onload = () => {
            this._refImage = img;
            this._refLoaded = true;
            this._canvasW = img.naturalWidth;
            this._canvasH = img.naturalHeight;
            this.node.setDirtyCanvas(true);
        };
        img.onerror = () => { this._refLoaded = false; };
        img.src = url;
    }

    // ── Build toolbar ────────────────────────────────────────────────
    _buildToolbar(w) {
        const btns = [];
        let x = 6;
        const mkBtn = (label, style, action) => {
            const colors = BTN_COLORS[style] || BTN_COLORS.default;
            const bw = Math.max(50, label.length * 8 + 16);
            btns.push({ x, y: 4, w: bw, h: TOOLBAR_H - 8, label, colors, action });
            x += bw + 4;
        };
        mkBtn("New Shape", "accent", () => this.newShape());
        mkBtn("Toggle Closed", "default", () => this.toggleClosed());
        mkBtn("Undo (Z)", "default", () => this.undo());
        mkBtn("Clear All", "danger", () => {
            this._pushUndo();
            this.shapes = [{ points: [], closed: true, type: this.activeShape?.type || "catmull_rom", handles: [] }];
            this.activeShapeIdx = 0;
            this._serialize();
        });

        // Shape selector (cycle through shapes)
        const shapeCount = this.shapes.length;
        if (shapeCount > 1) {
            mkBtn(`Shape ${this.activeShapeIdx + 1}/${shapeCount}`, "default", () => {
                this.activeShapeIdx = (this.activeShapeIdx + 1) % this.shapes.length;
            });
        }

        this._toolbarButtons = btns;
    }

    // ── Main draw ────────────────────────────────────────────────────
    draw(ctx, widgetX, widgetY, widgetW, widgetH) {
        this._updateRefImage();
        if (!this._hasAutoFitted) {
            this.autoFit(widgetW, widgetH);
            this._hasAutoFitted = true;
            this._deserialize();
        }

        // Background
        ctx.save();
        ctx.fillStyle = CANVAS_BG;
        ctx.fillRect(widgetX, widgetY, widgetW, widgetH);

        // Clip to widget
        ctx.beginPath();
        ctx.rect(widgetX, widgetY, widgetW, widgetH);
        ctx.clip();

        const ox = widgetX;
        const oy = widgetY;

        // ── Toolbar ──────────────────────────────────────────────────
        ctx.fillStyle = TOOLBAR_BG;
        ctx.fillRect(ox, oy, widgetW, TOOLBAR_H);
        this._buildToolbar(widgetW);
        for (const btn of this._toolbarButtons) {
            const isHover = this._hoveredButton === btn;
            const c = btn.colors;
            ctx.fillStyle = isHover ? c.hover : c.bg;
            _roundRect(ctx, ox + btn.x, oy + btn.y, btn.w, btn.h, 4);
            ctx.fill();
            ctx.fillStyle = c.fg;
            ctx.font = "11px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(btn.label, ox + btn.x + btn.w / 2, oy + btn.y + btn.h / 2);
        }

        // ── Draw area (below toolbar) ────────────────────────────────
        ctx.save();
        ctx.beginPath();
        ctx.rect(ox, oy + TOOLBAR_H, widgetW, widgetH - TOOLBAR_H);
        ctx.clip();
        ctx.translate(ox + this.panX, oy + TOOLBAR_H + this.panY - TOOLBAR_H);
        ctx.scale(this.zoom, this.zoom);

        // Grid
        const gridStep = 64;
        ctx.strokeStyle = GRID_COLOR;
        ctx.lineWidth = 1 / this.zoom;
        for (let gx = 0; gx <= this._canvasW; gx += gridStep) {
            ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, this._canvasH); ctx.stroke();
        }
        for (let gy = 0; gy <= this._canvasH; gy += gridStep) {
            ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(this._canvasW, gy); ctx.stroke();
        }

        // Reference image
        if (this._refImage && this._refLoaded) {
            ctx.globalAlpha = 0.5;
            ctx.drawImage(this._refImage, 0, 0, this._canvasW, this._canvasH);
            ctx.globalAlpha = 1.0;
        }

        // Canvas border
        ctx.strokeStyle = "#ffffff33";
        ctx.lineWidth = 1 / this.zoom;
        ctx.strokeRect(0, 0, this._canvasW, this._canvasH);

        // ── Draw all shapes ──────────────────────────────────────────
        for (let si = 0; si < this.shapes.length; si++) {
            const shape = this.shapes[si];
            const isActive = si === this.activeShapeIdx;
            const samples = this._sampleShape(shape);

            if (samples.length >= 2) {
                // Filled region (closed shapes)
                if (shape.closed && samples.length >= 3) {
                    ctx.beginPath();
                    ctx.moveTo(samples[0][0], samples[0][1]);
                    for (let i = 1; i < samples.length; i++) {
                        ctx.lineTo(samples[i][0], samples[i][1]);
                    }
                    ctx.closePath();
                    ctx.fillStyle = isActive ? FILL_COLOR : "#88888818";
                    ctx.fill();
                }

                // Curve line
                ctx.beginPath();
                ctx.moveTo(samples[0][0], samples[0][1]);
                for (let i = 1; i < samples.length; i++) {
                    ctx.lineTo(samples[i][0], samples[i][1]);
                }
                ctx.strokeStyle = isActive ? CURVE_COLOR : "#88888866";
                ctx.lineWidth = (isActive ? CURVE_WIDTH : 1) / this.zoom;
                ctx.stroke();
            }

            // Control points
            for (let i = 0; i < shape.points.length; i++) {
                const p = shape.points[i];
                const isHovered = isActive && i === this.hoveredPoint;
                const isSelected = isActive && i === this.selectedPoint;
                ctx.beginPath();
                ctx.arc(p.x, p.y, (isHovered ? POINT_RADIUS + 2 : POINT_RADIUS) / this.zoom, 0, Math.PI * 2);
                ctx.fillStyle = isSelected ? POINT_SELECTED : isHovered ? POINT_HOVER : (isActive ? POINT_COLOR : "#888888");
                ctx.fill();
                ctx.strokeStyle = "#000000aa";
                ctx.lineWidth = 1 / this.zoom;
                ctx.stroke();

                // Point number
                if (isActive) {
                    ctx.fillStyle = "#ffffffcc";
                    ctx.font = `${Math.round(10 / this.zoom)}px sans-serif`;
                    ctx.textAlign = "center";
                    ctx.textBaseline = "bottom";
                    ctx.fillText(String(i + 1), p.x, p.y - (POINT_RADIUS + 3) / this.zoom);
                }
            }

            // Bezier handles (active shape only)
            if (isActive && shape.type === "bezier") {
                for (let i = 0; i < shape.handles.length; i++) {
                    const h = shape.handles[i];
                    const p = shape.points[i];
                    if (!h || !p) continue;
                    for (const type of ["cp1", "cp2"]) {
                        const hx = h[type + "x"], hy = h[type + "y"];
                        if (hx == null || hy == null) continue;
                        // Line from point to handle
                        ctx.beginPath();
                        ctx.moveTo(p.x, p.y);
                        ctx.lineTo(hx, hy);
                        ctx.strokeStyle = HANDLE_LINE;
                        ctx.lineWidth = 1 / this.zoom;
                        ctx.stroke();
                        // Handle dot
                        ctx.beginPath();
                        ctx.arc(hx, hy, HANDLE_RADIUS / this.zoom, 0, Math.PI * 2);
                        ctx.fillStyle = HANDLE_COLOR;
                        ctx.fill();
                    }
                }
            }
        }

        ctx.restore();

        // ── Status bar ───────────────────────────────────────────────
        const shape = this.activeShape;
        const status = shape
            ? `Shape ${this.activeShapeIdx + 1}/${this.shapes.length} | ${shape.type} | ${shape.closed ? "closed" : "open"} | ${shape.points.length} pts`
            : "No shapes";
        ctx.fillStyle = "#cdd6f4aa";
        ctx.font = "10px sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillText(status, ox + 6, oy + widgetH - 4);

        ctx.restore();
    }

    // ── Event handling ───────────────────────────────────────────────
    onMouseDown(localX, localY, e) {
        // Toolbar hit test
        if (localY < TOOLBAR_H) {
            for (const btn of this._toolbarButtons) {
                if (localX >= btn.x && localX <= btn.x + btn.w &&
                    localY >= btn.y && localY <= btn.y + btn.h) {
                    btn.action();
                    return true;
                }
            }
            return false;
        }

        // Canvas coordinates
        const adjustedY = localY - TOOLBAR_H;
        const cx = (localX - this.panX) / this.zoom;
        const cy = (adjustedY - (this.panY - TOOLBAR_H)) / this.zoom;

        // Middle button or Alt = pan
        if (e.button === 1 || (e.button === 0 && e.altKey)) {
            this.isPanning = true;
            this.panStart = { x: localX - this.panX, y: adjustedY - (this.panY - TOOLBAR_H) };
            return true;
        }

        // Left button
        if (e.button === 0) {
            // Shift+click = delete
            if (e.shiftKey) {
                const pi = this.findPointAt(cx, cy);
                if (pi >= 0) {
                    this.deletePoint(pi);
                    return true;
                }
            }

            // Check handle hit (bezier)
            const handleHit = this.findHandleAt(cx, cy);
            if (handleHit) {
                this.isDraggingHandle = true;
                this.dragHandleIdx = handleHit.idx;
                this.dragHandleType = handleHit.type;
                this._pushUndo();
                return true;
            }

            // Check point hit for drag
            const pi = this.findPointAt(cx, cy);
            if (pi >= 0) {
                this.isDragging = true;
                this.dragPointIdx = pi;
                const p = this.activeShape.points[pi];
                this.dragOriginal = { x: p.x, y: p.y };
                this.selectedPoint = pi;
                this._pushUndo();
                return true;
            }

            // Add point (only within canvas bounds)
            if (cx >= 0 && cx <= this._canvasW && cy >= 0 && cy <= this._canvasH) {
                this.addPoint(cx, cy);
                return true;
            }
        }

        return false;
    }

    onMouseMove(localX, localY) {
        const adjustedY = localY - TOOLBAR_H;
        const cx = (localX - this.panX) / this.zoom;
        const cy = (adjustedY - (this.panY - TOOLBAR_H)) / this.zoom;

        this._mouseX = cx;
        this._mouseY = cy;

        // Toolbar hover
        if (localY < TOOLBAR_H) {
            this._hoveredButton = null;
            for (const btn of this._toolbarButtons) {
                if (localX >= btn.x && localX <= btn.x + btn.w &&
                    localY >= btn.y && localY <= btn.y + btn.h) {
                    this._hoveredButton = btn;
                    break;
                }
            }
            return;
        }
        this._hoveredButton = null;

        if (this.isPanning && this.panStart) {
            this.panX = localX - this.panStart.x;
            this.panY = TOOLBAR_H + adjustedY - this.panStart.y;
            return;
        }

        if (this.isDragging && this.dragPointIdx >= 0) {
            const shape = this.activeShape;
            if (shape) {
                const p = shape.points[this.dragPointIdx];
                // Move handles along with point (bezier)
                if (shape.type === "bezier" && shape.handles[this.dragPointIdx]) {
                    const h = shape.handles[this.dragPointIdx];
                    const dx = cx - p.x, dy = cy - p.y;
                    if (h.cp1x != null) { h.cp1x += dx; h.cp1y += dy; }
                    if (h.cp2x != null) { h.cp2x += dx; h.cp2y += dy; }
                }
                p.x = cx;
                p.y = cy;
                this._serialize();
            }
            return;
        }

        if (this.isDraggingHandle && this.dragHandleIdx >= 0) {
            const shape = this.activeShape;
            if (shape && shape.handles[this.dragHandleIdx]) {
                const h = shape.handles[this.dragHandleIdx];
                h[this.dragHandleType + "x"] = cx;
                h[this.dragHandleType + "y"] = cy;
                this._serialize();
            }
            return;
        }

        // Hover detection
        this.hoveredPoint = this.findPointAt(cx, cy);
    }

    onMouseUp() {
        this.isDragging = false;
        this.dragPointIdx = -1;
        this.dragOriginal = null;
        this.isPanning = false;
        this.panStart = null;
        this.isDraggingHandle = false;
        this.dragHandleIdx = -1;
        this.dragHandleType = null;
    }

    onWheel(localX, localY, deltaY) {
        if (localY < TOOLBAR_H) return;
        const adjustedY = localY - TOOLBAR_H;
        const cx = (localX - this.panX) / this.zoom;
        const cy = (adjustedY - (this.panY - TOOLBAR_H)) / this.zoom;

        const factor = deltaY > 0 ? 0.9 : 1.1;
        const newZoom = Math.max(0.1, Math.min(10, this.zoom * factor));
        this.panX = localX - cx * newZoom;
        this.panY = TOOLBAR_H + adjustedY - cy * newZoom + TOOLBAR_H;
        this.zoom = newZoom;
    }

    onKeyDown(e) {
        // Don't intercept when focused on a text input
        if (e.target?.tagName === "INPUT" || e.target?.tagName === "TEXTAREA") return false;

        switch (e.key.toLowerCase()) {
            case "z":
                if (!e.ctrlKey && !e.metaKey) { this.undo(); return true; }
                break;
            case "n":
                this.newShape(); return true;
            case "c":
                this.toggleClosed(); return true;
            case "escape":
                this.selectedPoint = -1; return true;
            case "delete":
            case "backspace":
                if (this.hoveredPoint >= 0) {
                    this.deletePoint(this.hoveredPoint);
                    return true;
                }
                if (this.selectedPoint >= 0) {
                    this.deletePoint(this.selectedPoint);
                    return true;
                }
                break;
        }
        return false;
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Register extension
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app.registerExtension({
    name: "Comfy.MEC.SplineMaskEditor",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SplineMaskEditorMEC") return;

        // Ensure spline_data widget is hidden
        const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            origGetExtraMenuOptions?.apply(this, arguments);
        };
    },

    nodeCreated(node) {
        if (node.comfyClass !== "SplineMaskEditorMEC") return;

        const editor = new SplineEditor(node);

        // Hide the spline_data text widget from display
        const dataWidget = node.widgets?.find(w => w.name === "spline_data");
        if (dataWidget) {
            dataWidget.type = "hidden";
            dataWidget.computeSize = () => [0, -4];
        }

        // Create custom widget for the interactive editor
        const widget = node.addCustomWidget({
            name: "spline_editor_canvas",
            type: "custom",
            value: "",
            draw(ctx, node, widgetW, widgetY, widgetH) {
                const realH = Math.max(300, widgetH || 400);
                editor.draw(ctx, node.pos[0], widgetY, widgetW, realH);
            },
            computeSize(w) {
                return [w, 420];
            },
            onMouseDown(e, pos, node) {
                const localX = pos[0];
                const localY = pos[1];
                if (editor.onMouseDown(localX, localY, e)) {
                    node.setDirtyCanvas(true);
                    return true;
                }
                return false;
            },
        });

        // Attach mouse move/up/wheel handlers to node
        const origMouseMove = node.onMouseMove;
        node.onMouseMove = function (e, pos) {
            origMouseMove?.apply(this, arguments);
            // Compute position relative to widget
            const widgetIdx = this.widgets?.indexOf(widget);
            if (widgetIdx < 0) return;
            let wy = 0;
            for (let i = 0; i < widgetIdx; i++) {
                const ws = this.widgets[i].computeSize?.(this.size[0]);
                if (ws) wy += ws[1] + 4;
            }
            const localX = pos[0];
            const localY = pos[1] - wy;
            editor.onMouseMove(localX, localY);
            this.setDirtyCanvas(true);
        };

        const origMouseUp = node.onMouseUp;
        node.onMouseUp = function (e) {
            origMouseUp?.apply(this, arguments);
            editor.onMouseUp();
        };

        const origWheel = node.onMouseWheel;
        node.onMouseWheel = function (e, pos) {
            origWheel?.apply(this, arguments);
            const widgetIdx = this.widgets?.indexOf(widget);
            if (widgetIdx < 0) return;
            let wy = 0;
            for (let i = 0; i < widgetIdx; i++) {
                const ws = this.widgets[i].computeSize?.(this.size[0]);
                if (ws) wy += ws[1] + 4;
            }
            editor.onWheel(pos[0], pos[1] - wy, e.deltaY);
            this.setDirtyCanvas(true);
            return true;
        };

        const origKeyDown = node.onKeyDown;
        node.onKeyDown = function (e) {
            origKeyDown?.apply(this, arguments);
            if (editor.onKeyDown(e)) {
                this.setDirtyCanvas(true);
                return true;
            }
        };

        // Sync on serialization
        const origOnSerialize = node.onSerialize;
        node.onSerialize = function (o) {
            editor._serialize();
            origOnSerialize?.apply(this, arguments);
        };

        // Deserialize on configure (loading workflow)
        const origOnConfigure = node.onConfigure;
        node.onConfigure = function (data) {
            origOnConfigure?.apply(this, arguments);
            setTimeout(() => editor._deserialize(), 100);
        };
    },
});

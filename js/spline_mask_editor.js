import { app } from "../../scripts/app.js";

/**
 * MaskEditControl – Spline Mask Editor v2
 *
 * Complete rewrite following Olm SplineMask patterns for professional UX.
 *
 * Data model:
 *   paths: [{ points: [[nx,ny], ...], closed: bool, mode: "smooth"|"sharp" }]
 *   All coordinates normalized [0,1] relative to canvas dimensions.
 *
 * Interactions:
 *   Left-click canvas        → add point to current path
 *   Left-click first point   → close current path (when ≥3 pts)
 *   Ctrl+click near segment  → insert point on curve
 *   Shift+click point        → delete point
 *   Right-click              → context menu
 *   Click & drag point       → move control point
 *   Middle-drag / Alt+drag   → pan canvas
 *   Scroll                   → zoom (centered on cursor)
 *   N                        → new path
 *   Z                        → undo
 *   Delete / Backspace       → delete hovered point
 *   C                        → toggle closed/open
 *   S                        → toggle smooth/sharp mode
 *   Escape                   → deselect / close context menu
 */

// ── Visual tuning ────────────────────────────────────────────────────
const CANVAS_BG       = "#181825";
const GRID_COLOR      = "#ffffff08";
const CURVE_COLOR_ACT = "#4488ffcc";
const CURVE_COLOR_DIM = "#88888866";
const FILL_ACT        = "#4488ff22";
const FILL_DIM        = "#88888810";
const POINT_FILL      = "#22d65a";
const POINT_HOVER     = "#80ffb0";
const POINT_SELECTED  = "#ffdd44";
const POINT_FIRST     = "#ff6644";
const POINT_DIM       = "#888888";
const HANDLE_COLOR    = "#ff8844aa";
const HANDLE_LINE     = "#ff884466";
const TOOLBAR_BG      = "#1e1e2eee";
const TOOLBAR_H       = 32;
const BTN_H           = 24;
const BTN_PAD         = 4;
const BTN_COLORS      = {
    normal:  { bg: "#45475a", fg: "#cdd6f4", hover: "#585b70" },
    accent:  { bg: "#2a6040", fg: "#80ffb0", hover: "#3a7850" },
    active:  { bg: "#4a9060", fg: "#ffffff", hover: "#5aaa70" },
    danger:  { bg: "#6c2030", fg: "#ffb0c0", hover: "#8c2840" },
};

// ── Detection thresholds (normalized) ────────────────────────────────
const NEAR_POINT_T   = 0.025;
const NEAR_SEGMENT_T = 0.015;
const NEAR_FIRST_T   = 0.035;

// ── Helpers ──────────────────────────────────────────────────────────
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

function dist2d(a, b) {
    const dx = a[0] - b[0], dy = a[1] - b[1];
    return Math.sqrt(dx * dx + dy * dy);
}

function isNear(p1, p2, threshold) {
    return dist2d(p1, p2) < threshold;
}

function isNearSegment(pt, a, b, threshold) {
    const abx = b[0] - a[0], aby = b[1] - a[1];
    const len2 = abx * abx + aby * aby;
    if (len2 < 1e-12) return { near: isNear(pt, a, threshold), t: 0 };
    let t = ((pt[0] - a[0]) * abx + (pt[1] - a[1]) * aby) / len2;
    t = Math.max(0, Math.min(1, t));
    const proj = [a[0] + t * abx, a[1] + t * aby];
    return { near: dist2d(pt, proj) < threshold, t };
}

function catmullRomSample(points, closed, samplesPerSeg) {
    const n = points.length;
    if (n < 2) return points.map(p => [...p]);
    if (n === 2) {
        const out = [];
        for (let i = 0; i <= samplesPerSeg; i++) {
            const t = i / samplesPerSeg;
            out.push([
                points[0][0] + t * (points[1][0] - points[0][0]),
                points[0][1] + t * (points[1][1] - points[0][1]),
            ]);
        }
        return out;
    }

    const ext = [];
    if (closed) {
        ext.push(points[n - 1]);
        for (const p of points) ext.push(p);
        ext.push(points[0], points[1]);
    } else {
        ext.push([2 * points[0][0] - points[1][0], 2 * points[0][1] - points[1][1]]);
        for (const p of points) ext.push(p);
        ext.push([2 * points[n-1][0] - points[n-2][0], 2 * points[n-1][1] - points[n-2][1]]);
    }

    const segs = closed ? n : n - 1;
    const out = [];
    for (let s = 0; s < segs; s++) {
        const p0 = ext[s], p1 = ext[s+1], p2 = ext[s+2], p3 = ext[s+3];
        for (let i = 0; i < samplesPerSeg; i++) {
            const t = i / samplesPerSeg;
            const t2 = t * t, t3 = t2 * t;
            const x = 0.5 * ((2*p1[0]) + (-p0[0]+p2[0])*t +
                (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2 +
                (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3);
            const y = 0.5 * ((2*p1[1]) + (-p0[1]+p2[1])*t +
                (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2 +
                (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3);
            out.push([x, y]);
        }
    }
    if (!closed && n > 0) out.push([...points[n-1]]);
    return out;
}

function sharpPolyline(points, closed) {
    const out = points.map(p => [...p]);
    if (closed && out.length >= 3) out.push([...points[0]]);
    return out;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SplineEditor {
    constructor(node) {
        this.node = node;

        // Path data — normalized [0,1] coordinates
        this.paths = [{ points: [], closed: true, mode: "smooth" }];
        this.activePathIdx = 0;

        // Canvas dimensions (set from reference image)
        this._canvasW = 512;
        this._canvasH = 512;
        this._hasAutoFitted = false;

        // Interaction state
        this.hoveredPointIdx = -1;
        this.selectedPointIdx = -1;
        this.isDragging = false;
        this.dragPointIdx = -1;
        this.isPanning = false;
        this.panStart = null;

        // Context menu
        this._contextMenu = null;

        // Undo
        this._undoStack = [];

        // Server preview image
        this._previewImage = null;
        this._previewLoaded = false;
        this._cacheKey = null;

        // Toolbar
        this._toolbarButtons = [];
        this._hoveredButton = null;

        // Preview bounds [x, y, w, h] in widget-local coords
        this._previewBounds = null;

        // Mouse tracking
        this._mouseLocal = [0, 0];
    }

    get activePath() {
        return this.paths[this.activePathIdx] || null;
    }

    // ── Coordinate transforms ────────────────────────────────────────
    normToScreen(nx, ny) {
        if (!this._previewBounds) return [0, 0];
        const [bx, by, bw, bh] = this._previewBounds;
        return [bx + nx * bw, by + ny * bh];
    }

    screenToNorm(sx, sy) {
        if (!this._previewBounds) return [0, 0];
        const [bx, by, bw, bh] = this._previewBounds;
        return [(sx - bx) / Math.max(bw, 1), (sy - by) / Math.max(bh, 1)];
    }

    clampNorm(nx, ny) {
        return [Math.min(Math.max(nx, 0), 1), Math.min(Math.max(ny, 0), 1)];
    }

    // ── Hit testing (normalized coords) ──────────────────────────────
    findPointAt(nx, ny) {
        const path = this.activePath;
        if (!path) return -1;
        for (let i = path.points.length - 1; i >= 0; i--) {
            if (isNear([nx, ny], path.points[i], NEAR_POINT_T)) return i;
        }
        return -1;
    }

    isNearFirstPoint(nx, ny) {
        const path = this.activePath;
        if (!path || path.points.length < 3 || path.closed) return false;
        return isNear([nx, ny], path.points[0], NEAR_FIRST_T);
    }

    findSegmentInsert(nx, ny) {
        const path = this.activePath;
        if (!path || path.points.length < 2) return null;
        const samples = this._samplePath(path);
        if (samples.length < 2) return null;

        const nPts = path.points.length;
        const samplesPerSeg = path.mode === "smooth" ? 12 : 1;
        const totalSegs = path.closed ? nPts : nPts - 1;

        for (let i = 0; i < samples.length - 1; i++) {
            const res = isNearSegment([nx, ny], samples[i], samples[i + 1], NEAR_SEGMENT_T);
            if (res.near) {
                const segIdx = Math.min(Math.floor(i / samplesPerSeg), totalSegs - 1);
                return { segIdx, normPt: this.clampNorm(nx, ny) };
            }
        }
        return null;
    }

    // ── Undo ─────────────────────────────────────────────────────────
    _pushUndo() {
        this._undoStack.push(JSON.stringify(this.paths));
        if (this._undoStack.length > 60) this._undoStack.shift();
    }

    undo() {
        if (this._undoStack.length === 0) return;
        this.paths = JSON.parse(this._undoStack.pop());
        if (this.activePathIdx >= this.paths.length) {
            this.activePathIdx = Math.max(0, this.paths.length - 1);
        }
        this._serialize();
    }

    // ── Serialize to widget + properties ─────────────────────────────
    _serialize() {
        // Write widget value: pixel coords for Python node compatibility
        const w = this.node.widgets?.find(w => w.name === "spline_data");
        if (w) {
            const shapes = this.paths.map(p => ({
                points: p.points.map(([nx, ny]) => ({
                    x: nx * this._canvasW,
                    y: ny * this._canvasH,
                })),
                closed: p.closed,
                type: p.mode === "smooth" ? "catmull_rom" : "polyline",
                handles: [],
            }));
            w.value = JSON.stringify(shapes);
        }

        // Store normalized data in properties for lossless persistence
        this.node.properties = this.node.properties || {};
        const flat = [];
        const closedFlags = [];
        const modes = [];
        for (const p of this.paths) {
            closedFlags.push(p.closed);
            modes.push(p.mode);
            for (const pt of p.points) flat.push([pt[0], pt[1]]);
            flat.push(null);
        }
        this.node.properties.spline_points = flat;
        this.node.properties.spline_closed_flags = JSON.stringify(closedFlags);
        this.node.properties.spline_modes = JSON.stringify(modes);
    }

    _deserialize() {
        // Try property-based first (normalized, lossless)
        const props = this.node.properties || {};
        if (props.spline_points && Array.isArray(props.spline_points) && props.spline_points.length > 0) {
            try {
                const closedFlags = JSON.parse(props.spline_closed_flags || "[]");
                const modes = JSON.parse(props.spline_modes || "[]");
                this.paths = this._restoreFromFlat(props.spline_points, closedFlags, modes);
                if (this.paths.length > 0 && this.paths.some(p => p.points.length > 0)) {
                    this.activePathIdx = Math.min(this.activePathIdx, this.paths.length - 1);
                    this._serialize();
                    return;
                }
            } catch (e) { /* fall through */ }
        }

        // Fallback: widget value (pixel coords → normalize)
        const w = this.node.widgets?.find(w => w.name === "spline_data");
        if (!w || !w.value) return;
        try {
            const data = JSON.parse(w.value);
            if (Array.isArray(data) && data.length > 0) {
                this.paths = data.map(s => ({
                    points: (s.points || []).map(p => {
                        if (Array.isArray(p)) return [p[0], p[1]];
                        const px = p.x ?? 0, py = p.y ?? 0;
                        if (px > 1 || py > 1) {
                            return [px / Math.max(this._canvasW, 1), py / Math.max(this._canvasH, 1)];
                        }
                        return [px, py];
                    }),
                    closed: s.closed !== false,
                    mode: s.type === "polyline" ? "sharp" : "smooth",
                }));
                this.activePathIdx = Math.min(this.activePathIdx, this.paths.length - 1);
            }
        } catch (e) { /* ignore */ }
    }

    _restoreFromFlat(flatData, closedFlags, modes) {
        const paths = [];
        let current = [];
        let pathIdx = 0;
        for (const item of flatData) {
            if (item === null) {
                if (current.length > 0) {
                    paths.push({
                        points: current,
                        closed: closedFlags[pathIdx] !== false,
                        mode: modes[pathIdx] || "smooth",
                    });
                }
                current = [];
                pathIdx++;
            } else if (Array.isArray(item) && item.length >= 2) {
                current.push([item[0], item[1]]);
            }
        }
        if (current.length > 0) {
            paths.push({
                points: current,
                closed: closedFlags[pathIdx] !== false,
                mode: modes[pathIdx] || "smooth",
            });
        }
        return paths.length > 0 ? paths : [{ points: [], closed: true, mode: "smooth" }];
    }

    // ── Path mutations ───────────────────────────────────────────────
    addPoint(nx, ny) {
        const path = this.activePath;
        if (!path) return;
        this._pushUndo();
        path.points.push(this.clampNorm(nx, ny));
        this._serialize();
    }

    insertPoint(segIdx, nx, ny) {
        const path = this.activePath;
        if (!path) return;
        this._pushUndo();
        path.points.splice(segIdx + 1, 0, this.clampNorm(nx, ny));
        this._serialize();
    }

    deletePoint(idx) {
        const path = this.activePath;
        if (!path || idx < 0 || idx >= path.points.length) return;
        this._pushUndo();
        path.points.splice(idx, 1);
        this.selectedPointIdx = -1;
        this.hoveredPointIdx = -1;
        this._serialize();
    }

    newPath() {
        this._pushUndo();
        const mode = this.activePath?.mode || "smooth";
        this.paths.push({ points: [], closed: true, mode });
        this.activePathIdx = this.paths.length - 1;
        this._serialize();
    }

    deletePath(idx) {
        if (this.paths.length <= 1) {
            this._pushUndo();
            this.paths[0].points = [];
            this._serialize();
            return;
        }
        this._pushUndo();
        this.paths.splice(idx, 1);
        if (this.activePathIdx >= this.paths.length) {
            this.activePathIdx = this.paths.length - 1;
        }
        this._serialize();
    }

    toggleClosed() {
        const path = this.activePath;
        if (!path) return;
        this._pushUndo();
        path.closed = !path.closed;
        this._serialize();
    }

    toggleMode() {
        const path = this.activePath;
        if (!path) return;
        this._pushUndo();
        path.mode = path.mode === "smooth" ? "sharp" : "smooth";
        this._serialize();
    }

    clearAll() {
        this._pushUndo();
        this.paths = [{ points: [], closed: true, mode: "smooth" }];
        this.activePathIdx = 0;
        this._serialize();
    }

    // ── Curve sampling ───────────────────────────────────────────────
    _samplePath(path) {
        if (!path || path.points.length < 2) return [];
        if (path.mode === "sharp") return sharpPolyline(path.points, path.closed);
        return catmullRomSample(path.points, path.closed, 12);
    }

    // ── Server preview image ─────────────────────────────────────────
    setPreviewImage(dataUrl) {
        if (!dataUrl) return;
        const img = new Image();
        img.onload = () => {
            this._previewImage = img;
            this._previewLoaded = true;
            const dimsChanged =
                img.naturalWidth !== this._canvasW ||
                img.naturalHeight !== this._canvasH;
            this._canvasW = img.naturalWidth;
            this._canvasH = img.naturalHeight;
            // Only re-fit when the underlying image dimensions actually
            // changed (e.g. first load, or user swapped to a different
            // resolution).  Otherwise preserve the user's zoom/pan to
            // prevent the "auto zoom in / zoom out" jitter on every
            // graph execution.
            if (dimsChanged) this._hasAutoFitted = false;
            this.node.setDirtyCanvas(true);
        };
        img.onerror = () => { this._previewLoaded = false; };
        img.src = dataUrl;
    }

    // ── Auto-fit preview bounds ──────────────────────────────────────
    autoFit(widgetW, widgetH) {
        const drawH = widgetH - TOOLBAR_H;
        if (drawH <= 0 || this._canvasW <= 0 || this._canvasH <= 0) return;
        const aspect = this._canvasW / this._canvasH;
        const padFrac = 0.95;

        let drawW = widgetW * padFrac;
        let dh = drawW / aspect;
        if (dh > drawH * padFrac) {
            dh = drawH * padFrac;
            drawW = dh * aspect;
        }

        this._previewBounds = [
            (widgetW - drawW) / 2,
            TOOLBAR_H + (drawH - dh) / 2,
            drawW,
            dh,
        ];
    }

    // ── Build toolbar ────────────────────────────────────────────────
    _buildToolbar(widgetW) {
        const btns = [];
        let x = BTN_PAD;

        const mkBtn = (label, style, action) => {
            const bw = Math.max(50, label.length * 7 + 14);
            btns.push({ x, y: BTN_PAD, w: bw, h: BTN_H, label, style, action });
            x += bw + BTN_PAD;
        };

        const path = this.activePath;
        mkBtn("New Path", "accent", () => this.newPath());
        mkBtn(path?.closed ? "Closed" : "Open", path?.closed ? "active" : "normal", () => this.toggleClosed());
        mkBtn(path?.mode === "smooth" ? "Smooth" : "Sharp", path?.mode === "smooth" ? "active" : "normal", () => this.toggleMode());
        mkBtn("Undo", "normal", () => this.undo());
        mkBtn("Clear", "danger", () => this.clearAll());

        if (this.paths.length > 1) {
            mkBtn(`Path ${this.activePathIdx + 1}/${this.paths.length}`, "normal", () => {
                this.activePathIdx = (this.activePathIdx + 1) % this.paths.length;
            });
        }

        this._toolbarButtons = btns;
    }

    // ── Context menu ─────────────────────────────────────────────────
    _showContextMenu(localX, localY) {
        const items = [];
        const [nx, ny] = this.screenToNorm(localX, localY);
        const pi = this.findPointAt(nx, ny);

        if (pi >= 0) {
            items.push({ label: "Delete Point", action: () => this.deletePoint(pi) });
        }

        const path = this.activePath;
        if (path) {
            items.push({ label: path.closed ? "Open Path" : "Close Path", action: () => this.toggleClosed() });
            items.push({ label: path.mode === "smooth" ? "Switch to Sharp" : "Switch to Smooth", action: () => this.toggleMode() });
            if (path.points.length > 0) {
                items.push({ label: "Delete This Path", action: () => this.deletePath(this.activePathIdx) });
            }
        }

        items.push({ label: "New Path", action: () => this.newPath() });
        items.push({ label: "Clear All", action: () => this.clearAll() });

        this._contextMenu = { x: localX, y: localY, items };
    }

    _hideContextMenu() { this._contextMenu = null; }

    _drawContextMenu(ctx, ox, oy) {
        const cm = this._contextMenu;
        if (!cm) return;
        const itemH = 24;
        const pad = 6;
        const maxW = Math.max(...cm.items.map(it => it.label.length * 7 + 20), 120);
        const totalH = cm.items.length * itemH + pad * 2;

        ctx.fillStyle = "#2a2a3eee";
        _roundRect(ctx, ox + cm.x, oy + cm.y, maxW, totalH, 6);
        ctx.fill();
        ctx.strokeStyle = "#555577";
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.font = "12px sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";

        for (let i = 0; i < cm.items.length; i++) {
            const iy = cm.y + pad + i * itemH;
            if (this._mouseLocal[1] >= iy && this._mouseLocal[1] < iy + itemH &&
                this._mouseLocal[0] >= cm.x && this._mouseLocal[0] < cm.x + maxW) {
                ctx.fillStyle = "#3a3a55";
                ctx.fillRect(ox + cm.x + 2, oy + iy, maxW - 4, itemH);
            }
            ctx.fillStyle = "#cdd6f4";
            ctx.fillText(cm.items[i].label, ox + cm.x + 10, oy + iy + itemH / 2);
        }
    }

    _hitContextMenu(localX, localY) {
        const cm = this._contextMenu;
        if (!cm) return -1;
        const itemH = 24;
        const pad = 6;
        const maxW = Math.max(...cm.items.map(it => it.label.length * 7 + 20), 120);
        if (localX < cm.x || localX > cm.x + maxW) return -1;
        const relY = localY - cm.y - pad;
        if (relY < 0 || relY >= cm.items.length * itemH) return -1;
        return Math.floor(relY / itemH);
    }

    // ── Main draw ────────────────────────────────────────────────────
    draw(ctx, widgetX, widgetY, widgetW, widgetH) {
        if (!this._hasAutoFitted) {
            this.autoFit(widgetW, widgetH);
            this._hasAutoFitted = true;
            this._deserialize();
        }
        // Re-fit the preview bounds if the widget was resized so that
        // the editor stays inside its container, but keep the user's
        // current zoom factor.  Without this guard, the bounds were
        // re-fitted on every draw — causing the user-visible jitter.
        if (this._previewBounds) {
            const [bx, by, bw, bh] = this._previewBounds;
            const drawH = widgetH - TOOLBAR_H;
            if (bw <= 0 || bh <= 0 || drawH <= 0) {
                this.autoFit(widgetW, widgetH);
            }
        } else {
            this.autoFit(widgetW, widgetH);
        }

        ctx.save();
        ctx.fillStyle = CANVAS_BG;
        ctx.fillRect(widgetX, widgetY, widgetW, widgetH);
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
            const c = BTN_COLORS[btn.style] || BTN_COLORS.normal;
            ctx.fillStyle = isHover ? c.hover : c.bg;
            _roundRect(ctx, ox + btn.x, oy + btn.y, btn.w, btn.h, 4);
            ctx.fill();
            ctx.fillStyle = c.fg;
            ctx.font = "11px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(btn.label, ox + btn.x + btn.w / 2, oy + btn.y + btn.h / 2);
        }

        // ── Preview area ─────────────────────────────────────────────
        const [bx, by, bw, bh] = this._previewBounds || [0, TOOLBAR_H, widgetW, widgetH - TOOLBAR_H];

        ctx.save();
        ctx.beginPath();
        ctx.rect(ox + bx - 2, oy + by - 2, bw + 4, bh + 4);
        ctx.clip();

        // Grid
        const gridStepPx = 64;
        ctx.strokeStyle = GRID_COLOR;
        ctx.lineWidth = 1;
        for (let gx = 0; gx <= bw; gx += gridStepPx) {
            ctx.beginPath(); ctx.moveTo(ox + bx + gx, oy + by); ctx.lineTo(ox + bx + gx, oy + by + bh); ctx.stroke();
        }
        for (let gy = 0; gy <= bh; gy += gridStepPx) {
            ctx.beginPath(); ctx.moveTo(ox + bx, oy + by + gy); ctx.lineTo(ox + bx + bw, oy + by + gy); ctx.stroke();
        }

        // Server preview image (full fidelity: image + mask overlay)
        if (this._previewImage && this._previewLoaded) {
            ctx.drawImage(this._previewImage, ox + bx, oy + by, bw, bh);
        } else {
            // No preview yet — show hint
            ctx.fillStyle = "#ffffff44";
            ctx.font = "13px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("Run graph to see preview", ox + bx + bw / 2, oy + by + bh / 2);
        }

        // Canvas border
        ctx.strokeStyle = "#ffffff33";
        ctx.lineWidth = 1;
        ctx.strokeRect(ox + bx, oy + by, bw, bh);

        // ── Draw all paths ───────────────────────────────────────────
        const pointR = Math.max(4, Math.min(8, bw * 0.008));

        for (let pi = 0; pi < this.paths.length; pi++) {
            const path = this.paths[pi];
            const isActive = pi === this.activePathIdx;
            const samples = this._samplePath(path);

            if (samples.length >= 2) {
                const sPts = samples.map(([nx, ny]) => {
                    const [sx, sy] = this.normToScreen(nx, ny);
                    return [ox + sx, oy + sy];
                });

                // Fill closed regions
                if (path.closed && sPts.length >= 3) {
                    ctx.beginPath();
                    ctx.moveTo(sPts[0][0], sPts[0][1]);
                    for (let i = 1; i < sPts.length; i++) ctx.lineTo(sPts[i][0], sPts[i][1]);
                    ctx.closePath();
                    ctx.fillStyle = isActive ? FILL_ACT : FILL_DIM;
                    ctx.fill();
                }

                // Curve stroke
                ctx.beginPath();
                ctx.moveTo(sPts[0][0], sPts[0][1]);
                for (let i = 1; i < sPts.length; i++) ctx.lineTo(sPts[i][0], sPts[i][1]);
                ctx.strokeStyle = isActive ? CURVE_COLOR_ACT : CURVE_COLOR_DIM;
                ctx.lineWidth = isActive ? 2 : 1;
                ctx.stroke();
            }

            // Control points
            for (let i = 0; i < path.points.length; i++) {
                const [nx, ny] = path.points[i];
                const [sx, sy] = this.normToScreen(nx, ny);
                const px = ox + sx, py = oy + sy;
                const isHovered = isActive && i === this.hoveredPointIdx;
                const isSelected = isActive && i === this.selectedPointIdx;
                const isFirst = i === 0 && !path.closed && path.points.length >= 3;

                const r = isHovered ? pointR * 1.4 : pointR;
                ctx.beginPath();
                ctx.arc(px, py, r, 0, Math.PI * 2);
                ctx.fillStyle = isFirst ? POINT_FIRST :
                                isSelected ? POINT_SELECTED :
                                isHovered ? POINT_HOVER :
                                isActive ? POINT_FILL : POINT_DIM;
                ctx.fill();
                ctx.strokeStyle = "#000000aa";
                ctx.lineWidth = 1;
                ctx.stroke();

                if (isActive) {
                    const fontSize = Math.max(9, Math.min(12, bw * 0.015));
                    ctx.fillStyle = "#ffffffcc";
                    ctx.font = `${Math.round(fontSize)}px sans-serif`;
                    ctx.textAlign = "center";
                    ctx.textBaseline = "bottom";
                    ctx.fillText(String(i + 1), px, py - r - 2);
                }
            }
        }

        ctx.restore(); // pop preview clip

        // Context menu
        this._drawContextMenu(ctx, ox, oy);

        // Status bar
        const path = this.activePath;
        const nPts = path ? path.points.length : 0;
        const status = `Path ${this.activePathIdx + 1}/${this.paths.length} | ${path?.mode || "-"} | ${path?.closed ? "closed" : "open"} | ${nPts} pts` +
            `  ·  Click:add  Shift+click:delete  Ctrl+click:insert  Right-click:menu`;
        ctx.fillStyle = "#cdd6f4aa";
        ctx.font = "10px sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillText(status, ox + 6, oy + widgetH - 4);

        ctx.restore();
    }

    // ── Event: Mouse Down ────────────────────────────────────────────
    onMouseDown(localX, localY, e) {
        // Context menu click
        if (this._contextMenu) {
            const hitIdx = this._hitContextMenu(localX, localY);
            if (hitIdx >= 0) this._contextMenu.items[hitIdx].action();
            this._hideContextMenu();
            return true;
        }

        // Right-click → context menu
        if (e.button === 2) {
            this._showContextMenu(localX, localY);
            return true;
        }

        // Toolbar
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

        // Pan (middle button or Alt+left)
        if (e.button === 1 || (e.button === 0 && e.altKey)) {
            this.isPanning = true;
            this.panStart = {
                x: localX, y: localY,
                pbx: this._previewBounds?.[0] || 0,
                pby: this._previewBounds?.[1] || 0,
            };
            return true;
        }

        const [nx, ny] = this.screenToNorm(localX, localY);

        if (e.button === 0) {
            // Shift+click = delete
            if (e.shiftKey) {
                const pi = this.findPointAt(nx, ny);
                if (pi >= 0) { this.deletePoint(pi); return true; }
            }

            // Click near first point → close path
            if (this.isNearFirstPoint(nx, ny)) {
                this._pushUndo();
                this.activePath.closed = true;
                this._serialize();
                return true;
            }

            // Drag existing point
            const pi = this.findPointAt(nx, ny);
            if (pi >= 0) {
                this.isDragging = true;
                this.dragPointIdx = pi;
                this.selectedPointIdx = pi;
                this._pushUndo();
                return true;
            }

            // Ctrl+click near segment → insert point
            if (e.ctrlKey) {
                const seg = this.findSegmentInsert(nx, ny);
                if (seg) { this.insertPoint(seg.segIdx, seg.normPt[0], seg.normPt[1]); return true; }
            }

            // Add point (within or slightly outside canvas)
            if (nx >= -0.05 && nx <= 1.05 && ny >= -0.05 && ny <= 1.05) {
                this.addPoint(nx, ny);
                return true;
            }
        }

        return false;
    }

    onMouseMove(localX, localY) {
        this._mouseLocal = [localX, localY];

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
            this.hoveredPointIdx = -1;
            return;
        }
        this._hoveredButton = null;

        const [nx, ny] = this.screenToNorm(localX, localY);

        // Panning
        if (this.isPanning && this.panStart && this._previewBounds) {
            this._previewBounds[0] = this.panStart.pbx + (localX - this.panStart.x);
            this._previewBounds[1] = this.panStart.pby + (localY - this.panStart.y);
            return;
        }

        // Dragging point
        if (this.isDragging && this.dragPointIdx >= 0) {
            const path = this.activePath;
            if (path && this.dragPointIdx < path.points.length) {
                path.points[this.dragPointIdx] = this.clampNorm(nx, ny);
                this._serialize();
            }
            return;
        }

        // Hover
        this.hoveredPointIdx = this.findPointAt(nx, ny);
    }

    onMouseUp() {
        this.isDragging = false;
        this.dragPointIdx = -1;
        this.isPanning = false;
        this.panStart = null;
    }

    onWheel(localX, localY, deltaY) {
        if (localY < TOOLBAR_H || !this._previewBounds) return;

        const [bx, by, bw, bh] = this._previewBounds;
        const factor = deltaY > 0 ? 0.92 : 1.08;
        // Hard clamps prevent the bounds from collapsing to zero or
        // exploding past the widget — both states caused the canvas
        // to "misfit" wildly after a few scroll events.
        const MIN_PX = 80;
        const MAX_PX = 8000;
        const newBw = Math.max(MIN_PX, Math.min(MAX_PX, bw * factor));
        const newBh = Math.max(MIN_PX, Math.min(MAX_PX, bh * factor));

        const relX = (localX - bx) / Math.max(bw, 1);
        const relY = (localY - by) / Math.max(bh, 1);
        this._previewBounds = [
            localX - relX * newBw,
            localY - relY * newBh,
            newBw,
            newBh,
        ];
    }

    onKeyDown(e) {
        if (e.target?.tagName === "INPUT" || e.target?.tagName === "TEXTAREA") return false;

        if (e.key === "Escape") {
            if (this._contextMenu) { this._hideContextMenu(); return true; }
            this.selectedPointIdx = -1;
            return true;
        }

        switch (e.key.toLowerCase()) {
            case "z":
                if (!e.ctrlKey && !e.metaKey) { this.undo(); return true; }
                break;
            case "n":
                this.newPath(); return true;
            case "c":
                this.toggleClosed(); return true;
            case "s":
                if (!e.ctrlKey && !e.metaKey) { this.toggleMode(); return true; }
                break;
            case "delete":
            case "backspace":
                if (this.hoveredPointIdx >= 0) { this.deletePoint(this.hoveredPointIdx); return true; }
                if (this.selectedPointIdx >= 0) { this.deletePoint(this.selectedPointIdx); return true; }
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

        const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            origGetExtraMenuOptions?.apply(this, arguments);
        };
    },

    nodeCreated(node) {
        if (node.comfyClass !== "SplineMaskEditorMEC") return;

        const editor = new SplineEditor(node);

        // Hide internal widgets
        for (const wName of ["spline_data", "mask_color", "mask_opacity"]) {
            const w = node.widgets?.find(w => w.name === wName);
            if (w) {
                w.type = "hidden";
                w.computeSize = () => [0, -4];
                w.draw = () => {};
            }
        }

        // ── Debounced preview update from server ─────────────────────
        let _previewTimeout = null;
        function requestPreviewUpdate() {
            if (!editor._cacheKey) return;
            clearTimeout(_previewTimeout);
            _previewTimeout = setTimeout(async () => {
                try {
                    const splineWidget = node.widgets?.find(w => w.name === "spline_data");
                    const splineData = splineWidget?.value || "[]";
                    const resp = await fetch("/mec/api/splinemask/preview", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            node_id: editor._cacheKey,
                            spline_data: splineData,
                        }),
                    });
                    const result = await resp.json();
                    if (result.status === "ok" && result.image) {
                        editor.setPreviewImage(result.image);
                    }
                } catch (e) {
                    // Silently fail — preview is non-critical
                }
            }, 150);
        }

        // Hook _serialize to trigger preview updates
        const origSerialize = editor._serialize.bind(editor);
        editor._serialize = function () {
            origSerialize();
            requestPreviewUpdate();
        };

        // ── Capture preview from execution results ───────────────────
        const origOnExecuted = node.onExecuted;
        node.onExecuted = function (message) {
            origOnExecuted?.apply(this, arguments);
            if (message?.cache_key?.[0]) {
                editor._cacheKey = message.cache_key[0];
            }
            if (message?.preview?.[0]) {
                editor.setPreviewImage(message.preview[0]);
            }
        };

        // Interactive editor widget
        const widget = node.addCustomWidget({
            name: "spline_editor_canvas",
            type: "custom",
            value: "",
            draw(ctx, node, widgetW, widgetY, widgetH) {
                const realH = Math.max(300, widgetH || 440);
                editor.draw(ctx, node.pos[0], widgetY, widgetW, realH);
            },
            computeSize(w) {
                return [w, 440];
            },
            onMouseDown(e, pos, node) {
                if (editor.onMouseDown(pos[0], pos[1], e)) {
                    node.setDirtyCanvas(true);
                    return true;
                }
                return false;
            },
        });

        // Mouse move
        const origMouseMove = node.onMouseMove;
        node.onMouseMove = function (e, pos) {
            origMouseMove?.apply(this, arguments);
            const widgetIdx = this.widgets?.indexOf(widget);
            if (widgetIdx < 0) return;
            let wy = 0;
            for (let i = 0; i < widgetIdx; i++) {
                const ws = this.widgets[i].computeSize?.(this.size[0]);
                if (ws) wy += ws[1] + 4;
            }
            editor.onMouseMove(pos[0], pos[1] - wy);
            this.setDirtyCanvas(true);
        };

        // Mouse up
        const origMouseUp = node.onMouseUp;
        node.onMouseUp = function (e) {
            origMouseUp?.apply(this, arguments);
            editor.onMouseUp();
        };

        // Wheel (zoom)
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

        // Keyboard
        const origKeyDown = node.onKeyDown;
        node.onKeyDown = function (e) {
            origKeyDown?.apply(this, arguments);
            if (editor.onKeyDown(e)) {
                this.setDirtyCanvas(true);
                return true;
            }
        };

        // Serialize
        const origOnSerialize = node.onSerialize;
        node.onSerialize = function (o) {
            editor._serialize();
            origOnSerialize?.apply(this, arguments);
        };

        // Deserialize
        const origOnConfigure = node.onConfigure;
        node.onConfigure = function (data) {
            origOnConfigure?.apply(this, arguments);
            setTimeout(() => editor._deserialize(), 100);
        };
    },
});
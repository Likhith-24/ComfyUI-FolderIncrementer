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
 *   CTRL + Shift + Z       → redo
 *   CTRL + C               → clear all
 *   R                      → reset view (zoom & pan)
 *
 * Data is stored as a single `editor_data` JSON widget:
 *   { "points": [{x, y, label, radius}, ...], "bboxes": [[x1,y1,x2,y2], ...] }
 *   bbox label is stored internally as 5th element but stripped for output.
 */

const POINT_COLORS = { positive: "#00ff00", negative: "#ff3333" };
const BBOX_COLORS  = { positive: "#00ff88", negative: "#ff4444" };
const CROSSHAIR_COLOR = "#ffffff88";
const GRID_COLOR      = "#ffffff0a";
const CANVAS_BG       = "#1a1a2e";

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
        this.bboxDragLabel = 1;   // 1 = positive, 0 = negative
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
    }

    // ── Coordinate transforms ────────────────────────────────────────
    screenToCanvas(sx, sy) {
        return {
            x: (sx - this.panX) / this.zoom,
            y: (sy - this.panY) / this.zoom,
        };
    }

    canvasToScreen(cx, cy) {
        return {
            x: cx * this.zoom + this.panX,
            y: cy * this.zoom + this.panY,
        };
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
            bboxes: this.bboxes.map(b => b.slice(0, 4)),  // strip label for SAM compat
        });
        const w = this.node.widgets?.find(w => w.name === "editor_data");
        if (w) {
            w.value = data;
            w.callback?.(data);
        }
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

            // Update node width/height widgets to match image
            const wWidget = this.node.widgets?.find(w => w.name === "width");
            const hWidget = this.node.widgets?.find(w => w.name === "height");
            if (wWidget) wWidget.value = img.naturalWidth;
            if (hWidget) hWidget.value = img.naturalHeight;

            // Auto-fit view to show entire image
            if (this._containerEl) {
                const rect = this._containerEl.getBoundingClientRect();
                if (rect.width > 0 && rect.height > 0) {
                    this.fitView(rect.width, rect.height);
                }
            }

            this.node._mecRender?.();
        };
        img.onerror = () => {
            console.warn("[MEC] Failed to load reference image:", imageUrl);
        };
        img.src = imageUrl;
    }

    // ── Fit view to canvas content ───────────────────────────────────
    fitView(containerW, containerH) {
        if (this._canvasW <= 0 || this._canvasH <= 0) return;
        const scaleX = containerW / this._canvasW;
        const scaleY = containerH / this._canvasH;
        this.zoom = Math.min(scaleX, scaleY) * 0.92;
        this.panX = (containerW - this._canvasW * this.zoom) / 2;
        this.panY = (containerH - this._canvasH * this.zoom) / 2;
    }

    // ── Drawing ──────────────────────────────────────────────────────
    draw(ctx, x, y, w, h) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(x, y, w, h);
        ctx.clip();

        // Background
        ctx.fillStyle = CANVAS_BG;
        ctx.fillRect(x, y, w, h);

        ctx.save();
        ctx.translate(x + this.panX, y + this.panY);
        ctx.scale(this.zoom, this.zoom);

        // Reference image
        if (this._refLoaded && this._refImage) {
            ctx.drawImage(this._refImage, 0, 0, this._canvasW, this._canvasH);
        } else {
            // Grid
            ctx.strokeStyle = GRID_COLOR;
            ctx.lineWidth = 1 / this.zoom;
            const step = 64;
            for (let gx = 0; gx <= this._canvasW; gx += step) {
                ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, this._canvasH); ctx.stroke();
            }
            for (let gy = 0; gy <= this._canvasH; gy += step) {
                ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(this._canvasW, gy); ctx.stroke();
            }

            // Canvas border
            ctx.strokeStyle = "#ffffff22";
            ctx.lineWidth = 2 / this.zoom;
            ctx.strokeRect(0, 0, this._canvasW, this._canvasH);
        }

        // ── Draw bboxes ──────────────────────────────────────────────
        for (let i = 0; i < this.bboxes.length; i++) {
            const b = this.bboxes[i];
            const isPos = (b[4] ?? 1) === 1;
            const color = isPos ? BBOX_COLORS.positive : BBOX_COLORS.negative;
            const isHovered = (i === this.hoveredBbox);

            // Fill
            ctx.fillStyle = color + (isHovered ? "33" : "18");
            ctx.fillRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);

            // Stroke
            ctx.strokeStyle = color + (isHovered ? "cc" : "88");
            ctx.lineWidth = (isHovered ? 2.5 : 1.5) / this.zoom;
            ctx.setLineDash(isPos ? [] : [6 / this.zoom, 4 / this.zoom]);
            ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
            ctx.setLineDash([]);

            // Label
            const labelText = isPos ? `+bbox${i}` : `-bbox${i}`;
            const fontSize = Math.max(9, 11 / this.zoom);
            ctx.font = `bold ${fontSize}px monospace`;
            ctx.fillStyle = color;
            ctx.fillText(labelText, b[0] + 3 / this.zoom, b[1] - 3 / this.zoom);
        }

        // ── Draw in-progress bbox drag ───────────────────────────────
        if (this.isBboxDrag && this.dragStart && this._dragEnd) {
            const bx1 = Math.min(this.dragStart.x, this._dragEnd.x);
            const by1 = Math.min(this.dragStart.y, this._dragEnd.y);
            const bx2 = Math.max(this.dragStart.x, this._dragEnd.x);
            const by2 = Math.max(this.dragStart.y, this._dragEnd.y);
            const color = this.bboxDragLabel === 1 ? BBOX_COLORS.positive : BBOX_COLORS.negative;

            ctx.fillStyle = color + "22";
            ctx.fillRect(bx1, by1, bx2 - bx1, by2 - by1);
            ctx.strokeStyle = color + "cc";
            ctx.lineWidth = 2 / this.zoom;
            ctx.setLineDash([4 / this.zoom, 3 / this.zoom]);
            ctx.strokeRect(bx1, by1, bx2 - bx1, by2 - by1);
            ctx.setLineDash([]);

            // Dimensions tooltip
            const ww = Math.abs(bx2 - bx1).toFixed(0);
            const hh = Math.abs(by2 - by1).toFixed(0);
            const dimText = `${ww}\u00d7${hh}`;
            const fs2 = Math.max(9, 10 / this.zoom);
            ctx.font = `${fs2}px monospace`;
            ctx.fillStyle = "#000000aa";
            const tw = ctx.measureText(dimText).width;
            ctx.fillRect(bx1, by2 + 2 / this.zoom, tw + 6 / this.zoom, fs2 + 4 / this.zoom);
            ctx.fillStyle = "#ffffffdd";
            ctx.fillText(dimText, bx1 + 3 / this.zoom, by2 + fs2 + 1 / this.zoom);
        }

        // ── Draw points ──────────────────────────────────────────────
        for (let i = 0; i < this.points.length; i++) {
            const p = this.points[i];
            const r = (p.radius || this.currentRadius);
            const isHovered = (i === this.hoveredPoint);
            const color = p.label === 1 ? POINT_COLORS.positive : POINT_COLORS.negative;

            // Glow
            const gr = r * 2.5;
            const gradient = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, gr);
            gradient.addColorStop(0, color + (isHovered ? "cc" : "88"));
            gradient.addColorStop(1, color + "00");
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(p.x, p.y, gr, 0, Math.PI * 2);
            ctx.fill();

            // Center dot
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(p.x, p.y, Math.max(2, 3 / this.zoom), 0, Math.PI * 2);
            ctx.fill();

            // Circle outline
            ctx.strokeStyle = color + "aa";
            ctx.lineWidth = 1.2 / this.zoom;
            ctx.beginPath();
            ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
            ctx.stroke();

            // Crosshair on hover
            if (isHovered) {
                ctx.strokeStyle = CROSSHAIR_COLOR;
                ctx.lineWidth = 1 / this.zoom;
                const ext = r + 5 / this.zoom;
                ctx.beginPath();
                ctx.moveTo(p.x - ext, p.y); ctx.lineTo(p.x + ext, p.y);
                ctx.moveTo(p.x, p.y - ext); ctx.lineTo(p.x, p.y + ext);
                ctx.stroke();
            }

            // Index label
            const fs3 = Math.max(8, 9 / this.zoom);
            ctx.fillStyle = "#ffffffdd";
            ctx.font = `${fs3}px monospace`;
            ctx.fillText(`${i}`, p.x + r + 3 / this.zoom, p.y - 3 / this.zoom);

            // Tooltip on hover
            if (isHovered) {
                const sign = p.label === 1 ? "+" : "\u2212";
                const text = `${sign} (${p.x.toFixed(1)}, ${p.y.toFixed(1)}) r=${r.toFixed(1)}`;
                const fs4 = Math.max(9, 10 / this.zoom);
                ctx.font = `${fs4}px monospace`;
                const ttw = ctx.measureText(text).width;
                ctx.fillStyle = "#000000cc";
                ctx.fillRect(p.x - ttw / 2 - 4 / this.zoom, p.y - r - (22 / this.zoom), ttw + 8 / this.zoom, (fs4 + 6) / this.zoom);
                ctx.fillStyle = "#ffffffdd";
                ctx.fillText(text, p.x - ttw / 2, p.y - r - (12 / this.zoom));
            }
        }

        ctx.restore(); // zoom/pan

        // ── HUD bar ──────────────────────────────────────────────────
        ctx.fillStyle = "#000000aa";
        ctx.fillRect(x, y, w, 22);
        ctx.fillStyle = "#ffffffcc";
        ctx.font = "11px monospace";
        const hudParts = [
            `${this.points.length} pts`,
            `${this.bboxes.length} bbox`,
            `radius:${this.currentRadius.toFixed(1)}`,
            `zoom:${this.zoom.toFixed(1)}x`,
        ];
        ctx.fillText(hudParts.join("  \u2502  "), x + 6, y + 15);

        // Help text at bottom
        ctx.fillStyle = "#ffffff66";
        ctx.font = "9px monospace";
        ctx.fillText("L/R=point  CTRL+drag=bbox  Shift=del  Scroll=radius  CTRL+Scroll=zoom", x + 6, y + h - 4);

        ctx.restore();
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Register with ComfyUI
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app.registerExtension({
    name: "MaskEditControl.PointsBBoxEditor",

    async nodeCreated(node) {
        if (!["PointsMaskEditor", "SAMMaskGeneratorMEC"].includes(node.comfyClass)) {
            return;
        }

        const editor = new PointsBBoxEditor(node);
        editor.saveState();

        // ── Sync canvas dimensions from widget values ────────────────
        function syncCanvasSize() {
            const wWidget = node.widgets?.find(w => w.name === "width");
            const hWidget = node.widgets?.find(w => w.name === "height");
            if (wWidget && hWidget) {
                editor._canvasW = wWidget.value;
                editor._canvasH = hWidget.value;
            }
        }
        syncCanvasSize();

        // Watch for width/height widget changes
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
        container.style.width = "100%";
        container.style.height = "340px";
        container.style.position = "relative";
        container.style.overflow = "hidden";
        container.style.cursor = "crosshair";
        container.style.borderRadius = "4px";
        container.style.border = "1px solid #444";
        container.style.boxSizing = "border-box";
        editor._containerEl = container;

        const canvas = document.createElement("canvas");
        canvas.style.width  = "100%";
        canvas.style.height = "100%";
        canvas.style.display = "block";
        container.appendChild(canvas);

        const widget = node.addDOMWidget("points_editor_canvas", "canvas", container, {
            serialize: false,
        });

        const ctx = canvas.getContext("2d");

        // ── Resize logic ─────────────────────────────────────────────
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

        // ── Try to load reference image from node input ──────────────
        function tryLoadRefImage() {
            if (!node.inputs) return;
            for (const inp of node.inputs) {
                if (inp.name === "reference_image" && inp.link != null) {
                    const linkInfo = app.graph.links[inp.link];
                    if (!linkInfo) continue;
                    const srcNode = app.graph.getNodeById(linkInfo.origin_id);
                    if (!srcNode) continue;

                    // Strategy 1: Post-execution preview images on source node
                    if (srcNode.imgs && srcNode.imgs.length > 0) {
                        editor.loadRefImage(srcNode.imgs[0].src);
                        return;
                    }

                    // Strategy 2: LoadImage node with "image" widget → /view API
                    const imageWidget = srcNode.widgets?.find(w => w.name === "image");
                    if (imageWidget && imageWidget.value) {
                        const val = imageWidget.value;
                        const parts = val.split("/");
                        const subfolder = parts.length > 1 ? parts.slice(0, -1).join("/") : "";
                        const filename = parts[parts.length - 1];
                        const url = `/view?filename=${encodeURIComponent(filename)}&subfolder=${encodeURIComponent(subfolder)}&type=input`;
                        editor.loadRefImage(url);
                        return;
                    }

                    // Strategy 3: Walk one level upstream to find source image
                    if (srcNode.inputs) {
                        for (const upInp of srcNode.inputs) {
                            if (upInp.link != null) {
                                const upLink = app.graph.links[upInp.link];
                                if (!upLink) continue;
                                const upNode = app.graph.getNodeById(upLink.origin_id);
                                if (!upNode) continue;
                                if (upNode.imgs?.length > 0) {
                                    editor.loadRefImage(upNode.imgs[0].src);
                                    return;
                                }
                                const upImgW = upNode.widgets?.find(w => w.name === "image");
                                if (upImgW?.value) {
                                    const uVal = upImgW.value;
                                    const uParts = uVal.split("/");
                                    const uSub = uParts.length > 1 ? uParts.slice(0, -1).join("/") : "";
                                    const uFile = uParts[uParts.length - 1];
                                    const url = `/view?filename=${encodeURIComponent(uFile)}&subfolder=${encodeURIComponent(uSub)}&type=input`;
                                    editor.loadRefImage(url);
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Detect connection changes to reload reference image
        const origConnChange = node.onConnectionsChange;
        node.onConnectionsChange = function(type, index, connected, link_info) {
            origConnChange?.apply(this, arguments);
            if (connected) {
                setTimeout(() => { tryLoadRefImage(); render(); }, 200);
            } else {
                editor._refLoaded = false;
                editor._refImage = null;
                editor._lastRefUrl = null;
                syncCanvasSize();
                render();
            }
        };

        // Periodically check for image (until loaded)
        const _imgInterval = setInterval(() => {
            if (!editor._refLoaded) tryLoadRefImage();
            else clearInterval(_imgInterval);
        }, 2000);

        // Also try on execution
        const origExecuted = node.onExecuted;
        node.onExecuted = function(output) {
            origExecuted?.apply(this, arguments);
            tryLoadRefImage();
            render();
        };

        // ── POINTER DOWN ─────────────────────────────────────────────
        canvas.addEventListener("pointerdown", (e) => {
            e.preventDefault();
            e.stopPropagation();
            canvas.focus();
            canvas.setPointerCapture(e.pointerId);
            const rect = canvas.getBoundingClientRect();
            const sx = e.clientX - rect.left;
            const sy = e.clientY - rect.top;
            const c = editor.screenToCanvas(sx, sy);

            // Middle mouse = pan
            if (e.button === 1) {
                editor.isPanning = true;
                editor.panStart = { x: e.clientX - editor.panX, y: e.clientY - editor.panY };
                canvas.style.cursor = "grab";
                return;
            }

            // CTRL held (without Shift) = BBox drag mode
            if (e.ctrlKey && !e.shiftKey && (e.button === 0 || e.button === 2)) {
                editor.isBboxDrag = true;
                editor.bboxDragLabel = (e.button === 0) ? 1 : 0;  // left=positive, right=negative
                editor.dragStart = { x: c.x, y: c.y };
                editor._dragEnd  = { x: c.x, y: c.y };
                canvas.style.cursor = "crosshair";
                return;
            }

            // Normal click = add/delete point
            if (e.button === 0 || e.button === 2) {
                // Shift+click = delete element under cursor
                if (e.shiftKey) {
                    const idx = editor.findPointAt(c.x, c.y);
                    if (idx >= 0) {
                        editor.saveState();
                        editor.points.splice(idx, 1);
                        editor.hoveredPoint = -1;
                        editor.updateWidgets();
                        render();
                        return;
                    }
                    const bidx = editor.findBboxAt(c.x, c.y);
                    if (bidx >= 0) {
                        editor.saveState();
                        editor.bboxes.splice(bidx, 1);
                        editor.hoveredBbox = -1;
                        editor.updateWidgets();
                        render();
                        return;
                    }
                    return;
                }

                // Add point
                editor.saveState();
                editor.points.push({
                    x: parseFloat(c.x.toFixed(2)),
                    y: parseFloat(c.y.toFixed(2)),
                    label: e.button === 0 ? 1 : 0,
                    radius: editor.currentRadius,
                });
                editor.updateWidgets();
                render();
            }
        });

        // ── POINTER MOVE ─────────────────────────────────────────────
        canvas.addEventListener("pointermove", (e) => {
            const rect = canvas.getBoundingClientRect();
            const sx = e.clientX - rect.left;
            const sy = e.clientY - rect.top;
            const c = editor.screenToCanvas(sx, sy);

            if (editor.isPanning) {
                editor.panX = e.clientX - editor.panStart.x;
                editor.panY = e.clientY - editor.panStart.y;
                render();
                return;
            }

            if (editor.isBboxDrag) {
                editor._dragEnd = { x: c.x, y: c.y };
                render();
                return;
            }

            // Hover detection
            const oldPoint = editor.hoveredPoint;
            const oldBbox  = editor.hoveredBbox;
            editor.hoveredPoint = editor.findPointAt(c.x, c.y);
            editor.hoveredBbox  = editor.hoveredPoint < 0 ? editor.findBboxAt(c.x, c.y) : -1;
            if (oldPoint !== editor.hoveredPoint || oldBbox !== editor.hoveredBbox) {
                render();
            }
        });

        // ── POINTER UP ───────────────────────────────────────────────
        canvas.addEventListener("pointerup", (e) => {
            try { canvas.releasePointerCapture(e.pointerId); } catch(_) {}
            if (editor.isPanning) {
                editor.isPanning = false;
                canvas.style.cursor = "crosshair";
                return;
            }

            if (editor.isBboxDrag && editor.dragStart && editor._dragEnd) {
                const x1 = Math.min(editor.dragStart.x, editor._dragEnd.x);
                const y1 = Math.min(editor.dragStart.y, editor._dragEnd.y);
                const x2 = Math.max(editor.dragStart.x, editor._dragEnd.x);
                const y2 = Math.max(editor.dragStart.y, editor._dragEnd.y);

                // Only commit if large enough (>3px in canvas space)
                if (x2 - x1 > 3 && y2 - y1 > 3) {
                    editor.saveState();
                    editor.bboxes.push([
                        Math.round(x1), Math.round(y1),
                        Math.round(x2), Math.round(y2),
                        editor.bboxDragLabel,
                    ]);
                    editor.updateWidgets();
                }

                editor.isBboxDrag = false;
                editor.dragStart  = null;
                editor._dragEnd   = null;
                render();
            }
        });

        // ── SCROLL ───────────────────────────────────────────────────
        canvas.addEventListener("wheel", (e) => {
            e.preventDefault();
            if (e.ctrlKey) {
                // CTRL+Scroll = zoom
                const factor = e.deltaY < 0 ? 1.12 : 0.89;
                const rect = canvas.getBoundingClientRect();
                const mx = e.clientX - rect.left;
                const my = e.clientY - rect.top;
                editor.panX = mx - (mx - editor.panX) * factor;
                editor.panY = my - (my - editor.panY) * factor;
                editor.zoom *= factor;
                editor.zoom = Math.max(0.1, Math.min(editor.zoom, 20));
            } else {
                // Scroll = radius adjust
                editor.currentRadius += e.deltaY < 0 ? 0.5 : -0.5;
                editor.currentRadius = Math.max(0.5, Math.min(editor.currentRadius, 256));
            }
            render();
        }, { passive: false });

        // ── Context menu prevention ──────────────────────────────────
        canvas.addEventListener("contextmenu", (e) => e.preventDefault());

        // ── Keyboard shortcuts ───────────────────────────────────────
        canvas.tabIndex = 0;
        canvas.style.outline = "none";
        canvas.addEventListener("keydown", (e) => {
            if (e.key === "z" && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                if (e.shiftKey) { editor.redo(); } else { editor.undo(); }
                render();
            } else if (e.key === "y" && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                editor.redo();
                render();
            } else if (e.key === "c" && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                editor.saveState();
                editor.points = [];
                editor.bboxes = [];
                editor.updateWidgets();
                render();
            } else if (e.key === "Delete" || e.key === "Backspace") {
                e.preventDefault();
                if (editor.hoveredPoint >= 0) {
                    editor.saveState();
                    editor.points.splice(editor.hoveredPoint, 1);
                    editor.hoveredPoint = -1;
                    editor.updateWidgets();
                    render();
                } else if (editor.hoveredBbox >= 0) {
                    editor.saveState();
                    editor.bboxes.splice(editor.hoveredBbox, 1);
                    editor.hoveredBbox = -1;
                    editor.updateWidgets();
                    render();
                }
            } else if (e.key === "r" || e.key === "R") {
                editor.zoom = 1.0; editor.panX = 0; editor.panY = 0;
                render();
            }
        });

        // ── Store references on node ─────────────────────────────────
        node._mecEditor = editor;
        node._mecRender = render;
    },

    // ── Serialization hooks ──────────────────────────────────────────
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!["PointsMaskEditor", "SAMMaskGeneratorMEC"].includes(nodeData.name)) {
            return;
        }

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
                data._mecState = {
                    points: this._mecEditor.points,
                    bboxes: this._mecEditor.bboxes,
                };
            }
            return data;
        };
    },
});

/**
 * ComfyUI-MaskEditControl – Points & BBox Editor Widget
 * 
 * Interactive canvas widget for:
 *  - Click to add positive/negative points (left=pos, right=neg)
 *  - Drag to draw bounding boxes
 *  - Adjustable point radius via scroll wheel
 *  - Sub-pixel coordinate display
 *  - Undo/redo support
 *  - Live JSON output to node
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const POINT_COLORS = {
    positive: "#00ff00",
    negative: "#ff0000",
};
const BBOX_COLOR = "#00aaff";
const CROSSHAIR_COLOR = "#ffffff80";

class PointsBBoxEditor {
    constructor(node, widget) {
        this.node = node;
        this.widget = widget;
        this.points = [];
        this.bbox = null;
        this.dragStart = null;
        this.isDragging = false;
        this.currentRadius = 5.0;
        this.mode = "points"; // "points" or "bbox"
        this.history = [];
        this.historyIndex = -1;
        this.hoveredPoint = -1;
        this.imageData = null;
        this.zoom = 1.0;
        this.panX = 0;
        this.panY = 0;
        this.isPanning = false;
        this.panStart = { x: 0, y: 0 };
        this.canvasWidth = 512;
        this.canvasHeight = 512;
    }

    saveState() {
        const state = {
            points: JSON.parse(JSON.stringify(this.points)),
            bbox: this.bbox ? [...this.bbox] : null,
        };
        this.history = this.history.slice(0, this.historyIndex + 1);
        this.history.push(state);
        this.historyIndex = this.history.length - 1;
        if (this.history.length > 100) {
            this.history.shift();
            this.historyIndex--;
        }
    }

    undo() {
        if (this.historyIndex > 0) {
            this.historyIndex--;
            const state = this.history[this.historyIndex];
            this.points = JSON.parse(JSON.stringify(state.points));
            this.bbox = state.bbox ? [...state.bbox] : null;
            this.updateWidgets();
        }
    }

    redo() {
        if (this.historyIndex < this.history.length - 1) {
            this.historyIndex++;
            const state = this.history[this.historyIndex];
            this.points = JSON.parse(JSON.stringify(state.points));
            this.bbox = state.bbox ? [...state.bbox] : null;
            this.updateWidgets();
        }
    }

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

    findPointAt(cx, cy) {
        const threshold = Math.max(8, this.currentRadius * 1.5) / this.zoom;
        for (let i = this.points.length - 1; i >= 0; i--) {
            const p = this.points[i];
            const dx = p.x - cx;
            const dy = p.y - cy;
            if (Math.sqrt(dx * dx + dy * dy) < threshold) {
                return i;
            }
        }
        return -1;
    }

    updateWidgets() {
        // Update points_json widget
        const pointsWidget = this.node.widgets?.find(w => w.name === "points_json");
        if (pointsWidget) {
            pointsWidget.value = JSON.stringify(this.points, null, 2);
        }
        // Update bbox_json widget  
        const bboxWidget = this.node.widgets?.find(w => w.name === "bbox_json");
        if (bboxWidget && this.bbox) {
            bboxWidget.value = JSON.stringify(this.bbox);
        }
        app.graph.setDirtyCanvas(true, false);
    }

    draw(ctx, widgetY, widgetWidth, widgetHeight) {
        const x = 0;
        const y = widgetY;
        const w = widgetWidth;
        const h = widgetHeight;

        // Background
        ctx.fillStyle = "#1a1a2e";
        ctx.fillRect(x, y, w, h);

        // Checkerboard
        ctx.save();
        ctx.beginPath();
        ctx.rect(x, y, w, h);
        ctx.clip();

        const checkSize = 10 * this.zoom;
        for (let cy = 0; cy < h / checkSize + 1; cy++) {
            for (let cx = 0; cx < w / checkSize + 1; cx++) {
                ctx.fillStyle = (cx + cy) % 2 === 0 ? "#2a2a3e" : "#222238";
                ctx.fillRect(
                    x + cx * checkSize + (this.panX % (checkSize * 2)),
                    y + cy * checkSize + (this.panY % (checkSize * 2)),
                    checkSize, checkSize
                );
            }
        }

        // Draw image if available
        if (this.imageData) {
            const sx = x + this.panX;
            const sy = y + this.panY;
            ctx.globalAlpha = 0.6;
            ctx.drawImage(this.imageData, sx, sy,
                this.canvasWidth * this.zoom, this.canvasHeight * this.zoom);
            ctx.globalAlpha = 1.0;
        }

        // Draw bbox
        if (this.bbox) {
            const s1 = this.canvasToScreen(this.bbox[0], this.bbox[1]);
            const s2 = this.canvasToScreen(this.bbox[0] + this.bbox[2], this.bbox[1] + this.bbox[3]);
            ctx.strokeStyle = BBOX_COLOR;
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 3]);
            ctx.strokeRect(x + s1.x, y + s1.y, s2.x - s1.x, s2.y - s1.y);
            ctx.setLineDash([]);

            // BBox label
            ctx.fillStyle = BBOX_COLOR;
            ctx.font = "10px monospace";
            ctx.fillText(
                `bbox: [${this.bbox[0]}, ${this.bbox[1]}, ${this.bbox[2]}, ${this.bbox[3]}]`,
                x + s1.x + 4, y + s1.y - 4
            );
        }

        // Draw dragging bbox
        if (this.isDragging && this.mode === "bbox" && this.dragStart) {
            ctx.strokeStyle = BBOX_COLOR + "80";
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            const ds = this.canvasToScreen(this.dragStart.x, this.dragStart.y);
            // dragEnd is tracked during mouse move
            if (this._dragEnd) {
                const de = this.canvasToScreen(this._dragEnd.x, this._dragEnd.y);
                ctx.strokeRect(x + ds.x, y + ds.y, de.x - ds.x, de.y - ds.y);
            }
            ctx.setLineDash([]);
        }

        // Draw points
        for (let i = 0; i < this.points.length; i++) {
            const p = this.points[i];
            const sp = this.canvasToScreen(p.x, p.y);
            const r = (p.radius || this.currentRadius) * this.zoom;
            const isHovered = i === this.hoveredPoint;

            // Soft circle
            const gradient = ctx.createRadialGradient(
                x + sp.x, y + sp.y, 0,
                x + sp.x, y + sp.y, r
            );
            const color = p.label === 1 ? POINT_COLORS.positive : POINT_COLORS.negative;
            gradient.addColorStop(0, color + (isHovered ? "cc" : "88"));
            gradient.addColorStop(1, color + "00");
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(x + sp.x, y + sp.y, r, 0, Math.PI * 2);
            ctx.fill();

            // Center dot
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(x + sp.x, y + sp.y, Math.max(2, 3 * this.zoom), 0, Math.PI * 2);
            ctx.fill();

            // Crosshair
            if (isHovered) {
                ctx.strokeStyle = CROSSHAIR_COLOR;
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(x + sp.x - r - 5, y + sp.y);
                ctx.lineTo(x + sp.x + r + 5, y + sp.y);
                ctx.moveTo(x + sp.x, y + sp.y - r - 5);
                ctx.lineTo(x + sp.x, y + sp.y + r + 5);
                ctx.stroke();
            }

            // Index label
            ctx.fillStyle = "#fff";
            ctx.font = `${Math.max(9, 10 * this.zoom)}px monospace`;
            ctx.fillText(`${i}`, x + sp.x + r + 3, y + sp.y - 3);

            // Coordinate tooltip on hover
            if (isHovered) {
                const label = p.label === 1 ? "+" : "-";
                const text = `${label} (${p.x.toFixed(1)}, ${p.y.toFixed(1)}) r=${(p.radius || this.currentRadius).toFixed(1)}`;
                ctx.fillStyle = "#000000cc";
                const tw = ctx.measureText(text).width;
                ctx.fillRect(x + sp.x - tw / 2 - 4, y + sp.y - r - 22, tw + 8, 16);
                ctx.fillStyle = "#ffffffdd";
                ctx.fillText(text, x + sp.x - tw / 2, y + sp.y - r - 10);
            }
        }

        // Mode indicator + toolbar
        ctx.fillStyle = "#00000088";
        ctx.fillRect(x, y, w, 20);
        ctx.fillStyle = "#ffffffcc";
        ctx.font = "11px monospace";
        const modeStr = this.mode === "points"
            ? `POINTS mode | ${this.points.length} pts | radius: ${this.currentRadius.toFixed(1)} | L=add R=sub scroll=radius`
            : `BBOX mode | drag to draw | ${this.bbox ? "bbox set" : "no bbox"}`;
        ctx.fillText(modeStr, x + 6, y + 14);

        // Toggle button
        ctx.fillStyle = this.mode === "points" ? "#00ff0044" : "#00aaff44";
        ctx.fillRect(x + w - 80, y, 80, 20);
        ctx.fillStyle = "#fff";
        ctx.fillText(this.mode === "points" ? "[B]Box" : "[P]oints", x + w - 72, y + 14);

        ctx.restore();
    }
}

// ── Register the widget with ComfyUI ─────────────────────────────────

app.registerExtension({
    name: "MaskEditControl.PointsBBoxEditor",

    async nodeCreated(node) {
        if (!["PointsMaskEditor", "SAMMaskGeneratorMEC"].includes(node.comfyClass)) {
            return;
        }

        const editor = new PointsBBoxEditor(node, null);
        editor.saveState();

        // Add canvas widget
        const widget = node.addDOMWidget("points_editor_canvas", "canvas", document.createElement("div"), {
            serialize: false,
        });

        // Create actual canvas element
        const container = widget.element;
        container.style.width = "100%";
        container.style.height = "300px";
        container.style.position = "relative";
        container.style.overflow = "hidden";
        container.style.cursor = "crosshair";
        container.style.borderRadius = "4px";
        container.style.border = "1px solid #444";

        const canvas = document.createElement("canvas");
        canvas.style.width = "100%";
        canvas.style.height = "100%";
        container.appendChild(canvas);

        const ctx = canvas.getContext("2d");

        function resize() {
            const rect = container.getBoundingClientRect();
            canvas.width = rect.width * (window.devicePixelRatio || 1);
            canvas.height = rect.height * (window.devicePixelRatio || 1);
            ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
            render();
        }

        function render() {
            const rect = container.getBoundingClientRect();
            ctx.clearRect(0, 0, rect.width, rect.height);
            editor.draw(ctx, 0, rect.width, rect.height);
        }

        // Resize observer
        const resizeObserver = new ResizeObserver(() => resize());
        resizeObserver.observe(container);
        setTimeout(resize, 100);

        // ── Mouse events ──────────────────────────────────────────────

        canvas.addEventListener("mousedown", (e) => {
            e.preventDefault();
            const rect = canvas.getBoundingClientRect();
            const sx = e.clientX - rect.left;
            const sy = e.clientY - rect.top;
            const c = editor.screenToCanvas(sx, sy);

            // Middle button = pan
            if (e.button === 1) {
                editor.isPanning = true;
                editor.panStart = { x: e.clientX - editor.panX, y: e.clientY - editor.panY };
                canvas.style.cursor = "grab";
                return;
            }

            if (editor.mode === "points") {
                if (e.button === 0 || e.button === 2) {
                    // Check if clicking on existing point to delete (shift+click)
                    if (e.shiftKey) {
                        const idx = editor.findPointAt(c.x, c.y);
                        if (idx >= 0) {
                            editor.saveState();
                            editor.points.splice(idx, 1);
                            editor.updateWidgets();
                            render();
                            return;
                        }
                    }

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
            } else if (editor.mode === "bbox") {
                if (e.button === 0) {
                    editor.isDragging = true;
                    editor.dragStart = { x: c.x, y: c.y };
                    editor._dragEnd = { x: c.x, y: c.y };
                }
            }
        });

        canvas.addEventListener("mousemove", (e) => {
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

            if (editor.isDragging && editor.mode === "bbox") {
                editor._dragEnd = { x: c.x, y: c.y };
                render();
                return;
            }

            // Hover detection
            const oldHover = editor.hoveredPoint;
            editor.hoveredPoint = editor.findPointAt(c.x, c.y);
            if (oldHover !== editor.hoveredPoint) {
                render();
            }
        });

        canvas.addEventListener("mouseup", (e) => {
            if (editor.isPanning) {
                editor.isPanning = false;
                canvas.style.cursor = "crosshair";
                return;
            }

            if (editor.isDragging && editor.mode === "bbox" && editor.dragStart && editor._dragEnd) {
                editor.saveState();
                const x1 = Math.min(editor.dragStart.x, editor._dragEnd.x);
                const y1 = Math.min(editor.dragStart.y, editor._dragEnd.y);
                const x2 = Math.max(editor.dragStart.x, editor._dragEnd.x);
                const y2 = Math.max(editor.dragStart.y, editor._dragEnd.y);
                if (x2 - x1 > 2 && y2 - y1 > 2) {
                    editor.bbox = [
                        Math.round(x1), Math.round(y1),
                        Math.round(x2 - x1), Math.round(y2 - y1)
                    ];
                    editor.updateWidgets();
                }
                editor.isDragging = false;
                editor.dragStart = null;
                editor._dragEnd = null;
                render();
            }
        });

        // Scroll = adjust radius
        canvas.addEventListener("wheel", (e) => {
            e.preventDefault();
            if (e.ctrlKey) {
                // Zoom
                const factor = e.deltaY < 0 ? 1.1 : 0.9;
                const rect = canvas.getBoundingClientRect();
                const mx = e.clientX - rect.left;
                const my = e.clientY - rect.top;
                editor.panX = mx - (mx - editor.panX) * factor;
                editor.panY = my - (my - editor.panY) * factor;
                editor.zoom *= factor;
                editor.zoom = Math.max(0.1, Math.min(editor.zoom, 20));
            } else {
                editor.currentRadius += e.deltaY < 0 ? 0.5 : -0.5;
                editor.currentRadius = Math.max(0.5, Math.min(editor.currentRadius, 256));
            }
            render();
        });

        // Right-click context menu prevention
        canvas.addEventListener("contextmenu", (e) => e.preventDefault());

        // Keyboard shortcuts
        canvas.tabIndex = 0;
        canvas.addEventListener("keydown", (e) => {
            if (e.key === "b" || e.key === "B") {
                editor.mode = editor.mode === "points" ? "bbox" : "points";
                render();
            } else if (e.key === "p" || e.key === "P") {
                editor.mode = "points";
                render();
            } else if (e.key === "z" && (e.ctrlKey || e.metaKey)) {
                if (e.shiftKey) {
                    editor.redo();
                } else {
                    editor.undo();
                }
                render();
            } else if (e.key === "c" && (e.ctrlKey || e.metaKey)) {
                // Clear all
                editor.saveState();
                editor.points = [];
                editor.bbox = null;
                editor.updateWidgets();
                render();
            } else if (e.key === "Delete" || e.key === "Backspace") {
                // Delete hovered point
                if (editor.hoveredPoint >= 0) {
                    editor.saveState();
                    editor.points.splice(editor.hoveredPoint, 1);
                    editor.hoveredPoint = -1;
                    editor.updateWidgets();
                    render();
                }
            } else if (e.key === "r" || e.key === "R") {
                // Reset view
                editor.zoom = 1.0;
                editor.panX = 0;
                editor.panY = 0;
                render();
            }
        });

        // Store ref
        node._mecEditor = editor;
        node._mecRender = render;
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Add serialization hooks
        if (["PointsMaskEditor", "SAMMaskGeneratorMEC"].includes(nodeData.name)) {
            const origOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                origOnConfigure?.apply(this, arguments);
                if (this._mecEditor && info._mecState) {
                    this._mecEditor.points = info._mecState.points || [];
                    this._mecEditor.bbox = info._mecState.bbox || null;
                    this._mecEditor.updateWidgets();
                    this._mecRender?.();
                }
            };

            const origSerialize = nodeType.prototype.serialize;
            nodeType.prototype.serialize = function () {
                const data = origSerialize?.apply(this, arguments) || {};
                if (this._mecEditor) {
                    data._mecState = {
                        points: this._mecEditor.points,
                        bbox: this._mecEditor.bbox,
                    };
                }
                return data;
            };
        }
    },
});

// ─────────────────────────────────────────────────────────────────────
// MEC Advanced Paint Canvas widget
// ─────────────────────────────────────────────────────────────────────
//   • Left-drag  = paint
//   • Right-drag = erase (transparency)
//   • Mouse wheel scroll over canvas = brush size
//   • Cursor matches brush_size + brush_hardness (hard outline / soft glow)
//   • Bresenham-style linear interpolation between samples so fast drags
//     never produce dotted strokes.
//   • On serialise the canvas pixels are dumped as a PNG → base64 string and
//     stored in the hidden `canvas_data` widget so the Python node can decode
//     them.
// ─────────────────────────────────────────────────────────────────────
import { app } from "../../scripts/app.js";

const NODE_NAME = "MECAdvancedPaintCanvas";

function hexToRgb(hex) {
    if (!hex) return [0, 0, 0];
    let h = hex.trim();
    if (h.startsWith("#")) h = h.slice(1);
    if (h.length === 3) h = h.split("").map((c) => c + c).join("");
    const v = parseInt(h, 16);
    if (isNaN(v)) return [0, 0, 0];
    return [(v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff];
}

function getWidget(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

class PaintCanvasController {
    constructor(node) {
        this.node = node;
        this.size = [512, 512];
        this.dpr = window.devicePixelRatio || 1;

        // Wrapper that ComfyUI sizes for us
        this.root = document.createElement("div");
        Object.assign(this.root.style, {
            position: "relative",
            width: "100%",
            minHeight: "320px",
            background: "#1c1c1c",
            border: "1px solid #444",
            borderRadius: "4px",
            overflow: "hidden",
            userSelect: "none",
            touchAction: "none",
        });

        // Drawing canvas (RGBA, persistent)
        this.draw = document.createElement("canvas");
        this.draw.width = this.size[0];
        this.draw.height = this.size[1];
        Object.assign(this.draw.style, {
            position: "absolute", inset: 0,
            width: "100%", height: "100%",
            cursor: "crosshair",
        });
        this.ctx = this.draw.getContext("2d");

        // Cursor overlay
        this.cursor = document.createElement("canvas");
        Object.assign(this.cursor.style, {
            position: "absolute", inset: 0,
            width: "100%", height: "100%",
            pointerEvents: "none",
        });
        this.cctx = this.cursor.getContext("2d");

        this.root.appendChild(this.draw);
        this.root.appendChild(this.cursor);

        // pointer state
        this._down = false;
        this._eraser = false;
        this._last = null;
        this._mouse = null;

        this._bind();
    }

    // ─── events ──────────────────────────────────────────────────────
    _bind() {
        const c = this.draw;
        c.addEventListener("contextmenu", (e) => e.preventDefault());
        c.addEventListener("pointerdown", (e) => this._onDown(e));
        c.addEventListener("pointermove", (e) => this._onMove(e));
        window.addEventListener("pointerup",   (e) => this._onUp(e));
        c.addEventListener("pointerleave", () => { this._mouse = null; this._drawCursor(); });
        c.addEventListener("wheel", (e) => this._onWheel(e), { passive: false });
    }

    _localPos(e) {
        const r = this.draw.getBoundingClientRect();
        const x = ((e.clientX - r.left) / r.width)  * this.size[0];
        const y = ((e.clientY - r.top)  / r.height) * this.size[1];
        return [x, y];
    }

    _onDown(e) {
        e.preventDefault();
        this.draw.setPointerCapture?.(e.pointerId);
        this._down = true;
        this._eraser = e.button === 2;          // right-button erases
        this._last = this._localPos(e);
        this._stamp(this._last[0], this._last[1]);
    }
    _onMove(e) {
        this._mouse = this._localPos(e);
        if (this._down) {
            this._stroke(this._last, this._mouse);
            this._last = this._mouse;
            this._serialiseSoon();
        }
        this._drawCursor();
    }
    _onUp(e) {
        if (!this._down) return;
        this._down = false;
        this._serialiseSoon();
    }
    _onWheel(e) {
        e.preventDefault();
        const w = getWidget(this.node, "brush_size");
        if (!w) return;
        const dir = e.deltaY > 0 ? -2 : 2;
        w.value = Math.max(1, Math.min(500, (w.value | 0) + dir));
        if (w.callback) w.callback(w.value);
        this.node.setDirtyCanvas(true, true);
        this._drawCursor();
    }

    // ─── drawing ─────────────────────────────────────────────────────
    _brushParams() {
        const size = +(getWidget(this.node, "brush_size")?.value     ?? 20);
        const hard = +(getWidget(this.node, "brush_hardness")?.value ?? 0.8);
        const op   = +(getWidget(this.node, "brush_opacity")?.value  ?? 1.0);
        const col  = (getWidget(this.node, "brush_color")?.value     ?? "#000000");
        return { size, hard, op, col };
    }

    _stamp(x, y) {
        const { size, hard, op, col } = this._brushParams();
        const r = Math.max(1, size * 0.5);
        const ctx = this.ctx;
        ctx.save();
        if (this._eraser) {
            ctx.globalCompositeOperation = "destination-out";
        } else {
            ctx.globalCompositeOperation = "source-over";
        }
        const [R, G, B] = hexToRgb(col);
        const grad = ctx.createRadialGradient(x, y, 0, x, y, r);
        // hardness=1 → solid disc; hardness=0 → gaussian-ish falloff.
        const inner = Math.max(0, Math.min(1, hard));
        grad.addColorStop(0,                `rgba(${R},${G},${B},${op})`);
        grad.addColorStop(inner,            `rgba(${R},${G},${B},${op})`);
        grad.addColorStop(1,                `rgba(${R},${G},${B},0)`);
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
    }

    _stroke(a, b) {
        // Linear interpolation along the segment with spacing = 0.25 * radius.
        const { size } = this._brushParams();
        const r = Math.max(1, size * 0.5);
        const dx = b[0] - a[0], dy = b[1] - a[1];
        const dist = Math.hypot(dx, dy);
        const step = Math.max(1, r * 0.25);
        const n = Math.ceil(dist / step);
        for (let i = 1; i <= n; i++) {
            const t = i / n;
            this._stamp(a[0] + dx * t, a[1] + dy * t);
        }
    }

    // ─── cursor overlay ──────────────────────────────────────────────
    _drawCursor() {
        const cv = this.cursor;
        const r2 = this.draw.getBoundingClientRect();
        // Match overlay backing size to drawing canvas pixel ratio
        if (cv.width !== this.size[0] || cv.height !== this.size[1]) {
            cv.width = this.size[0];
            cv.height = this.size[1];
        }
        const ctx = this.cctx;
        ctx.clearRect(0, 0, cv.width, cv.height);
        if (!this._mouse) return;
        const { size, hard } = this._brushParams();
        const r = Math.max(1, size * 0.5);
        const [x, y] = this._mouse;
        ctx.save();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = "rgba(255,255,255,0.95)";
        if (hard >= 0.99) {
            ctx.beginPath();
            ctx.arc(x, y, r, 0, Math.PI * 2);
            ctx.stroke();
        } else {
            // soft halo: 4 dotted rings shrinking by hardness
            ctx.setLineDash([3, 3]);
            for (let i = 0; i < 4; i++) {
                const f = 1.0 - (1.0 - hard) * (i / 4);
                ctx.beginPath();
                ctx.arc(x, y, r * f, 0, Math.PI * 2);
                ctx.stroke();
            }
        }
        ctx.restore();
    }

    // ─── serialise to hidden widget ──────────────────────────────────
    _serialiseSoon() {
        if (this._serTimer) return;
        this._serTimer = setTimeout(() => {
            this._serTimer = null;
            this.serialise();
        }, 60);
    }
    serialise() {
        const w = getWidget(this.node, "canvas_data");
        if (!w) return;
        try {
            w.value = this.draw.toDataURL("image/png");
        } catch (e) {
            console.warn("[MEC paint] serialise failed", e);
        }
    }

    // ─── public api ──────────────────────────────────────────────────
    setSize(w, h) {
        if (this.size[0] === w && this.size[1] === h) return;
        // resample existing pixels into new size
        const tmp = document.createElement("canvas");
        tmp.width  = this.draw.width;
        tmp.height = this.draw.height;
        tmp.getContext("2d").drawImage(this.draw, 0, 0);
        this.draw.width = w;
        this.draw.height = h;
        this.cursor.width = w;
        this.cursor.height = h;
        this.size = [w, h];
        this.ctx.drawImage(tmp, 0, 0, w, h);
        this._drawCursor();
    }
    clear() {
        this.ctx.clearRect(0, 0, this.draw.width, this.draw.height);
        this._serialiseSoon();
    }
    loadFromDataURL(url) {
        if (!url) return;
        const img = new Image();
        img.onload = () => {
            this.ctx.clearRect(0, 0, this.draw.width, this.draw.height);
            this.ctx.drawImage(img, 0, 0, this.draw.width, this.draw.height);
        };
        img.src = url;
    }
}

app.registerExtension({
    name: "mec.advanced_paint",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onCreated?.apply(this, arguments);

            const ctrl = new PaintCanvasController(this);
            this._mecPaint = ctrl;

            // Hidden data widget — created once if not present
            let dataW = getWidget(this, "canvas_data");
            if (!dataW) {
                dataW = this.addWidget("text", "canvas_data", "", () => {}, { multiline: false });
            }
            // Hide the textbox visually
            if (dataW.element) dataW.element.style.display = "none";
            dataW.serializeValue = () => dataW.value || "";
            dataW.computeSize = () => [0, -4];
            dataW.type = "hidden";

            // Clear button
            this.addWidget("button", "Clear Canvas", null, () => ctrl.clear());

            // DOM widget mount
            this.addDOMWidget("paint_canvas", "canvas", ctrl.root, {
                serialize: false,
                getMinHeight: () => 320,
            });

            // Sync canvas dimensions with width/height widgets
            const sync = () => {
                const w = +(getWidget(this, "canvas_width")?.value  ?? 512);
                const h = +(getWidget(this, "canvas_height")?.value ?? 512);
                ctrl.setSize(w, h);
            };
            const wW = getWidget(this, "canvas_width");
            const hW = getWidget(this, "canvas_height");
            if (wW) { const cb = wW.callback; wW.callback = (v) => { cb?.(v); sync(); }; }
            if (hW) { const cb = hW.callback; hW.callback = (v) => { cb?.(v); sync(); }; }
            setTimeout(sync, 0);

            // Restore previous serialised image if the node was loaded from a saved workflow
            setTimeout(() => {
                const v = dataW.value;
                if (v && typeof v === "string" && v.startsWith("data:")) ctrl.loadFromDataURL(v);
            }, 30);

            // Reference image — we don't currently decode upstream IMAGE here
            // (that requires running the graph), but we leave the hook so a
            // future preview endpoint can paint the background.
        };

        // Make sure the latest canvas pixels are written before save / queue
        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (o) {
            try { this._mecPaint?.serialise(); } catch (e) {}
            onSerialize?.apply(this, arguments);
        };
    },
});

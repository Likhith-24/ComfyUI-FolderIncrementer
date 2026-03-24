import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * ImageComparerMEC – interactive before/after comparison widget.
 * Modes: Slider (drag divider), Overlay (drag to blend), Diff (heatmap).
 */

const MODES = ["◧ Compare", "⊕ Overlay", "≠ Diff"];
const BTN_W = 62, BTN_H = 22, BTN_GAP = 3, BTN_PAD = 6;
const HANDLE_R = 14;
const LABEL_FADE_MS = 200;

app.registerExtension({
    name: "MEC.ImageComparer",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ImageComparerMEC") return;

        /* ── onNodeCreated ─────────────────────────────── */
        const _created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            _created?.apply(this, arguments);
            const node = this;

            const el = document.createElement("div");
            el.style.cssText =
                "position:relative;width:100%;min-height:200px;background:#111;border-radius:4px;overflow:hidden;";

            const cvs = document.createElement("canvas");
            cvs.style.cssText = "display:block;width:100%;cursor:col-resize;";
            el.appendChild(cvs);

            const S = {
                cvs,
                ctx: cvs.getContext("2d"),
                el,
                imgA: null,
                imgB: null,
                diffCvs: null,
                mode: 0,
                divPos: 0.5,
                alpha: 0.5,
                drag: false,
                labelA: "Before",
                labelB: "After",
                labelOpacity: 1.0,
                _fadeTimer: null,
            };
            node._S = S;

            /* ── pointer helpers ── */
            const canvasXY = (e) => {
                const r = cvs.getBoundingClientRect();
                return [
                    (e.clientX - r.left) * (cvs.width / (r.width || 1)),
                    (e.clientY - r.top) * (cvs.height / (r.height || 1)),
                ];
            };

            cvs.addEventListener("pointerdown", (e) => {
                const [cx, cy] = canvasXY(e);

                // Hit-test mode buttons (top-right)
                const bx0 =
                    cvs.width -
                    (BTN_W * MODES.length + BTN_GAP * (MODES.length - 1)) -
                    BTN_PAD;
                if (cy >= BTN_PAD && cy <= BTN_PAD + BTN_H && cx >= bx0) {
                    for (let i = 0; i < MODES.length; i++) {
                        const bx = bx0 + i * (BTN_W + BTN_GAP);
                        if (cx >= bx && cx <= bx + BTN_W) {
                            S.mode = i;
                            cvs.style.cursor =
                                i === 2 ? "default" : "col-resize";
                            if (i === 2 && !S.diffCvs) node._buildDiff();
                            node._render();
                            return;
                        }
                    }
                }

                // Start drag
                cvs.setPointerCapture(e.pointerId);
                S.drag = true;
                S.labelOpacity = 0.15;
                if (S._fadeTimer) clearTimeout(S._fadeTimer);
                const nx = cx / cvs.width;
                if (S.mode === 0)
                    S.divPos = Math.max(0.01, Math.min(0.99, nx));
                else if (S.mode === 1)
                    S.alpha = Math.max(0, Math.min(1, nx));
                node._render();
            });

            cvs.addEventListener("pointermove", (e) => {
                if (!S.drag) return;
                const [cx] = canvasXY(e);
                const nx = cx / cvs.width;
                if (S.mode === 0)
                    S.divPos = Math.max(0.01, Math.min(0.99, nx));
                else if (S.mode === 1)
                    S.alpha = Math.max(0, Math.min(1, nx));
                node._render();
            });

            const up = () => {
                S.drag = false;
                if (S._fadeTimer) clearTimeout(S._fadeTimer);
                S._fadeTimer = setTimeout(() => {
                    S.labelOpacity = 1.0;
                    node._render();
                }, LABEL_FADE_MS);
            };
            cvs.addEventListener("pointerup", up);
            cvs.addEventListener("pointercancel", up);

            node.addDOMWidget("comparer_view", "COMPARER", el, {
                serialize: false,
            });
            node.setSize([420, 380]);
        };

        /* ── onExecuted ────────────────────────────────── */
        const _exec = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (msg) {
            _exec?.apply(this, arguments);
            if (!msg?.image_a?.[0] || !msg?.image_b?.[0]) return;

            const S = this._S;
            S.labelA = msg.label_a?.[0] ?? "Before";
            S.labelB = msg.label_b?.[0] ?? "After";
            S.diffCvs = null;

            const mkURL = (info) =>
                api.apiURL(
                    `/view?filename=${encodeURIComponent(info.filename)}` +
                    `&type=${info.type}` +
                    `&subfolder=${encodeURIComponent(info.subfolder || "")}`
                );

            let loaded = 0;
            const node = this;
            const done = () => {
                if (++loaded < 2) return;
                node._render();
                const aspect = S.imgA.height / S.imgA.width;
                const w = Math.max(node.size[0], 320);
                node.setSize([w, Math.round((w - 20) * aspect) + 140]);
                app.graph.setDirtyCanvas(true);
            };

            S.imgA = new Image();
            S.imgB = new Image();
            S.imgA.onload = done;
            S.imgB.onload = done;
            S.imgA.src = mkURL(msg.image_a[0]);
            S.imgB.src = mkURL(msg.image_b[0]);
        };

        /* ── Build difference heatmap (cached) ─────────── */
        nodeType.prototype._buildDiff = function () {
            const S = this._S;
            if (!S.imgA || !S.imgB) return;
            const w = S.imgA.width,
                h = S.imgA.height;

            const pixels = (img) => {
                const c = document.createElement("canvas");
                c.width = w;
                c.height = h;
                c.getContext("2d").drawImage(img, 0, 0, w, h);
                return c.getContext("2d").getImageData(0, 0, w, h).data;
            };
            const pA = pixels(S.imgA),
                pB = pixels(S.imgB);

            const dc = document.createElement("canvas");
            dc.width = w;
            dc.height = h;
            const dctx = dc.getContext("2d");
            const id = dctx.createImageData(w, h);
            const p = id.data;

            for (let i = 0; i < pA.length; i += 4) {
                const d =
                    (Math.abs(pA[i] - pB[i]) +
                        Math.abs(pA[i + 1] - pB[i + 1]) +
                        Math.abs(pA[i + 2] - pB[i + 2])) /
                    3;
                // Amplify: 20% brightness diff → maps to full red
                const t = Math.min(1, d / 51);
                let r, g, b;
                if (t < 0.33) {
                    const s = t / 0.33;
                    r = 0;
                    g = Math.round(s * 255);
                    b = Math.round((1 - s) * 255);
                } else if (t < 0.66) {
                    const s = (t - 0.33) / 0.33;
                    r = Math.round(s * 255);
                    g = 255;
                    b = 0;
                } else {
                    const s = (t - 0.66) / 0.34;
                    r = 255;
                    g = Math.round((1 - s) * 255);
                    b = 0;
                }
                p[i] = r;
                p[i + 1] = g;
                p[i + 2] = b;
                p[i + 3] = 255;
            }
            dctx.putImageData(id, 0, 0);
            S.diffCvs = dc;
        };

        /* ── Main render ───────────────────────────────── */
        nodeType.prototype._render = function () {
            const S = this._S;
            if (!S.imgA || !S.imgB) return;

            const cvs = S.cvs,
                ctx = S.ctx;
            const cw = S.el.clientWidth || 400;
            const asp = S.imgA.height / S.imgA.width;
            const ch = Math.max(100, Math.round(cw * asp));

            if (cvs.width !== cw || cvs.height !== ch) {
                cvs.width = cw;
                cvs.height = ch;
            }
            ctx.clearRect(0, 0, cw, ch);

            if (S.mode === 0) {
                /* ── Slider ── */
                const dx = Math.round(cw * S.divPos);

                ctx.save();
                ctx.beginPath();
                ctx.rect(0, 0, dx, ch);
                ctx.clip();
                ctx.drawImage(S.imgA, 0, 0, cw, ch);
                ctx.restore();

                ctx.save();
                ctx.beginPath();
                ctx.rect(dx, 0, cw - dx, ch);
                ctx.clip();
                ctx.drawImage(S.imgB, 0, 0, cw, ch);
                ctx.restore();

                // Divider line + shadow
                ctx.save();
                ctx.shadowColor = "rgba(0,0,0,0.6)";
                ctx.shadowBlur = 6;
                ctx.strokeStyle = "#fff";
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(dx, 0);
                ctx.lineTo(dx, ch);
                ctx.stroke();
                ctx.shadowBlur = 0;

                // Handle circle with grip dots
                const hy = ch / 2;
                ctx.beginPath();
                ctx.arc(dx, hy, HANDLE_R, 0, Math.PI * 2);
                ctx.fillStyle = "rgba(255,255,255,0.92)";
                ctx.fill();
                ctx.strokeStyle = "#666";
                ctx.lineWidth = 1.5;
                ctx.stroke();
                // Grip dots (⋮ pattern)
                ctx.fillStyle = "#555";
                for (let dy = -6; dy <= 6; dy += 6) {
                    ctx.beginPath();
                    ctx.arc(dx - 3, hy + dy, 1.5, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.beginPath();
                    ctx.arc(dx + 3, hy + dy, 1.5, 0, Math.PI * 2);
                    ctx.fill();
                }
                ctx.restore();
            } else if (S.mode === 1) {
                /* ── Overlay ── */
                ctx.globalAlpha = 1;
                ctx.drawImage(S.imgA, 0, 0, cw, ch);
                ctx.globalAlpha = S.alpha;
                ctx.drawImage(S.imgB, 0, 0, cw, ch);
                ctx.globalAlpha = 1;

                // Scrubber bar (20px height, bottom)
                const bh = 6, by = ch - 20;
                const barX = 16, barW = cw - 32;
                ctx.fillStyle = "rgba(0,0,0,0.5)";
                ctx.beginPath();
                ctx.roundRect(barX, by, barW, bh, 3);
                ctx.fill();
                ctx.fillStyle = "#5bf";
                ctx.beginPath();
                ctx.roundRect(barX, by, barW * S.alpha, bh, 3);
                ctx.fill();
                // Scrubber handle
                const hx = barX + barW * S.alpha;
                ctx.beginPath();
                ctx.arc(hx, by + bh / 2, 7, 0, Math.PI * 2);
                ctx.fillStyle = "#fff";
                ctx.fill();
                ctx.strokeStyle = "#5bf";
                ctx.lineWidth = 2;
                ctx.stroke();
            } else {
                /* ── Diff ── */
                if (!S.diffCvs) this._buildDiff();
                ctx.drawImage(S.diffCvs || S.imgA, 0, 0, cw, ch);
            }

            // Labels (bottom-left)
            _drawLabels(ctx, S, cw, ch);
            // Mode buttons (top-right)
            _drawBtns(ctx, S, cw);
        };
    },
});

/* ── Helpers (module-private) ──────────────────────────── */

function _pill(ctx, x, y, text) {
    const tw = ctx.measureText(text).width;
    const pw = tw + 12,
        ph = 18;
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.beginPath();
    ctx.roundRect(x, y, pw, ph, 4);
    ctx.fill();
    ctx.fillStyle = "#eee";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(text, x + 6, y + ph / 2);
}

function _drawLabels(ctx, S, cw, ch) {
    ctx.save();
    ctx.globalAlpha = S.labelOpacity;
    ctx.font = "bold 11px sans-serif";
    if (S.mode === 0) {
        _pill(ctx, 5, ch - 24, S.labelA);
        const tw = ctx.measureText(S.labelB).width;
        _pill(ctx, cw - tw - 17, ch - 24, S.labelB);
    } else if (S.mode === 1) {
        _pill(
            ctx,
            5,
            ch - 24,
            `${S.labelA}  ${Math.round(S.alpha * 100)}%  ${S.labelB}`
        );
    } else {
        _pill(ctx, 5, ch - 24, `Diff: ${S.labelA} vs ${S.labelB}`);
    }
    ctx.restore();
}

function _drawBtns(ctx, S, cw) {
    const x0 =
        cw -
        (BTN_W * MODES.length + BTN_GAP * (MODES.length - 1)) -
        BTN_PAD;
    ctx.save();
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (let i = 0; i < MODES.length; i++) {
        const x = x0 + i * (BTN_W + BTN_GAP);
        const active = i === S.mode;
        ctx.fillStyle = active
            ? "rgba(60,140,240,0.88)"
            : "rgba(40,40,40,0.72)";
        ctx.beginPath();
        ctx.roundRect(x, BTN_PAD, BTN_W, BTN_H, 4);
        ctx.fill();
        ctx.fillStyle = active ? "#fff" : "#bbb";
        ctx.fillText(MODES[i], x + BTN_W / 2, BTN_PAD + BTN_H / 2);
    }
    ctx.restore();
}

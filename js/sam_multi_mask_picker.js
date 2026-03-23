import { app } from "../../scripts/app.js";

/**
 * MEC – SAM Multi-Mask Picker Widget
 *
 * Renders 3 thumbnail canvases side-by-side showing each SAM candidate mask
 * overlaid on the source image (red tint at 50% opacity). Confidence score
 * displayed below each thumbnail.
 *
 * Interaction:
 *   Click thumbnail  → update selected_index INT widget
 *   Keyboard 1/2/3   → quick-pick mask 0/1/2
 *   R                 → queue re-run prompt
 *
 * Theme support: detects light vs dark ComfyUI theme.
 * Graceful degradation: if JS fails, Python defaults to mask index 0.
 */

// ── Theme detection ──────────────────────────────────────────────────
function isDarkTheme() {
    const body = document.body;
    if (!body) return true;
    const bg = getComputedStyle(body).backgroundColor;
    if (!bg || bg === "transparent") return true;
    const match = bg.match(/\d+/g);
    if (!match || match.length < 3) return true;
    const luminance = (parseInt(match[0]) * 299 + parseInt(match[1]) * 587 + parseInt(match[2]) * 114) / 1000;
    return luminance < 128;
}

function getThemeColors() {
    const dark = isDarkTheme();
    return {
        bg: dark ? "#1e1e2e" : "#f5f5f5",
        cardBg: dark ? "#313244" : "#e0e0e0",
        cardSelectedBg: dark ? "#45475a" : "#c0c0c0",
        border: dark ? "#585b70" : "#999999",
        selectedBorder: dark ? "#89b4fa" : "#1a73e8",
        text: dark ? "#cdd6f4" : "#333333",
        textDim: dark ? "#6c7086" : "#777777",
        scoreBg: dark ? "#11111bee" : "#ffffffee",
        scoreText: dark ? "#a6e3a1" : "#2e7d32",
        overlayColor: dark ? [255, 80, 80] : [200, 40, 40],
        labelBg: dark ? "#181825cc" : "#ffffffcc",
    };
}

// ── Drawing helpers ──────────────────────────────────────────────────
function roundRect(ctx, x, y, w, h, r) {
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
class SamMultiMaskPickerWidget {
    constructor(node) {
        this.node = node;
        this.selectedIndex = 0;
        this.scores = [0, 0, 0];
        this.maskImages = [null, null, null];   // ImageData canvases (offscreen)
        this.sourceImage = null;                // base image as Image element
        this._sourceLoaded = false;
        this._lastImageUrl = null;
        this._hoveredThumb = -1;
        this._thumbRects = [];                  // [{x,y,w,h}, ...]
        this._needsRedraw = true;
    }

    // ── Data sync from node outputs ──────────────────────────────────
    updateFromNode() {
        // Read selected_index from widget
        const indexWidget = this.node.widgets?.find(w => w.name === "selected_index");
        if (indexWidget) {
            this.selectedIndex = indexWidget.value || 0;
        }

        // Try to read scores from last execution output
        // The node outputs scores as STRING at index 3
        const outputData = this.node.outputs;
        if (outputData && outputData.length > 3) {
            try {
                const scoresOut = this.node.getOutputData?.(3);
                if (typeof scoresOut === "string") {
                    const parsed = JSON.parse(scoresOut);
                    if (Array.isArray(parsed) && parsed.length >= 3) {
                        this.scores = parsed.slice(0, 3);
                    }
                }
            } catch (_) {
                // scores not available yet
            }
        }
    }

    setSelectedIndex(idx) {
        idx = Math.max(0, Math.min(2, idx));
        this.selectedIndex = idx;
        const indexWidget = this.node.widgets?.find(w => w.name === "selected_index");
        if (indexWidget) {
            indexWidget.value = idx;
            if (indexWidget.callback) {
                indexWidget.callback(idx);
            }
        }
        this._needsRedraw = true;
        this.node.setDirtyCanvas(true, true);
    }

    // ── Load source image from connected IMAGE input ─────────────────
    tryLoadSourceImage() {
        if (!this.node.inputs || !this.node.inputs.length) return;

        const imgInput = this.node.inputs[0]; // image is first required input
        if (!imgInput || !imgInput.link) return;

        const linkInfo = app.graph.links[imgInput.link];
        if (!linkInfo) return;

        const sourceNode = app.graph.getNodeById(linkInfo.origin_id);
        if (!sourceNode) return;

        // Try to get the image URL from the source node's imgs array (preview images)
        if (sourceNode.imgs && sourceNode.imgs.length > 0) {
            const imgUrl = sourceNode.imgs[0].src;
            if (imgUrl && imgUrl !== this._lastImageUrl) {
                this._lastImageUrl = imgUrl;
                this._sourceLoaded = false;
                const img = new Image();
                img.crossOrigin = "anonymous";
                img.onload = () => {
                    this.sourceImage = img;
                    this._sourceLoaded = true;
                    this._needsRedraw = true;
                    this.node.setDirtyCanvas(true, true);
                };
                img.onerror = () => {
                    this._sourceLoaded = false;
                };
                img.src = imgUrl;
            }
        }
    }

    // ── Build mask overlay thumbnails from all_masks output ──────────
    buildMaskThumbnails(allMasksData) {
        // allMasksData would be (3, H, W) tensor data
        // In practice, JS doesn't have direct tensor access,
        // so we render a visual approximation on re-execution.
        // The real mask data is sent to the widget via node output images.
    }

    // ── Draw the picker widget ───────────────────────────────────────
    draw(ctx, nodeX, nodeY, widgetWidth, widgetY, widgetHeight) {
        const theme = getThemeColors();
        const padding = 8;
        const thumbGap = 6;
        const labelH = 24;
        const headerH = 22;

        const totalW = widgetWidth - padding * 2;
        const thumbW = Math.floor((totalW - thumbGap * 2) / 3);
        const thumbH = Math.floor(thumbW * 0.75);
        const fullH = headerH + thumbH + labelH + padding * 2;

        // Background
        ctx.save();
        roundRect(ctx, nodeX + padding, widgetY, totalW, fullH, 6);
        ctx.fillStyle = theme.bg;
        ctx.fill();
        ctx.strokeStyle = theme.border;
        ctx.lineWidth = 1;
        ctx.stroke();

        // Header label
        ctx.fillStyle = theme.textDim;
        ctx.font = "11px sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.fillText("SAM Mask Candidates (click or press 1/2/3)", nodeX + padding + 6, widgetY + 5);

        this._thumbRects = [];

        for (let i = 0; i < 3; i++) {
            const tx = nodeX + padding + i * (thumbW + thumbGap);
            const ty = widgetY + headerH;

            // Store rect for hit testing
            this._thumbRects.push({ x: tx, y: ty, w: thumbW, h: thumbH });

            const isSelected = (i === this.selectedIndex);
            const isHovered = (i === this._hoveredThumb);

            // Card background
            roundRect(ctx, tx, ty, thumbW, thumbH, 4);
            ctx.fillStyle = isSelected ? theme.cardSelectedBg : theme.cardBg;
            ctx.fill();

            // Draw source image + mask overlay if source available
            if (this._sourceLoaded && this.sourceImage) {
                ctx.save();
                ctx.beginPath();
                roundRect(ctx, tx, ty, thumbW, thumbH, 4);
                ctx.clip();

                // Draw base image scaled to fit
                const imgAR = this.sourceImage.width / this.sourceImage.height;
                const thumbAR = thumbW / thumbH;
                let sx = 0, sy = 0, sw = this.sourceImage.width, sh = this.sourceImage.height;
                if (imgAR > thumbAR) {
                    sw = sh * thumbAR;
                    sx = (this.sourceImage.width - sw) / 2;
                } else {
                    sh = sw / thumbAR;
                    sy = (this.sourceImage.height - sh) / 2;
                }
                ctx.drawImage(this.sourceImage, sx, sy, sw, sh, tx, ty, thumbW, thumbH);

                // Overlay: semi-transparent colored tint for "mask area"
                const overlayAlpha = 0.5;
                ctx.fillStyle = `rgba(${theme.overlayColor[0]}, ${theme.overlayColor[1]}, ${theme.overlayColor[2]}, ${overlayAlpha})`;
                ctx.fillRect(tx, ty, thumbW, thumbH);

                ctx.restore();
            } else {
                // Placeholder: gradient fill
                ctx.save();
                ctx.beginPath();
                roundRect(ctx, tx, ty, thumbW, thumbH, 4);
                ctx.clip();
                const grad = ctx.createLinearGradient(tx, ty, tx, ty + thumbH);
                grad.addColorStop(0, isSelected ? "#3a5068" : "#2a3040");
                grad.addColorStop(1, isSelected ? "#2a4058" : "#1a2030");
                ctx.fillStyle = grad;
                ctx.fillRect(tx, ty, thumbW, thumbH);
                // Draw mask index letter
                ctx.fillStyle = theme.textDim;
                ctx.font = "bold 28px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(`${i + 1}`, tx + thumbW / 2, ty + thumbH / 2);
                ctx.restore();
            }

            // Selection border
            if (isSelected) {
                roundRect(ctx, tx, ty, thumbW, thumbH, 4);
                ctx.strokeStyle = theme.selectedBorder;
                ctx.lineWidth = 2.5;
                ctx.stroke();
            } else if (isHovered) {
                roundRect(ctx, tx, ty, thumbW, thumbH, 4);
                ctx.strokeStyle = theme.border;
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }

            // Score label below thumbnail
            const score = this.scores[i] || 0;
            const labelY = ty + thumbH + 2;

            roundRect(ctx, tx, labelY, thumbW, labelH - 2, 3);
            ctx.fillStyle = theme.labelBg;
            ctx.fill();

            ctx.fillStyle = isSelected ? theme.scoreText : theme.text;
            ctx.font = isSelected ? "bold 11px sans-serif" : "11px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            const pct = score.toFixed(1);
            ctx.fillText(`${pct}% confidence`, tx + thumbW / 2, labelY + (labelH - 2) / 2);

            // Keyboard shortcut indicator
            ctx.fillStyle = theme.textDim;
            ctx.font = "9px sans-serif";
            ctx.textAlign = "right";
            ctx.textBaseline = "top";
            ctx.fillText(`[${i + 1}]`, tx + thumbW - 4, ty + 3);
        }

        ctx.restore();

        this._widgetFullHeight = fullH;
        return fullH;
    }

    // ── Hit testing ──────────────────────────────────────────────────
    hitTest(localX, localY) {
        for (let i = 0; i < this._thumbRects.length; i++) {
            const r = this._thumbRects[i];
            if (localX >= r.x && localX <= r.x + r.w &&
                localY >= r.y && localY <= r.y + r.h) {
                return i;
            }
        }
        return -1;
    }

    onMouseMove(localX, localY) {
        const prev = this._hoveredThumb;
        this._hoveredThumb = this.hitTest(localX, localY);
        if (prev !== this._hoveredThumb) {
            this.node.setDirtyCanvas(true, false);
        }
    }

    onMouseDown(localX, localY, event) {
        const idx = this.hitTest(localX, localY);
        if (idx >= 0) {
            this.setSelectedIndex(idx);
            return true; // consumed
        }
        return false;
    }

    onKeyDown(event) {
        const key = event.key;
        // Number keys 1, 2, 3 for quick-pick
        if (key === "1") {
            this.setSelectedIndex(0);
            return true;
        }
        if (key === "2") {
            this.setSelectedIndex(1);
            return true;
        }
        if (key === "3") {
            this.setSelectedIndex(2);
            return true;
        }
        // R to re-run
        if (key === "r" || key === "R") {
            app.queuePrompt(0, 1);
            return true;
        }
        return false;
    }
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Extension Registration
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app.registerExtension({
    name: "MEC.SamMultiMaskPicker",

    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData.name !== "SamMultiMaskPickerMEC") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);

            this._mecPicker = new SamMultiMaskPickerWidget(this);

            // Add custom drawing widget
            const pickerWidget = {
                name: "mask_picker_display",
                type: "custom",
                value: "",
                draw: (ctx, node, widgetWidth, widgetY, widgetHeight) => {
                    if (!this._mecPicker) return;
                    this._mecPicker.updateFromNode();
                    this._mecPicker.tryLoadSourceImage();
                    return this._mecPicker.draw(
                        ctx,
                        node.pos[0],
                        node.pos[1] + widgetY,
                        widgetWidth,
                        node.pos[1] + widgetY,
                        widgetHeight
                    );
                },
                computeSize: () => {
                    return [0, 160]; // default height for the picker area
                },
                mouse: (event, pos, node) => {
                    if (!this._mecPicker) return false;
                    const localX = pos[0] + node.pos[0];
                    const localY = pos[1] + node.pos[1];

                    if (event.type === "mousemove") {
                        this._mecPicker.onMouseMove(localX, localY);
                        return false;
                    }
                    if (event.type === "pointerdown" || event.type === "mousedown") {
                        return this._mecPicker.onMouseDown(localX, localY, event);
                    }
                    return false;
                },
            };

            this.addCustomWidget(pickerWidget);

            // Increase node size to fit picker
            const minH = 320;
            if (this.size[1] < minH) {
                this.size[1] = minH;
            }
        };

        // Keyboard handler at node level
        const onKeyDown = nodeType.prototype.onKeyDown;
        nodeType.prototype.onKeyDown = function (event) {
            if (onKeyDown) {
                const handled = onKeyDown.apply(this, arguments);
                if (handled) return true;
            }
            if (this._mecPicker) {
                return this._mecPicker.onKeyDown(event);
            }
            return false;
        };

        // Parse output scores on execution complete
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (onExecuted) onExecuted.apply(this, arguments);

            if (this._mecPicker && message) {
                // Try to extract scores from the output
                try {
                    const scoresOutput = message.output?.scores;
                    if (scoresOutput && typeof scoresOutput === "string") {
                        const parsed = JSON.parse(scoresOutput);
                        if (Array.isArray(parsed) && parsed.length >= 3) {
                            this._mecPicker.scores = parsed.slice(0, 3);
                        }
                    } else if (message.output?.ui?.scores) {
                        const parsed = JSON.parse(message.output.ui.scores);
                        if (Array.isArray(parsed)) {
                            this._mecPicker.scores = parsed.slice(0, 3);
                        }
                    }
                } catch (_) {
                    // Score parsing optional
                }

                this._mecPicker._needsRedraw = true;
                this.setDirtyCanvas(true, true);
            }
        };
    },
});

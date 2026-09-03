/*
 * curve_editor.js - GeneralCurveEditor frontend extension.
 *
 * Graph is a DOM widget (node.addDOMWidget), not a litegraph canvas
 * widget - see MajoorWaldi/ComfyUI-Majoor-H3-GuideMaster for the pattern.
 * Canvas pixel buffer is kept matched to its actual rendered size via
 * ResizeObserver (no CSS-stretch distortion). node.size is self-healed
 * on an interval since Classic mode has repeatedly reset it otherwise.
 *
 * Values are dB offsets (0 = no change), mirrors core.py.
 */

import { app } from "../../scripts/app.js";

const NODE_NAME = "GeneralCurveEditor";
const MIN_NODE_WIDTH = 340;
const MIN_NODE_HEIGHT = 440;
const GRAPH_MIN_CSS_HEIGHT = 220;
const SEEK_BAR_HEIGHT = 26;
const WIDGET_FIXED_HEIGHT = GRAPH_MIN_CSS_HEIGHT + SEEK_BAR_HEIGHT;

const KEYFRAME_RADIUS = 6;
const HIT_RADIUS = 12;
const VALUE_MIN = -24.0;
const VALUE_MAX = 24.0;
const NEUTRAL_VALUE = 0.0;
const INTERP_MODES = ["linear", "smooth", "step"];
const PAD_X = 14;
const PAD_TOP = 18;
const PAD_BOTTOM = 8;
const WAVEFORM_DISPLAY_CEILING = 0.85;

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function defaultCurve() {
    return { keyframes: [{ t: 0.0, value: 0.0, interp: "linear" }, { t: 1.0, value: 0.0, interp: "linear" }] };
}

function sortKeyframesKeepingSelection(curveData, selectedRef) {
    const kfs = curveData.keyframes;
    const sel = selectedRef.index >= 0 && selectedRef.index < kfs.length ? kfs[selectedRef.index] : null;
    kfs.sort((a, b) => a.t - b.t);
    if (sel !== null) selectedRef.index = kfs.indexOf(sel);
}

function hermiteEval(t0, v0, t1, v1, vPrev, vNext, t) {
    if (t1 === t0) return v0;
    const u = (t - t0) / (t1 - t0);
    const m0 = (v1 - (vPrev !== null ? vPrev : v0)) / 2.0;
    const m1 = ((vNext !== null ? vNext : v1) - v0) / 2.0;
    const u2 = u * u, u3 = u2 * u;
    return (2 * u3 - 3 * u2 + 1) * v0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * v1 + (u3 - u2) * m1;
}

// Mirrors core.py's evaluate_curve, including the final-keyframe-step fix.
function evalCurveAt(curve, t) {
    const kfs = curve.keyframes;
    if (kfs.length === 0) return 0.0;
    if (kfs.length === 1) return kfs[0].value;
    t = clamp(t, 0.0, 1.0);

    const lastKf = kfs[kfs.length - 1];
    if (t >= lastKf.t) return lastKf.value;

    let si = 0;
    for (let i = 0; i < kfs.length - 1; i++) {
        if (kfs[i].t <= t) si = i; else break;
    }
    const k0 = kfs[si], k1 = kfs[si + 1];
    if (k1.t === k0.t) return k1.value;

    const mode = k1.interp || "linear";
    if (mode === "step") return k0.value;
    if (mode === "smooth") {
        const vPrev = si - 1 >= 0 ? kfs[si - 1].value : null;
        const vNext = si + 2 < kfs.length ? kfs[si + 2].value : null;
        return hermiteEval(k0.t, k0.value, k1.t, k1.value, vPrev, vNext, t);
    }
    const u = (t - k0.t) / (k1.t - k0.t);
    return k0.value + u * (k1.value - k0.value);
}

// Nodes 2.0 reads widget.options.hidden through a reactive store proxy
// (widget._state), not widget.hidden - see NODES_2_COMPAT.md section 1.
function setOptionsHidden(widget, hidden) {
    const state = widget._state;
    if (state?.options) state.options.hidden = hidden;
    else if (widget.options) widget.options.hidden = hidden;
}

function setWidgetHidden(widget, hidden) {
    if (!widget) return;
    setOptionsHidden(widget, !!hidden);
    widget.hidden = !!hidden;
    if (hidden) widget.computeSize = () => [0, -4];
}

function findWidget(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : undefined;
}

function syncBackingValue(node) {
    const backing = findWidget(node, "curve_editor");
    if (backing) backing.value = JSON.stringify(node._curveData);
}

function syncInterpDropdown(node) {
    const kfs = node._curveData.keyframes;
    const idx = node._selected.index;
    if (node._interpWidget && idx >= 0 && idx < kfs.length) {
        node._interpWidget.value = kfs[idx].interp || "linear";
    }
}

function normalizePeaksForDisplay(peaks) {
    let maxPeak = 0;
    for (const p of peaks) if (p > maxPeak) maxPeak = p;
    if (maxPeak <= 0) return peaks;
    const scale = WAVEFORM_DISPLAY_CEILING / maxPeak;
    return peaks.map((p) => clamp(p * scale, 0, 1));
}

function getPlotRect(w, h) {
    const plotX0 = PAD_X, plotX1 = w - PAD_X;
    const plotY0 = PAD_TOP, plotH = Math.max(1, h - PAD_TOP - PAD_BOTTOM);
    return { plotX0, plotX1, plotW: Math.max(1, plotX1 - plotX0), plotY0, plotH };
}
function tToPixelX(t, r) { return r.plotX0 + t * r.plotW; }
function pixelXToT(px, r) { return clamp((px - r.plotX0) / r.plotW, 0.0, 1.0); }
function valueToPixelY(v, r) {
    const f = clamp((v - VALUE_MIN) / (VALUE_MAX - VALUE_MIN), 0, 1);
    return r.plotY0 + r.plotH * (1 - f);
}
function pixelYToValue(py, r) {
    const f = clamp((py - r.plotY0) / r.plotH, 0, 1);
    return VALUE_MAX - f * (VALUE_MAX - VALUE_MIN);
}

function redrawGraph(node, canvas) {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w = canvas.width, h = canvas.height;
    const rect = getPlotRect(w, h);

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, w, h);

    const peaks = node._waveformPeaks;
    if (peaks && peaks.length > 0) {
        ctx.fillStyle = "rgba(255, 159, 28, 0.35)";
        const n = peaks.length, barW = rect.plotW / n, midY = rect.plotY0 + rect.plotH / 2;
        for (let i = 0; i < n; i++) {
            const barH = clamp(peaks[i], 0, 1) * (rect.plotH / 2);
            ctx.fillRect(rect.plotX0 + i * barW, midY - barH, Math.max(1, barW - 1), barH * 2);
        }
    }

    const neutralY = valueToPixelY(NEUTRAL_VALUE, rect);
    ctx.strokeStyle = "#555555";
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(rect.plotX0, neutralY);
    ctx.lineTo(rect.plotX1, neutralY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = "#888888";
    ctx.font = "11px sans-serif";
    for (let i = 0; i <= 4; i++) {
        const frac = i / 4;
        const label = node._durationSeconds
            ? (frac * node._durationSeconds).toFixed(2) + "s"
            : Math.round(frac * 100) + "%";
        ctx.fillText(label, clamp(rect.plotX0 + frac * rect.plotW, rect.plotX0, rect.plotX1 - 32), 14);
    }

    const kfs = node._curveData.keyframes;
    ctx.strokeStyle = "#4fd8ff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    const steps = Math.max(2, Math.floor(rect.plotW / 2));
    for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        const px = tToPixelX(t, rect), py = valueToPixelY(evalCurveAt(node._curveData, t), rect);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();

    for (let i = 0; i < kfs.length; i++) {
        const px = tToPixelX(kfs[i].t, rect), py = valueToPixelY(kfs[i].value, rect);
        ctx.fillStyle = node._selected.index === i ? "#ffdd55" : "#ffffff";
        ctx.beginPath();
        ctx.arc(px, py, KEYFRAME_RADIUS, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#333333";
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    if (node._dragging && node._selected.index >= 0 && node._selected.index < kfs.length) {
        const sel = kfs[node._selected.index];
        const px = tToPixelX(sel.t, rect), py = valueToPixelY(sel.value, rect);
        const timeLabel = node._durationSeconds
            ? (sel.t * node._durationSeconds).toFixed(2) + "s"
            : Math.round(sel.t * 100) + "%";
        const valueLabel = (sel.value >= 0 ? "+" : "") + sel.value.toFixed(1) + "dB";
        drawFloatingLabel(ctx, px, py, `${timeLabel}, ${valueLabel}`, w);
    }
}

// Small floating tooltip-style label following the point being dragged -
// purely a redraw-time visual, no effect on drag math or hit-testing.
function drawFloatingLabel(ctx, px, py, text, canvasW) {
    ctx.font = "12px sans-serif";
    const textW = ctx.measureText(text).width;
    const boxW = textW + 12, boxH = 20;
    let boxX = px - boxW / 2;
    boxX = clamp(boxX, 2, canvasW - boxW - 2);
    const boxY = py - KEYFRAME_RADIUS - boxH - 6;

    ctx.fillStyle = "rgba(20, 20, 20, 0.9)";
    ctx.strokeStyle = "#4fd8ff";
    ctx.lineWidth = 1;
    if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(boxX, boxY, boxW, boxH, 4);
        ctx.fill();
        ctx.stroke();
    } else {
        ctx.fillRect(boxX, boxY, boxW, boxH);
        ctx.strokeRect(boxX, boxY, boxW, boxH);
    }
    ctx.fillStyle = "#ffffff";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, boxX + boxW / 2, boxY + boxH / 2 + 1);
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
}

function toInternalCoords(evt, canvas) {
    const r = canvas.getBoundingClientRect();
    const sx = canvas.width / Math.max(1, r.width), sy = canvas.height / Math.max(1, r.height);
    return { px: (evt.clientX - r.left) * sx, py: (evt.clientY - r.top) * sy };
}

function attachPointerHandlers(node, canvas, onChange) {
    node._dragging = false;

    canvas.addEventListener("contextmenu", (evt) => evt.preventDefault());

    canvas.addEventListener("pointerdown", (evt) => {
        evt.preventDefault();
        canvas.setPointerCapture?.(evt.pointerId);
        const { px, py } = toInternalCoords(evt, canvas);
        const rect = getPlotRect(canvas.width, canvas.height);
        const tAtX = pixelXToT(px, rect), valueAtY = pixelYToValue(py, rect);
        const kfs = node._curveData.keyframes;

        let hitIndex = -1, hitDist = Infinity;
        for (let i = 0; i < kfs.length; i++) {
            const kx = tToPixelX(kfs[i].t, rect), ky = valueToPixelY(kfs[i].value, rect);
            const dist = Math.hypot(px - kx, py - ky);
            if (dist <= HIT_RADIUS && dist < hitDist) { hitIndex = i; hitDist = dist; }
        }

        if ((evt.button === 2 || evt.shiftKey) && hitIndex >= 0) {
            if (hitIndex !== 0 && hitIndex !== kfs.length - 1) {
                kfs.splice(hitIndex, 1);
                node._selected.index = clamp(hitIndex - 1, 0, kfs.length - 1);
                onChange();
            }
            return;
        }

        if (hitIndex >= 0) {
            node._selected.index = hitIndex;
            node._dragging = true;
        } else {
            const newKf = { t: tAtX, value: valueAtY, interp: "linear" };
            kfs.push(newKf);
            kfs.sort((a, b) => a.t - b.t);
            node._selected.index = kfs.indexOf(newKf);
            node._dragging = true;
        }
        onChange();
    });

    canvas.addEventListener("pointermove", (evt) => {
        if (!node._dragging || node._selected.index < 0) return;
        const { px, py } = toInternalCoords(evt, canvas);
        const rect = getPlotRect(canvas.width, canvas.height);
        const kfs = node._curveData.keyframes;
        const idx = node._selected.index, kf = kfs[idx];
        const isEdge = idx === 0 || idx === kfs.length - 1;
        if (!isEdge) kf.t = pixelXToT(px, rect);
        kf.value = pixelYToValue(py, rect);
        sortKeyframesKeepingSelection(node._curveData, node._selected);
        onChange();
    });

    const endDrag = (evt) => { node._dragging = false; canvas.releasePointerCapture?.(evt.pointerId); };
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);
}

function buildCurveGraphDOM() {
    const container = document.createElement("div");
    // Leave a small transparent hit-safe border around the DOM content in
    // Classic mode so LiteGraph's native resize handles remain reachable.
    // The actual graph still tracks the full widget size minus this border.
    container.style.cssText = `width:100%; min-height:${GRAPH_MIN_CSS_HEIGHT}px; height:100%; box-sizing:border-box; overflow:hidden; pointer-events:none;`;
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "width:calc(100% - 12px); height:calc(100% - 12px); margin:6px; display:block; cursor:crosshair; pointer-events:auto;";
    container.appendChild(canvas);
    return { container, canvas };
}

function attachCanvasResizeObserver(node, canvas, redraw) {
    const ro = new ResizeObserver((entries) => {
        for (const entry of entries) {
            const w = Math.round(entry.contentRect.width), h = Math.round(entry.contentRect.height);
            if (w > 0 && h > 0 && (canvas.width !== w || canvas.height !== h)) {
                canvas.width = w;
                canvas.height = h;
                redraw();
            }
        }
    });
    ro.observe(canvas);
    node._curveGraphResizeObserver = ro;
}

// --- Seek bar / playback ---

function buildSeekBar() {
    const bar = document.createElement("div");
    bar.style.cssText = `display:flex; align-items:center; gap:8px; width:calc(100% - 12px); margin:0 6px; height:${SEEK_BAR_HEIGHT}px; padding:0 4px; background:#222; box-sizing:border-box; pointer-events:auto;`;

    const playBtn = document.createElement("button");
    playBtn.textContent = "\u25B6";
    playBtn.disabled = true;
    playBtn.style.cssText = "width:24px; height:18px; font-size:10px; cursor:pointer; line-height:1;";

    const track = document.createElement("div");
    track.style.cssText = "flex:1; height:6px; background:#333; position:relative; border-radius:3px;";

    const fill = document.createElement("div");
    fill.style.cssText = "position:absolute; left:0; top:0; height:100%; width:0%; background:#4fd8ff; border-radius:3px; pointer-events:none;";
    track.appendChild(fill);

    const head = document.createElement("div");
    head.style.cssText = "position:absolute; top:50%; left:0%; width:12px; height:12px; margin-left:-6px; margin-top:-6px; border-radius:50%; background:#fff; border:2px solid #333; cursor:grab;";
    track.appendChild(head);

    bar.appendChild(playBtn);
    bar.appendChild(track);
    return { bar, playBtn, track, fill, head };
}

function attachSeekBar(node, { playBtn, track, fill, head }) {
    let audioEl = null;
    let dragging = false;
    let wasPlaying = false;

    function setProgress(frac) {
        frac = clamp(frac, 0, 1);
        fill.style.width = frac * 100 + "%";
        head.style.left = frac * 100 + "%";
    }

    node._loadSeekAudio = (url) => {
        audioEl?.pause();
        audioEl = new Audio(url);
        audioEl.addEventListener("timeupdate", () => {
            if (!dragging && audioEl.duration) setProgress(audioEl.currentTime / audioEl.duration);
        });
        audioEl.addEventListener("ended", () => { playBtn.textContent = "\u25B6"; });
        playBtn.disabled = false;
    };

    playBtn.addEventListener("click", () => {
        if (!audioEl) return;
        if (audioEl.paused) { audioEl.play(); playBtn.textContent = "\u23F8"; }
        else { audioEl.pause(); playBtn.textContent = "\u25B6"; }
    });

    function scrubTo(evt) {
        if (!audioEl || !audioEl.duration) return;
        const r = track.getBoundingClientRect();
        const frac = clamp((evt.clientX - r.left) / r.width, 0, 1);
        audioEl.currentTime = frac * audioEl.duration;
        setProgress(frac);
    }

    head.addEventListener("pointerdown", (evt) => {
        evt.preventDefault();
        evt.stopPropagation();
        if (!audioEl) return;
        dragging = true;
        wasPlaying = !audioEl.paused;
        audioEl.play();
        head.setPointerCapture?.(evt.pointerId);
        scrubTo(evt);
    });
    head.addEventListener("pointermove", (evt) => { if (dragging) scrubTo(evt); });
    const endScrub = (evt) => {
        if (!dragging) return;
        dragging = false;
        if (!wasPlaying) { audioEl?.pause(); playBtn.textContent = "\u25B6"; }
        else playBtn.textContent = "\u23F8";
        head.releasePointerCapture?.(evt.pointerId);
    };
    head.addEventListener("pointerup", endScrub);
    head.addEventListener("pointercancel", endScrub);
}

app.registerExtension({
    name: "audio_amplifier.CurveEditorWidget",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;

            if (typeof node.addDOMWidget !== "function") {
                console.warn("[AudioAmplifier] node.addDOMWidget unavailable in this frontend.");
                return result;
            }
            const backing = findWidget(node, "curve_editor");
            if (!backing) {
                console.warn("[AudioAmplifier] curve_editor backing widget not found.");
                return result;
            }

            let initialCurve;
            try { initialCurve = JSON.parse(backing.value); }
            catch (e) { initialCurve = defaultCurve(); }
            initialCurve.keyframes.sort((a, b) => a.t - b.t);

            node._curveData = initialCurve;
            node._selected = { index: 0 };
            node._waveformPeaks = null;
            node._durationSeconds = null;
            setWidgetHidden(backing, true);

            const outer = document.createElement("div");
            outer.style.cssText = `display:flex; flex-direction:column; width:100%; height:${WIDGET_FIXED_HEIGHT}px; min-height:${WIDGET_FIXED_HEIGHT}px; max-height:${WIDGET_FIXED_HEIGHT}px; pointer-events:none;`;

            const { bar, playBtn, track, fill, head } = buildSeekBar();
            attachSeekBar(node, { playBtn, track, fill, head });
            outer.appendChild(bar);

            const { container, canvas } = buildCurveGraphDOM();
            node._curveCanvas = canvas;
            outer.appendChild(container);

            const redraw = () => redrawGraph(node, canvas);
            const onChange = () => { syncBackingValue(node); syncInterpDropdown(node); redraw(); };
            attachPointerHandlers(node, canvas, onChange);
            attachCanvasResizeObserver(node, canvas, redraw);

            const domWidget = node.addDOMWidget("curve_graph_dom", "curve_graph", outer, {
                hideOnZoom: false,
                getMinHeight: () => WIDGET_FIXED_HEIGHT,
                getMaxHeight: () => WIDGET_FIXED_HEIGHT,
                // Classic-only visual sizing: this changes the rendered DOM
                // height after the node has already been resized. It does NOT
                // participate in node.computeSize(), so it cannot create a
                // node-size feedback loop. Nodes 2.0 ignores this legacy
                // computedHeight for layout.
                afterResize: function () {
                    const widgets = node.widgets || [];
                    const graphIndex = widgets.indexOf(domWidget);
                    if (graphIndex < 0) return;

                    let trailingHeight = 0;
                    for (let i = graphIndex + 1; i < widgets.length; i++) {
                        const w = widgets[i];
                        if (!w || w.hidden) continue;
                        if (typeof w.computeSize === "function") {
                            const size = w.computeSize(node.size?.[0]);
                            if (Array.isArray(size) && Number.isFinite(size[1])) {
                                trailingHeight += size[1];
                            }
                        } else {
                            trailingHeight += 24;
                        }
                    }

                    const graphY = Number.isFinite(domWidget.y) ? domWidget.y : 0;
                    const available = (node.size?.[1] || MIN_NODE_HEIGHT) - graphY - trailingHeight - 4;
                    domWidget.computedHeight = Math.max(WIDGET_FIXED_HEIGHT, available);
                },
            });
            // Keep Classic's minimum-size calculation stable. Never derive
            // computeSize() from node.size: that creates a direct resize
            // feedback loop and is what caused the previous infinite growth.
            domWidget.computeSize = (w) => [w, WIDGET_FIXED_HEIGHT];

            // Nodes 2.0 does not use the legacy computeSize() path above.
            // Give its Vue/layout system an explicit fixed layout height as
            // well; otherwise the DOM widget can be treated as an auto-sized
            // flex row and participate in the node's live height calculation,
            // causing the node to grow repeatedly while switching/resizing.
            // This is intentionally constant: the Classic afterResize hook
            // above is the only path that visually stretches the graph.
            domWidget.computeLayoutSize = () => ({
                minHeight: WIDGET_FIXED_HEIGHT,
                maxHeight: WIDGET_FIXED_HEIGHT,
                minWidth: 0,
            });

            // In Classic, DOM widgets can retain the width from the moment
            // they were created. Force the layout to use the live node width.
            // This has no effect on the Vue/Nodes 2.0 layout path.
            try {
                Object.defineProperty(domWidget, "width", {
                    configurable: true,
                    enumerable: true,
                    get: () => undefined,
                    set: () => {},
                });
            } catch (_) {
                domWidget.width = undefined;
            }
            domWidget.serialize = false;
            if (domWidget.options) domWidget.options.serialize = false;

            node._interpWidget = node.addWidget(
                "combo", "interpolation (selected point)", node._curveData.keyframes[0].interp || "linear",
                (v) => {
                    const idx = node._selected.index;
                    if (idx >= 0 && idx < node._curveData.keyframes.length) {
                        node._curveData.keyframes[idx].interp = v;
                        syncBackingValue(node);
                        redraw();
                    }
                },
                { values: INTERP_MODES }
            );

            node._bulkInterpValue = "linear";
            node.addWidget("combo", "interpolation (all points)", node._bulkInterpValue,
                (v) => { node._bulkInterpValue = v; }, { values: INTERP_MODES });
            node.addWidget("button", "Apply to All Points", null, () => {
                for (const kf of node._curveData.keyframes) kf.interp = node._bulkInterpValue;
                syncBackingValue(node);
                syncInterpDropdown(node);
                redraw();
            });
            node.addWidget("button", "Reset Curve", null, () => {
                node._curveData = defaultCurve();
                node._selected.index = 0;
                syncBackingValue(node);
                syncInterpDropdown(node);
                redraw();
            });

            syncBackingValue(node);
            node.setSize?.([Math.max(node.size?.[0] || 0, MIN_NODE_WIDTH), Math.max(node.size?.[1] || 0, MIN_NODE_HEIGHT)]);

            const onRemoved = node.onRemoved;
            node.onRemoved = function () {
                node._curveGraphResizeObserver?.disconnect();
                return onRemoved ? onRemoved.apply(this, arguments) : undefined;
            };

            redraw();
            return result;
        };

        // Restore the curve editor's actual keyframes after LiteGraph has
        // configured the node from a saved workflow. onNodeCreated runs before
        // widgets_values are applied, so reading the backing widget there alone
        // is too early; without this hook the editor would revert to its default
        // two points after reopening ComfyUI.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            const node = this;
            const backing = findWidget(node, "curve_editor");
            if (backing) {
                try {
                    const restored = JSON.parse(backing.value);
                    if (restored && Array.isArray(restored.keyframes) && restored.keyframes.length >= 2) {
                        restored.keyframes.sort((a, b) => a.t - b.t);
                        node._curveData = restored;
                        node._selected = { index: 0 };
                        syncInterpDropdown(node);
                        node._curveCanvas && redrawGraph(node, node._curveCanvas);
                    }
                } catch (_) {
                    // Keep the already-created default curve if the saved value is invalid.
                }
            }
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            const node = this;
            if (!node._curveCanvas || !message) return result;

            const duration = message.duration_seconds ? message.duration_seconds[0] : null;
            const peaks = message.waveform_peaks ? message.waveform_peaks[0] : null;
            if (typeof duration === "number" && duration > 0) node._durationSeconds = duration;
            if (Array.isArray(peaks) && peaks.length > 0) node._waveformPeaks = normalizePeaksForDisplay(peaks);
            redrawGraph(node, node._curveCanvas);

            const audioInfo = message.audio ? message.audio[0] : null;
            if (audioInfo && node._loadSeekAudio) {
                const params = new URLSearchParams({
                    filename: audioInfo.filename, subfolder: audioInfo.subfolder || "",
                    type: audioInfo.type || "temp", rand: Date.now(),
                });
                node._loadSeekAudio(`/view?${params.toString()}`);
            }
            return result;
        };
    },
});

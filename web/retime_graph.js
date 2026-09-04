/*
 * retime_graph.js - RetimeGraph frontend extension.
 *
 * Deliberately reuses the same DOM-widget / ResizeObserver / seek-bar
 * scaffolding proven out in curve_editor.js.
 *
 * The actual graph differs in three ways from the Curve Editor's:
 *   1. Markers move in X (output_time, real seconds) only. Y
 *      (source_time) is fixed at whatever value the point had when
 *      created - this guarantees the mapping stays monotonic and
 *      matches the "Y axis is locked" design confirmed with the user.
 *   2. The X axis is real seconds (0..output_duration), not a
 *      normalized 0-1 fraction - output_duration is a widget, and
 *      raising it gives the last marker more room to be dragged out
 *      to slow a tail down, without needing to touch source content
 *      that doesn't exist.
 *   3. Always a straight-line (linear) piecewise mapping - no
 *      interpolation-mode picker.
 */

import { app } from "../../scripts/app.js";

const NODE_NAME = "RetimeGraph";
const MIN_NODE_WIDTH = 340;
const MIN_NODE_HEIGHT = 440;
const GRAPH_MIN_CSS_HEIGHT = 220;
const SEEK_BAR_HEIGHT = 26;
const WIDGET_FIXED_HEIGHT = GRAPH_MIN_CSS_HEIGHT + SEEK_BAR_HEIGHT;

const MARKER_RADIUS = 6;
const HIT_RADIUS = 12;
const PAD_X = 14;
const PAD_TOP = 18;
const PAD_BOTTOM = 8;
const WAVEFORM_DISPLAY_CEILING = 0.85;

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function defaultTimeMap() {
    return { output_duration: 5.0, points: [{ output_time: 0.0, source_time: 0.0 }, { output_time: 5.0, source_time: 5.0 }] };
}

// Mirrors core_retime.evaluate_source_time.
function evalSourceTime(points, tOut) {
    if (tOut <= points[0].output_time) return points[0].source_time;
    const last = points[points.length - 1];
    if (tOut >= last.output_time) return last.source_time;
    let i = 0;
    for (; i < points.length - 1; i++) if (points[i].output_time <= tOut) { if (points[i + 1].output_time > tOut) break; } else break;
    const p0 = points[i], p1 = points[i + 1];
    if (p1.output_time === p0.output_time) return p1.source_time;
    const u = (tOut - p0.output_time) / (p1.output_time - p0.output_time);
    return p0.source_time + u * (p1.source_time - p0.source_time);
}

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
    const backing = findWidget(node, "retime_graph");
    if (backing) backing.value = JSON.stringify(node._timeMap);
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
// X: real seconds along the output timeline, 0..node._outputDuration.
function outTimeToPixelX(t, r, outputDuration) {
    return r.plotX0 + clamp(t / Math.max(0.001, outputDuration), 0, 1) * r.plotW;
}
function pixelXToOutTime(px, r, outputDuration) {
    return clamp((px - r.plotX0) / r.plotW, 0, 1) * outputDuration;
}
// Y is a flat, centered reference line - source_time is intentionally not
// given a vertical position. Marker spacing along X is the only signal that
// matters visually, so Y is just centerY for every point, always.
function centerY(r) {
    return r.plotY0 + r.plotH / 2;
}

function redrawGraph(node, canvas) {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w = canvas.width, h = canvas.height;
    const rect = getPlotRect(w, h);
    const outputDuration = node._timeMap.output_duration;
    const cy = centerY(rect);

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, w, h);

    // Waveform backdrop only spans the source's real duration: it's a
    // static reference, not warped to the current mapping, so it's only
    // honest to draw it out to where source content actually exists.
    // Anything to the right (an extended tail) stays visually blank -
    // that's the "new output-time space" the user is stretching into.
    const peaks = node._waveformPeaks;
    if (peaks && peaks.length > 0 && node._durationSeconds) {
        const srcSpanPx = outTimeToPixelX(node._durationSeconds, rect, outputDuration) - rect.plotX0;
        ctx.fillStyle = "rgba(255, 159, 28, 0.30)";
        const n = peaks.length, barW = srcSpanPx / n, midY = rect.plotY0 + rect.plotH / 2;
        for (let i = 0; i < n; i++) {
            const barH = clamp(peaks[i], 0, 1) * (rect.plotH / 2);
            ctx.fillRect(rect.plotX0 + i * barW, midY - barH, Math.max(1, barW - 1), barH * 2);
        }
    }

    // Seconds ruler along X (real output-time seconds, not a percentage -
    // this is what makes the extendable timeline legible).
    ctx.fillStyle = "#888888";
    ctx.font = "11px sans-serif";
    for (let i = 0; i <= 4; i++) {
        const frac = i / 4;
        const label = (frac * outputDuration).toFixed(2) + "s";
        ctx.fillText(label, clamp(rect.plotX0 + frac * rect.plotW, rect.plotX0, rect.plotX1 - 32), 14);
    }

    // The piecewise-linear time-map itself.
    const pts = node._timeMap.points;
    ctx.strokeStyle = "#4fd8ff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {
        const px = outTimeToPixelX(pts[i].output_time, rect, outputDuration);
        if (i === 0) ctx.moveTo(px, cy); else ctx.lineTo(px, cy);
    }
    ctx.stroke();

    for (let i = 0; i < pts.length; i++) {
        const px = outTimeToPixelX(pts[i].output_time, rect, outputDuration);
        const py = cy;
        ctx.fillStyle = node._selected.index === i ? "#ffdd55" : "#ffffff";
        ctx.beginPath();
        ctx.arc(px, py, MARKER_RADIUS, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#333333";
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    if (node._dragging && node._selected.index >= 0 && node._selected.index < pts.length) {
        const sel = pts[node._selected.index];
        const px = outTimeToPixelX(sel.output_time, rect, outputDuration);
        drawFloatingLabel(ctx, px, cy, sel.output_time.toFixed(2) + "s", w);
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
    const boxY = py - MARKER_RADIUS - boxH - 6;

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
        const outputDuration = node._timeMap.output_duration;
        const pts = node._timeMap.points;

        let hitIndex = -1, hitDist = Infinity;
        for (let i = 0; i < pts.length; i++) {
            const kx = outTimeToPixelX(pts[i].output_time, rect, outputDuration);
            const ky = centerY(rect);
            const dist = Math.hypot(px - kx, py - ky);
            if (dist <= HIT_RADIUS && dist < hitDist) { hitIndex = i; hitDist = dist; }
        }

        // Right-click / shift-click a middle marker to delete it. First
        // and last points define the mapping's domain and can't be removed.
        if ((evt.button === 2 || evt.shiftKey) && hitIndex >= 0) {
            if (hitIndex !== 0 && hitIndex !== pts.length - 1) {
                pts.splice(hitIndex, 1);
                node._selected.index = clamp(hitIndex - 1, 0, pts.length - 1);
                onChange();
            }
            return;
        }

        if (hitIndex >= 0) {
            node._selected.index = hitIndex;
            node._dragging = true;
        } else {
            // New marker starts exactly on the current line at the click
            // position, so adding a point never introduces a jump.
            const tOut = pixelXToOutTime(px, rect, outputDuration);
            const srcTime = evalSourceTime(pts, tOut);
            const newPt = { output_time: tOut, source_time: srcTime };
            pts.push(newPt);
            pts.sort((a, b) => a.output_time - b.output_time);
            node._selected.index = pts.indexOf(newPt);
            node._dragging = true;
        }
        onChange();
    });

    canvas.addEventListener("pointermove", (evt) => {
        if (!node._dragging || node._selected.index < 0) return;
        const { px } = toInternalCoords(evt, canvas);
        const rect = getPlotRect(canvas.width, canvas.height);
        const outputDuration = node._timeMap.output_duration;
        const pts = node._timeMap.points;
        const idx = node._selected.index;

        // Y (source_time) is locked - only output_time moves. The first
        // point is pinned at output_time 0 (playback always starts at
        // the start); the last point can move but never past
        // output_duration (raise the widget for more room, rather than
        // dragging past the visible edge).
        if (idx === 0) { onChange(); return; }
        let t = pixelXToOutTime(px, rect, outputDuration);
        const prevT = pts[idx - 1].output_time;
        const nextT = idx + 1 < pts.length ? pts[idx + 1].output_time : outputDuration;
        t = clamp(t, prevT + 0.01, nextT - (idx + 1 < pts.length ? 0.01 : 0));
        pts[idx].output_time = t;
        onChange();
    });

    const endDrag = (evt) => { node._dragging = false; canvas.releasePointerCapture?.(evt.pointerId); };
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);
}

function buildGraphDOM() {
    const container = document.createElement("div");
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
    node._retimeGraphResizeObserver = ro;
}

// --- Seek bar / playback (previews the original, un-retimed source audio -
// this is a calibration aid, not a preview of the retimed result, which
// only exists after Audio Retimer actually runs) ---

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
    name: "audio_amplifier.RetimeGraphWidget",

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
            const backing = findWidget(node, "retime_graph");
            if (!backing) {
                console.warn("[AudioAmplifier] retime_graph backing widget not found.");
                return result;
            }

            let initialMap;
            try { initialMap = JSON.parse(backing.value); }
            catch (e) { initialMap = defaultTimeMap(); }
            if (!initialMap.points || initialMap.points.length < 2) initialMap = defaultTimeMap();
            initialMap.points.sort((a, b) => a.output_time - b.output_time);

            node._timeMap = initialMap;
            node._selected = { index: 0 };
            node._waveformPeaks = null;
            node._durationSeconds = null;
            setWidgetHidden(backing, true);

            const outer = document.createElement("div");
            outer.style.cssText = `display:flex; flex-direction:column; width:100%; height:${WIDGET_FIXED_HEIGHT}px; min-height:${WIDGET_FIXED_HEIGHT}px; max-height:${WIDGET_FIXED_HEIGHT}px; pointer-events:none;`;

            const { bar, playBtn, track, fill, head } = buildSeekBar();
            attachSeekBar(node, { playBtn, track, fill, head });
            outer.appendChild(bar);

            const { container, canvas } = buildGraphDOM();
            node._retimeCanvas = canvas;
            outer.appendChild(container);

            const redraw = () => redrawGraph(node, canvas);
            const onChange = () => { syncBackingValue(node); redraw(); };
            attachPointerHandlers(node, canvas, onChange);
            attachCanvasResizeObserver(node, canvas, redraw);

            const domWidget = node.addDOMWidget("retime_graph_dom", "retime_graph", outer, {
                hideOnZoom: false,
                getMinHeight: () => WIDGET_FIXED_HEIGHT,
                getMaxHeight: () => WIDGET_FIXED_HEIGHT,
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
                            if (Array.isArray(size) && Number.isFinite(size[1])) trailingHeight += size[1];
                        } else trailingHeight += 24;
                    }
                    const graphY = Number.isFinite(domWidget.y) ? domWidget.y : 0;
                    const available = (node.size?.[1] || MIN_NODE_HEIGHT) - graphY - trailingHeight - 4;
                    domWidget.computedHeight = Math.max(WIDGET_FIXED_HEIGHT, available);
                },
            });
            domWidget.computeSize = (w) => [w, WIDGET_FIXED_HEIGHT];
            domWidget.computeLayoutSize = () => ({ minHeight: WIDGET_FIXED_HEIGHT, maxHeight: WIDGET_FIXED_HEIGHT, minWidth: 0 });
            try {
                Object.defineProperty(domWidget, "width", { configurable: true, enumerable: true, get: () => undefined, set: () => {} });
            } catch (_) { domWidget.width = undefined; }
            domWidget.serialize = false;
            if (domWidget.options) domWidget.options.serialize = false;

            // The output_duration widget already exists (declared in
            // INPUT_TYPES, renders as an ordinary number widget) - hook
            // its callback so raising it live-widens the graph's timeline
            // instead of requiring a run.
            const applyOutputDuration = (v) => {
                node._timeMap.output_duration = Math.max(v, node._timeMap.points[node._timeMap.points.length - 1].output_time);
                syncBackingValue(node);
                redraw();
            };
            const durationWidget = findWidget(node, "output_duration");
            if (durationWidget) {
                const origCallback = durationWidget.callback;
                durationWidget.callback = function (v) {
                    const r = origCallback ? origCallback.apply(this, arguments) : undefined;
                    applyOutputDuration(v);
                    return r;
                };
            }
            // Belt-and-suspenders: some frontend builds route widget value
            // changes through node.onWidgetChanged instead of (or in
            // addition to) the widget's own .callback, especially when
            // something else in the widget-init chain reassigns .callback
            // after this point. Hooking both means the live-update keeps
            // working regardless of which path actually fires.
            const onWidgetChanged = node.onWidgetChanged;
            node.onWidgetChanged = function (name, value, old_value, widget) {
                const r = onWidgetChanged ? onWidgetChanged.apply(this, arguments) : undefined;
                if (name === "output_duration" && typeof value === "number") applyOutputDuration(value);
                return r;
            };

            node.addWidget("button", "Reset Graph", null, () => {
                const dur = node._timeMap.output_duration;
                node._timeMap = { output_duration: dur, points: [{ output_time: 0.0, source_time: 0.0 }, { output_time: dur, source_time: node._durationSeconds || dur }] };
                node._selected.index = 0;
                syncBackingValue(node);
                redraw();
            });

            syncBackingValue(node);
            node.setSize?.([Math.max(node.size?.[0] || 0, MIN_NODE_WIDTH), Math.max(node.size?.[1] || 0, MIN_NODE_HEIGHT)]);

            const onRemoved = node.onRemoved;
            node.onRemoved = function () {
                node._retimeGraphResizeObserver?.disconnect();
                return onRemoved ? onRemoved.apply(this, arguments) : undefined;
            };

            redraw();
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            const node = this;
            const backing = findWidget(node, "retime_graph");
            if (backing) {
                try {
                    const restored = JSON.parse(backing.value);
                    if (restored && Array.isArray(restored.points) && restored.points.length >= 2) {
                        restored.points.sort((a, b) => a.output_time - b.output_time);
                        node._timeMap = restored;
                        node._selected = { index: 0 };
                        node._retimeCanvas && redrawGraph(node, node._retimeCanvas);
                    }
                } catch (_) { /* keep the already-created default */ }
            }
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            const node = this;
            if (!node._retimeCanvas || !message) return result;

            const duration = message.duration_seconds ? message.duration_seconds[0] : null;
            const peaks = message.waveform_peaks ? message.waveform_peaks[0] : null;
            const timeMapJson = message.time_map ? message.time_map[0] : null;
            const outDur = message.output_duration ? message.output_duration[0] : null;
            if (typeof duration === "number" && duration > 0) node._durationSeconds = duration;
            if (Array.isArray(peaks) && peaks.length > 0) node._waveformPeaks = normalizePeaksForDisplay(peaks);
            // Adopt the full corrected time-map (points included) after a
            // run - this is what makes the auto-fit-to-real-audio-duration
            // correction (see nodes_retime.py) actually move the graph's
            // line/timeline, not just the output_duration number. Falls
            // back to the narrower output_duration-only sync if an older
            // backend response doesn't include time_map.
            if (typeof timeMapJson === "string") {
                try {
                    const restored = JSON.parse(timeMapJson);
                    if (restored && Array.isArray(restored.points) && restored.points.length >= 2) {
                        restored.points.sort((a, b) => a.output_time - b.output_time);
                        node._timeMap = restored;
                        syncBackingValue(node);
                    }
                } catch (_) { /* keep current state */ }
            } else if (typeof outDur === "number" && outDur > 0) {
                node._timeMap.output_duration = outDur;
            }
            redrawGraph(node, node._retimeCanvas);

            const audioInfo = message.audio ? message.audio[0] : null;
            if (audioInfo && node._loadSeekAudio) {
                const params = new URLSearchParams({ filename: audioInfo.filename, subfolder: audioInfo.subfolder || "", type: audioInfo.type || "temp", rand: Date.now() });
                node._loadSeekAudio(`/view?${params.toString()}`);
            }
            return result;
        };
    },
});

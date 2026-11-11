"""
core_retime.py - Shared data model and DSP for the Retime Graph / Audio
Retimer node pair.

Deliberately a separate module from core.py (which backs Audio Amplifier /
General Curve Editor) rather than an extension of it: core.py's curve
schema is a time->value envelope (used as a dB offset), while this
module's schema is a time-remap (output_time->source_time), a different
shape with different invariants (monotonicity, an extendable timeline).
Reusing the same JSON schema for both would blur two genuinely different
concepts. core.py is not imported or modified here.

Schema ("retime_v1"):
    {
        "schema": "retime_v1",
        "output_duration": <float seconds - width of the graph's timeline>,
        "points": [
            {"output_time": 0.0, "source_time": 0.0},
            ...
            {"output_time": <float>, "source_time": <float>}
        ]
    }

Invariants enforced by normalize_time_map():
    - >= 2 points.
    - output_time strictly increasing point-to-point (no zero-width gaps -
      that would mean infinite local speed).
    - source_time non-decreasing point-to-point (playback never runs
      backwards; a flat run is allowed - that's a freeze/hold).
    - first point's output_time == 0.0 (playback always starts at the
      beginning of the output timeline).
    - source_time values are NOT clamped to a known audio duration here,
      since this module doesn't always have the audio at hand (the graph
      node may run before an AUDIO input is connected). Audio Retimer
      clamps against the real waveform length at execution time instead.

The "last point" extendable-tail design: output_time is free to exceed
the source audio's own duration (that's what output_duration widens for).
source_time for the last point is what actually determines whether a
tail is slowed down (same source_time span, spread over a larger
output_time span = slower), not anything about extrapolating past the
source's real content.
"""

import json
import bisect

SCHEMA_NAME = "retime_v1"

OUTPUT_DURATION_MIN = 0.1
OUTPUT_DURATION_MAX = 3600.0

RATE_MIN = 0.1
RATE_MAX = 4.0
RATE_DEFAULT = 1.0


class RetimeError(Exception):
    """Raised for malformed or invalid time-map data."""


def default_time_map(source_duration=5.0):
    d = max(OUTPUT_DURATION_MIN, float(source_duration))
    return {
        "schema": SCHEMA_NAME,
        "output_duration": d,
        "points": [
            {"output_time": 0.0, "source_time": 0.0},
            {"output_time": d, "source_time": d},
        ],
    }


def normalize_time_map(data):
    """Validates and coerces a parsed time-map dict. Raises RetimeError
    on anything that can't be made sane; never silently reorders points
    (a caller feeding out-of-order points has a bug worth surfacing)."""
    if not isinstance(data, dict):
        raise RetimeError("Time-map must be a JSON object.")
    points = data.get("points")
    if not isinstance(points, list) or len(points) < 2:
        raise RetimeError("Time-map needs at least 2 points.")

    out_pts = []
    prev_out, prev_src = None, None
    for i, p in enumerate(points):
        try:
            t_out = float(p["output_time"])
            t_src = float(p["source_time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RetimeError(f"Malformed point at index {i}: {exc}") from exc
        if t_out < 0 or t_src < 0:
            raise RetimeError(f"Point {i} has a negative time value.")
        if prev_out is not None:
            if t_out <= prev_out:
                raise RetimeError(
                    f"Point {i}'s output_time ({t_out}) must be strictly "
                    f"greater than the previous point's ({prev_out})."
                )
            if t_src < prev_src:
                raise RetimeError(
                    f"Point {i}'s source_time ({t_src}) cannot be earlier "
                    f"than the previous point's ({prev_src}) - playback "
                    f"cannot run backwards."
                )
        out_pts.append({"output_time": t_out, "source_time": t_src})
        prev_out, prev_src = t_out, t_src

    if out_pts[0]["output_time"] != 0.0:
        raise RetimeError("The first point's output_time must be 0.0.")

    output_duration = data.get("output_duration", out_pts[-1]["output_time"])
    try:
        output_duration = float(output_duration)
    except (TypeError, ValueError) as exc:
        raise RetimeError(f"Malformed output_duration: {exc}") from exc
    output_duration = max(output_duration, out_pts[-1]["output_time"])
    output_duration = min(max(output_duration, OUTPUT_DURATION_MIN), OUTPUT_DURATION_MAX)

    return {"schema": SCHEMA_NAME, "output_duration": output_duration, "points": out_pts}


def time_map_to_json_string(time_map):
    return json.dumps(normalize_time_map(time_map))


def time_map_from_json_string(s):
    try:
        data = json.loads(s)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RetimeError(f"Invalid JSON: {exc}") from exc
    return normalize_time_map(data)


def evaluate_source_time(points, t_out):
    """Piecewise-linear lookup: given an output-timeline moment, returns
    the corresponding source-timeline moment. Clamped at both ends."""
    outs = [p["output_time"] for p in points]
    if t_out <= outs[0]:
        return points[0]["source_time"]
    if t_out >= outs[-1]:
        return points[-1]["source_time"]
    i = bisect.bisect_right(outs, t_out) - 1
    o0, s0 = points[i]["output_time"], points[i]["source_time"]
    o1, s1 = points[i + 1]["output_time"], points[i + 1]["source_time"]
    frac = (t_out - o0) / (o1 - o0)
    return s0 + frac * (s1 - s0)


def scale_time_map_by_rate(time_map, rate):
    """Applies the constant `rate` multiplier on top of a graph's local
    shape: output_time values are divided by rate (rate=2 -> whole thing
    takes half the output time, local relative speed variations
    preserved), source_time values are untouched. This is the exact
    behavior confirmed with the user: rate acts as a uniform multiplier
    over the graph's shape, not a replacement for it."""
    rate = max(RATE_MIN, min(RATE_MAX, float(rate)))
    points = [
        {"output_time": p["output_time"] / rate, "source_time": p["source_time"]}
        for p in time_map["points"]
    ]
    return {
        "schema": SCHEMA_NAME,
        "output_duration": time_map["output_duration"] / rate,
        "points": points,
    }


def constant_rate_time_map(rate, source_duration):
    """Builds a flat (no graph connected) time-map for a plain constant
    rate - the degenerate case Audio Retimer uses when no curve is wired
    in. Kept in this module so both nodes agree on one definition of
    'what a plain rate means' instead of each re-deriving it."""
    rate = max(RATE_MIN, min(RATE_MAX, float(rate)))
    out_dur = source_duration / rate
    return {
        "schema": SCHEMA_NAME,
        "output_duration": out_dur,
        "points": [
            {"output_time": 0.0, "source_time": 0.0},
            {"output_time": out_dur, "source_time": source_duration},
        ],
    }


# ---------------------------------------------------------------------------
# DSP: time-varying WSOLA
# ---------------------------------------------------------------------------
#
# Rather than chunking the audio at each graph marker and phase-vocoding
# each chunk at its own constant rate (which needs a crossfade patched in
# at every chunk boundary to avoid clicks), this runs a single WSOLA pass
# across the whole clip where the analysis position at each synthesis step
# is looked up directly from the time-map. A local speed change is then
# just a smoothly varying analysis-frame position, not a seam - so there
# is no per-marker discontinuity to patch over in the first place.
#
# WSOLA works in the time domain (pick source frames, phase-align them
# against the tail of what's already been synthesized via cross-
# correlation, window and overlap-add), which is simpler to get right
# for a variable rate than STFT phase-vocoding, at some cost in transient
# sharpness on very extreme stretches - acceptable given the 0.1x-4x
# range this is clamped to.

import numpy as np


def _hann(n):
    if n <= 1:
        return np.ones(max(n, 1), dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))


def wsola_time_remap(mono, sample_rate, points, output_duration,
                      synthesis_hop_ms=20.0, search_ms=8.0):
    """mono: 1-D numpy float array (single channel). points/output_duration:
    a normalized time-map (already rate-scaled if applicable). Returns a
    1-D numpy float array of length round(output_duration * sample_rate).

    source_time values in `points` are clamped to the actual clip length
    here (this is the one place that matters - the graph node itself
    doesn't always know the true audio length when it runs)."""
    n_src = len(mono)
    src_duration = n_src / sample_rate
    clamped_points = [
        {"output_time": p["output_time"], "source_time": min(p["source_time"], src_duration)}
        for p in points
    ]

    synthesis_hop = max(1, int(round(synthesis_hop_ms / 1000.0 * sample_rate)))
    frame_len = synthesis_hop * 2  # 50% nominal overlap
    search_radius = max(1, int(round(search_ms / 1000.0 * sample_rate)))
    window = _hann(frame_len)

    n_out = max(1, int(round(output_duration * sample_rate)))
    out = np.zeros(n_out + frame_len, dtype=np.float64)
    norm = np.zeros(n_out + frame_len, dtype=np.float64)

    def src_at(t_out_samples):
        t_out_sec = t_out_samples / sample_rate
        t_src_sec = evaluate_source_time(clamped_points, t_out_sec)
        pos = int(round(t_src_sec * sample_rate))
        return max(0, min(n_src - frame_len, pos))

    prev_tail = None
    write_pos = 0
    t_out_samples = 0
    while write_pos < n_out:
        target_pos = src_at(t_out_samples)

        if prev_tail is not None:
            lo = max(0, target_pos - search_radius)
            hi = min(n_src - frame_len, target_pos + search_radius)
            if hi > lo:
                candidates = np.arange(lo, hi + 1)
                overlap_len = len(prev_tail)
                best_pos, best_score = target_pos, -np.inf
                for c in candidates:
                    seg = mono[c:c + overlap_len]
                    if len(seg) < overlap_len:
                        continue
                    score = float(np.dot(seg, prev_tail))
                    if score > best_score:
                        best_score, best_pos = score, c
                target_pos = best_pos

        frame = mono[target_pos:target_pos + frame_len]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        windowed = frame * window

        out[write_pos:write_pos + frame_len] += windowed
        norm[write_pos:write_pos + frame_len] += window

        prev_tail = frame[-synthesis_hop:] * window[-synthesis_hop:]
        write_pos += synthesis_hop
        t_out_samples += synthesis_hop

    norm[norm < 1e-8] = 1.0
    out = out / norm
    return out[:n_out].astype(np.float32)


def linear_time_remap(mono, sample_rate, points, output_duration):
    """The preserve_pitch=False path: directly samples the source at each
    output sample's mapped source position via linear interpolation. No
    windowing/phase-alignment - cheap, and pitch shifts with speed (like
    changing tape/turntable speed), which is sometimes the wanted effect."""
    n_src = len(mono)
    src_duration = n_src / sample_rate
    n_out = max(1, int(round(output_duration * sample_rate)))
    t_out = np.arange(n_out) / sample_rate
    t_src = np.array([min(evaluate_source_time(points, t), src_duration) for t in t_out])
    src_positions = t_src * sample_rate
    src_idx = np.arange(n_src)
    return np.interp(src_positions, src_idx, mono).astype(np.float32)


def apply_time_map_to_waveform_numpy(waveform_np, sample_rate, time_map, preserve_pitch=True):
    """waveform_np: (channels, samples) float numpy array. Returns a new
    (channels, new_samples) array, each channel remapped identically
    (source_time lookups are shared across channels so they stay in
    phase with each other)."""
    points = time_map["points"]
    # Deliberately points[-1]["output_time"], not time_map["output_duration"]:
    # output_duration is the Retime Graph's editable-canvas ceiling (room to
    # drag the last marker into), not the actual synthesized length. Using
    # the ceiling here was the bug that held the last note out to a stale
    # widget value instead of stopping where the mapped content really ends.
    output_duration = points[-1]["output_time"]
    remap_fn = wsola_time_remap if preserve_pitch else \
        (lambda m, sr, p, d: linear_time_remap(m, sr, p, d))
    channels = [
        remap_fn(waveform_np[c], sample_rate, points, output_duration)
        for c in range(waveform_np.shape[0])
    ]
    max_len = max(len(c) for c in channels)
    channels = [np.pad(c, (0, max_len - len(c))) for c in channels]
    return np.stack(channels, axis=0)

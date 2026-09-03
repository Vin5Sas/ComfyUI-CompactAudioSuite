"""
core.py - DSP core for the Audio Amplifier / General Curve Editor node pair.

Framework-agnostic: no ComfyUI imports here, so this module can be tested
headless (pytest, or any script) without loading ComfyUI at all. Only numpy
and torch are used, matching the offline/bundled-dependency requirement.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Curve data model
# ---------------------------------------------------------------------------
# A Curve is a plain dict:
#   {
#       "keyframes": [
#           {"t": 0.0, "value": 0.0, "interp": "linear"},
#           ...
#       ]
#   }
# "t" is normalized time in [0.0, 1.0] - not seconds, not samples. This is
# what lets the same curve apply correctly regardless of the connected
# audio's actual duration; the mapping from t to real time happens only at
# apply time (build_curve_envelope), never in the curve data itself.
#
# "value" is a dB offset, applied at that point in time. 0.0 means no
# change - true unity gain, not silence. Positive values boost, negative
# values cut (attenuate) - a curve pulled down still means quieter, never
# phase-inverted or muted, since it's a dB quantity, not a linear
# multiplier. When paired with Audio Amplifier, the node also exposes an
# `amplitude_db` widget: the two are independent, additive dB offsets -
# final_db(t) = amplitude_db + curve_value(t) - so the curve always
# applies exactly as drawn, including when amplitude_db is 0.0, and
# amplitude_db's effect always comes through even where the curve is flat
# at 0.0.
#
# "interp" describes the interpolation INTO that keyframe from the previous
# one. Valid values: "linear", "smooth", "step".

DEFAULT_CURVE = {
    "keyframes": [
        {"t": 0.0, "value": 0.0, "interp": "linear"},
        {"t": 1.0, "value": 0.0, "interp": "linear"},
    ]
}

VALID_INTERP_MODES = ("linear", "smooth", "step")


class CurveError(ValueError):
    """Raised for malformed curve data. Kept as a distinct type so callers
    (node execute() methods) can catch this specifically and surface a
    clean user-facing error instead of a raw stack trace."""
    pass


def normalize_curve(curve):
    """
    Return a sanitized copy of curve:
      - keyframes sorted by t
      - t clamped to [0, 1]
      - a keyframe forced to exist at t=0.0 and t=1.0 (edge-extended from
        the nearest existing keyframe if not explicitly authored there) -
        this is what guarantees the curve always spans the full range
      - degenerate curves (0 or 1 keyframes) expanded to a flat two-point
        curve at that same value

    Raises CurveError on malformed input rather than silently guessing.
    """
    if not isinstance(curve, dict) or "keyframes" not in curve:
        raise CurveError("Curve must be a dict with a 'keyframes' list")

    kfs = curve["keyframes"]
    if not isinstance(kfs, list):
        raise CurveError("Curve['keyframes'] must be a list")

    if len(kfs) == 0:
        return {"keyframes": [
            {"t": 0.0, "value": 0.0, "interp": "linear"},
            {"t": 1.0, "value": 0.0, "interp": "linear"},
        ]}

    cleaned = []
    for kf in kfs:
        try:
            t = float(kf["t"])
            value = float(kf["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CurveError(f"Malformed keyframe {kf!r}: {exc}") from exc
        t = min(1.0, max(0.0, t))
        interp = kf.get("interp", "linear")
        if interp not in VALID_INTERP_MODES:
            raise CurveError(
                f"Unknown interpolation mode '{interp}' "
                f"(expected one of {VALID_INTERP_MODES})"
            )
        cleaned.append({"t": t, "value": value, "interp": interp})

    cleaned.sort(key=lambda k: k["t"])

    if len(cleaned) == 1:
        only = cleaned[0]
        return {"keyframes": [
            {"t": 0.0, "value": only["value"], "interp": "linear"},
            {"t": 1.0, "value": only["value"], "interp": "linear"},
        ]}

    if cleaned[0]["t"] > 0.0:
        cleaned.insert(0, {"t": 0.0, "value": cleaned[0]["value"], "interp": "linear"})
    if cleaned[-1]["t"] < 1.0:
        cleaned.append({"t": 1.0, "value": cleaned[-1]["value"], "interp": cleaned[-1]["interp"]})

    return {"keyframes": cleaned}


def _hermite_interp(t0, v0, t1, v1, v_prev, v_next, t):
    """
    Cubic Hermite (Catmull-Rom style) smooth interpolation between (t0, v0)
    and (t1, v1), using neighboring keyframe values to compute tangents so
    the curve eases smoothly through each point without discontinuous
    slope changes. Falls back to a one-sided tangent at curve edges, where
    a neighbor does not exist.
    """
    if t1 == t0:
        return v0
    u = (t - t0) / (t1 - t0)

    m0 = (v1 - (v_prev if v_prev is not None else v0)) / 2.0
    m1 = ((v_next if v_next is not None else v1) - v0) / 2.0

    u2 = u * u
    u3 = u2 * u
    h00 = 2 * u3 - 3 * u2 + 1
    h10 = u3 - 2 * u2 + u
    h01 = -2 * u3 + 3 * u2
    h11 = u3 - u2

    return h00 * v0 + h10 * m0 + h01 * v1 + h11 * m1


def evaluate_curve(curve, t_query):
    """
    Evaluate a curve at one or more normalized time positions.

    curve: curve dict (normalized internally; caller does not need to
           pre-normalize)
    t_query: float, or 1D array-like of floats, each in [0, 1]

    Returns: float if t_query was scalar, else a 1D np.ndarray matching
    the input shape.
    """
    curve = normalize_curve(curve)
    kfs = curve["keyframes"]

    scalar_input = np.isscalar(t_query)
    t_arr = np.atleast_1d(np.asarray(t_query, dtype=np.float64))
    t_arr = np.clip(t_arr, 0.0, 1.0)

    ts = np.array([k["t"] for k in kfs])
    vs = np.array([k["value"] for k in kfs])
    interps = [k["interp"] for k in kfs]

    out = np.empty_like(t_arr)

    # searchsorted gives, for each query t, the index of the segment
    # [ts[i], ts[i+1]] it falls in.
    seg_idx = np.searchsorted(ts, t_arr, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(ts) - 2)

    for i, t in enumerate(t_arr):
        si = int(seg_idx[i])
        t0, t1 = ts[si], ts[si + 1]
        v0, v1 = vs[si], vs[si + 1]
        mode = interps[si + 1]  # interp describes interpolation INTO the right keyframe

        if t1 == t0:
            out[i] = v1
            continue

        if mode == "step":
            out[i] = v0
        elif mode == "linear":
            u = (t - t0) / (t1 - t0)
            out[i] = v0 + u * (v1 - v0)
        elif mode == "smooth":
            v_prev = vs[si - 1] if si - 1 >= 0 else None
            v_next = vs[si + 2] if si + 2 < len(vs) else None
            out[i] = _hermite_interp(t0, v0, t1, v1, v_prev, v_next, t)
        else:
            raise CurveError(f"Unknown interpolation mode: {mode}")

    # Exactly at the curve's final t, always return the final keyframe's
    # own authored value, regardless of interpolation mode. Interior step
    # transitions work correctly because evaluating exactly at a shared
    # boundary point hands off to the NEXT segment (whose own t0 naturally
    # equals the step target's value); the very last keyframe has no next
    # segment to hand off to, so without this, a 'step' mode on the final
    # keyframe would never actually take effect even at t=1.0 exactly.
    last_t = ts[-1]
    at_end = t_arr >= last_t
    out[at_end] = vs[-1]

    return float(out[0]) if scalar_input else out


# ---------------------------------------------------------------------------
# Envelope application to audio
# ---------------------------------------------------------------------------

CONTROL_RATE_HZ = 200.0   # curve evaluated at this rate, then upsampled to
                          # full sample rate; 200Hz = 5ms control resolution,
                          # well below the audible stepping-artifact threshold
SMOOTH_MS = 8.0           # moving-average window applied at any discontinuity
                          # (including "step" jumps) to avoid zipper noise/clicks


def build_curve_envelope(curve, num_samples, sample_rate):
    """
    Build a per-sample dB-offset envelope array of length num_samples for
    audio at the given sample_rate, from a normalized-time curve. This is
    where normalized curve time (0..1) gets mapped onto the actual
    duration of whatever audio is connected - the curve itself never
    needs to know or store real seconds.

    The result is the curve's own dB contribution (0.0 = no change,
    positive = boost, negative = cut) - the caller (apply_envelope_to_audio)
    adds amplitude_db's independent contribution on top afterward.
    """
    if num_samples <= 0:
        return np.zeros(0, dtype=np.float64)
    if sample_rate <= 0:
        raise CurveError(f"Invalid sample_rate: {sample_rate}")

    duration_s = num_samples / float(sample_rate)
    control_samples = max(2, int(np.ceil(duration_s * CONTROL_RATE_HZ)) + 1)
    control_t = np.linspace(0.0, 1.0, control_samples)
    control_vals = evaluate_curve(curve, control_t)

    full_t = np.linspace(0.0, 1.0, num_samples)
    envelope_db = np.interp(full_t, control_t, control_vals)

    envelope_db = _smooth_discontinuities(envelope_db, sample_rate)
    return envelope_db


def _smooth_discontinuities(envelope, sample_rate):
    """
    Short moving-average smoothing pass to suppress any remaining zipper
    noise / clicks at sharp transitions (notably from 'step' segments).

    Uses edge-replicated padding (not zero-padding): zero-padding would
    bias smoothed values toward 0 right at the start/end of every clip -
    for a dB-offset envelope, 0 means "no change", so that's actually the
    correct neutral value to bias toward at boundaries in general, but
    ONLY when the curve's own edge value actually is 0 - zero-padding
    would incorrectly pull toward 0 even when the curve was deliberately
    authored with a nonzero value at the very edge. Replicating the edge
    sample keeps the smoothing local to genuine discontinuities in the
    curve instead of overriding what the curve actually says at its
    boundary.
    """
    window = max(1, int(sample_rate * (SMOOTH_MS / 1000.0)))
    if window <= 1 or len(envelope) < window:
        return envelope
    kernel = np.ones(window) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(envelope, (pad_left, pad_right), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed


def soft_limit(waveform, threshold_db=-1.0):
    """
    Soft-knee limiter: tanh-based soft clip applied above threshold_db
    (relative to full scale, i.e. amplitude 1.0). Only used when
    clip_protection is enabled - default behavior leaves audio untouched
    even if it exceeds +-1.0.

    waveform: torch.Tensor, any shape, nominal range approximately [-1, 1]
    """
    threshold = 10 ** (threshold_db / 20.0)
    abs_wave = torch.abs(waveform)
    over = abs_wave > threshold
    if not torch.any(over):
        return waveform

    sign = torch.sign(waveform)
    excess = (abs_wave - threshold).clamp(min=0)
    headroom = max(1e-6, 1.0 - threshold)
    limited_excess = headroom * torch.tanh(excess / headroom)
    limited_abs = torch.where(over, threshold + limited_excess, abs_wave)
    return sign * limited_abs


def audio_duration_seconds(audio_dict):
    """
    Compute duration in seconds from a ComfyUI AUDIO dict
    ({"waveform": torch.Tensor[B,C,T], "sample_rate": int}).
    """
    waveform = audio_dict["waveform"]
    sample_rate = audio_dict["sample_rate"]
    if sample_rate <= 0:
        raise CurveError(f"Invalid sample_rate: {sample_rate}")
    return waveform.shape[-1] / float(sample_rate)


def compute_waveform_peaks(waveform, num_buckets=400):
    """
    Downsample a waveform into num_buckets peak-amplitude values, for cheap
    client-side waveform preview rendering in the Curve Editor widget -
    avoids shipping full-resolution audio to the browser just to draw a
    backdrop.

    waveform: torch.Tensor, shape [B, C, T]
    Returns: list[float] of length num_buckets, each the max abs amplitude
    across channels/batch within that bucket.
    """
    if not torch.is_tensor(waveform) or waveform.ndim != 3:
        raise CurveError(
            f"Expected waveform shape [B, C, T], got "
            f"{tuple(waveform.shape) if torch.is_tensor(waveform) else type(waveform)}"
        )

    num_samples = waveform.shape[-1]
    if num_samples == 0:
        return [0.0] * num_buckets

    mono = waveform.abs().amax(dim=(0, 1))  # [T] - max across batch and channel dims
    num_buckets = max(1, min(num_buckets, num_samples))
    bucket_size = num_samples / num_buckets

    peaks = []
    for i in range(num_buckets):
        start = int(i * bucket_size)
        end = max(start + 1, int((i + 1) * bucket_size))
        end = min(end, num_samples)
        peaks.append(float(mono[start:end].max().item()))
    return peaks


def curve_from_json_string(s):
    """
    Parse a JSON string into a normalized curve dict. Used because the
    curve editor widget is backed by a STRING input (carrying JSON) rather
    than a true custom widget type - see nodes_v1.py / nodes_v3.py for why.
    Raises CurveError on invalid JSON, same error type as other curve
    validation failures, so callers can handle both uniformly.
    """
    import json
    try:
        data = json.loads(s)
    except (json.JSONDecodeError, TypeError) as exc:
        raise CurveError(f"Invalid curve JSON: {exc}") from exc
    return normalize_curve(data)


def curve_to_json_string(curve):
    """Serialize a curve dict to a JSON string (the widget's default value)."""
    import json
    return json.dumps(curve)


AMPLITUDE_DB_MIN = -60.0
AMPLITUDE_DB_MAX = 60.0
CURVE_INTENSITY_MIN = -4.0
CURVE_INTENSITY_MAX = 4.0
CURVE_INTENSITY_DEFAULT = 1.0


def apply_envelope_to_audio(waveform, sample_rate, curve, amplitude_db=0.0,
                             curve_intensity=CURVE_INTENSITY_DEFAULT, clip_protection=False):
    """
    Core DSP entry point used by the Audio Amplifier node.

    waveform: torch.Tensor, shape [B, C, T] (ComfyUI AUDIO convention)
    sample_rate: int
    curve: curve dict (normalized time, dB-offset values - 0.0 = no change)
    amplitude_db: float - a plain, independent dB offset (a volume trim),
                  added on top of the curve's own contribution:
                      final_db(t) = amplitude_db + curve_value(t) * curve_intensity
                      final_linear_gain(t) = 10 ** (final_db(t) / 20)
                  Always applies uniformly, including with a flat curve -
                  this is what makes it work as a plain gain knob even if
                  the curve is never touched.
    curve_intensity: float - scales the curve's own deviation from 0dB
                  (its "contrast"), default 1.0 (curve applies exactly as
                  drawn). Values above 1.0 stretch the curve - crests get
                  louder AND troughs get deeper, simultaneously - values
                  between 0 and 1 flatten it toward neutral, and 0.0
                  removes the curve's contribution entirely (only
                  amplitude_db then applies). This is deliberately
                  separate from amplitude_db: a single control can't both
                  act as a plain volume trim (needs to do something even
                  when the curve is flat) and as a contrast/depth control
                  (which does nothing when the curve is flat, since there
                  is no deviation to scale) without conflict - see
                  VERSIONING_LOG.md for the fuller reasoning.
    clip_protection: bool - apply soft_limit() as a final pass if True

    Returns: torch.Tensor, same shape and dtype/device as the input waveform.
    """
    if not torch.is_tensor(waveform):
        raise CurveError(f"waveform must be a torch.Tensor, got {type(waveform)}")
    if waveform.ndim != 3:
        raise CurveError(f"Expected waveform shape [B, C, T], got {tuple(waveform.shape)}")

    num_samples = waveform.shape[-1]
    curve_db = build_curve_envelope(curve, num_samples, sample_rate)
    total_db = amplitude_db + curve_db * curve_intensity
    gain_linear = 10.0 ** (total_db / 20.0)

    gain_tensor = torch.as_tensor(gain_linear, dtype=waveform.dtype, device=waveform.device)
    gain_tensor = gain_tensor.view(1, 1, -1)  # broadcast across batch and channel dims

    out = waveform * gain_tensor

    if clip_protection:
        out = soft_limit(out)

    return out

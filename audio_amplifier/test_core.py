"""
test_core.py - headless tests for core.py. Run with: python3 test_core.py
No ComfyUI import required - this exercises the DSP core in isolation.
"""

import numpy as np
import torch

import core


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        raise AssertionError(name)


def test_normalize_curve_edge_extension():
    curve = {"keyframes": [{"t": 0.3, "value": 0.5, "interp": "linear"},
                            {"t": 0.7, "value": 2.0, "interp": "linear"}]}
    norm = core.normalize_curve(curve)
    ts = [k["t"] for k in norm["keyframes"]]
    check("edge extension adds t=0.0 and t=1.0", ts[0] == 0.0 and ts[-1] == 1.0)
    check("t=0.0 value holds the first authored value",
          norm["keyframes"][0]["value"] == 0.5)
    check("t=1.0 value holds the last authored value",
          norm["keyframes"][-1]["value"] == 2.0)


def test_normalize_curve_degenerate():
    flat = core.normalize_curve({"keyframes": [{"t": 0.5, "value": 3.0, "interp": "linear"}]})
    check("single keyframe expands to flat 2-point curve",
          len(flat["keyframes"]) == 2 and
          flat["keyframes"][0]["value"] == 3.0 and
          flat["keyframes"][1]["value"] == 3.0)

    empty = core.normalize_curve({"keyframes": []})
    check("empty curve expands to flat 0.0 dB (no change) curve",
          len(empty["keyframes"]) == 2 and
          empty["keyframes"][0]["value"] == 0.0)


def test_normalize_curve_malformed_raises():
    raised = False
    try:
        core.normalize_curve({"keyframes": [{"t": 0.0, "value": 0.0, "interp": "bogus"}]})
    except core.CurveError:
        raised = True
    check("unknown interp mode raises CurveError", raised)


def test_evaluate_curve_linear():
    curve = {"keyframes": [{"t": 0.0, "value": 0.0, "interp": "linear"},
                            {"t": 1.0, "value": 10.0, "interp": "linear"}]}
    check("linear midpoint", abs(core.evaluate_curve(curve, 0.5) - 5.0) < 1e-9)
    check("linear start", abs(core.evaluate_curve(curve, 0.0) - 0.0) < 1e-9)
    check("linear end", abs(core.evaluate_curve(curve, 1.0) - 10.0) < 1e-9)


def test_evaluate_curve_step():
    curve = {"keyframes": [{"t": 0.0, "value": 0.0, "interp": "linear"},
                            {"t": 0.5, "value": 6.0, "interp": "step"},
                            {"t": 1.0, "value": 6.0, "interp": "linear"}]}
    check("step holds previous value just before the keyframe",
          abs(core.evaluate_curve(curve, 0.49) - 0.0) < 1e-9)
    check("step jumps exactly at the keyframe",
          abs(core.evaluate_curve(curve, 0.5) - 6.0) < 1e-9)


def test_evaluate_curve_smooth_passes_through_keyframes():
    curve = {"keyframes": [{"t": 0.0, "value": 0.0, "interp": "linear"},
                            {"t": 0.3, "value": 8.0, "interp": "smooth"},
                            {"t": 0.6, "value": 2.0, "interp": "smooth"},
                            {"t": 1.0, "value": 0.0, "interp": "linear"}]}
    check("smooth passes exactly through keyframe at t=0.3",
          abs(core.evaluate_curve(curve, 0.3) - 8.0) < 1e-6)
    check("smooth passes exactly through keyframe at t=0.6",
          abs(core.evaluate_curve(curve, 0.6) - 2.0) < 1e-6)


def test_evaluate_curve_step_on_final_keyframe():
    # Bug found while verifying curve_intensity: interior step points work
    # (evaluating exactly at the boundary hands off to the next segment,
    # whose own t0 naturally equals the step target's value) - but the
    # very LAST keyframe has no next segment to hand off to, so a 'step'
    # mode there never took effect, even at t=1.0 exactly, before this fix.
    curve = {"keyframes": [{"t": 0.0, "value": 6.0, "interp": "linear"},
                            {"t": 1.0, "value": -6.0, "interp": "step"}]}
    check("value just before the final step is still the pre-jump value",
          abs(core.evaluate_curve(curve, 0.99) - 6.0) < 1e-9)
    check("value exactly at t=1.0 is the final keyframe's own value, not held",
          abs(core.evaluate_curve(curve, 1.0) - (-6.0)) < 1e-9)


def test_evaluate_curve_array_input():
    curve = {"keyframes": [{"t": 0.0, "value": 0.0, "interp": "linear"},
                            {"t": 1.0, "value": 10.0, "interp": "linear"}]}
    out = core.evaluate_curve(curve, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    expected = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
    check("array evaluation matches expected linear ramp",
          np.allclose(out, expected, atol=1e-9))


def test_default_curve_with_zero_amplitude_leaves_audio_unchanged():
    sr = 48000
    waveform = torch.randn(1, 2, sr)  # 1 second, stereo
    out = core.apply_envelope_to_audio(waveform, sr, core.DEFAULT_CURVE,
                                        amplitude_db=0.0, clip_protection=False)
    check("amplitude_db=0.0 with default (flat 0.0dB) curve leaves audio numerically unchanged",
          torch.allclose(out, waveform, atol=1e-5))


def test_curve_still_applies_when_amplitude_db_is_zero():
    # The curve is an independent dB offset - it must always apply as
    # drawn, even when amplitude_db is 0.0. (An earlier design had the
    # curve act as a scalar or multiplier tied to amplitude_db, so it had
    # no effect whenever amplitude_db was 0 - reported as unexpected and
    # corrected.)
    sr = 48000
    waveform = torch.ones(1, 1, sr) * 0.2
    curve = {"keyframes": [{"t": 0.0, "value": 6.0, "interp": "linear"},
                            {"t": 1.0, "value": 6.0, "interp": "linear"}]}
    out = core.apply_envelope_to_audio(waveform, sr, curve,
                                        amplitude_db=0.0, clip_protection=False)
    ratio = (out[0, 0, sr // 2] / waveform[0, 0, sr // 2]).item()
    expected_ratio = 10 ** (6.0 / 20.0)
    check(f"curve value +6dB applies fully even with amplitude_db=0.0 "
          f"(expected ratio ~{expected_ratio:.4f}, got {ratio:.4f})",
          abs(ratio - expected_ratio) < 0.01)


def test_amplitude_alone_acts_as_simple_gain():
    sr = 48000
    waveform = torch.ones(1, 1, sr) * 0.1
    out = core.apply_envelope_to_audio(waveform, sr, core.DEFAULT_CURVE,
                                        amplitude_db=6.0, clip_protection=False)
    ratio = (out[0, 0, sr // 2] / waveform[0, 0, sr // 2]).item()
    check(f"amplitude_db=6 with default curve (flat 0.0dB) gives ~2x amplitude (got {ratio:.4f})",
          abs(ratio - 2.0) < 0.01)


def test_curve_and_amplitude_add_as_independent_db_offsets():
    sr = 48000
    waveform = torch.ones(1, 1, sr) * 0.1
    curve = {"keyframes": [{"t": 0.0, "value": -6.0, "interp": "linear"},
                            {"t": 1.0, "value": -6.0, "interp": "linear"}]}
    out = core.apply_envelope_to_audio(waveform, sr, curve,
                                        amplitude_db=12.0, clip_protection=False)
    ratio = (out[0, 0, sr // 2] / waveform[0, 0, sr // 2]).item()
    # final_db = amplitude_db + curve_value = 12 + (-6) = 6dB
    expected_ratio = 10 ** (6.0 / 20.0)
    check(f"curve=-6dB and amplitude_db=+12dB add to a net +6dB, "
          f"expected ratio ~{expected_ratio:.4f} (got {ratio:.4f})",
          abs(ratio - expected_ratio) < 0.01)


def test_curve_value_zero_means_no_change_not_silence():
    # This is the core semantic the design was corrected to guarantee:
    # curve=0.0 is a dB offset of zero, so it contributes NO change and
    # lets amplitude_db's own effect pass through untouched - it does
    # NOT silence the audio. (An earlier version treated curve values as
    # a direct linear multiplier, where 0 meant "multiply by zero", i.e.
    # true silence - reported as not matching the intended design, since
    # a flat/neutral curve position should mean "unchanged", not "mute".)
    sr = 48000
    waveform = torch.ones(1, 1, sr) * 0.3
    curve = {"keyframes": [{"t": 0.0, "value": 0.0, "interp": "linear"},
                            {"t": 1.0, "value": 0.0, "interp": "linear"}]}
    out = core.apply_envelope_to_audio(waveform, sr, curve,
                                        amplitude_db=20.0, clip_protection=False)
    ratio = (out[0, 0, sr // 2] / waveform[0, 0, sr // 2]).item()
    expected_ratio = 10 ** (20.0 / 20.0)  # curve contributes 0dB, amplitude_db passes through fully
    check(f"curve=0.0 does not silence audio; amplitude_db's effect passes through "
          f"unaffected (expected ratio ~{expected_ratio:.4f}, got {ratio:.4f})",
          abs(ratio - expected_ratio) < 0.01)


def test_negative_curve_value_attenuates_not_inverts():
    # A curve pulled down means quieter (attenuation), same polarity as
    # the input - not phase-inverted or muted. This is a dB cut, exactly
    # like turning a volume knob down, never a sign flip.
    sr = 48000
    waveform = torch.ones(1, 1, sr) * 0.3
    curve = {"keyframes": [{"t": 0.0, "value": -12.0, "interp": "linear"},
                            {"t": 1.0, "value": -12.0, "interp": "linear"}]}
    out = core.apply_envelope_to_audio(waveform, sr, curve,
                                        amplitude_db=0.0, clip_protection=False)
    sample_out = out[0, 0, sr // 2].item()
    sample_in = waveform[0, 0, sr // 2].item()
    expected_ratio = 10 ** (-12.0 / 20.0)
    check(f"negative curve value (-12dB) attenuates while keeping the same sign "
          f"(in={sample_in:.3f}, out={sample_out:.3f}, expected ratio ~{expected_ratio:.4f})",
          sample_out > 0 and abs(sample_out / sample_in - expected_ratio) < 0.01)


def test_curve_stretches_to_audio_duration_regardless_of_authoring():
    curve = {"keyframes": [{"t": 0.0, "value": -60.0, "interp": "linear"},
                            {"t": 0.5, "value": 0.0, "interp": "step"},
                            {"t": 1.0, "value": 0.0, "interp": "linear"}]}
    sr = 48000
    short_audio = torch.ones(1, 1, sr) * 0.1
    long_audio = torch.ones(1, 1, sr * 10) * 0.1

    short_out = core.apply_envelope_to_audio(short_audio, sr, curve, amplitude_db=12.0)
    long_out = core.apply_envelope_to_audio(long_audio, sr, curve, amplitude_db=12.0)

    short_ratio = (short_out[0, 0, -1] / short_audio[0, 0, -1]).item()
    long_ratio = (long_out[0, 0, -1] / long_audio[0, 0, -1]).item()
    # at t=1.0, curve contributes 0dB, so total = amplitude_db alone = 12dB
    check(f"short clip end reaches ~12dB gain (ratio {short_ratio:.3f})",
          abs(short_ratio - 10 ** (12 / 20)) < 0.05)
    check(f"long clip end reaches ~12dB gain (ratio {long_ratio:.3f})",
          abs(long_ratio - 10 ** (12 / 20)) < 0.05)


def test_step_transition_has_no_hard_sample_to_sample_jump():
    sr = 48000
    curve = {"keyframes": [{"t": 0.0, "value": 0.0, "interp": "linear"},
                            {"t": 0.5, "value": 24.0, "interp": "step"},
                            {"t": 1.0, "value": 24.0, "interp": "linear"}]}
    envelope_db = core.build_curve_envelope(curve, sr, sr)
    max_delta = np.max(np.abs(np.diff(envelope_db)))
    check(f"smoothed step has no instant full jump (max delta {max_delta:.4f} dB/sample)",
          max_delta < 1.0)


def test_clip_protection_off_by_default_can_exceed_unity():
    sr = 48000
    waveform = torch.ones(1, 1, sr) * 0.9
    out = core.apply_envelope_to_audio(waveform, sr, core.DEFAULT_CURVE,
                                        amplitude_db=12.0, clip_protection=False)
    check("clip_protection=False allows amplitude beyond 1.0",
          out.abs().max().item() > 1.0)


def test_clip_protection_on_keeps_near_unity():
    sr = 48000
    waveform = torch.ones(1, 1, sr) * 0.9
    out = core.apply_envelope_to_audio(waveform, sr, core.DEFAULT_CURVE,
                                        amplitude_db=12.0, clip_protection=True)
    check("clip_protection=True keeps amplitude close to 1.0",
          out.abs().max().item() <= 1.01)


def test_multichannel_broadcast():
    sr = 48000
    waveform = torch.ones(2, 4, sr) * 0.1
    out = core.apply_envelope_to_audio(waveform, sr, core.DEFAULT_CURVE, amplitude_db=6.0)
    check("gain applied identically across all batches/channels (linked envelope)",
          torch.allclose(out[0, 0], out[1, 3], atol=1e-6))


def test_bad_waveform_shape_raises():
    raised = False
    try:
        core.apply_envelope_to_audio(torch.zeros(10), 48000, core.DEFAULT_CURVE)
    except core.CurveError:
        raised = True
    check("non-[B,C,T] waveform raises CurveError", raised)


def test_flat_curve_with_all_defaults_is_pass_through():
    # amplitude_db=0, curve=flat 0dB, curve_intensity=default -> audio
    # completely unchanged, regardless of curve_intensity's value (since
    # a flat curve has nothing to scale).
    sr = 48000
    waveform = torch.randn(1, 2, sr)
    out = core.apply_envelope_to_audio(waveform, sr, core.DEFAULT_CURVE,
                                        amplitude_db=0.0, curve_intensity=2.5,
                                        clip_protection=False)
    check("flat curve passes through unchanged regardless of curve_intensity",
          torch.allclose(out, waveform, atol=1e-5))


def test_curve_intensity_stretches_crests_and_troughs_together():
    # amplitude_db=0, curve has a +6dB crest and a -6dB trough,
    # curve_intensity=2.0 -> crest becomes +12dB (louder), trough becomes
    # -12dB (deeper/quieter) - both move further from neutral together,
    # unlike amplitude_db which would shift both in the same direction.
    sr = 48000
    crest_curve = {"keyframes": [{"t": 0.0, "value": 6.0, "interp": "linear"},
                                  {"t": 1.0, "value": 6.0, "interp": "linear"}]}
    trough_curve = {"keyframes": [{"t": 0.0, "value": -6.0, "interp": "linear"},
                                   {"t": 1.0, "value": -6.0, "interp": "linear"}]}
    waveform = torch.ones(1, 1, sr) * 0.1

    crest_out = core.apply_envelope_to_audio(waveform, sr, crest_curve,
                                              amplitude_db=0.0, curve_intensity=2.0)
    trough_out = core.apply_envelope_to_audio(waveform, sr, trough_curve,
                                               amplitude_db=0.0, curve_intensity=2.0)

    crest_ratio = (crest_out[0, 0, sr // 2] / waveform[0, 0, sr // 2]).item()
    trough_ratio = (trough_out[0, 0, sr // 2] / waveform[0, 0, sr // 2]).item()
    expected_crest = 10 ** (12.0 / 20.0)
    expected_trough = 10 ** (-12.0 / 20.0)
    check(f"crest (+6dB) at intensity=2.0 becomes +12dB (expected ratio ~{expected_crest:.4f}, got {crest_ratio:.4f})",
          abs(crest_ratio - expected_crest) < 0.01)
    check(f"trough (-6dB) at intensity=2.0 becomes -12dB (expected ratio ~{expected_trough:.4f}, got {trough_ratio:.4f})",
          abs(trough_ratio - expected_trough) < 0.01)


def test_curve_intensity_zero_removes_curve_but_amplitude_still_applies():
    sr = 48000
    curve = {"keyframes": [{"t": 0.0, "value": 12.0, "interp": "linear"},
                            {"t": 1.0, "value": 12.0, "interp": "linear"}]}
    waveform = torch.ones(1, 1, sr) * 0.1
    out = core.apply_envelope_to_audio(waveform, sr, curve,
                                        amplitude_db=6.0, curve_intensity=0.0)
    ratio = (out[0, 0, sr // 2] / waveform[0, 0, sr // 2]).item()
    expected_ratio = 10 ** (6.0 / 20.0)  # curve contributes nothing; amplitude_db alone applies
    check(f"curve_intensity=0.0 removes the curve's contribution; amplitude_db still applies "
          f"(expected ratio ~{expected_ratio:.4f}, got {ratio:.4f})",
          abs(ratio - expected_ratio) < 0.01)


def test_amplitude_and_curve_intensity_combine_correctly():
    # amplitude_db=+3 (uniform shift) combined with curve_intensity=2.0
    # (contrast stretch) on a curve with a -6dB trough:
    # final_db = 3 + (-6 * 2.0) = 3 - 12 = -9dB
    sr = 48000
    curve = {"keyframes": [{"t": 0.0, "value": -6.0, "interp": "linear"},
                            {"t": 1.0, "value": -6.0, "interp": "linear"}]}
    waveform = torch.ones(1, 1, sr) * 0.1
    out = core.apply_envelope_to_audio(waveform, sr, curve,
                                        amplitude_db=3.0, curve_intensity=2.0)
    ratio = (out[0, 0, sr // 2] / waveform[0, 0, sr // 2]).item()
    expected_ratio = 10 ** (-9.0 / 20.0)
    check(f"amplitude_db and curve_intensity combine as amplitude_db + curve*intensity "
          f"(expected ratio ~{expected_ratio:.4f}, got {ratio:.4f})",
          abs(ratio - expected_ratio) < 0.01)


def test_audio_duration_seconds():
    sr = 44100
    audio = {"waveform": torch.zeros(1, 2, sr * 3), "sample_rate": sr}
    check("duration computed correctly (3s clip)",
          abs(core.audio_duration_seconds(audio) - 3.0) < 1e-9)


def test_compute_waveform_peaks_length_and_range():
    sr = 48000
    waveform = torch.zeros(1, 1, sr)
    waveform[0, 0, 1000] = 0.7   # a single peak in the first bucket region
    waveform[0, 0, sr - 1000] = -0.9  # a single peak near the end (abs -> 0.9)
    peaks = core.compute_waveform_peaks(waveform, num_buckets=100)
    check("peaks list has requested length", len(peaks) == 100)
    check("peaks capture the inserted amplitude near start",
          max(peaks[:5]) >= 0.69)
    check("peaks capture the inserted amplitude near end",
          max(peaks[-5:]) >= 0.89)
    check("silent region produces near-zero peaks",
          max(peaks[40:60]) < 1e-6)


def test_compute_waveform_peaks_empty_audio():
    waveform = torch.zeros(1, 1, 0)
    peaks = core.compute_waveform_peaks(waveform, num_buckets=50)
    check("empty audio returns flat zero peaks of requested length",
          len(peaks) == 50 and max(peaks) == 0.0)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} tests passed.")

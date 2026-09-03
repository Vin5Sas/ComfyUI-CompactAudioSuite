"""
nodes_v1.py - Classic (dict-based INPUT_TYPES / NODE_CLASS_MAPPINGS) node
definitions for Audio Amplifier and General Curve Editor.

This is the sole registration path (see __init__.py for why V3-only
registration was dropped: it broke classic-mode compatibility on testing).
Both node classes delegate to core.py for the actual logic.

The 'curve' data passed between the two nodes is wire-typed as a plain
STRING carrying JSON, not a custom "CURVE" type. A custom, unrecognized
socket type caused ComfyUI's frontend to attach its own generic fallback
curve/spline widget to Audio Amplifier's unconnected 'curve' input - a
widget this project never wrote and does not want there. STRING is a
well-defined core type with predictable behavior (a socket when connected,
forced to stay a socket-only via forceInput so no stray text box appears
either), which avoids that fallback rendering entirely.

The Curve Editor's on-screen UI (a single draggable curve graph plus one
interpolation dropdown) lives entirely in web/curve_editor.js; this file
only defines the backing data model and execution logic.
"""

from . import core

import os
try:
    import folder_paths
    import torchaudio
except ImportError:
    folder_paths = None
    torchaudio = None


def _save_temp_audio(audio, filename_prefix="curve_editor_preview"):
    """Saves audio as a temp WAV for browser playback, using ComfyUI's
    standard temp-preview convention (ui.audio = [{filename, subfolder,
    type}], matching PreviewAudio's format)."""
    if folder_paths is None or torchaudio is None:
        return None
    try:
        full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_temp_directory()
        )
        file = f"{filename}_{counter:05}.wav"
        torchaudio.save(os.path.join(full_folder, file), audio["waveform"][0].cpu(), audio["sample_rate"])
        return {"filename": file, "subfolder": subfolder, "type": "temp"}
    except Exception as exc:
        print(f"[GeneralCurveEditor] Could not save audio preview: {exc}")
        return None


class GeneralCurveEditorV1:
    """
    Draws a generic value-over-time curve (keyframes + interpolation).
    Not audio-specific: the 'audio' input is optional and used only for
    waveform preview and second-accurate ruler calibration in the widget.

    Marked as an output node (OUTPUT_NODE=True) so it gets its own
    Partial-Execution "run" button in the ComfyUI frontend and can be run
    on its own - connect audio, hit run, see the waveform/ruler update -
    without needing a full chain to a downstream output node first.
    """

    CATEGORY = "utils/curve"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("curve",)
    FUNCTION = "execute"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Draws a time-varying curve for other nodes to consume (e.g. Audio "
        "Amplifier). Click the graph to add a point, drag a point to move "
        "it, and Shift+click an interior point to delete it. Use the "
        "dropdown to set the selected point's interpolation, or Apply to "
        "All to set every point at once. Connect audio and run once to "
        "preview its waveform and calibrate the ruler to real seconds."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Single source-of-truth value carrier: a plain single-line
                # STRING widget holding JSON-encoded curve data. Kept
                # single-line (not multiline) deliberately - a multiline
                # STRING widget renders as a real DOM textarea in current
                # ComfyUI frontends, which cannot be hidden by manipulating
                # the widget object alone (that only hides canvas-drawn
                # widgets) and was showing as an oversized floating text
                # box. The frontend extension (web/curve_editor.js) hides
                # this widget and replaces it on-screen with a single
                # draggable curve graph plus an interpolation dropdown.
                "curve_editor": ("STRING", {
                    "multiline": False,
                    "default": core.curve_to_json_string(core.DEFAULT_CURVE),
                }),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    def execute(self, curve_editor, audio=None):
        try:
            curve = core.curve_from_json_string(curve_editor)
        except core.CurveError as exc:
            print(f"[GeneralCurveEditor] Invalid curve data, falling back to default: {exc}")
            curve = core.normalize_curve(core.DEFAULT_CURVE)

        ui = {}
        if audio is not None:
            try:
                ui["duration_seconds"] = [core.audio_duration_seconds(audio)]
                ui["waveform_peaks"] = [core.compute_waveform_peaks(audio["waveform"])]
            except (core.CurveError, KeyError, AttributeError) as exc:
                print(f"[GeneralCurveEditor] Could not compute audio preview data: {exc}")
            preview = _save_temp_audio(audio)
            if preview:
                ui["audio"] = [preview]

        # Output is the JSON string, matching Audio Amplifier's STRING
        # input on the other end of the wire.
        return {"ui": ui, "result": (core.curve_to_json_string(curve),)}


class AudioAmplifierV1:
    """
    Applies a curve to AUDIO as a dB offset, with two independent
    controls that do different jobs:
        final_db(t) = amplitude_db + curve_value(t) * curve_intensity
        final_linear_gain(t) = 10**(final_db(t)/20)
    amplitude_db is a plain volume trim - it shifts everything uniformly
    and works even with a flat curve. curve_intensity scales the curve's
    own contrast: values above 1.0 stretch it (crests get louder AND
    troughs get deeper, simultaneously), 0.0 removes the curve's
    contribution entirely (only amplitude_db then applies), and the
    default of 1.0 applies the curve exactly as drawn. A single control
    can't do both jobs (a volume trim needs to work on a flat curve; a
    contrast control has nothing to scale on a flat curve), so these are
    deliberately separate.

    'curve' is a plain STRING (JSON-encoded) input, forced to be a
    connection-only socket (forceInput=True) so no stray editable text box
    or fallback widget appears on this node - it should only ever be fed
    from a General Curve Editor's output.
    """

    CATEGORY = "audio/amplifier"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "execute"
    DESCRIPTION = (
        "Boosts or cuts an audio clip's volume. The connected curve (from "
        "General Curve Editor) is a dB offset applied at every moment - 0 "
        "means no change, positive boosts, negative cuts. amplitude_db "
        "adds a plain volume trim on top, working even with a flat curve. "
        "curve_intensity instead scales the curve's own contrast - above "
        "1.0 stretches it so crests get louder and troughs get deeper "
        "together, 0.0 removes the curve's effect entirely. Enable "
        "clip_protection to soft-limit the output instead of letting it "
        "clip."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "curve": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Connect a General Curve Editor node's output here.",
                }),
                "amplitude_db": ("FLOAT", {
                    "default": 0.0,
                    "min": core.AMPLITUDE_DB_MIN,
                    "max": core.AMPLITUDE_DB_MAX,
                    "step": 0.1,
                    "tooltip": "Plain volume trim in dB, added on top of the curve. "
                               "Works even with a flat curve, unlike curve_intensity.",
                }),
                "curve_intensity": ("FLOAT", {
                    "default": core.CURVE_INTENSITY_DEFAULT,
                    "min": core.CURVE_INTENSITY_MIN,
                    "max": core.CURVE_INTENSITY_MAX,
                    "step": 0.05,
                    "tooltip": "Scales the curve's own contrast. Above 1.0 stretches it "
                               "(crests louder, troughs deeper, together); 0.0 removes "
                               "the curve's effect (amplitude_db still applies); 1.0 "
                               "(default) applies the curve exactly as drawn.",
                }),
                "clip_protection": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Soft-limit output to prevent clipping from positive gain.",
                }),
            },
        }

    def execute(self, audio, curve, amplitude_db=0.0,
                curve_intensity=core.CURVE_INTENSITY_DEFAULT, clip_protection=False):
        try:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
        except (KeyError, TypeError) as exc:
            raise core.CurveError(f"Malformed AUDIO input: {exc}") from exc

        curve_dict = core.curve_from_json_string(curve)

        out_waveform = core.apply_envelope_to_audio(
            waveform, sample_rate, curve_dict,
            amplitude_db=amplitude_db, curve_intensity=curve_intensity,
            clip_protection=clip_protection,
        )
        return ({"waveform": out_waveform, "sample_rate": sample_rate},)


NODE_CLASS_MAPPINGS = {
    "AudioAmplifier": AudioAmplifierV1,
    "GeneralCurveEditor": GeneralCurveEditorV1,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioAmplifier": "Audio Amplifier",
    "GeneralCurveEditor": "General Curve Editor",
}

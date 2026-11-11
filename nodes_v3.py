"""
nodes_v3.py - V3 (comfy_api.latest / Nodes 2.0) node definitions for
Audio Amplifier and General Curve Editor.

STATUS: NOT CURRENTLY ACTIVE. __init__.py registers exclusively via the
classic V1 path (nodes_v1.py) after real-world testing showed V3-exclusive
registration broke classic-mode compatibility - Audio Amplifier worked
only under Nodes 2.0 rendering, not classic rendering, the opposite of
the "works both ways" requirement. This file is kept for reference and in
case ComfyUI's V3+Nodes-2.0 rendering support matures to the point where
revisiting it makes sense. It has been updated to match nodes_v1.py's
STRING-based curve wire type (see nodes_v1.py's docstring for why), but
is otherwise unverified against a real install since it is not loaded.
"""

from . import core

try:
    from comfy_api.latest import ComfyExtension, io
    V3_AVAILABLE = True
except ImportError:
    V3_AVAILABLE = False


if V3_AVAILABLE:

    class GeneralCurveEditor(io.ComfyNode):
        """Draws a generic value-over-time curve. See nodes_v1.py's
        GeneralCurveEditorV1 for the active implementation and full notes."""

        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="GeneralCurveEditor",
                display_name="General Curve Editor",
                category="utils/curve",
                is_output_node=True,
                inputs=[
                    io.Audio.Input("audio", optional=True),
                    io.String.Input(
                        "curve_editor",
                        multiline=True,
                        default=core.curve_to_json_string(core.DEFAULT_CURVE),
                    ),
                ],
                outputs=[
                    io.String.Output("curve"),
                ],
            )

        @classmethod
        def execute(cls, audio=None, curve_editor=None) -> io.NodeOutput:
            try:
                curve = core.curve_from_json_string(curve_editor) if curve_editor else core.normalize_curve(core.DEFAULT_CURVE)
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

            return io.NodeOutput(core.curve_to_json_string(curve), ui=ui)


    class AudioAmplifier(io.ComfyNode):
        """Applies a curve as an independent dB offset, added to
        amplitude_db. See nodes_v1.py's AudioAmplifierV1 for the active
        implementation."""

        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="AudioAmplifier",
                display_name="Audio Amplifier",
                category="audio/amplifier",
                inputs=[
                    io.Audio.Input("audio"),
                    io.String.Input("curve", force_input=True),
                    io.Float.Input(
                        "amplitude_db", default=0.0,
                        min=core.AMPLITUDE_DB_MIN, max=core.AMPLITUDE_DB_MAX, step=0.1,
                    ),
                    io.Boolean.Input("clip_protection", default=False),
                ],
                outputs=[
                    io.Audio.Output("audio"),
                ],
            )

        @classmethod
        def execute(cls, audio, curve, amplitude_db=0.0, clip_protection=False) -> io.NodeOutput:
            try:
                waveform = audio["waveform"]
                sample_rate = audio["sample_rate"]
            except (KeyError, TypeError) as exc:
                raise core.CurveError(f"Malformed AUDIO input: {exc}") from exc

            curve_dict = core.curve_from_json_string(curve)
            out_waveform = core.apply_envelope_to_audio(
                waveform, sample_rate, curve_dict,
                amplitude_db=amplitude_db, clip_protection=clip_protection,
            )
            return io.NodeOutput({"waveform": out_waveform, "sample_rate": sample_rate})


    class AudioAmplifierExtension(ComfyExtension):
        async def get_node_list(self):
            return [GeneralCurveEditor, AudioAmplifier]


    async def comfy_entrypoint() -> ComfyExtension:
        return AudioAmplifierExtension()

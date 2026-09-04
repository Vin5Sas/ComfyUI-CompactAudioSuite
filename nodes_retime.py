"""
nodes_retime.py - Retime Graph (UI) and Audio Retimer (DSP) node
definitions.

Mirrors the split used by General Curve Editor / Audio Amplifier in
nodes_v1.py: a graph-drawing node that outputs a JSON-encoded STRING,
and a DSP node that consumes it. Kept in their own module/schema
(core_retime.py) rather than extending core.py or nodes_v1.py, since
neither of those should be touched.

The 'time_map' wire is a plain STRING (JSON), for the same reason
Audio Amplifier's 'curve' input is: a custom socket type causes ComfyUI's
frontend to attach a generic fallback widget we don't want. See
nodes_v1.py's module docstring for the full reasoning.
"""

from . import core
from . import core_retime as cr

import os
import numpy as np

try:
    import folder_paths
    import torch
    import torchaudio
except ImportError:
    folder_paths = None
    torch = None
    torchaudio = None


def _save_temp_audio(audio, filename_prefix="retime_graph_preview"):
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
        print(f"[RetimeGraph] Could not save audio preview: {exc}")
        return None


class RetimeGraphV1:
    """
    Draws time-remap graph: a straight line from
    (0,0), markers draggable in output-time (X) only - source-time (Y) is
    fixed per point, so the graph can never point at content out of order.
    Pulling two adjacent markers closer together in X compresses that
    span of source content into less output time (faster); spreading them
    apart stretches it across more output time (slower).

    """

    CATEGORY = "audio/retime"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("time_map",)
    FUNCTION = "execute"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Draws a time-remap curve for Audio Retimer. Click the line to add "
        "a marker, drag a marker left/right to change its timing (markers "
        "move in output-time only - the source position they point to is "
        "fixed). Pull two markers closer together to speed that section "
        "up; spread them apart to slow it down. Raise output_duration to "
        "give the last marker room to slow a tail down. Connect audio and "
        "run once to preview its waveform and calibrate the ruler."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "retime_graph": ("STRING", {
                    "multiline": False,
                    "default": cr.time_map_to_json_string(cr.default_time_map()),
                }),
                "output_duration": ("FLOAT", {
                    "default": 5.0,
                    "min": cr.OUTPUT_DURATION_MIN,
                    "max": cr.OUTPUT_DURATION_MAX,
                    "step": 0.1,
                    "tooltip": "Width of the graph's timeline in seconds. Raise this "
                               "to give yourself room to drag the last marker further "
                               "right and slow down a tail - no clipping needed.",
                }),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    def execute(self, retime_graph, output_duration=5.0, audio=None):
        try:
            time_map = cr.time_map_from_json_string(retime_graph)
        except cr.RetimeError as exc:
            print(f"[RetimeGraph] Invalid time-map data, falling back to default: {exc}")
            time_map = cr.normalize_time_map(cr.default_time_map())

        # Auto-fit to the connected audio's real duration, but only while
        # the graph is still an untouched 1:1 identity (exactly 2 points,
        # starting at (0,0), no retime drawn yet) AND its endpoint doesn't
        # already match the real duration.
        if audio is not None:
            try:
                real_duration = core.audio_duration_seconds(audio)
            except (core.CurveError, KeyError, AttributeError):
                real_duration = None
            if real_duration and real_duration > 0:
                pts = time_map["points"]
                is_identity = (
                    len(pts) == 2
                    and pts[0]["output_time"] == 0.0 and pts[0]["source_time"] == 0.0
                    and pts[-1]["output_time"] == pts[-1]["source_time"]
                )
                if is_identity and abs(pts[-1]["output_time"] - real_duration) > 1e-6:
                    time_map = cr.normalize_time_map(cr.default_time_map(real_duration))

        # A widget-driven output_duration that's larger than what the
        # stored points imply should win (the whole point of the widget
        # is to pre-widen the timeline before the user drags a marker
        # into the new space) - but never shrink below the last point.
        time_map["output_duration"] = max(time_map["output_duration"], float(output_duration))

        ui = {
            "output_duration": [time_map["output_duration"]],
            "time_map": [cr.time_map_to_json_string(time_map)],
        }
        if audio is not None:
            try:
                ui["duration_seconds"] = [core.audio_duration_seconds(audio)]
                ui["waveform_peaks"] = [core.compute_waveform_peaks(audio["waveform"])]
            except (core.CurveError, KeyError, AttributeError) as exc:
                print(f"[RetimeGraph] Could not compute audio preview data: {exc}")
            preview = _save_temp_audio(audio)
            if preview:
                ui["audio"] = [preview]

        return {"ui": ui, "result": (cr.time_map_to_json_string(time_map),)}


class AudioRetimerV1:
    """
    Retimes AUDIO. With no time_map connected, `rate` is a plain constant
    speed multiplier (2.0 = twice as fast/half duration, 0.5 = half speed/
    double duration). With a time_map connected (from Retime Graph),
    `rate` becomes a uniform multiplier applied on top of the graph's own
    shape: it scales overall pacing while preserving whatever local
    fast/slow variation the graph draws, so you don't have to redraw the
    graph just to nudge the whole thing faster or slower.
    """

    CATEGORY = "audio/retime"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "execute"
    DESCRIPTION = (
        "Changes audio playback speed/timing. rate alone is a plain constant "
        "multiplier. Connect a Retime Graph's output to time_map for a "
        "variable speed-ramp retime instead - rate then acts as a uniform "
        "multiplier on top of the graph's shape rather than replacing it. "
        "preserve_pitch keeps tone the same while speed changes (like a "
        "professional time-stretch); turn it off for a natural pitch shift "
        "with speed, like a tape or turntable speed change."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "rate": ("FLOAT", {
                    "default": cr.RATE_DEFAULT,
                    "min": cr.RATE_MIN,
                    "max": cr.RATE_MAX,
                    "step": 0.05,
                    "tooltip": "Constant speed multiplier. >1.0 faster/shorter, "
                               "<1.0 slower/longer. Acts as a uniform multiplier on "
                               "top of time_map's shape when one is connected.",
                }),
                "preserve_pitch": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "On: time-stretch keeps pitch unchanged. Off: pitch "
                               "shifts with speed, like a tape/turntable speed change.",
                }),
            },
            "optional": {
                "time_map": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Connect a Retime Graph node's output here for a "
                               "variable speed-ramp retime.",
                }),
            },
        }

    def execute(self, audio, rate=cr.RATE_DEFAULT, preserve_pitch=True, time_map=None):
        if torch is None:
            raise cr.RetimeError("torch/torchaudio not available in this environment.")
        try:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
        except (KeyError, TypeError) as exc:
            raise cr.RetimeError(f"Malformed AUDIO input: {exc}") from exc

        source_duration = waveform.shape[-1] / float(sample_rate)

        if time_map is not None:
            base_map = cr.time_map_from_json_string(time_map)
            tm = cr.scale_time_map_by_rate(base_map, rate)
        else:
            tm = cr.constant_rate_time_map(rate, source_duration)

        orig_dtype = waveform.dtype
        orig_device = waveform.device
        batches = []
        for b in range(waveform.shape[0]):
            wf_np = waveform[b].detach().cpu().numpy().astype(np.float32)
            out_np = cr.apply_time_map_to_waveform_numpy(
                wf_np, sample_rate, tm, preserve_pitch=preserve_pitch,
            )
            batches.append(torch.from_numpy(out_np))

        max_len = max(t.shape[-1] for t in batches)
        batches = [
            torch.nn.functional.pad(t, (0, max_len - t.shape[-1])) for t in batches
        ]
        out_waveform = torch.stack(batches, dim=0).to(device=orig_device, dtype=orig_dtype)

        return ({"waveform": out_waveform, "sample_rate": sample_rate},)


NODE_CLASS_MAPPINGS = {
    "RetimeGraph": RetimeGraphV1,
    "AudioRetimer": AudioRetimerV1,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RetimeGraph": "Retime Graph",
    "AudioRetimer": "Audio Retimer",
}

"""
nodes_waveform_overlay.py - Print Waveform onto Images node.

A decorative node: takes an IMAGE sequence and AUDIO, and burns a
waveform overlay into every frame. Independent of the retime/amplifier
DSP nodes - no shared schema, just IMAGE in, IMAGE out.


"""

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "DejaVuSans.ttf",
    "Arial.ttf",
    "arial.ttf",
]


def _load_font(size):
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _hex_to_rgb(s, default=(79, 216, 255)):
    try:
        s = s.strip().lstrip("#")
        if len(s) == 3:
            s = "".join(c * 2 for c in s)
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return default


def _compute_envelope(mono, num_buckets):
    """Downsamples mono audio (numpy, 1-D) into num_buckets peak-abs
    values, normalized to 0..1."""
    n = len(mono)
    if n == 0 or num_buckets <= 0:
        return np.zeros(max(num_buckets, 1), dtype=np.float32)
    edges = np.linspace(0, n, num_buckets + 1).astype(int)
    peaks = np.zeros(num_buckets, dtype=np.float32)
    for i in range(num_buckets):
        seg = mono[edges[i]:max(edges[i + 1], edges[i] + 1)]
        peaks[i] = float(np.max(np.abs(seg))) if len(seg) else 0.0
    m = float(np.max(peaks)) if len(peaks) else 0.0
    if m > 1e-8:
        peaks = peaks / m
    return peaks


def _draw_waveform_band(size_w, size_h, orientation, envelope, playhead_frac,
                         waveform_rgb, waveform_opacity, background, bg_rgb, bg_opacity,
                         title_text, timer_text, frame_text, text_size, mode="static",
                         live_trace=None):
    """Renders one waveform band as an RGBA PIL Image of size
    (size_w, size_h). orientation is 'horizontal' (band is wide/short,
    waveform runs left-to-right) or 'vertical' (band is tall/narrow,
    waveform runs top-to-bottom).

    mode='static': the whole clip is drawn on every frame, with a thin
    playhead line marking the current position - a scrubber reference,
    readable on a single paused frame.
    mode='growing': only the already-played portion is drawn (a growing
    reveal as frames advance), with a brighter leading bar at the
    current position - closer to a live console/DAW level meter. Not
    meaningfully readable on a single paused frame in isolation, but
    animates like a live waveform when played back.
    mode='live': a real oscilloscope-style trace - a short raw-sample
    window centered on the current frame's timestamp, drawn as a single
    wiggling polyline across the full band (classic Winamp waveform
    visualizer style), rather than the clip-wide envelope used by the
    other two modes. `live_trace` is the array of amplitudes for that
    window (envelope is unused in this mode).
    """
    band = Image.new("RGBA", (size_w, size_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(band)

    if background:
        draw.rectangle([0, 0, size_w, size_h], fill=(*bg_rgb, int(bg_opacity * 255)))

    wf_alpha = int(waveform_opacity * 255)

    if mode == "live":
        trace = live_trace if live_trace is not None and len(live_trace) > 1 else [0.0, 0.0]
        m = len(trace)
        if orientation == "horizontal":
            mid = size_h / 2.0
            xs = np.linspace(0, size_w, m)
            points = [(float(xs[i]), float(mid - trace[i] * (size_h * 0.9) / 2.0)) for i in range(m)]
        else:
            mid = size_w / 2.0
            ys = np.linspace(0, size_h, m)
            points = [(float(mid - trace[i] * (size_w * 0.9) / 2.0), float(ys[i])) for i in range(m)]
        draw.line(points, fill=(*waveform_rgb, wf_alpha), width=2, joint="curve")
    else:
        n = len(envelope)
        played_n = n if mode == "static" else min(n, max(1, int(round(playhead_frac * n))))

        if orientation == "horizontal":
            mid = size_h / 2.0
            col_w = size_w / max(n, 1)
            for i in range(played_n):
                v = envelope[i]
                h = v * (size_h * 0.9) / 2.0
                x0 = i * col_w
                is_leading = (mode == "growing" and i == played_n - 1)
                fill = (255, 255, 255, wf_alpha) if is_leading else (*waveform_rgb, wf_alpha)
                draw.rectangle([x0, mid - h, x0 + max(col_w - 1, 1), mid + h], fill=fill)
            if mode == "static":
                px = playhead_frac * size_w
                draw.line([(px, 0), (px, size_h)], fill=(255, 255, 255, 220), width=2)
        else:
            mid = size_w / 2.0
            row_h = size_h / max(n, 1)
            for i in range(played_n):
                v = envelope[i]
                wgt = v * (size_w * 0.9) / 2.0
                y0 = i * row_h
                is_leading = (mode == "growing" and i == played_n - 1)
                fill = (255, 255, 255, wf_alpha) if is_leading else (*waveform_rgb, wf_alpha)
                draw.rectangle([mid - wgt, y0, mid + wgt, y0 + max(row_h - 1, 1)], fill=fill)
            if mode == "static":
                py = playhead_frac * size_h
                draw.line([(0, py), (size_w, py)], fill=(255, 255, 255, 220), width=2)

    font = _load_font(text_size)
    pad = max(4, text_size // 3)
    if title_text:
        draw.text((pad, pad), title_text, font=font, fill=(255, 255, 255, 235))
    if timer_text:
        tw = draw.textlength(timer_text, font=font)
        draw.text((size_w - tw - pad, pad), timer_text, font=font, fill=(255, 255, 255, 235))
    if frame_text:
        fw = draw.textlength(frame_text, font=font)
        draw.text((size_w - fw - pad, size_h - text_size - pad), frame_text, font=font, fill=(255, 255, 255, 235))

    return band


def _extract_live_trace(mono, sample_rate, t, global_max, window_seconds=0.05, n_points=200):
    """Raw-sample oscilloscope window centered on time t, decimated to
    n_points and normalized against the whole clip's peak (global_max)
    so the live trace sits at the same visual scale as the other modes
    rather than auto-gaining per-window."""
    half = window_seconds / 2.0
    start = t - half
    n_src = len(mono)
    idx = np.linspace(start, start + window_seconds, n_points) * sample_rate
    src_idx = np.arange(n_src)
    window = np.interp(idx, src_idx, mono, left=0.0, right=0.0)
    scale = global_max if global_max > 1e-8 else 1.0
    return np.clip(window / scale, -1.0, 1.0)


class PrintWaveformOntoImagesV1:
    """
    Burns a waveform overlay into every frame of an image sequence. The
    waveform shown is the whole clip (identical on every frame) with a
    moving playhead marking the current frame's position - a burned-in
    scrubber reference, readable on a single paused frame as well as
    during playback.
    """

    CATEGORY = "audio/overlay"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    DESCRIPTION = (
        "Overlays a waveform of the connected audio across an image sequence, "
        "with a moving playhead marking each frame's position. padding: on "
        "grows the image to fit the waveform band alongside the picture; off "
        "overlays it directly on top of the existing frame. fps=0 spreads "
        "frames evenly across the audio's duration; set fps for exact timing."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "waveform_mode": (["static", "growing", "live"], {"default": "static",
                    "tooltip": "static: full clip drawn every frame with a moving playhead "
                               "line (readable on a single paused frame). growing: only the "
                               "already-played portion is drawn, building up as frames "
                               "advance, like a console/DAW level meter. live: a real "
                               "oscilloscope-style trace of the current moment (classic "
                               "Winamp waveform visualizer style) - pulses with the audio."}),
                "orientation": (["horizontal", "vertical"], {"default": "horizontal"}),
                "position": (["top", "bottom", "left", "right"], {"default": "bottom",
                    "tooltip": "top/bottom apply when orientation is horizontal; left/right when vertical."}),
                "band_size": ("INT", {"default": 120, "min": 20, "max": 2000,
                    "tooltip": "Thickness (px) of the waveform band."}),
                "padding": ("BOOLEAN", {"default": True,
                    "tooltip": "On: grows the canvas to fit the band (image resolution increases). "
                               "Off: overlays the band directly on top of the existing frame."}),
                "position_x": ("INT", {"default": 0, "min": -4096, "max": 4096,
                    "tooltip": "Extra horizontal offset (px) for the band's placement."}),
                "position_y": ("INT", {"default": 0, "min": -4096, "max": 4096,
                    "tooltip": "Extra vertical offset (px) for the band's placement."}),
                "waveform_color": ("STRING", {"default": "#4FD8FF"}),
                "waveform_opacity": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "background": ("BOOLEAN", {"default": True}),
                "background_color": ("STRING", {"default": "#000000"}),
                "background_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "text_size": ("INT", {"default": 16, "min": 8, "max": 96}),
                "fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01,
                    "tooltip": "0 = spread frames evenly across the audio's duration. "
                               ">0 = exact frame timing (frame i at i/fps seconds)."}),
            },
            "optional": {
                "title_text": ("STRING", {"default": "", "tooltip": "Optional label. Empty = off."}),
                "show_timer": ("BOOLEAN", {"default": False, "tooltip": "Show current time (MM:SS.ms)."}),
                "show_frame_number": ("BOOLEAN", {"default": False, "tooltip": "Show current frame index."}),
            },
        }

    def execute(self, images, audio, waveform_mode="static", orientation="horizontal", position="bottom",
                band_size=120, padding=True, position_x=0, position_y=0,
                waveform_color="#4FD8FF", waveform_opacity=0.9,
                background=True, background_color="#000000", background_opacity=0.5,
                text_size=16, fps=0.0, title_text="", show_timer=False, show_frame_number=False):
        if torch is None:
            raise RuntimeError("torch not available in this environment.")

        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        mono = waveform[0].mean(dim=0).detach().cpu().numpy().astype(np.float32)
        audio_duration = len(mono) / float(sample_rate)

        n_frames = images.shape[0]
        img_h, img_w = images.shape[1], images.shape[2]

        wf_rgb = _hex_to_rgb(waveform_color)
        bg_rgb = _hex_to_rgb(background_color, default=(0, 0, 0))

        num_buckets = img_w if orientation == "horizontal" else img_h
        envelope = _compute_envelope(mono, max(num_buckets, 8))
        global_max = float(np.max(np.abs(mono))) if len(mono) else 0.0

        if padding:
            if orientation == "horizontal":
                out_h = img_h + band_size
                out_w = img_w
            else:
                out_w = img_w + band_size
                out_h = img_h
        else:
            out_h, out_w = img_h, img_w

        out_frames = np.zeros((n_frames, out_h, out_w, images.shape[3]), dtype=np.float32)

        for i in range(n_frames):
            t = (i / fps) if fps > 0 else (
                (i / max(n_frames - 1, 1)) * audio_duration
            )
            playhead_frac = 0.0 if audio_duration <= 0 else min(1.0, t / audio_duration)

            timer_text = ""
            if show_timer:
                mm = int(t // 60)
                ss = t - mm * 60
                timer_text = f"{mm:02d}:{ss:05.2f}"
            frame_text = f"frame {i}" if show_frame_number else ""
            live_trace = _extract_live_trace(mono, sample_rate, t, global_max) if waveform_mode == "live" else None

            frame_np = (images[i].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            base = Image.fromarray(frame_np).convert("RGBA")

            if padding:
                canvas = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 255))
                if orientation == "horizontal":
                    img_pos = (0, 0) if position == "bottom" else (0, band_size)
                    band_pos = (0, img_h) if position == "bottom" else (0, 0)
                    band_w, band_h = out_w, band_size
                else:
                    img_pos = (0, 0) if position == "right" else (band_size, 0)
                    band_pos = (img_w, 0) if position == "right" else (0, 0)
                    band_w, band_h = band_size, out_h
                canvas.paste(base, img_pos)
                band = _draw_waveform_band(band_w, band_h, orientation, envelope, playhead_frac,
                                            wf_rgb, waveform_opacity, background, bg_rgb, background_opacity,
                                            title_text, timer_text, frame_text, text_size, mode=waveform_mode, live_trace=live_trace)
                canvas.alpha_composite(band, band_pos)
            else:
                canvas = base
                if orientation == "horizontal":
                    band_w, band_h = img_w, band_size
                    x = position_x
                    y = position_y if position == "top" else (img_h - band_size + position_y)
                else:
                    band_w, band_h = band_size, img_h
                    y = position_y
                    x = position_x if position == "left" else (img_w - band_size + position_x)
                band = _draw_waveform_band(band_w, band_h, orientation, envelope, playhead_frac,
                                            wf_rgb, waveform_opacity, background, bg_rgb, background_opacity,
                                            title_text, timer_text, frame_text, text_size, mode=waveform_mode, live_trace=live_trace)
                canvas.alpha_composite(band, (int(x), int(y)))

            out_frames[i] = np.asarray(canvas.convert("RGB"), dtype=np.float32) / 255.0

        out_tensor = torch.from_numpy(out_frames).to(images.device, images.dtype)
        return (out_tensor,)


NODE_CLASS_MAPPINGS = {
    "PrintWaveformOntoImages": PrintWaveformOntoImagesV1,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrintWaveformOntoImages": "Print Waveform onto Images",
}

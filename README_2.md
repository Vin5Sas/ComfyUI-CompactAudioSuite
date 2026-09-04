# Audio Amplifier Suite

Five ComfyUI custom nodes, packaged together:

- **General Curve Editor** - a single draggable curve graph plus an
  interpolation dropdown. The curve it produces can
  be consumed by any node that wants a time-varying value.
- **Audio Amplifier** - applies that curve as a per-time multiplier on a
  base gain (in dB) to an AUDIO input.
- **Retime Graph** - a simple time-remap graph for
  variable-speed audio retiming (speed ramps, not just a flat multiplier).
- **Audio Retimer** - changes audio playback speed/timing, either a
  plain constant rate or a Retime Graph-driven speed ramp.
- **Print Waveform onto Images** - burns a waveform overlay into an
  image sequence, in static/growing/live (oscilloscope) styles.

## Install

Drop this folder into `ComfyUI/custom_nodes/` (e.g. as
`ComfyUI/custom_nodes/audio_amplifier/`) and restart ComfyUI.

No third-party Python dependencies - only `numpy`, `torch`, and `Pillow`
(Print Waveform onto Images only), all already bundled with ComfyUI.

## Using the nodes

**General Curve Editor** (category `utils/curve`):
- Optionally connect `audio` - used only to preview the waveform and
  calibrate the graph's ruler to real seconds. It has no effect on the
  curve's logic. This node is marked as an output node, so it has its
  own "run" button (ComfyUI's Partial Execution feature) - run it by
  itself right after connecting audio, before wiring anything downstream.
- The graph: **click** empty space to add a point, **drag** a point to
  move it, **Shift+click** an interior point to delete it (right-click
  also works as a fallback, but Shift+click is the reliable one - plain
  right-click can get intercepted by ComfyUI's own canvas menu). The two
  edge points (start/end) can't be deleted and only move vertically -
  their time is pinned at 0 and 1 so the curve always spans the full
  connected audio's duration, whatever that duration is. The graph
  resizes with the node - drag the node's corner to make it taller or
  shorter. While dragging a point, a small floating label follows it
  showing time (seconds, or % if no audio duration is known yet) and
  its dB value.
- The **interpolation dropdown** sets the interpolation mode
  (linear/smooth/step) of whichever point you last clicked or dragged
  (highlighted yellow on the graph).
- **interpolation (all points) + Apply to All Points**: pick a mode in
  the dropdown, click the button, and every existing point switches to
  that mode at once. You can still change individual points afterward.
- **Reset Curve** discards all points and restores the flat default
  (0.0dB - no change).
- The graph has a fixed height (a Nodes-2.0 layout bug made variable
  height unsafe - see VERSIONING_LOG v1.9.0). Width follows the node
  normally. Node minimum size is 340x440.
- A play/pause + seek bar sits above the graph. Drag the head (not the
  track) to scrub; it plays live and restores your prior play/pause
  state on release. Disabled until the node has run with audio connected.
- Each point's value is a **dB offset**, draggable within +/-24dB in the
  UI - the dashed reference line sits at 0.0, which means no change.
  Pulling a point down cuts (attenuates); pushing it up boosts. Not a
  linear multiplier - down should mean quieter, never silenced/inverted.
  The +/-24dB range is a UI drag limit only; the gain math has no cap.

**Audio Amplifier** (category `audio/amplifier`):
- Connect `audio` and a `curve` (from General Curve Editor).
- **amplitude_db**: a plain volume trim in dB, added on top of the
  curve. Works even with a flat/untouched curve - this is your simple
  "just make it louder/quieter" control.
- **curve_intensity**: scales the curve's own contrast (default 1.0 =
  applies exactly as drawn). Above 1.0 stretches the curve - crests get
  louder *and* troughs get deeper, simultaneously. 0.0 removes the
  curve's contribution entirely (only amplitude_db then applies). Unlike
  amplitude_db, this does nothing if the curve is flat, since there's no
  deviation to scale - the two controls do genuinely different jobs and
  can't be combined into one.
- Formula: `final_db(t) = amplitude_db + curve_value(t) * curve_intensity`.
- `clip_protection` (off by default) applies a soft limiter so gain can't
  push the signal past +/-1.0 amplitude.

**Retime Graph** (category `audio/retime`):
- Optionally connect `audio` for a waveform preview, same Partial
  Execution "run by itself" pattern as General Curve Editor. On a fresh
  graph (no retime drawn yet), running with audio connected auto-fits
  the timeline to that audio's real duration - it won't touch a graph
  you've already customized, or a timeline you've deliberately widened
  ahead of dragging a marker.
- The line is flat and centered - it's a timeline, not a value graph.
  Markers move in **output time (X) only**; where a marker points to in
  the source audio (Y) is fixed the moment it's created, so the mapping
  can never go out of order. **Click** the line to add a marker (it
  starts exactly on the current mapping, so adding one never causes a
  jump), **drag** a marker left/right, **Shift+click**/right-click an
  interior marker to delete it. Pull two markers closer together to
  speed that span up; spread them apart to slow it down. A floating
  seconds label follows a marker while it's being dragged.
- **output_duration**: widens the graph's own timeline (live). This is
  what gives you room to drag the last marker further right to slow a
  tail down - it's an editing-canvas ceiling, not the final audio
  length (that's always wherever your last marker actually sits).
- The waveform backdrop only spans the source audio's real duration;
  anything to the right of that (an extended tail you've made room for)
  is intentionally left blank - that's "new" output-time space you're
  stretching into, not existing content.
- Outputs a STRING-encoded time-map for Audio Retimer's `time_map` input.

**Audio Retimer** (category `audio/retime`):
- Connect `audio`. `rate` alone (no `time_map` connected) is a plain
  constant speed multiplier: >1.0 faster/shorter, <1.0 slower/longer,
  clamped 0.1x-4x.
- Connect a Retime Graph's output to `time_map` for a variable
  speed-ramp retime instead. `rate` then acts as a **uniform multiplier
  on top of the graph's shape** - it scales overall pacing without
  needing to redraw the graph.
- `preserve_pitch` (default on): time-stretches via a time-varying WSOLA
  pass, so pitch stays put while speed changes. Off: a direct resample,
  so pitch shifts with speed (tape/turntable-style).
- Output audio length always matches where the time-map's content
  actually ends, not any editing-canvas ceiling from the Retime Graph.

**Print Waveform onto Images** (category `audio/overlay`):
- Connect `images` (a sequence) and `audio`. `fps=0` (default) spreads
  frames evenly across the audio's real duration; set `fps` for exact
  per-frame timing instead.
- **waveform_mode**:
  - `static` - the whole clip's waveform, identical on every frame, with
    a thin playhead line marking the current position. Readable on a
    single paused frame, not just during playback.
  - `growing` - only the already-played portion is drawn, building up as
    frames advance, with a brighter leading edge - a console/DAW level
    meter feel.
  - `live` - a real oscilloscope-style trace: a short raw-sample window
    centered on the current frame's timestamp, drawn as a single
    wiggling polyline (classic Winamp waveform-visualizer look), rather
    than the clip-wide envelope the other two modes use.
- `orientation` (horizontal/vertical) + `position` (top/bottom for
  horizontal, left/right for vertical) place the band.
- `padding` on grows the canvas/resolution to fit the band alongside the
  picture; off overlays the band directly on top of existing pixels at
  `position_x`/`position_y`.
- `title_text`, `show_timer`, `show_frame_number` are all optional and
  off/empty by default - the waveform itself is the only always-on part.
- Font is a generic system sans-serif (DejaVu Sans / Liberation Sans,
  falling back to Pillow's built-in default) - no decorative fonts.

## Compatibility

Registers via classic `NODE_CLASS_MAPPINGS`, merged from every node
module in `__init__.py`. `nodes_v3.py` (a V3/Nodes-2.0 schema version of
General Curve Editor and Audio Amplifier only) is kept in the package
for reference but is not currently used - see its docstring and
`HANDOFF.md` for why.

## Files

- `core.py` - DSP core for General Curve Editor / Audio Amplifier (curve
  evaluation, envelope building, gain application). Framework-agnostic,
  no ComfyUI imports - testable standalone with `python3 test_core.py`.
- `core_retime.py` - shared data model + DSP for Retime Graph / Audio
  Retimer (time-map schema, piecewise-linear evaluation, time-varying
  WSOLA time-stretch). Kept separate from `core.py` since the schemas
  and invariants genuinely differ (a time-remap isn't a gain envelope).
- `nodes_v1.py` - the active General Curve Editor / Audio Amplifier node
  definitions.
- `nodes_v3.py` - inactive V3 reference implementation of those two
  nodes, not currently used.
- `nodes_retime.py` - Retime Graph + Audio Retimer node definitions.
- `nodes_waveform_overlay.py` - Print Waveform onto Images node definition.
- `__init__.py` - registration entrypoint; merges every node module's
  `NODE_CLASS_MAPPINGS`.
- `web/curve_editor.js` - the curve graph + interpolation dropdown widget.
- `web/retime_graph.js` - the time-remap graph widget. Deliberately
  reuses the same DOM-widget/ResizeObserver/seek-bar scaffolding as
  `curve_editor.js` rather than re-solving canvas resize/sizing from
  scratch - see that file's header comment for the history of why it's
  built the way it is.

---

## Disclaimer

By installing and using this node, you acknowledge and agree to the following terms:

* **Run at Your Own Risk:** You assume full responsibility for the installation, execution, and operation of this custom node.
* **Total Exclusion of Liability:** The creator is not liable for any direct or indirect damages, data loss, hardware issues, or software corruption that may arise from its use or inability to use this tool.
* **Compatibility:** Compatibility with your specific system configuration, hardware, operating system, or future versions of ComfyUI is not guaranteed.
* **Testing:** It is highly recommended to test this node in a safe, non-production sandbox environment before introducing it into your critical workflows.
* **Environment & Dependencies:** This node may require external packages. You are responsible for managing your environment. The creator is not responsible for dependency conflicts that may disrupt your existing ComfyUI setup.
* **Compatibility & Future Updates:** Compatibility with your specific setup is not guaranteed. Future updates to ComfyUI core or dependent libraries may break this node's functionality without prior notice.
> **Notice:** This ComfyUI custom node is developed through AI-assisted coding. While carefully tested, it is provided "as is" without warranty of any kind.

---

# ComfyUI-CompactAudioSuite
This repository is a compact set of 5 Audio helper nodes developed for ComfyUI to manipulate an input audio and perform operations like Amplification, Attenuation, Linear Retime, Variable Retime and Print waveforms on Images.

---

# Five ComfyUI custom nodes, packaged together:

- **General Curve Editor** - a single draggable curve graph plus an
  interpolation dropdown. the curve it produces can
  be consumed by any node that wants a time-varying value.
- **Audio Amplifier** - applies that curve as a per-time multiplier on a
  base gain (in dB) to an AUDIO input.
- **Retime Graph** - a simple time-remap graph for
  variable-speed audio retiming (speed ramps, not just a flat multiplier).
- **Audio Retimer** - changes audio playback speed/timing, either a
  plain constant rate or a Retime Graph-driven speed ramp.
- **Print Waveform onto Images** - burns a waveform overlay into an
  image sequence, in static/growing/live (oscilloscope) styles.

<img width="1845" height="771" alt="Audio_Compact_Pack_ss" src="https://github.com/user-attachments/assets/0608827a-32ae-4c08-bea1-eb9e86541736" />


---

## Install

Drop this folder into `ComfyUI/custom_nodes/` (e.g. as
`ComfyUI/custom_nodes/audio_amplifier/`) and restart ComfyUI.

No third-party Python dependencies - only `numpy`, `torch`, and `Pillow`
(Print Waveform onto Images only), all already bundled with ComfyUI.

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

# Image Prompts

Generate pending images manually, review them against the criteria, then embed only existing files.

## 01-melodyne-fixed-notes-adjustable-tone.png

Status: generated (MAI-Image-2.6)

Used in: 02-melodyne-backprop-synthesizer.ipynb — Build the Differentiable Forward Pass

> **Copilot image prompt**
> - **Target file:** `images/01-melodyne-fixed-notes-adjustable-tone.png`
> - **Teaches:** fixed note events stay unchanged while audio error guides four tone controls.
> - **Aspect ratio:** 16:9
> - **Prompt:** A clean hand-drawn whiteboard signal-flow diagram on white, 16:9, left to right. Far left, a blue box "Fixed notes (never change)" listing A2 C#3 E3 D3 G3 E3 B2 A2, with an arrow to a blue box "dry waveform". Arrow onward into an amber box "tone shaping (FFT -> adjust -> inverse FFT)", then an amber box "drive", then an amber box "candidate audio". Under the tone and drive boxes, four amber dial icons labelled "bass" "mid" "treble" "drive" with thin arrows up into the boxes they control. On the right, a green box "target audio" and both candidate audio and target audio feed a black box "error". From the error box, dashed coral arrows run back only to the four dials, labelled "backprop: who is responsible?". A separate green box "optimizer nudges the dials" sits below the dials. No arrows go back to the fixed notes or the target. Large, sparse, correctly spelled text.
> - **Must show:** fixed notes separate from four learned controls; FFT before inverse FFT; target compared with candidate audio; backward arrows reach raw controls but not target or notes; update distinct from backward reporting.
> - **Must not show:** invented numbers, unreadable text, decorative clutter, logos, copyrighted characters
> - **Alt text:** The fixed eight-note phrase passes through adjustable tone and drive controls. Audio error traces responsibility back to four raw controls before a separate optimizer changes them.

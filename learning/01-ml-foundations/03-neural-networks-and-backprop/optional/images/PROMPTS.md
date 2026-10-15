# Image Prompts

Generate pending images manually, review them against the criteria, then embed only existing files.

## 01-melodyne-fixed-notes-adjustable-tone.png

Status: pending

Used in: 02-melodyne-backprop-synthesizer.ipynb — Build the Differentiable Forward Pass

> **Copilot image prompt**
> - **Target file:** `images/01-melodyne-fixed-notes-adjustable-tone.png`
> - **Teaches:** fixed note events stay unchanged while audio error guides four tone controls.
> - **Aspect ratio:** 16:9
> - **Prompt:** Draw a clean hand-drawn whiteboard signal path. Use blue for the fixed phrase A2, C#3, E3, D3, G3, E3, B2, A2 feeding a dry waveform. Beneath it show exactly four adjustable raw controls labelled bass, mid, treble, drive in amber. Route bass, mid, treble through bounded controls into FFT frequency shaping, then inverse FFT, then smooth drive into candidate audio. Draw a separate green target-audio box joining candidate audio at an error comparison. Coral arrows return from error to the four raw controls only. Add a separate green optimizer-update box after the responsibility report. Keep labels large and the layout left to right.
> - **Must show:** fixed notes separate from four learned controls; FFT before inverse FFT; target compared with candidate audio; backward arrows reach raw controls but not target or notes; update distinct from backward reporting.
> - **Must not show:** invented numbers, unreadable text, decorative clutter, logos, copyrighted characters
> - **Alt text:** The fixed eight-note phrase passes through adjustable tone and drive controls. Audio error traces responsibility back to four raw controls before a separate optimizer changes them.

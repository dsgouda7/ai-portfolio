# Image Prompts

Generate these images manually when ready. Keep placeholders in the lesson until the image file exists.

## 03-first-use-concept-map.png

Status: generated (MAI-Image-2.6; regenerated after accuracy review)
Used in: `../README.md` — Recommended reader route

- **Target file:** `images/03-first-use-concept-map.png`
- **Teaches:** each text-tensor idea appears only when the current task needs it, while its formal owner remains easy to find.
- **Aspect ratio:** 16:9
- **Prompt:** A clean hand-drawn whiteboard route diagram on white, 16:9. Title: "Text machinery appears when a task needs it". Five cards left to right joined by arrows. Card 1 "token pieces": shows "D3  A3  D4  F4" and note "one note = one token". Card 2 "integer IDs": shows "D3 -> id" and note "an ID is an address, not a size". Card 3 "learned rows": shows "id -> embedding row" and note "rows learned in training". Card 4 "shifted targets": two labelled lines "inputs:  D3 A3 D4" and "targets: A3 D4 F4" and note "predict the next note". Card 5 "padding and masks": shows a short row "D3 A3 <pad>" with note "pad = placeholder", then two small labelled tags "input mask: which positions are real" and "loss mask: which targets get graded". Above cards 1 to 5 a thin bracket labelled "first used in Chapter 01B sequence-memory lab". Below all cards a wide box "formally explained in Chapter 02: tokenization and embeddings" with dashed arrows up to each card. Small footnote bottom left: "numeric sequences in Chapter 01A need none of this". Large, sparse, correctly spelled text; nothing else.
- **Must show:** Five concepts in route order using D3 A3 D4 F4; correct one-step shift (inputs D3 A3 D4, targets A3 D4 F4); separate input and loss masks; first use in Chapter 01B; formal home in Chapter 02; no measurements.
- **Must not show:** invented numbers, unreadable text, decorative clutter, logos, copyrighted characters, or a claim that tokenization is required for numeric recurrence.
- **Alt text:** Piano notes become token pieces, integer addresses, and learned rows, then shifted next-note targets, padding, and separate input and loss masks. All are first used in the Chapter 01B lab and formally explained in Chapter 02.
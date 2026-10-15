# Image Prompts

Generate images manually, save the target, replace its pending line with an embed, then set status to generated. Until then do not add image links.

## 03-evo-01-model-evolution-map.png

Status: generated (MAI-Image-2.6)

Used in: README.md — The route is not a leaderboard; 01-counts-and-fixed-window-language-models.ipynb — The route is a chain of complaints

> **Copilot image prompt**
> - **Target file:** `images/03-evo-01-model-evolution-map.png`
> - **Teaches:** A route of complaints, not a ranking of newer models.
> - **Aspect ratio:** 16:9
> - **Prompt:** A clean hand-drawn whiteboard flow diagram on white, 16:9. Title at top: "Each model answers the previous model's complaint". Six rounded cards left to right joined by arrows: "Counts", "Fixed window", "Running note (RNN)", "Reader + writer (one summary card)", "Writer looks back (attention)", "Transformer". Each arrow carries a short red complaint label written above it: between Counts and Fixed window: "unseen combinations"; between Fixed window and Running note: "clue outside the window"; between Running note and Reader + writer: "answer is a different sequence"; between Reader + writer and Writer looks back: "one card cannot hold everything"; between Writer looks back and Transformer: "reading is still one step at a time". Directly below the third card "Running note (RNN)" (and only that card), draw a short dashed arrow pointing straight down from the bottom edge of "Running note (RNN)" to a smaller dashed card "Protected note (LSTM)" with caption "side fix: memory fades". Nothing else connects to the LSTM card. Cards with blue borders on the left fading to green on the right. No numbers, no other text.
> - **Must show:** Seven legible cards; LSTM side refinement; arrows name complaints; no accuracy claims.
> - **Must not show:** invented numbers, unreadable text, decorative clutter, logos, copyrighted characters
> - **Alt text:** Counts lead to a bounded window, a running note, and a reader and writer. Looking back removes a source-access restriction.

## 03-evo-02-fixed-window-blind-spot.png

Status: generated (MAI-Image-2.6)

Used in: 01-counts-and-fixed-window-language-models.ipynb — The clue is one seat too far away

> **Copilot image prompt**
> - **Target file:** `images/03-evo-02-fixed-window-blind-spot.png`
> - **Teaches:** A predictor cannot use a clue outside its input.
> - **Aspect ratio:** 16:9
> - **Prompt:** A clean hand-drawn whiteboard diagram on white, 16:9. Title: "Same window, different answers". Two horizontal rows. Top row: a red word card "red" on the far left, then a blue rectangle labelled "window" containing three word cards "wait" "quietly" "now", then an arrow to a green card "warm". Bottom row: identical layout but the far-left card is "blue" and the green answer card is "cool". The two window rectangles are identical. The far-left cards "red" and "blue" sit clearly outside the window rectangles, each with a crossed-out grey arrow showing they cannot reach the predictor. A caption under both rows: "The window sees the same three words, so it cannot tell warm from cool." Large, sparse, correctly spelled text.
> - **Must show:** Both windows contain exactly wait quietly now; outside clue and required answer differ; boundary explicit.
> - **Must not show:** invented numbers, unreadable text, decorative clutter, logos, copyrighted characters
> - **Alt text:** Red and blue lie outside identical windows. A window-only predictor cannot distinguish their required answers.

## 03-evo-03-fading-notecard.png

Status: generated (MAI-Image-2.6)

Used in: 02-fixed-windows-to-recurrent-language-models.ipynb — Keep the old failure in view

> **Copilot image prompt**
> - **Target file:** `images/03-evo-03-fading-notecard.png`
> - **Teaches:** A running note has a route beyond the window, but recall is not guaranteed.
> - **Aspect ratio:** 16:9
> - **Prompt:** Show two parallel handoffs reading red -> rest -> calmly -> now and blue -> rest -> calmly -> now. At every handoff show an orange card labelled carried state; words blue, warm/cool green. In a caution inset fade red ink over more handoffs, labelled possible forgetting, not measured result. Clean hand-drawn whiteboard style.
> - **Must show:** Current word and previous note enter every update; no future word enters; fading inset labelled conceptual.
> - **Must not show:** invented numbers, unreadable text, decorative clutter, logos, copyrighted characters
> - **Alt text:** A recurrent reader passes a note between words. The note can carry the clue beyond a window but can also forget.

## 03-evo-04-one-card-bottleneck.png

Status: generated (MAI-Image-2.6)

Used in: 03-recurrent-encoder-decoder-to-attention.ipynb — Read one sequence, write another

> **Copilot image prompt**
> - **Target file:** `images/03-evo-04-one-card-bottleneck.png`
> - **Teaches:** An ordered reader may give the writer only one source card.
> - **Aspect ratio:** 16:9
> - **Prompt:** Show source [3, 1, 4, 1] entering a reader one digit at a time. Its orange final card crosses a narrow interface to a writer producing [1, 4, 1, 3]. Keep intermediate notes behind a closed door labelled unavailable to writer. Label reader learns what to keep. No averaging. Blue source, green target, clean hand-drawn whiteboard style.
> - **Must show:** Ordered recurrent reading; only final learned state crosses; no pooling symbol; reversed output.
> - **Must not show:** invented numbers, unreadable text, decorative clutter, logos, copyrighted characters
> - **Alt text:** The reader compresses an ordered source into one learned card. The writer cannot reopen earlier source positions.

## 03-evo-05-writer-looks-back.png

Status: generated (MAI-Image-2.6)

Used in: 03-recurrent-encoder-decoder-to-attention.ipynb — Build a learned recurrent reader and writer

> **Copilot image prompt**
> - **Target file:** `images/03-evo-05-writer-looks-back.png`
> - **Teaches:** Each writing step can ask all reader notes for a useful detail.
> - **Aspect ratio:** 16:9
> - **Prompt:** Show blue source [3, 1, 4, 1], each digit over an orange reader-state card labelled by position. A green writer labelled first output 1 sends a question arrow toward all four cards. Emphasize last card as expected reversal address, not measured weight. Keep recurrent arrows between writer steps. Clean hand-drawn whiteboard style.
> - **Must show:** All source positions available; query comes from writer; expected first address is last source position; recurrence remains.
> - **Must not show:** invented numbers, unreadable text, decorative clutter, logos, copyrighted characters
> - **Alt text:** A recurrent writer looks back at all source states. End-to-start is the expected reversal route, not a fabricated measured map.

## 03-evo-06-practice-vs-generation.png

Status: generated (MAI-Image-2.6)

Used in: 03-recurrent-encoder-decoder-to-attention.ipynb — Practice and generation give different help

> **Copilot image prompt**
> - **Target file:** `images/03-evo-06-practice-vs-generation.png`
> - **Teaches:** Practice and generation use different previous answers.
> - **Aspect ratio:** 16:9
> - **Prompt:** A clean hand-drawn whiteboard diagram on white, 16:9, split into two panels side by side. Left panel title "Practice (teacher forcing)": a row of writer steps; above each step an input card and below each step an output card. Inputs: "start" "1" "4" "1" "3". Outputs: "1" "4" "1" "3" "end". A small green tag beside the inputs: "correct previous answer handed in". Right panel title "Generation": the same writer steps; inputs "start" then a curved arrow from each output back up into the next input, labelled "its own guess". Outputs shown as "1" "?" "?" "?" "?". A small orange tag: "no answer key, mistakes can carry forward". Top strip spanning both panels: "source 3 1 4 1 -> reversed target 1 4 1 3". Large, sparse, correctly spelled text; no other captions.
> - **Must show:** Correct shifted alignment; generated guess becomes next input; current answer never supplied as own input.
> - **Must not show:** invented numbers, unreadable text, decorative clutter, logos, copyrighted characters
> - **Alt text:** Practice supplies the correct previous answer. Generation feeds back its own guess, changing the next input.

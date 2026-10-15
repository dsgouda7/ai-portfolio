# Image Prompts

Generate images manually, save the target, replace its pending line with an embed, then set status to generated. Until then do not add image links.

## 03-evo-01-model-evolution-map.png

Status: pending

Used in: README.md — The route is not a leaderboard; 01-counts-and-fixed-window-language-models.ipynb — The route is a chain of complaints

> **Copilot image prompt**
> - **Target file:** `images/03-evo-01-model-evolution-map.png`
> - **Teaches:** A route of complaints, not a ranking of newer models.
> - **Aspect ratio:** 16:9
> - **Prompt:** Arrange cards: counts, ordered window, running note (RNN), reader and writer with one card, writer looks back, Transformer. Above counts show red wait quietly now warm. Label arrows: unseen combinations; clue outside window; different answer sequence; cannot look back; reading remains sequential. Add protected note (LSTM) as a side refinement from running note labelled memory can fade, not a required predecessor of attention. Use blue input, orange notes, green output, clean hand-drawn whiteboard style.
> - **Must show:** Seven legible cards; LSTM side refinement; arrows name complaints; no accuracy claims.
> - **Must not show:** invented numbers, unreadable text, decorative clutter, logos, copyrighted characters
> - **Alt text:** Counts lead to a bounded window, a running note, and a reader and writer. Looking back removes a source-access restriction.

## 03-evo-02-fixed-window-blind-spot.png

Status: pending

Used in: 01-counts-and-fixed-window-language-models.ipynb — The clue is one seat too far away

> **Copilot image prompt**
> - **Target file:** `images/03-evo-02-fixed-window-blind-spot.png`
> - **Teaches:** A predictor cannot use a clue outside its input.
> - **Aspect ratio:** 16:9
> - **Prompt:** Stack red wait quietly now -> warm over blue wait quietly now -> cool. Outline only wait quietly now in identical blue rectangles. Put red and blue just outside the left edge; draw crossed-out arrows from them. Label same window and different required words. Use blue/orange/green palette, clean hand-drawn whiteboard style.
> - **Must show:** Both windows contain exactly wait quietly now; outside clue and required answer differ; boundary explicit.
> - **Must not show:** invented numbers, unreadable text, decorative clutter, logos, copyrighted characters
> - **Alt text:** Red and blue lie outside identical windows. A window-only predictor cannot distinguish their required answers.

## 03-evo-03-fading-notecard.png

Status: pending

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

Status: pending

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

Status: pending

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

Status: pending

Used in: 03-recurrent-encoder-decoder-to-attention.ipynb — Practice and generation give different help

> **Copilot image prompt**
> - **Target file:** `images/03-evo-06-practice-vs-generation.png`
> - **Teaches:** Practice and generation use different previous answers.
> - **Aspect ratio:** 16:9
> - **Prompt:** Draw aligned lanes for source [3, 1, 4, 1], target [1, 4, 1, 3, end]. Practice inputs start, 1, 4, 1, 3 point to expected next answers. Generation shows start -> guessed 4 -> next input 4. Label correct previous input would be 1; wrong 4 is hypothetical trace, not measured prediction. Blue/orange/green palette, clean hand-drawn whiteboard style.
> - **Must show:** Correct shifted alignment; generated guess becomes next input; current answer never supplied as own input.
> - **Must not show:** invented numbers, unreadable text, decorative clutter, logos, copyrighted characters
> - **Alt text:** Practice supplies the correct previous answer. Generation feeds back its own guess, changing the next input.

# Image Prompts

Each image was generated from its prompt with Azure AI Foundry `MAI-Image-2.6`; regenerate by editing the prompt and rerunning the generator. Pending images stay as placeholders until a file exists.

## 04-reader-map.png

Status: generated (MAI-Image-2.6)

Used in: README.md — Reader map: what changes in each Part

> **Copilot image prompt**
> - **Target file:** `images/04-reader-map.png`
> - **Teaches:** You can distinguish familiar foundations from each new Transformer mechanism.
> - **Aspect ratio:** 16:9
> - **Prompt:** A clean hand-drawn whiteboard diagram on white, 16:9. Title at top: "Stage 04 route (teaching order, not history)". Six rounded cards in one left-to-right row joined by arrows. Card 1 (blue border): "Part 1" / "pieces and lookup rows" / tag "recap". Card 2 (orange): "Part 2" / "tokens look at tokens + position" / tag "new". Card 3 (orange): "Part 3" / "reusable block" / tag "new". Card 4 (orange with green stripe): "Part 4A / 4B" / "one growing tape (causal writing)" / tag "new + reuse". Card 5 (green with orange stripe): "Part 5A / 5B" / "source reader + target writer" / tag "reuse + new". Card 6 (green): "Part 6" / "modern block" / tag "refine + reuse". Under cards 1-3 a small caption "the cat sat on the mat"; under card 4 "aria heard"; under card 5 "source 3 1 4 1 -> target 1 4 1 3". Bottom-right legend with three swatches: blue "recap", orange "new mechanism", green "deliberate reuse". Large, sparse, correctly spelled text; nothing else.
> - **Must show:** All six Parts; both theory/lab pairs; distinct recap, new mechanism, and deliberate reuse labels; explicit teaching-order label.
> - **Must not show:** invented numbers, unreadable text, decorative clutter, logos, copyrighted characters
> - **Alt text:** Six cards show how token foundations lead to attention, blocks, causal writing, separate source memory, and modern refinements. Labels distinguish recap from new work and deliberate reuse.

## 04-block-family-comparison.png

Status: generated (MAI-Image-2.6; regenerated after accuracy review)

Used in: README.md — What is inside each block family?

> **Copilot image prompt**
> - **Target file:** `images/04-block-family-comparison.png`
> - **Teaches:** The writer with a separate source has one extra place to look: cross-attention.
> - **Aspect ratio:** 16:9
> - **Prompt:** A clean hand-drawn whiteboard diagram on white, 16:9. Title: "Three block families". Subtitle: "residual paths and normalization omitted here, not absent". Three vertical stacks side by side, each read top to bottom with downward arrows only. Stack 1 titled "Encoder": blue card "source 3 1 4 1", down to orange box "self-attention (sees all source positions)", down to green box "feed-forward (each position alone)". Stack 2 titled "Decoder-only": blue card "aria heard", down to orange box "causal self-attention (earlier positions only)", down to green box "feed-forward (each position alone)". Stack 3 titled "Encoder-decoder writer": blue card "target so far: start 1 4", down to orange box "causal self-attention (earlier target positions only)", down to orange box "cross-attention (looks at source memory)", down to green box "feed-forward (each position alone)". To the right of stack 3, a dashed blue box "source memory 3 1 4 1 (fixed, from encoder)" with ONE arrow pointing left into the cross-attention box only. No arrow points into the source memory. Large, sparse, correctly spelled text; nothing else.
> - **Must show:** Exact operation order in all three stacks; fixed source memory enters cross-attention only, with no arrow back into it; residual/normalization omission caveat.
> - **Must not show:** invented numbers, unreadable text, decorative clutter, logos, copyrighted characters
> - **Alt text:** Encoder and decoder-only blocks each have attention then a feed-forward step. A source-conditioned writer adds cross-attention between its causal self-attention and feed-forward step.

## 04-rope-in-real-path.png

Status: generated (MAI-Image-2.6; regenerated after accuracy review)

Used in: 02-attention-and-position.ipynb — Before the dials: where the question and label come from

> **Copilot image prompt**
> - **Target file:** `images/04-rope-in-real-path.png`
> - **Teaches:** Position turns the question and matching label before scoring, not the information that is delivered.
> - **Aspect ratio:** 16:9
> - **Prompt:** A clean hand-drawn whiteboard flow diagram on white, 16:9, one single left-to-right pipeline, no second panel. Title: "Where position enters attention". Far left: a row of six small blue cards "the" "cat" "sat" "on" "the" "mat" labelled "token states". One arrow from them splits into three blue boxes stacked vertically: top "Q: what am I looking for?", middle "K: what do I offer?", bottom "V: what do I deliver?". Q and K each pass through their own orange round dial icon labelled "rotate by position". V has NO dial; its arrow runs straight along the bottom labelled "V is not rotated". The rotated Q and rotated K arrows both enter one orange box "compare -> scaled scores", then arrow to orange box "mask (no illegal routes)", then arrow to orange box "softmax (weights sum to 1)". The softmax arrow and the long bottom V arrow both enter one green box "weighted blend of V", then arrow to green box "output". V never touches scores, mask, or softmax. Large, sparse, correctly spelled text; no numbers; nothing else.
> - **Must show:** Q and K rotate by position; V bypasses rotation and never touches scores, mask, or softmax; softmax weights and V meet only at the weighted blend.
> - **Must not show:** invented numbers, unreadable text, decorative clutter, logos, copyrighted characters
> - **Alt text:** Token states become queries, keys, and values. Only queries and keys rotate by position before scaled, masked comparison and softmax; the resulting weights blend unrotated values.

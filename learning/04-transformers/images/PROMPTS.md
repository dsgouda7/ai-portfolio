# Image Prompts

Generate each pending image manually with its Copilot prompt. Save it at the target path before replacing the matching placeholder with an image embed; then mark its status generated.

## 04-reader-map.png

Status: pending

Used in: README.md — Reader map: what changes in each Part

> **Copilot image prompt**
> - **Target file:** `images/04-reader-map.png`
> - **Teaches:** You can distinguish familiar foundations from each new Transformer mechanism.
> - **Aspect ratio:** 16:9
> - **Prompt:** Draw a left-to-right route of six labelled cards: Part 1 pieces and lookup rows (recap), Part 2 token-to-token lookup and position (new mechanism), Part 3 reusable block (new mechanism), Part 4A/4B one growing tape (new mechanism plus deliberate reuse), Part 5A/5B source reader and target writer (deliberate reuse plus new internals), Part 6 modern block (refinement and deliberate reuse). Put the sentence the cat sat on the mat under Parts 1–3; put aria heard under Part 4; put source [3,1,4,1] and target [1,4,1,3] under Part 5. Arrows show teaching dependencies, not historical chronology. Use a clean hand-drawn whiteboard style, blue inputs, orange new questions, green working components, and a labelled recap/new/reuse legend.
> - **Must show:** All six Parts; both theory/lab pairs; distinct recap, new mechanism, and deliberate reuse labels; explicit teaching-order label.
> - **Must not show:** invented numbers, unreadable text, decorative clutter, logos, copyrighted characters
> - **Alt text:** Six cards show how token foundations lead to attention, blocks, causal writing, separate source memory, and modern refinements. Labels distinguish recap from new work and deliberate reuse.

## 04-block-family-comparison.png

Status: pending

Used in: README.md — What is inside each block family?

> **Copilot image prompt**
> - **Target file:** `images/04-block-family-comparison.png`
> - **Teaches:** The writer with a separate source has one extra place to look: cross-attention.
> - **Aspect ratio:** 16:9
> - **Prompt:** Draw three side-by-side vertical stacks in a clean hand-drawn whiteboard style. Encoder: source [3,1,4,1] into self-attention (all valid source positions) then FFN (private per-position refinement). Decoder-only: aria heard into causal self-attention (prefix only) then FFN. Encoder-decoder writer: target [BOS,1,4] into causal self-attention then cross-attention then FFN; a separate blue source-memory bank [3,1,4,1] feeds only cross-attention. Use blue input cards, orange visibility boundaries, and green processed states. Label residual paths and normalization omitted from this comparison, not absent.
> - **Must show:** Exact operation order in all three stacks; fixed source memory enters cross-attention; future target positions blocked; residual/normalization omission caveat.
> - **Must not show:** invented numbers, unreadable text, decorative clutter, logos, copyrighted characters
> - **Alt text:** Encoder and decoder-only blocks each have attention then a feed-forward step. A source-conditioned writer adds cross-attention between its causal self-attention and feed-forward step.

## 04-rope-in-real-path.png

Status: pending

Used in: 02-attention-and-position.ipynb — Before the dials: where the question and label come from

> **Copilot image prompt**
> - **Target file:** `images/04-rope-in-real-path.png`
> - **Teaches:** Position turns the question and matching label before scoring, not the information that is delivered.
> - **Aspect ratio:** 16:9
> - **Prompt:** Draw the actual attention path left to right with the sentence the cat sat on the mat as the input row. Blue token-state cards feed learned Q/K/V projections, then separate head lanes. Label Q what cat asks, K what each position advertises, V what each position delivers. Q and K pass through orange paired dials labelled rotate by position; V bypasses the dials. Q/K arrows meet at scaled scores, then masks, then softmax (weights that sum to one), then weighted values; the V bypass enters this final blend. Use clean hand-drawn whiteboard style with blue inputs, orange position and legal-route controls, and green retrieved output. Clearly mark the early dial lesson as a visual preview and Sections 4–5 as the full computation connection.
> - **Must show:** States, learned Q/K/V, head split, Q/K-only rotation, scaled scores, masks before softmax, weighted V in exact order; an explicit unrotated V bypass; preview versus real-path labels.
> - **Must not show:** invented numbers, unreadable text, decorative clutter, logos, copyrighted characters
> - **Alt text:** Projected queries and keys split into heads and rotate before scaled, masked comparisons. The resulting weights blend unrotated values.

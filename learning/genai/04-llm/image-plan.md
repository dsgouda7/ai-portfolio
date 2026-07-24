# 04-LLM Image Plan

## Purpose

Use diagrams to make the decision points and data flow in the LLM chapter visible. Every image should explain a mechanism or trade-off that a learner needs before the adjacent code; decorative art is out of scope.

## Asset Rules

- Store final assets in `images/` with descriptive lowercase filenames.
- Use a wide 16:9 composition, at least 1600×900 PNG.
- Use a dark graphite background, muted teal for data flow, amber for trainable components, coral for failures, and light text.
- Keep generated text to short labels only; notebook Markdown provides the full explanation.
- Do not use logos, photorealistic people, UI mockups, gradients, or unreadably small text.
- Generate planned assets with Perchance only. If Perchance is unavailable, do not create a substitute image or generator script.

## Planned Assets

| Asset | Notebook placement | Teaching job |
|---|---|---|
| `lora-low-rank-adaptation.png` | 02 Concept 6; combined notebook Section 6 | Show $B(Ax)$, the rank bottleneck, and frozen base weights. Existing asset, relocate only. |
| `data-objectives-pipeline.png` | 01 opening roadmap | Distinguish continued pretraining, SFT, and DPO by their data and behavior change. |
| `parameter-strategies-spectrum.png` | 02 before full/partial/LoRA comparison | Compare what trains and what stays frozen across full FT, partial FT, LoRA, and QLoRA. |
| `finetuning-decision-matrix.png` | 03 before deployment recommendation | Map task requirement, budget, and latency to the right training strategy. |
| `hybrid-retrieval-storyboard.png` | 04 before RRF fusion | Show BM25 winning an exact-term query, dense retrieval winning a synonym query, and hybrid retrieval retaining both. |
| `rag-failure-location-map.png` | 05 before metrics implementation | Attach retrieval, grounding, relevance, and correctness metrics to the RAG stage they diagnose. |
| `gateway-request-lifecycle.png` | 06 architecture introduction | Show cache, rate limit, routing, fallback, provider call, and telemetry on one request path. |
| `end-to-end-finetuning-workflow.png` | fine-tuning-in-action opening | Show corpus to checkpoints and evaluation across pretraining, SFT, DPO, and LoRA. |
| `combined-notebook-checkpoints.png` | combined notebook opening | Show which checkpoint/artifact each section produces and reuses. |
| `exercise-finetuning-roadmap.png` | exercise notebook opening | Give learners a compact workflow map without exposing implementation solutions. |

## Perchance Prompts

Run these prompts in Perchance. No local or programmatic image fallback is permitted.

### `data-objectives-pipeline.png`

```text
Flat vector editorial infographic, wide 16:9, dark graphite background. Three left-to-right training lanes for an LLM: lane 1 raw domain documents flowing into "continued pretraining" and changing "domain language"; lane 2 instruction and response cards flowing into "SFT" and changing "follows instructions"; lane 3 chosen and rejected response pairs flowing into "DPO" and changing "preferences". One small frozen transformer silhouette is reused across lanes. Muted teal data flow, amber trainable path, coral rejected examples, ivory labels. Clean technical diagrams, strong hierarchy, no logo, no photorealism, no gradients, no tiny text.
```

### `parameter-strategies-spectrum.png`

```text
Flat vector technical infographic, wide 16:9, dark graphite background. Four aligned transformer stacks labeled full fine-tuning, partial freezing, LoRA, QLoRA. Full fine-tuning: all blocks amber. Partial freezing: top blocks amber, lower blocks slate with lock icons. LoRA: slate frozen attention blocks with small amber adapter side paths. QLoRA: compact purple-gray quantized frozen blocks with amber adapter side paths. Beneath each: simple proportional bars for trainable parameters and memory, using only large readable short labels. Muted teal arrows, ivory text, coral warning for high-memory full fine-tune. No logos, no photorealism, no gradients, no UI mockup.
```

### `finetuning-decision-matrix.png`

```text
Flat vector decision diagram, wide 16:9, dark graphite background. A clear branching path starts with "What must change?" and branches to domain knowledge, instruction behavior, preference/style, and severe distribution shift. Each path ends in a compact recommendation badge: continued pretraining plus LoRA, SFT plus LoRA, DPO plus LoRA, or full fine-tuning. Add two visible decision dials: budget and latency. Use muted teal paths, amber recommended choices, coral expensive path, ivory labels. Visual language of an engineering decision tree, no logos, no photorealism, no gradients, no tiny text.
```

### `hybrid-retrieval-storyboard.png`

```text
Flat vector technical storyboard, wide 16:9, dark graphite background, three panels. Panel one exact rare product code query: BM25 retrieves the correct document with a teal check, dense search misses with a coral x. Panel two synonym query: dense search retrieves the correct document with a teal check, BM25 misses with a coral x. Panel three RRF fusion combines both candidate lists into one ranked result list with the correct documents retained. Use document cards, simple token highlights and vector dots, ivory labels, amber fusion box. No logos, no photorealism, no gradients, no tiny text.
```

### `rag-failure-location-map.png`

```text
Flat vector systems diagram, wide 16:9, dark graphite background. Left-to-right RAG pipeline: question, retriever, retrieved context, generator, answer. Attach four large diagnostic callouts directly to the stages: retrieval recall at retriever, groundedness at context-to-generator boundary, answer relevance at generator, correctness at answer. Include one subtle coral broken-link path for a retrieval miss. Muted teal main flow, amber callouts, ivory labels. Clean instructional architecture, no logos, no photorealism, no gradients, no tiny text.
```

### `gateway-request-lifecycle.png`

```text
Flat vector production architecture diagram, wide 16:9, dark graphite background. A single request flows through semantic cache, rate limiter, router, selected provider, response, telemetry. Router branches to three provider boxes; one provider failure visibly triggers a coral fallback arrow to another provider. Cost and latency telemetry line returns to a small dashboard icon. Muted teal flow arrows, amber controls, coral fallback, ivory labels. Clean engineering infographic, no logos, no photorealism, no gradients, no tiny text.
```

### `end-to-end-finetuning-workflow.png`

```text
Flat vector workflow map, wide 16:9, dark graphite background. A curated text corpus enters an LLM fine-tuning workshop and separates into continued pretraining, instruction SFT, preference DPO, and LoRA parameter strategy. Each branch emits a named artifact icon: base checkpoint, instruction adapter, preference adapter, LoRA adapter. All converge into evaluation with output quality, cost, and latency gauges. Muted teal arrows, amber active training components, slate frozen base model, ivory labels, coral for evaluation failure. No logos, no photorealism, no gradients, no tiny text.
```

### `combined-notebook-checkpoints.png`

```text
Flat vector checkpoint dependency map, wide 16:9, dark graphite background. Five numbered but unlabeled workflow stages from data preparation through pretraining, SFT, DPO, LoRA, and evaluation. Each stage produces a small checkpoint file icon and arrows show exactly which later stages reuse it. Alternate paths use a clear "exercise" outline and a filled "solution" outline without source code. Muted teal arrows, amber artifacts, slate base model, ivory labels. No logos, no photorealism, no gradients, no tiny text.
```

### `exercise-finetuning-roadmap.png`

```text
Flat vector learner roadmap, wide 16:9, dark graphite background. Six large connected milestones: inspect corpus, tokenize, train objective, configure adapter, compare generations, evaluate. Each milestone has a simple line icon and an empty square completion marker. Use muted teal path, amber current-step highlight, ivory labels, a single coral warning marker beside evaluation. Clean practical course graphic, no logos, no photorealism, no gradients, no tiny text, no code.
```

## Review Rubric

Accept an image only when it passes all of these checks:

1. The mechanism can be explained without relying on text generated inside the image.
2. Shapes, arrows, and short labels remain legible at notebook content width.
3. The visual makes no technical claim that conflicts with its adjacent notebook.
4. It uses the shared palette and contains no provider logo or copyrighted character.
5. It adds information that the notebook's existing plots do not already show.

## Embedding Convention

Use a Markdown image on its own line, immediately after the paragraph that establishes the learner's problem. Follow it with two to four sentences that explain what the reader should notice in the graphic.

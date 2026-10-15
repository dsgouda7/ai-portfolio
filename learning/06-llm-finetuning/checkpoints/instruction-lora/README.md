---
base_model: HuggingFaceTB/SmolLM2-360M-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
  - base_model:adapter:HuggingFaceTB/SmolLM2-360M-Instruct
  - lora
  - transformers
---

# Instruction LoRA Teaching Artifact

This directory contains a LoRA adapter produced for the response-masked instruction-tuning demonstration in [Part 1](../../01-llm-finetuning-data-techniques.ipynb). It is a local teaching artifact, not a promoted or independently evaluated model release.

## Recorded Configuration

The committed `adapter_config.json` records:

- base model family: `HuggingFaceTB/SmolLM2-360M-Instruct`;
- task: causal language modeling;
- rank and scale: $r=8$, $\alpha=16$;
- dropout: `0.05`;
- target modules: `q_proj`, `k_proj`, `v_proj`, and `o_proj`;
- bias: none;
- PEFT version: `0.19.1`.

The adapter weights are stored in `adapter_model.safetensors`. A tokenizer is not bundled in this directory; use the tokenizer from the matching base model.

## Provenance and Evaluation Limits

This artifact does not include an experiment manifest. Its exact immutable base revision, source split hashes, seed, training arguments, runtime, and held-out metrics therefore cannot be reconstructed from this directory alone. Do not use it as release evidence or compare it with another adapter as though the runs were controlled.

Regenerate the adapter through Part 1 when auditable results are required. Use the all-novel GPU practice notebook for chapter-disjoint validation/test manifests across the full corpus.

## Intended Use

Use this adapter only to inspect PEFT structure, local loading, and the notebook's one-sentence continuation exercise. Validate behavior against held-out requests before any downstream use.

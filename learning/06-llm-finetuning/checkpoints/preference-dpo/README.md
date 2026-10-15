---
base_model: HuggingFaceTB/SmolLM2-360M-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
  - base_model:adapter:HuggingFaceTB/SmolLM2-360M-Instruct
  - dpo
  - lora
  - transformers
  - trl
---

# Preference DPO Teaching Artifact

This directory contains a LoRA adapter produced for the preference-optimization demonstration in [Part 1](../../01-llm-finetuning-data-techniques.ipynb). It is a local teaching artifact, not a promoted or independently evaluated model release.

## Recorded Configuration

The committed `adapter_config.json` records:

- base model family: `HuggingFaceTB/SmolLM2-360M-Instruct`;
- task: causal language modeling;
- rank and scale: $r=8$, $\alpha=16$;
- dropout: `0.05`;
- target modules: `q_proj`, `k_proj`, `v_proj`, and `o_proj`;
- bias: none;
- PEFT version: `0.19.1`.

The directory also contains adapter weights and tokenizer/chat-template files used by the teaching run.

## Provenance and Evaluation Limits

This artifact does not include an experiment manifest. The exact immutable base revision, accepted SFT parent, preference-pair hashes, DPO $\beta$, seed, training arguments, runtime, and held-out ranking/retention metrics cannot be reconstructed from this directory alone. The folder name indicates the intended DPO role, but the complete lineage is not auditable from the committed files.

Do not treat this adapter as evidence that generated responses improved. Regenerate the SFT and DPO stages through Part 1 and retain their manifests when a controlled comparison is required.

## Intended Use

Use this artifact only to inspect PEFT/DPO checkpoint structure and local loading. Any behavioral claim requires held-out preference edges, contract-retention checks, and matched generation review from the originating run.

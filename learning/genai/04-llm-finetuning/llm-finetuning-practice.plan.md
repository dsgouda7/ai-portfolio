# LLM Fine-Tuning Practice Notebook Plan

Status: decision-gated plan only. Do not create the notebook until the calibration gates below are satisfied.

## Goal

Create `07-llm-finetuning-practice.ipynb` in this directory. It will use each Riverside novel as an independent fine-tuning lab and show measured before/after behavior. No model will be trained sequentially across novels; every lab starts from a fresh pinned base so one novel cannot contaminate another lab.

## Candidate Models

| Use | Primary CPU candidate | Immutable revision | Parameters |
| --- | --- | --- | ---: |
| Continued pretraining | `HuggingFaceTB/SmolLM2-135M` | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | 134,515,008 |
| SFT and DPO | `HuggingFaceTB/SmolLM2-135M-Instruct` | `12fd25f77366fa6b3b4b768ec3050bf629380bac` | 134,515,008 |
| Fallback reference | `HuggingFaceTB/SmolLM2-360M-Instruct` | `a10cc1512eabd3dde888204e902eca88bddb4951` | 361,821,120 |

Use the 135M models first because the current 360M outputs took roughly 10-12 minutes for ten CPT/SFT steps and about 32 minutes for ten DPO steps on CPU. The smaller model has 37% as many parameters, but measured seconds per step, not parameter count alone, decides suitability.

## Calibration Before Implementation

Create a temporary benchmark script or scratch cell, not the practice notebook, and record:

1. CPU model, logical cores, available RAM, PyTorch thread count, and package versions.
2. Three warm training steps and five measured steps for CPT, LoRA SFT, and DPO.
3. Seconds per step, evaluation seconds, peak process RSS, and checkpoint size.
4. The same fixed sequence lengths and batch size intended for the notebook.
5. One 135M run first; benchmark 360M only if 135M misses a behavior gate.

Reject a configuration if one evidence-mode lab is projected to exceed 20 CPU minutes or peak memory leaves less than 25% system RAM free.

## Adaptive Step Policy

Do not select one arbitrary `max_steps` value for every objective.

1. Evaluate the untouched base before training.
2. Train to step 10 and run the reserved metric.
3. If the gate is not met and there is no retention failure, resume to steps 25, 50, then 100.
4. Stop at the first passing checkpoint.
5. Stop early on retention regression, unstable loss, or memory pressure.
6. Repeat the first passing configuration with a second seed before accepting it for the notebook.

Quick mode may stop at ten steps but must label the result `MECHANISM ONLY`. Evidence mode is the default committed-output path and must report `PASS`, `FAIL`, or `INCONCLUSIVE`.

## Data Contract

For every novel:

- split by complete chapter before constructing examples;
- reserve at least three chapters and approximately 15% of chapters for evaluation;
- hash and record every train and evaluation source file;
- build prompts, SFT targets, and DPO pairs only after the split;
- keep request wording variants disjoint between SFT training and evaluation;
- keep DPO chosen/rejected response lengths equal or report length-normalized metrics;
- write an `experiment-manifest.json` beside every adapter or checkpoint.

## Objective Gates

### Continued Pretraining

- Primary metric: token-weighted perplexity on reserved chapters.
- Pass: at least 5% improvement over the matching base on two consecutive evaluation checkpoints.
- Retention gate: no more than 5% perplexity regression on the fixed general-language set.
- Visible check: show at least four matched base/trained continuations under deterministic decoding. The notebook may claim visible story adaptation only when qualified review marks at least three trained continuations as more manuscript-consistent without copying the reference continuation.
- A perplexity pass without the visible check is a likelihood result, not a claim that open-ended prose visibly improved.

### SFT Specialization

- Primary metric: complete-contract pass rate on reserved contexts and unseen request wording.
- Pass: at least 60% complete passes and at least 25 percentage points over the instruction base.
- Every target must decode to a complete sentence plus exactly one EOS token.
- Compare against the instruction base because SFT here demonstrates specialization, not invention of general instruction following.
- Visible check: print at least eight matched base/SFT outputs. At least six SFT outputs must satisfy the complete contract, and at least two cases must visibly repair a base-model contract failure.

### DPO

- Primary metric: mean held-out preference edge and positive-edge rate.
- Build at least 64 training triplets and 16 held-out triplets from chapter-disjoint contexts.
- Use multiple semantically distinct rejected-response families; do not train every pair against one repeated stalling template.
- Validate that every chosen and rejected response is a complete grammatical sentence. Length matching may not be achieved by cutting a sentence mid-clause or leaving quotation marks unbalanced.
- Ensure every training triplet is consumed at least once before claiming a corpus-level DPO result.
- Ranking pass: positive mean edge and at least 75% positive edges across at least 16 reserved pairs.
- Retention gate: SFT complete-contract pass rate may drop by no more than 10 percentage points.
- Visible check: print matched-seed SFT/DPO generations for at least eight held-out prompts. DPO must change at least half of them, and blinded review must prefer DPO on a majority without reducing contract compliance.
- A margin pass without the visible check proves ranking movement only, not improved generated answers.

## Novel Labs

| Novel | Independent lab | Evidence shown |
| --- | --- | --- |
| *The Weight of Distant Light* | Baseline and continued pretraining | Reserved prose perplexity and general-language retention |
| *The Tidebound Accord* | Response-masked LoRA SFT | Unseen-wording contract pass rate |
| *The Cartographer's Cipher* | DPO after an accepted SFT adapter | Held-out preference edge and SFT retention |
| *The Silk Merchant's Daughter* | Full update versus LoRA on matched SFT data | Same behavior metric first, trainable state and artifact size second |
| *Neural Drift* | Partial freezing versus LoRA | Matched behavior gate and measured CPU time |
| *The Hollow Beneath* | Step-budget and overfitting lab | Training versus reserved loss across 10/25/50/100 steps |
| *The Weight of Tides* | Retention and forgetting lab | Domain improvement versus general-language regression |
| *The Everglades Cipher* | Adapter reconstruction and blind comparison | Manifest verification, reload parity, and randomized output review |

QLoRA remains conceptual on this CPU path unless the selected environment has a verified supported quantized-training backend. Do not present dynamic int8 inference as QLoRA training.

## Notebook Structure

1. Reproducible setup and hardware report.
2. Shared chapter-split, hashing, scoring, and manifest helpers.
3. One self-contained lab per novel.
4. A compact evidence ledger after every lab.
5. Final table containing only objective-aligned metrics, elapsed time, peak memory, trainable parameters, and artifact size.
6. Cleanup cell that releases models between labs.

Do not embed resumable orchestration, promotion, serving, or monitoring stubs in the practice notebook. Those belong in a future production fine-tuning lifecycle notebook.

## Implementation Gate

Implementation may begin only when a calibration report records all of the following:

- selected model id and immutable revision for each objective;
- sequence lengths, LoRA rank/targets, learning rates, and adaptive step ceilings;
- measured seconds per step and peak memory on the target CPU;
- at least one passing two-seed pilot for CPT, SFT, and DPO;
- the visible before/after generation gate for each claimed behavior;
- projected evidence-mode runtime for every lab and the full notebook;
- confirmed chapter-disjoint train/evaluation manifests.

If 135M cannot clear an objective gate by step 100, do not silently switch to longer training. Compare one controlled 360M pilot, inspect data quality and metric sensitivity, then make an explicit model-versus-runtime decision.

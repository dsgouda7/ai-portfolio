# Learning Directory Restructure Plan

> **Status:** Proposed — not yet executed
> **Authoring standard:** `learning/genai/authoring-guide.md`
> **Gold-standard notebook:** `learning/genai/transformers/transformers.ipynb`

This plan converts the asks from the backlog into discrete, independently executable
tasks sized for parallel subagent execution.  Tasks within the same Phase can run
in parallel unless a dependency arrow (→) is shown.

---

## Current State

```
learning/
  data-engineering/          IBM DS course notebooks (SpaceX data; 3 notebooks)
  ml/                        IBM DS course notebooks (housing, Titanic; 2 notebooks)
  genai/
    authoring-guide.md
    conversation-analysis/   ibm_genai_conversation_analyzer.py  [Flask/Gradio app]
    conversational-ai/       ibm_genai_conversational.py         [Flask app]
    encoder-decoder/         encoder_decoder.ipynb               [EMPTY]
    image-captioning/        ibm_genai_image_captioning.py       [Gradio app]
    llm/                     hybrid-search, llm-gateway, rag-evaluation
    llm-tuning/              llm_finetuning_deep_dive.ipynb      [PyTorch]
    rnns/MIT/                TF_Part1_Intro.ipynb, PT_Part1_Intro.ipynb [Keras+PyTorch]
    text-translation/        ibm_genai_translator.py             [Flask app]
    transformers/            transformers.ipynb                  [Keras only]
    voice-assistant/         Full multi-file Flask project
  README.md
```

---

## Decision: data-engineering/ and ml/

**Recommendation: Move to `playground/`.**

Rationale:
- All six notebooks are IBM Data Science course artefacts using IBM-specific datasets
  (SpaceX API, King County housing).  They have no shared running example, no
  intuition-building narrative, and no connection to the GenAI learning arc.
- Applying the authoring-guide treatment to IBM course material would require
  completely rewriting them around a different domain — the IBM data and exercises
  would become cosmetic.
- `playground/` already holds exploratory and third-party material.  These notebooks
  fit that contract exactly.

**Action:** Move `learning/data-engineering/` and `learning/ml/` to
`playground/data-engineering/` and `playground/ml/` respectively.

---

## Pedagogical Ordering for `learning/genai/`

The natural learning arc, numbered for directory renaming:

| New name | Current name | Rationale for position |
|---|---|---|
| `01-rnns/` | `rnns/` | Context before transformers: sequential models, their limits |
| `02-transformers/` | `transformers/` | Gold-standard; the conceptual core |
| `03-encoder-decoder/` | `encoder-decoder/` | Architecture built on transformer blocks |
| `04-llm/` | `llm/` | Applied LLM patterns: search, gateways, evaluation |
| `05-llm-tuning/` | `llm-tuning/` | Adapting pretrained models |

Directories with Python scripts that become notebooks (see Phase 4):

| New name | Current name |
|---|---|
| `06-conversation-analysis/` | `conversation-analysis/` |
| `07-conversational-ai/` | `conversational-ai/` |
| `08-image-captioning/` | `image-captioning/` |
| `09-text-translation/` | `text-translation/` |

**`voice-assistant/`:** Keep as-is; it is a full multi-file project
(`controllers.py`, `model_manager.py`, templates, model cache) not convertible to
a single notebook without losing the architecture.  Rename to `10-voice-assistant/`
for ordering completeness.

---

## Tasks

### Phase 1 — Move exploratory material (prerequisite for Phase 2)

**TASK-1**  `move-data-eng-ml-to-playground`
- Move `learning/data-engineering/` → `playground/data-engineering/`
- Move `learning/ml/` → `playground/ml/`
- Use `git mv` to preserve history
- Update `learning/README.md` to remove references to these directories
- Update `playground/README.md` to document the new additions

---

### Phase 2 — Rename and order genai directories (parallel; prerequisite for Phase 3)

**TASK-2a**  `rename-genai-dirs`  *(depends on TASK-1)*
- Rename directories inside `learning/genai/` using `git mv`:
  - `rnns/` → `01-rnns/`
  - `transformers/` → `02-transformers/`
  - `encoder-decoder/` → `03-encoder-decoder/`
  - `llm/` → `04-llm/`
  - `llm-tuning/` → `05-llm-tuning/`
  - `conversation-analysis/` → `06-conversation-analysis/`
  - `conversational-ai/` → `07-conversational-ai/`
  - `image-captioning/` → `08-image-captioning/`
  - `text-translation/` → `09-text-translation/`
  - `voice-assistant/` → `10-voice-assistant/`
- Update any cross-references in the authoring guide and existing READMEs

**TASK-2b**  `write-genai-readme`  *(can run in parallel with TASK-2a; merge after)*
- Create `learning/genai/README.md` with:
  - One-paragraph description of the full learning arc
  - Table: `| Dir | Topic | What you build | What you can do when done |`
  - Row per numbered directory (01 through 10)
  - Prerequisites column pointing to the prior chapter
  - Cross-link to `authoring-guide.md`

---

### Phase 3 — Notebook audit (fully parallel; run one subagent per notebook)

Each subagent receives the notebook path and the authoring-guide and answers:
- Does the notebook have a single threaded running example?
- Does it have "predict before you run" cells?
- Does it have "What just happened" reflection cells?
- Does it use section-banner code comments?
- Are print statements pedagogical (→ arrows, conclusions)?
- What framework: Keras / PyTorch / neither?

Output: a one-page audit summary per notebook appended to
`learning/genai/notebook-audit.md`.

**TASK-3-01a** Audit `01-rnns/MIT/TF_Part1_Intro.ipynb`
**TASK-3-01b** Audit `01-rnns/MIT/PT_Part1_Intro.ipynb`
**TASK-3-02**  Audit `02-transformers/transformers.ipynb`
**TASK-3-03**  Audit `03-encoder-decoder/encoder_decoder.ipynb`  (currently empty — note this)
**TASK-3-04a** Audit `04-llm/hybrid-search.ipynb`
**TASK-3-04b** Audit `04-llm/llm-gateway.ipynb`
**TASK-3-04c** Audit `04-llm/rag-evaluation.ipynb`
**TASK-3-05**  Audit `05-llm-tuning/llm_finetuning_deep_dive.ipynb`

---

### Phase 4 — Keras → PyTorch conversions (parallel per notebook; after Phase 2)

For every notebook that uses Keras/TensorFlow, create a parallel PyTorch
implementation.  Rename the original to `<stem>-keras.ipynb`; the new PyTorch
version takes the original stem name.

| Original path | Keras rename | New PyTorch notebook |
|---|---|---|
| `02-transformers/transformers.ipynb` | `transformers-keras.ipynb` | `transformers.ipynb` |
| `01-rnns/MIT/TF_Part1_Intro.ipynb` | `TF_Part1_Intro-keras.ipynb` | `TF_Part1_Intro.ipynb` (PyTorch) |

Note: `01-rnns/MIT/PT_Part1_Intro.ipynb` already has PyTorch content but also imports
Keras — audit (TASK-3-01b) must determine if it is truly PyTorch-primary before
deciding whether it also needs renaming.

**TASK-4a** `pytorch-transformers`  *(depends on TASK-2a)*
- Read `02-transformers/transformers-keras.ipynb` (full 97-cell notebook)
- Produce `02-transformers/transformers.ipynb` with identical pedagogical structure:
  same parts, same running example ("the cat sat on the mat"), same intuition flow
- Replace every TF/Keras call with an equivalent `torch` / `torch.nn` call
- Variable names mirror the math identically to the Keras version
- All code cells must execute cleanly in the venv Python (no API keys)
- Apply authoring-guide conventions throughout

**TASK-4b** `pytorch-rnns-tf`  *(depends on TASK-2a)*
- Read `01-rnns/MIT/TF_Part1_Intro-keras.ipynb`
- Produce a PyTorch version `TF_Part1_Intro.ipynb` following the same structure
- Use same running example and same pedagogical flow as the original

---

### Phase 5 — Script-to-notebook conversions (parallel per script; after Phase 2)

Each script is a standalone HuggingFace-based demo (no API keys needed if models
are locally cached).  Convert to a notebook applying the authoring guide:
- Part 1 — motivation and the problem the model solves
- Part 2 — model and architecture overview (inline formula where relevant)
- Part 3 — implementation walkthrough (cell by cell)
- Part 4 — live demo cell
- Part 5 — "Your turn" exercise cell

Both PyTorch and Keras variants should be created if the underlying model framework
supports both.  For these scripts, all use HuggingFace `transformers` (PyTorch
backend) — a Keras variant is only needed where a `TFAuto*` equivalent exists.

| Script | Dir after rename | Frameworks | Notes |
|---|---|---|---|
| `ibm_genai_conversation_analyzer.py` | `06-conversation-analysis/` | PyTorch (FLAN-T5) | Create `conversation_analyzer.ipynb` |
| `ibm_genai_conversational.py` | `07-conversational-ai/` | PyTorch (Qwen 2.5-1.5B) | Create `conversational_ai.ipynb` |
| `ibm_genai_image_captioning.py` | `08-image-captioning/` | PyTorch (BLIP-2) | Create `image_captioning.ipynb` |
| `ibm_genai_translator.py` | `09-text-translation/` | PyTorch (Whisper + Helsinki) | Create `text_translation.ipynb` |

**TASK-5a** `notebook-conversation-analysis`
**TASK-5b** `notebook-conversational-ai`
**TASK-5c** `notebook-image-captioning`
**TASK-5d** `notebook-text-translation`

---

### Phase 6 — Cell execution, error fixing, output clearing (parallel per notebook; after Phase 4+5)

For every notebook under `learning/genai/` (after renaming):
1. Open in the venv kernel
2. Run all cells top-to-bottom
3. Fix any errors (import errors, missing packages, deprecated APIs)
4. Clear all cell outputs (`nbconvert --clear-output` or equivalent)
5. Commit the cleaned notebook

One subagent per notebook.  Subagent receives: notebook path, venv Python path
(`learning/genai/<dir>/.venv/Scripts/python.exe` if local, else repo root `.venv`).

**TASK-6-01a** Execute + fix + clear `01-rnns/MIT/TF_Part1_Intro-keras.ipynb`
**TASK-6-01b** Execute + fix + clear `01-rnns/MIT/TF_Part1_Intro.ipynb`  (new PyTorch)
**TASK-6-01c** Execute + fix + clear `01-rnns/MIT/PT_Part1_Intro.ipynb`
**TASK-6-02a** Execute + fix + clear `02-transformers/transformers-keras.ipynb`
**TASK-6-02b** Execute + fix + clear `02-transformers/transformers.ipynb`  (new PyTorch)
**TASK-6-03**  Execute + fix + clear `03-encoder-decoder/encoder_decoder.ipynb`  (after Phase 5 if rebuilt)
**TASK-6-04a** Execute + fix + clear `04-llm/hybrid-search.ipynb`
**TASK-6-04b** Execute + fix + clear `04-llm/llm-gateway.ipynb`
**TASK-6-04c** Execute + fix + clear `04-llm/rag-evaluation.ipynb`
**TASK-6-05**  Execute + fix + clear `05-llm-tuning/llm_finetuning_deep_dive.ipynb`
**TASK-6-06**  Execute + fix + clear `06-conversation-analysis/conversation_analyzer.ipynb`
**TASK-6-07**  Execute + fix + clear `07-conversational-ai/conversational_ai.ipynb`
**TASK-6-08**  Execute + fix + clear `08-image-captioning/image_captioning.ipynb`
**TASK-6-09**  Execute + fix + clear `09-text-translation/text_translation.ipynb`

---

### Phase 7 — Final commit and push

**TASK-7**  `final-commit`  *(depends on all Phase 6 tasks)*
- `git add -A`
- `git commit -m "restructure(learning): numbered genai dirs, pytorch variants, script-to-notebooks, move data-eng+ml to playground"`
- `git push`

---

## Execution order summary

```
TASK-1
  └─→ TASK-2a (parallel with TASK-2b)
      TASK-2b
        └─→ [Phase 3: TASK-3-* all parallel]
            [Phase 4: TASK-4a, TASK-4b in parallel]
            [Phase 5: TASK-5a, TASK-5b, TASK-5c, TASK-5d in parallel]
              └─→ [Phase 6: all TASK-6-* in parallel]
                    └─→ TASK-7
```

---

## Key reference paths

| Resource | Path |
|---|---|
| Authoring guide | `learning/genai/authoring-guide.md` |
| Gold-standard notebook | `learning/genai/transformers/transformers.ipynb` |
| venv Python | `c:\repos\ai-portfolio\.venv\Scripts\python.exe` |
| Transformers venv | `learning/genai/transformers/.venv/Scripts/python.exe` |
| LLM-tuning venv | `learning/genai/llm-tuning/.venv/Scripts/python.exe` |

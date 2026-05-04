# Contributing

## What This Repo Sounds Like

The notes are written for working engineers, not for marketing copy or academic prose. Keep the tone:

- practical and specific
- historically grounded when introducing concepts
- opinionated when it helps a reader choose between approaches
- clear about tradeoffs, costs, and failure modes
- dense with meaning, but still easy to scan

Prefer direct explanations over hype. When a chapter or supplement explains a concept, it should answer: what it is, why it exists, what problem it solves, and when it is the wrong tool.

## Editing Notes

When you change chapter content:

- keep the existing voice and reading level of the surrounding note
- preserve the chapter's historical arc and running example
- update linked READMEs and supplements when the structure changes
- make sure references to generated images or animations still match the files on disk

If you are revising a chapter with a companion `_Supplement.md` or `_Supplement.ipynb`, keep the main chapter as the primary narrative and use the supplement for depth, edge cases, or GPU-only material.

## Generated Images And Animations

A lot of visuals in this repo are generated from Python scripts checked into the chapter folders.

Typical locations:

- `notes/<Track>/<Chapter>/gen_scripts/` for the generator scripts
- `notes/<Track>/<Chapter>/img/` for generated PNG, GIF, or other image assets

There are two common patterns:

- chapter illustrations generated with `matplotlib` and saved as PNGs
- flow or process animations generated through shared helpers and saved as PNG + GIF pairs

Some scripts use `matplotlib.use("Agg")` so they can run headless. Others call shared renderers or helper modules in the repo.

## How To Regenerate Assets Locally

Use the repo root virtual environment if one exists, then run the chapter generator directly with Python.

Examples:

```powershell
# Windows
.\.venv\Scripts\Activate.ps1
python notes\04-MultimodalAI\VisionTransformers\gen_scripts\gen_vision_transformers.py
python notes\04-MultimodalAI\VisionTransformers\gen_scripts\gen_vision_transformers_flow.py
```

```bash
# macOS / Linux
source .venv/bin/activate
python notes/05-MultimodalAI/VisionTransformers/gen_scripts/gen_vision_transformers.py
python notes/05-MultimodalAI/VisionTransformers/gen_scripts/gen_vision_transformers_flow.py
```

For other chapters, use the same pattern: run the chapter-local `gen_scripts/*.py` file that owns the asset you want to change.

If you edit a generator script:

- rerun it locally
- verify the output landed in the chapter's `img/` folder
- commit both the script change and the regenerated asset when the generated file is part of the source of truth

## Local Workflow

1. Make the note or generator change.
2. Rerun the relevant chapter script.
3. Review the generated image or animation in the chapter `img/` folder.
4. Update the surrounding note text if the visual or caption changed.
5. Keep the diff minimal and avoid unrelated reformatting.

## What To Avoid

- Do not rewrite the notes into a generic tutorial voice.
- Do not change the historical ordering or chapter structure unless the chapter itself needs it.
- Do not hand-edit generated images if the script is the source of truth.
- Do not add new generated assets without also updating the note that explains them.

## Questions To Ask Before Large Changes

If a change affects chapter structure, visual assets, or generated notebooks, check whether the chapter already has:

- a `README.md` that explains the chapter arc
- a `notebook.ipynb_solution.ipynb` (reference) or `notebook.ipynb_exercise.ipynb` (practice) or `notebook_supplement.ipynb_solution.ipynb` (reference) or `notebook_supplement.ipynb_exercise.ipynb` (practice)
- a `gen_scripts/` folder that owns the visuals
- links from the parent track README

If not, update the surrounding documentation together with the asset change.
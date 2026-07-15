"""Runner: combines Part 1 + Part 2 builder scripts and saves the notebook."""

import json
import pathlib

NB_PATH = pathlib.Path(
    r"c:\repos\ai-portfolio\learning\genai\02-transformers\transformers.ipynb"
)

# Execute Part 1 (defines helpers + cells list with cells 0..54ish)
exec(
    open(
        r"c:\repos\ai-portfolio\scripts\build_pytorch_transformer_nb.py",
        encoding="utf-8",
    ).read()
)

# Execute Part 2 (appends cells 55..96 to existing `cells` list)
exec(
    open(
        r"c:\repos\ai-portfolio\scripts\build_pytorch_transformer_nb_part2.py",
        encoding="utf-8",
    ).read()
)

print(f"\nTotal cells: {len(cells)}")

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0",
        },
    },
    "cells": cells,
}

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
size = NB_PATH.stat().st_size
print(f"Written {len(cells)} cells, {size:,} bytes -> {NB_PATH}")

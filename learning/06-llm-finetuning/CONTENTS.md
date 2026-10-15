# Contents: LLM Fine-Tuning

Decide what to teach an existing model, where its changes live, and which evidence supports a release.

## Required route

- [LLM Fine-Tuning, Part 1 of 4: What Should the Model Learn?](01-llm-finetuning-data-techniques.ipynb) — Choose what an existing model should learn by changing the examples and the behavior rewarded during practice.
- [LLM Fine-Tuning Data Techniques: Handwritten Theory Notes](01-llm-finetuning-data-techniques-theory.md) — Review how continuation, instruction examples, and preference comparisons teach different behavior.
- [LLM Fine-Tuning, Part 2 of 4: What Must Riverside Pay to Change?](02-llm-finetuning-parameter-techniques.ipynb) — Compare changing all weights with changing a small added correction and account for the memory each approach needs.
- [LLM Fine-Tuning Parameter Techniques: Handwritten Theory Notes](02-llm-finetuning-parameter-techniques-theory.md) — Review where an adaptation is stored and why a smaller update does not remove every memory cost.
- [LLM Fine-Tuning, Part 3 of 4: Evaluate First, Compare Second, Decide Last](03-llm-finetuning-comparison-and-decision.ipynb) — Compare isolated candidates using independent quality, safety, speed, and source-history evidence before deciding whether to release.
- [LLM Fine-Tuning Comparison and Decision: Handwritten Theory Notes](03-llm-finetuning-comparison-and-decision-theory.md) — Review why a favorable score cannot replace independent checks and traceable candidate identity.
- [LLM Fine-Tuning, Part 4 of 4: Staged CUDA Evidence](04-llm-finetuning-practice.ipynb) — Run staged graphics-card training and evaluation checks across the Riverside novels while retaining evidence for review.
- [LLM Fine-Tuning Practice: Evidence Before Claims](04-llm-finetuning-practice-theory.md) — Review the staged training evidence and hardware boundaries without treating an incomplete run as a release success.

## Optional

No optional lessons are listed in this directory.

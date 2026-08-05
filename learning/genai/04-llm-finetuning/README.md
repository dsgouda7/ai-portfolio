# LLM Fine-Tuning

Riverside House changes model behavior only when an observed failure justifies a new training signal.

1. [What Should the Model Learn?](01-llm-finetuning-data-techniques.ipynb) covers continued pretraining, response-masked SFT, DPO, and objective-aligned evidence.
2. [Where Should the Update Live?](02-llm-finetuning-parameter-techniques.ipynb) compares full fine-tuning, partial freezing, LoRA, and the QLoRA path.
3. [What Does the Evidence Support?](03-llm-finetuning-comparison-and-decision.ipynb) separates visible behavior from release evidence and makes workload-aware decisions.
4. [GPU Practice: Fine-Tune and Evaluate Every Riverside Novel](04-llm-finetuning-practice.ipynb) runs independent, chapter-disjoint LoRA experiments across all eight novels with validation-selected checkpoints and untouched test sets.

The notebooks share the committed `content/` corpus, generated `data/`, calibration artifacts, and local teaching checkpoints in this directory. Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS; both use the environment defined in [`../_llm-shared/`](../_llm-shared/README.md).

Parts 1-3 select a CPU or CUDA profile. Part 4 is intentionally CUDA-only and stops immediately when PyTorch cannot see a compatible GPU.

# LLM Fine-Tuning

Riverside House changes model behavior only when an observed failure justifies a new training signal.

1. [What Should the Model Learn?](01-llm-finetuning-data-techniques.ipynb) covers continued pretraining, response-masked SFT, DPO, and objective-aligned evidence.
2. [Where Should the Update Live?](02-llm-finetuning-parameter-techniques.ipynb) compares full fine-tuning, partial freezing, LoRA, and the QLoRA path.
3. [What Does the Evidence Support?](03-llm-finetuning-comparison-and-decision.ipynb) separates visible behavior from release evidence and makes workload-aware decisions.

The notebooks share the committed `content/` corpus, generated `data/`, calibration artifacts, and local teaching checkpoints in this directory. Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS; both use the environment defined in [`../_llm-shared/`](../_llm-shared/README.md).

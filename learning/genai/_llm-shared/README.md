# Shared Applied-LLM Environment

This environment serves the sibling chapters:

- [`04-llm-finetuning`](../04-llm-finetuning/)
- [`05-rag`](../05-rag/)
- [`06-llm-gateway`](../06-llm-gateway/)

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS. Both register the `genai-llm` kernel used by the notebooks. The environment is shared because the three chapters reuse PyTorch, Transformers, plotting, retrieval, and gateway dependencies while remaining separate pedagogical units.

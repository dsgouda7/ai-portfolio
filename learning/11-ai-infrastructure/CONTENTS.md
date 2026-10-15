# Contents: AI Infrastructure

Locate memory, computation, queue, and ownership limits before changing a training or serving system.

## Required route

- [GPU Hardware Foundations](01-gpu-hardware/gpu-hardware-foundations.ipynb) — Compare arithmetic speed with memory movement to explain which hardware limit matters for a workload.
- [Mixed Precision and Memory](02-mixed-precision/mixed-precision-and-memory.ipynb) — Account for model memory and compare number formats before deciding what fits on a graphics card.
- [PyTorch Profiling](03-profiling/pytorch-profiling.ipynb) — Read an operation timeline to locate the real training bottleneck before trying a speedup.
- [FlashAttention Internals](04-flash-attention/flash-attention-internals.ipynb) — Process context comparisons in small blocks instead of repeatedly moving one large score table through slow memory.
- [Distributed Training](05-distributed-training/distributed-training.ipynb) — Compare ways to divide examples, model state, and computation across devices while accounting for communication.
- [Quantization in Depth](06-quantization/quantization-in-depth.ipynb) — Store model numbers with fewer bits and measure the accuracy and memory trade-offs of the compressed artifact.
- [Inference Systems](07-inference-systems/inference-systems.ipynb) — Reuse previous token calculations and share serving work across requests to improve generation capacity.
- [Custom Kernels with Triton](08-triton-kernels/triton-kernels.ipynb) — Write small graphics-card programs and measure how splitting work into blocks or combining steps changes memory movement and execution cost.
- [Systems Composition](09-systems-composition/README.md) — Find which service boundary prevents added replicas from improving long waits.
- [Azure Operational LLM Serving](10-azure-operational-llm-serving/README.md) — Test a bounded local serving contract before mapping it to cloud services.

## Optional

No optional lessons are listed in this directory.

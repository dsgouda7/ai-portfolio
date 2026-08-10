# LLM Evaluation

This four-notebook sequence covers reference and semantic metrics, LLM-as-judge and safety evaluation, hallucination detection, and confidence calibration.

1. [Automated Metrics and Benchmarks](01-llm-evaluation-metrics-and-benchmarks.ipynb) · [Theory notes](01-llm-evaluation-metrics-and-benchmarks-theory.md)
2. [LLM-as-Judge, Safety, and Eval Pipeline](02-llm-as-judge-safety-and-pipeline.ipynb) · [Theory notes](02-llm-as-judge-safety-and-pipeline-theory.md)
3. [Hallucination Detection](03-hallucination-detection.ipynb) · [Theory notes](03-hallucination-detection-theory.md)
4. [Calibration and Confidence](04-calibration-and-confidence.ipynb) · [Theory notes](04-calibration-and-confidence-theory.md)

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS. Either script creates or reuses this chapter's `.venv`, installs the adjacent `requirements.txt`, downloads the required NLTK data, registers the chapter-unique `genai-11-llm-evaluation` Jupyter kernel, and assigns it to all four notebooks.

## Continue Into Operations

This chapter owns evaluator design, metric limits, hallucination checks, safety evaluation, and
calibration. Continue with:

- [AI Engineer: Prompt Release and Experimentation](../../role-based-tracks/ai-engineer/02-prompt-release-and-experimentation/README.md)
	for paired comparison, critical-slice gates, uncertainty, shadow design, and rollback evidence;
- [AI Engineer: Production Feedback and Drift](../../role-based-tracks/ai-engineer/05-production-feedback-and-drift/README.md)
	for privacy-safe monitoring, reviewed failure clusters, and evaluation-set updates;
- [Riverside Evaluation Assets](../../../projects/riverside-ai-platform/evaluations/README.md) and
	[evaluation strategy](../../../projects/riverside-ai-platform/docs/evaluation-strategy.md) for
	versioned production-oriented gate inputs and evidence policy.

The AI Engineer and FDE operational learning notebooks were executed successfully locally and then
cleared. Riverside evaluation source assets exist, and its non-cloud suite passed 142 tests with 5
cloud tests deselected; model, endpoint, rollout, live Azure evaluation results, and production
readiness remain **live-unvalidated**.

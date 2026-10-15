# Contents: LLM Evaluation

Build independent checks for quality, safety, unsupported claims, and decisions to decline an answer.

## Required route

- [LLM Evaluation 1 of 4: What Can Each Metric Observe?](01-llm-evaluation-metrics-and-benchmarks.ipynb) — Compare wording scores, model-assigned ratings, and how unexpected text looks to a model without treating any reading as correctness.
- [LLM Evaluation Metrics and Benchmarks: Handwritten Theory Notes](01-llm-evaluation-metrics-and-benchmarks-theory.md) — Review why quality measures disagree and why that disagreement is evidence to inspect.
- [LLM Evaluation 2 of 4: Trust the Judge and the Release Pipeline](02-llm-as-judge-safety-and-pipeline.ipynb) — Test judging rules, swap answer order, and combine independent safety checks into a repeatable release review.
- [LLM-as-Judge, Safety, and Evaluation Pipelines: Intuition Notes](02-llm-as-judge-safety-and-pipeline-theory.md) — Review how to check a model-based judge before using its scores in a release pipeline.
- [LLM Evaluation 3 of 4: Localize Unsupported Claims](03-hallucination-detection.ipynb) — Split an answer into checkable claims and choose review thresholds from the costs of missed and false warnings.
- [Hallucination Detection and Factual Grounding](03-hallucination-detection-theory.md) — Review how to locate unsupported claims instead of giving the whole answer one unexplained score.
- [LLM Evaluation 4 of 4: Turn Confidence into Abstention Policy](04-calibration-and-confidence.ipynb) — Compare stated confidence with observed correctness and decide when declining to answer reduces risk.
- [Calibration and Confidence: Handwritten Theory Notes](04-calibration-and-confidence-theory.md) — Review why confidence adjustment and independent risk gates solve different problems.

## Optional

No optional lessons are listed in this directory.

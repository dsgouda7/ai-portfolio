# Contents: LLM Evaluation

Follow one problem: a convincing answer can still change a fact. Check what the score,
grader, evidence and confidence each establish before deciding what may be returned.

## Read in Order

1. [What Can a Score Really Tell Us?](01-llm-evaluation-metrics-and-benchmarks.ipynb) —
   Compare word matches, contextual similarity and expected wording, then inspect actual
   CPU-generated answers. [Theory notes](01-llm-evaluation-metrics-and-benchmarks-theory.md).
2. [Who Checks the Judge?](02-llm-as-judge-safety-and-pipeline.ipynb) —
   Make grading questions explicit, swap positions, inspect declared reviewer labels and
   keep safety separate. [Theory notes](02-llm-as-judge-safety-and-pipeline-theory.md).
3. [Which Claim Does the Passage Support?](03-hallucination-detection.ipynb) —
   Find specific review questions and see why crude splitting and word rules are not
   a validated detector. [Theory notes](03-hallucination-detection-theory.md).
4. [Does 80% Confidence Mean 80% Right?](04-calibration-and-confidence.ipynb) —
   Compare confidence with outcomes, keep a failed calibration attempt visible, and
   reject empty returned sets as zero-error success.
   [Theory notes](04-calibration-and-confidence-theory.md).

Each notebook ends with a **Concept recap** and short code sketches.
Part 1 runs real pretrained models on CPU; Parts 2-4 use declared teaching examples.
The [chapter guide](README.md) explains setup, retained results, image provenance and the
handoff to the gateway.

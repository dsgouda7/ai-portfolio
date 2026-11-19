# Stage 08: LLM Evaluation

**Contents:** [one-line map of this directory](CONTENTS.md)

**Previous:** [Stage 07 · RAG](../07-rag/README.md)

**Current:** Stage 08 · LLM Evaluation

**Next:** [Stage 09 · LLM Gateway](../09-llm-gateway/README.md)

An answer can sound helpful, repeat the source's names, and still change what happened.
This stage asks what each score actually checks, who checks the grader, which claim needs
evidence, and whether a confidence number justifies returning an answer.

Only Part 1 runs a real pretrained answer model. Parts 2-4 use small declared examples and
rules to open the evaluation mechanics; they do not call a live LLM judge or validate a
production hallucination detector.

## Keep the Evidence from the RAG Lesson

Bring separate answers to these questions, not just one combined RAG score:

| Evidence link | Stage 08 responsibility |
| --- | --- |
| Did retrieval find the needed passage? | Distinguish missing evidence from a generator failing to use it |
| Does that passage support the answer? | Check particular claims rather than only familiar wording |
| Where did each claim come from? | Keep the claim beside its source and review reason |
| What does the confidence score mean? | Compare scores with outcomes before using a cutoff to answer, decline or review |

For a failed answer, try supplying the known right passage while holding the generator fixed.
If that repairs it, retrieval is a likely place to investigate. If not, inspect how the answer
or evaluator used the passage. This is a diagnostic experiment, not a feature available at runtime.

## Read in Order

| Chapter | What we try | What the result can tell us |
| --- | --- | --- |
| [What Can a Score Really Tell Us?](01-llm-evaluation-metrics-and-benchmarks.ipynb) · [Theory notes](01-llm-evaluation-metrics-and-benchmarks-theory.md) | Open one ROUGE calculation, compare related metrics, then score real CPU-generated answers | Matching words, contextual closeness and predictable wording are different from factual support |
| [Who Checks the Judge?](02-llm-as-judge-safety-and-pipeline.ipynb) · [Theory notes](02-llm-as-judge-safety-and-pipeline-theory.md) | Write grading questions, swap answer order, compare declared labels and check refusal behavior | A passing small control does not validate a judge; failed safety cases still block the example release |
| [Which Claim Does the Passage Support?](03-hallucination-detection.ipynb) · [Theory notes](03-hallucination-detection-theory.md) | Try a crude splitter, inspect names/relations and vary a review cutoff | The workflow can focus review without earning a claim of reliable hallucination detection |
| [Does 80% Confidence Mean 80% Right?](04-calibration-and-confidence.ipynb) · [Theory notes](04-calibration-and-confidence-theory.md) | Group scores by outcomes, try an adjustment and inspect errors against coverage | Calibration can worsen; returning nothing is not zero-risk success or an approved serving policy |

Part 1 starts with two answers to the same pendant question, then asks what each
score actually noticed. Short reading guides explain the matching grid, word-grouping
toy, surprise scores and real-model tables before drawing conclusions. Its final
exercise changes one evaluation answer while keeping the policy miss visible.

Each notebook ends with an abstractly titled **Concept recap**, combining short explanations
and code sketches. The notes use the same examples without repeating the full implementation.

## What Was Actually Tested?

| Evidence | Boundary |
| --- | --- |
| Part 1's retained CPU answers and metric values | Local measured results on four visible teaching questions, not a sealed quality benchmark |
| Parts 2-4's deterministic tables and checks | Local fixtures, not a human annotation study, live judge evaluation or deployed confidence model |
| Concept diagrams and plots | Explanations of the mechanism; the current code tables determine this fixture's results |

The default 0.04 judge bonus changes no winners; the stronger controlled probe does.
Temperature scaling worsens the small second-half ECE. The default 20% risk ceiling has no
qualifying positive-coverage row on its grid. These are findings to explain, not failures to
hide with more reassuring prose.
The older threshold-sweep GIF remains in the image directory for preservation, but is no
longer presented in Part 4: it shows a different twelve-row example and a fallback cutoff
that is not a valid replacement for the explicit no-policy result.

## What the Gateway May Act On

Stage 09 may enforce a decision only when the matching, versioned evidence justifies it:

| Evidence we would need | Control it can justify | What happens when evidence is missing or fails |
| --- | --- | --- |
| The request's task/group passes its release rule | Which deployments are allowed for that request | A failing or unmeasured group is not approved |
| The required safety checks pass | A hard safety gate | Quality averages cannot cancel a safety failure |
| Claims have adequate source support and attribution | Whether an answer can be returned with its evidence | Decline or send unresolved claims to review |
| A calibrated score meets an independently checked risk/coverage policy | Answer, abstain or escalate | No qualifying policy means hold, not an invented threshold |
| Judge disagreement is below its reviewed limit | Whether an automated verdict can be acted on | Uncertain evaluations need review, not automatic authority |
| A version still passes its regression/drift rules | Whether to keep using or roll back the release | Retain the failed evidence version and take the declared action |

Latency and cost help choose **among approved options**. They do not repair failed quality,
safety, support or confidence evidence. Part 4's independent routing illustration is not
an approved output of its failed default operating-point search.

The [shared helper](evaluation_shared.py) contains the four practice cases, repeatable seeds
and simple word normalization. The real metric runner and visual renderer keep their
separate measured provenance.

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS. Either script creates or reuses this
chapter's `.venv`, installs the adjacent `requirements.txt`, registers the chapter-unique
`learning-08-llm-evaluation` Jupyter kernel, and assigns it to all four notebooks.

## Real CPU Metric Run

Part 1 checks both the public Hugging Face model/cache and saved Stage 06 HF checkpoints
or adapters, then **explicitly selects** its model. It does not silently replace a broken
local artifact. A full local checkpoint needs its tokenizer; an adapter needs its declared
base. Stage 05's custom training packages require their own loader and are not assumed
to be compatible Hugging Face checkpoints.

The default is pinned **SmolLM2-135M-Instruct**, CPU float32, four threads and greedy generation.
A separate pinned **DistilBERT-base-uncased** encoder supplies real BERTScore.
First use downloads missing public model files and English WordNet to user caches; inference
and scoring stay local. No GPU, account, fine-tuning, remote code or paid API is required.

The [measured run record](results/metrics-cpu-smollm2-135m.json) retains all four generated
answers, references, per-item ROUGE-1/2/L P/R/F1, BLEU-4, METEOR, BERTScore P/R/F1 and
context-conditioned fixed-reference perplexity, plus revisions, scorer settings and provenance.
ROUGE-S and BLEU-1/2/3 are explained in the evolution tables, not measured.
The manual word-mapping toy is explicitly not BERTScore.

Reproduce the record from this directory after setup:

```powershell
.\.venv\Scripts\python.exe evaluation_metrics.py --output results\metrics-cpu-smollm2-135m.json
```

The notebook runs the same components without overwriting the retained record.
The [library runner](evaluation_metrics.py) keeps scorer calls and variant-switch comments
inspectable. [Focused checks](test_evaluation_metrics.py) verify manual/library agreement,
local-source boundaries, reference-only loss and the saved score contract.
Run `python -m unittest test_evaluation_metrics` in this directory using the chapter environment.

These are **Local measured** results on four visible teaching questions, not a sealed quality
benchmark, an adapter comparison or production evidence. Low reference perplexity and high
similarity coexist with an invented award and a changed policy condition in the actual answers.
Part 3's award example therefore treats "won" versus "longlisted" as insufficient support,
not proof of a contradiction.

## Visuals and Concept Recap

Two additional checks sit beside the existing metric explanations:

- [Probability, loss and perplexity](01-llm-evaluation-metrics-and-benchmarks.ipynb#loss-probability-check)
  uses hand-picked one-token probabilities and library-calculated costs. Loss can exceed one;
  perplexity is not runtime efficiency.
- [One versus two accepted reference wordings](01-llm-evaluation-metrics-and-benchmarks.ipynb#multiple-reference-check)
  keeps the actual generated pendant answer fixed and uses the two prewritten fixture answers.
  It changes the reference protocol, not the model or the original saved report.

The notebook ends with a [Concept recap](01-llm-evaluation-metrics-and-benchmarks.ipynb#concept-recap):
short explanations and code sketches for matching, precision/recall/F1, BLEU/METEOR,
contextual similarity, and reference perplexity.

Five illustrations sit beside the concepts rather than in a detached gallery:

| Visual | What to watch |
| --- | --- |
| [ROUGE choices](images/01-metrics-rouge.png) | One inserted word breaks an adjacent pair without breaking the ordered match |
| [BLEU safeguards](images/01-metrics-bleu.png) | Repeated words cannot claim unlimited matches; a tiny answer needs a length penalty |
| [METEOR alignment](images/01-metrics-meteor.png) | Related words can align, while scattered matches form more chunks |
| [BERTScore matching](images/01-metrics-bertscore.png) | Actual contextual-token similarities and each answer token's strongest reference match |
| [Perplexity](images/01-metrics-perplexity.png) | Actual reference-token probabilities become losses before averaging |

[MAI prompts](images/01-metric-illustration-prompts.json) define the layouts.
The [renderer](metric_visuals.py) corrects generated token labels/ribbon endpoints and inserts
exact numerical panels from [checked visual evidence](images/01-metric-visual-evidence.json).
The heatmap uses the pinned DistilBERT encoder; token losses use the pinned SmolLM2 prompt
and agree with the retained CPU run. Constructed word-match examples are labeled separately
from model measurements. Original `-mai.png` layouts are retained for reconstruction, not
embedded as the final lesson images.

To remeasure and rebuild the final overlays after reviewing the model/evidence changes:

```powershell
python metric_visuals.py --measure
python metric_visuals.py
```

Only intentional artwork replacement needs the existing
[MAI generator](../03-language-model-foundations/03-language-models/generate_illustrations.py)
with this chapter's prompt manifest. That can incur image-generation charges.
After replacing images, refresh their matching notebook attachments as well.
Normal notebook execution displays the embedded images; it never calls MAI.

## Continue Into Operations

This chapter owns evaluator design, metric limits, hallucination checks, safety evaluation, and
calibration. Continue with:

- [AI Engineer: Prompt Release and Experimentation](../12-role-based-tracks/ai-engineer/02-prompt-release-and-experimentation/README.md)
	for paired comparison, critical-slice gates, uncertainty, shadow design, and rollback evidence;
- [AI Engineer: Production Feedback and Drift](../12-role-based-tracks/ai-engineer/05-production-feedback-and-drift/README.md)
	for privacy-safe monitoring, reviewed failure clusters, and evaluation-set updates;
- [Riverside Evaluation Assets](../../projects/riverside-ai-platform/evaluations/README.md) and
	[evaluation strategy](../../projects/riverside-ai-platform/docs/evaluation-strategy.md) for
	versioned production-oriented gate inputs and evidence policy.

Previously recorded operational checks report successful local execution of the role-track
notebooks and 142 non-cloud Riverside tests, with 5 cloud tests deselected. Those checks concern
the operational assets, not an LLM's quality. They do not supply a live Azure model, endpoint,
rollout or production-readiness claim; those remain **live-unvalidated**.

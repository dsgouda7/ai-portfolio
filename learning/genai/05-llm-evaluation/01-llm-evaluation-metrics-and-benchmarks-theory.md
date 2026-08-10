# LLM Evaluation Metrics and Benchmarks: Handwritten Theory Notes

These notes turn the notebook's Riverside House example into a compact evaluation framework. They are written to be copied verbatim by hand. The central claim is simple: **LLM quality is multidimensional, so no single metric can certify a model.** A useful evaluation combines evidence about wording, meaning, model confidence, task capability, and known failure cases.

![Evaluation taxonomy from questions to evidence](images/01-llm-evaluation-metrics-and-benchmarks-theory-01.png)

## 1. Evaluation taxonomy

Start with the question that a stakeholder actually asks. "Does the answer resemble a reference?" is different from "Is it factually correct?" or "Has the model learned our domain?" The metric must match the claim.

**Reference-based string metrics** compare a candidate answer with one or more human references. BLEU measures clipped n-gram precision and applies a brevity penalty. ROUGE-L measures the longest common subsequence, so matching words may be separated but must remain in order. METEOR adds unigram alignment, WordNet synonym matching, and a fragmentation penalty. These metrics are reproducible and cheap, but they inherit the wording of the reference.

**Model-based semantic metrics** use a pretrained model as part of the measurement. BERTScore embeds candidate and reference tokens contextually, independently matches each token to its most similar token on the other side, and reports precision, recall, and F1. This is greedy maximum similarity, not a one-to-one alignment. It can recognize that "reclaim" and "recover" are close in meaning even when string metrics cannot. The price is greater compute, model dependence, and sensitivity to the evaluator model. LLM-as-judge is another model-based family: a stronger model follows a rubric, scores or compares answers, and may explain its decision. The notebook reserves the full judge protocol for Part 2 because judge bias, order effects, and rubric design need separate controls.

**Reference-free model metrics** ask how a language model scores a sequence without using a gold answer. Perplexity is the exponential of mean token negative log-likelihood:

$$\operatorname{PPL}(x)=\exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log P_\theta(x_t\mid x_{<t})\right).$$

Lower perplexity means the sequence is less surprising to the model. On Riverside manuscript text, lower fine-tuned perplexity is evidence of domain and style fit. It is not evidence of truth: a fluent hallucination can have excellent perplexity.

**Capability benchmarks** are curated tasks with known answers and a scoring rule. MMLU tests broad knowledge, HellaSwag commonsense completion, TruthfulQA resistance to common falsehoods, HumanEval Python generation through unit tests, and HELM multiple scenarios and dimensions. Riverside also needs a private benchmark because public scores do not directly answer manuscript-specific questions.

The taxonomy therefore separates four claims: surface fidelity, semantic fidelity, language-model fit, and task capability. Human review, safety evaluation, hallucination detection, and calibration remain necessary companion layers, even though they are developed in later notebooks.

## 2. Reference-based metrics

BLEU asks whether candidate n-grams occur in the reference. For each order $n$, clipped precision caps a candidate n-gram's count at its reference count. A common BLEU-4 setup geometrically combines unigram through four-gram precision, while the brevity penalty discourages a one-word answer that happens to match perfectly. Sentence-level BLEU also needs a stated smoothing rule when an n-gram order has zero matches. BLEU is appropriate when valid outputs are close paraphrases, especially translation. It is usually low for open-ended editorial answers because many correct phrasings exist.

ROUGE-L finds the longest sequence of shared tokens in the same relative order. Its precision is $|LCS|/|candidate|$, recall is $|LCS|/|reference|$, and the F-score balances them. It tolerates inserted words better than BLEU, which helps summarization, but an unseen synonym still breaks the sequence. It may also reward extractive copying more than concise synthesis.

METEOR is a useful middle ground. It aligns exact words, stems, and synonyms, then penalizes alignments split across many chunks. A candidate containing the right concepts in a scrambled or list-like form therefore loses credit. METEOR is more paraphrase-tolerant than BLEU or ROUGE, but WordNet coverage and language-specific resources limit it.

Reference quality controls all three. A single reference is one acceptable expression, not the entire space of correct answers. Use multiple references when possible, keep references independent of model outputs, and record who wrote and reviewed them.

## 3. Model-based and reference-free metrics

BERTScore compares contextual token vectors with cosine similarity. Reference-side greedy matching acts like semantic recall; candidate-side matching acts like semantic precision; their harmonic mean is semantic F1. It is valuable for Riverside because a fine-tuned answer may state the correct plot fact with different wording. However, semantic closeness is not entailment. "The colony was destroyed" and "the colony was thriving" share topic and syntax while contradicting each other. A high BERTScore must not become a factuality certificate.

Perplexity measures model surprise. It is useful for regression alarms: if held-out in-domain perplexity rises sharply after a model change, tokenization, weights, data, or serving behavior may have shifted. Compare perplexity only under the same tokenizer, context policy, and evaluation corpus. Do not compare raw values across unrelated model families as if they shared one scale.

An LLM judge can target correctness, completeness, relevance, or style with an explicit rubric. Prefer pairwise comparison when relative quality matters, randomize answer order, hide model identity, require structured output, and audit judge-human agreement. A judge extends the suite; it does not remove the need for examples reviewed by people.

## 4. Worked Riverside micro-example

Question: **What is significant about the jade pendant?**

Reference: "The pendant hides a letter proving Mei-Lin's noble birth, giving her standing to reclaim the family trade route."

Candidate A: "The pendant contains a document confirming Mei-Lin's lineage, allowing her to recover the family trade permit."

Candidate B: "The pendant is a valuable jade heirloom symbolizing status and family heritage."

Candidate A changes "letter" to "document," "proving noble birth" to "confirming lineage," and "reclaim" to "recover." BLEU may be low because longer exact n-grams disappear. ROUGE-L finds some ordered overlap, but its recall is limited. METEOR gains synonym matches and BERTScore should be high because the legal function and outcome are preserved.

Candidate B is fluent and topically related. It shares "pendant," "jade," and "family," so string overlap may look respectable; perplexity may also be low. Yet it omits the hidden letter and legal consequence. This is the notebook's fluency trap: plausible nouns are not sufficient evidence of correctness. A factual MCQ, rubric-based judge, or human reviewer should prefer A decisively.

The micro-example shows why disagreement is diagnostic. Low BLEU plus high BERTScore may indicate a valid paraphrase. Moderate overlap plus low factual judgment may indicate a fluent hallucination. Do not average away that pattern before inspecting it.

## 5. Benchmark construction

Define the capability before writing questions. Riverside wants manuscript factual recall, general literary knowledge, and meta-evaluation knowledge. Its 20-question benchmark therefore labels each item by domain and uses four answer options with one ground truth.

For a causal language model, score each complete prompt-option sequence by cross-entropy and select the option with the lowest loss. Keep prompt format and option treatment fixed. A stronger harness should score only the answer continuation rather than allowing common prompt tokens to dominate, normalize for option length when appropriate, and test whether shuffling option order changes predictions.

Build items from a specification, not from whatever examples are easiest to write. Record source, skill, difficulty, answer, plausible distractors, reviewer, and version. Separate development items from a sealed test set. Include positive, negative, contrastive, and boundary cases. Remove ambiguous questions and distractors that leak the answer through grammar or length.

Public benchmarks support external comparison, but contamination is a major threat because questions and answers may appear in training data. Private benchmarks better test Riverside's proprietary domain, although they sacrifice direct comparison with published leaderboards. Use both only when each supports a distinct claim.

MCQ accuracy is not generation quality. A model may recognize the right option but fail to produce a complete answer from scratch. Pair the MCQ benchmark with representative generation prompts and semantic or judged scoring.

![Benchmark construction, slicing, uncertainty, and leakage loop](images/01-llm-evaluation-metrics-and-benchmarks-theory-02.png)

## 6. Slicing, aggregation, and uncertainty

Always retain per-example results. Then report slices that match risk: Riverside-specific versus general questions, novel, query type, answer length, rare entities, and known critical cases. Overall accuracy can hide a severe regression in one manuscript or one user workflow.

Localize a failure before changing the model. Move from dashboard metric, to affected slice, to individual item, to claim or token span, and finally to a likely source such as retrieval, prompt, decoding, reference, evaluator, or model behavior. Metric disagreement narrows the search: low overlap with high semantic similarity suggests paraphrase; high similarity with failed factual review suggests unsupported or contradictory content.

Use macro-averaging when every slice should count equally and micro-averaging when every example should count equally. State which one is used. A radar chart can summarize BLEU, ROUGE-L, BERTScore, METEOR, and MCQ accuracy, but its area is not a statistically meaningful universal quality score. Different axes have different scales and semantics. Show the underlying table beside the chart.

A measured score is an estimate. For accuracy on $n$ independent items, the rough standard error is $\sqrt{p(1-p)/n}$; with only 20 questions it is large. Prefer bootstrap confidence intervals over examples, and use paired bootstrap differences when two models answer the same items. Report the point estimate, interval, sample size, and number of wins, ties, and losses. Do not claim a model improved because its average rose slightly while the paired interval still includes zero.

## 7. Leakage and metric failure modes

**Benchmark leakage:** evaluation items or close paraphrases enter pretraining, fine-tuning, prompt examples, retrieval indexes, or evaluator context. Freeze and access-control the test set; hash or search for near-duplicates; version every release; rotate only through a documented process.

**Entity leak:** the question itself contains names that inflate BLEU or ROUGE without proving domain knowledge. Add contrastive questions and measure answer-only facts.

**Fluent-but-wrong:** shared vocabulary and low perplexity mask incorrect claims. Add factual checks, judges, or human review.

**Correct-but-penalized:** terse answers and paraphrases lose n-gram credit. Add BERTScore, METEOR, multiple references, or rubric scoring.

**Goodhart collapse:** once a metric becomes the optimization target, the model learns metric-shaped behavior. Keep sealed tests, inspect outputs, and avoid training directly against the complete reporting suite.

**Evaluator dependence:** BERTScore and judges inherit their model's biases, language coverage, and blind spots. Pin evaluator versions and periodically compare them with human labels.

## 8. Metric selection rules

1. Translate close paraphrases: use BLEU, with semantic and human checks for freer translation.
2. Evaluate summaries: combine ROUGE-L for ordered coverage with BERTScore for paraphrase tolerance.
3. Evaluate open-ended Riverside QA: use BERTScore plus METEOR, then add factual MCQ or judged correctness.
4. Monitor domain style fit: use held-out perplexity as a regression alarm, never as an accuracy score.
5. Test factual recall: use a private capability benchmark plus generated-answer evaluation.
6. Evaluate safety, hallucination, and calibration: use their specialized protocols; overlap metrics are not substitutes.
7. When metrics disagree, inspect the examples and let the task claim decide. Never choose the largest number after seeing results.

Riverside's minimum viable suite is semantic fidelity with BERTScore, domain-fit regression with perplexity, manuscript recall with the private MCQ benchmark, and a small recurring sample of manual reviews. String metrics remain useful diagnostics, especially for explaining why results disagree.

## 9. Final breadth checklist

- [ ] State the product decision and capability claim before choosing metrics.
- [ ] Include reference-based, semantic/model-based, reference-free, and capability evidence where relevant.
- [ ] Keep references independent, reviewed, and versioned.
- [ ] Preserve per-example outputs and inspect canonical disagreement cases.
- [ ] Report critical slices, not only one overall average.
- [ ] State aggregation rules, sample sizes, confidence intervals, and paired differences.
- [ ] Check public and private data for exact and near-duplicate leakage.
- [ ] Separate MCQ recognition from free-form generation quality.
- [ ] Pin model, tokenizer, prompt, evaluator, dataset, and code versions.
- [ ] Add factuality, safety, hallucination, calibration, and human review when the deployment claim requires them.
- [ ] Maintain a sealed holdout so Goodhart's Law cannot consume the entire suite.
- [ ] Treat the dashboard as a map of evidence, not a single truth score.

The final principle is: **measure the claim, preserve disagreement, quantify uncertainty, and keep humans close to the failures that automated metrics cannot see.**

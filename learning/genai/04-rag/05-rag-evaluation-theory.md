# RAG Evaluation: A Diagnostic Theory Note

RAG evaluation is not one score attached to one answer. It is a method for locating where a two-stage system failed:

$$
\text{answer}=\text{Generate}\bigl(q,\text{Retrieve}(q)\bigr)
$$

The retriever chooses evidence. The generator uses, ignores, or distorts that evidence. A fluent answer can therefore be correct, faithfully wrong, hallucinated, or off-topic. Reading the answer alone does not reliably separate those cases. You need measurements tied to the component that could have caused the failure.

This note uses the notebook's Riverside rehearsal: 14 internal documents, eight labeled question/reference/gold-document cases, and one running question:

> How does ReAct combine reasoning and acting?

The fixture proves metric mechanics and exposes known blind spots. It does not calibrate production thresholds. Real thresholds require representative Riverside labels, versioned scorers, and repeated evaluation.

## 1. Start With Failure Localization

The basic failure map is a two-by-two table:

| Retriever | Generator | Observable result |
| --- | --- | --- |
| Correct | Faithful to context | Correct answer |
| Wrong | Faithful to context | Coherent wrong answer |
| Correct | Ignores context | Hallucination |
| Wrong | Ignores context | Double failure |

The coherent wrong answer is especially dangerous. The prose is fluent, the cited document is real, and the generator may summarize it perfectly. The defect is upstream: the wrong document entered the answer path. Improving generation cannot repair missing evidence.

![RAG evaluation component and failure-location map](images/05-rag-evaluation-theory-01.png)

Use four questions in order:

1. **Retrieval:** Did the system fetch the required document?
2. **Groundedness:** Are the answer's claims supported by the retrieved context?
3. **Answer relevance:** Does the answer address the user's question?
4. **Correctness:** Does it cover the required content in the reference answer?

These questions overlap, but they are not interchangeable. The pattern across them is the diagnosis.

## 2. Retrieval Metrics: Coverage, Noise, and Semantic Proximity

When there can be several required documents, retrieval has two faces:

$$
\text{Context Recall@k}=\frac{|D_{ret}\cap D_{gold}|}{|D_{gold}|}
$$

Recall asks how much required evidence was found.

$$
\text{Context Precision@k}=\frac{|D_{ret}\cap D_{gold}|}{|D_{ret}|}
$$

Precision asks how much retrieved material was actually required. Returning the whole corpus can give perfect recall and terrible precision. Returning one correct page can give perfect precision while missing other required pages.

The notebook has one gold document per question, so rank-one recall becomes binary: the gold document is either first or it is not. That label is strong but coarse. The notebook therefore also uses average embedding cosine between the question and retrieved documents as a continuous retrieval-relevance proxy.

Keyword overlap is weaker. A wrong LoRA document can share words such as "model" and "parameters" with a ReAct-related query. Conversely, a correct paraphrase may share almost no exact wording with the source. Embeddings better preserve topical meaning, but they are still proxies: a topically adjacent document can be semantically close without answering the question.

## 3. Answer Metrics: Three Different Questions

### Groundedness: did the answer use the evidence?

Groundedness, or faithfulness, asks whether the context entails every answer claim. The notebook's cheap proxy is token recall:

$$
\text{Ground}(a,C)=\frac{|\operatorname{tok}(a)\cap\operatorname{tok}(C)|}{|\operatorname{tok}(a)|}
$$

Invented phrases such as "quantum entanglement" introduce tokens absent from the ReAct context, so the score falls. This catches obvious invention cheaply.

It does not understand entailment. "ReAct does not interleave thought and action" reuses nearly all the source vocabulary while reversing the fact. A verbatim context copy can also score almost perfectly even if it is not a focused answer. Therefore groundedness must be read with relevance, and subtle negation or distortion needs an NLI-capable or LLM judge.

### Answer relevance: did it answer this question?

Answer relevance compares the question and answer embeddings:

$$
\text{AnsRel}(q,a)=\cos(e_q,e_a)
$$

Cosine normalizes vector magnitude, so the comparison focuses on direction in embedding space rather than answer length. It catches topic drift: an answer about fine-tuning should be far from a question about ReAct.

But an invented answer full of ReAct vocabulary can remain highly relevant. Relevance means "about the requested topic," not "factually supported."

### Correctness: did it cover the labeled answer?

Correctness requires a reference answer. Exact word-set overlap such as Jaccard punishes synonyms and ignores order. The notebook advances to recall-oriented ROUGE-L:

$$
\text{ROUGE-L}(a,r)=\frac{|\operatorname{LCS}(a,r)|}{|r|}
$$

The longest common subsequence rewards reference words preserved in the same relative order, even when they are not adjacent. It gives useful partial credit to light paraphrases.

This notebook uses the recall-oriented form: it asks how much of the reference appears in the answer. It does not penalize unsupported extra claims. Read it with groundedness, or a verbose answer can cover the reference and still add inventions.

ROUGE-L still cannot recognize meaning when vocabulary changes heavily. "Employ" and "use" remain different tokens. This is where an embedding-based correctness metric or an audited LLM judge must replace surface overlap.

## 4. Composite Scores Are Summaries; Fingerprints Are Diagnoses

The notebook computes a simple mean:

$$
\text{Composite}=\frac{\text{RetRel}+\text{Ground}+\text{AnsRel}+\text{ROUGE-L}}{4}
$$

That number can summarize a run, but it hides cause. Two systems can have the same mean and require opposite repairs. Read the four-dimensional fingerprint instead:

| Fingerprint | Likely failure | First place to inspect |
| --- | --- | --- |
| Retrieval relevance low | Wrong evidence neighborhood | Index, embeddings, retrieval settings |
| Groundedness low while relevance stays high | On-topic invention | Prompt and generation constraints |
| Answer relevance low | Topic drift | Retrieval-generation interface or prompt |
| Correctness low | Required content omitted or changed | Answer construction and reference coverage |

Dependencies matter. Retrieval is upstream. If required evidence is absent, a context-bound generator cannot produce the correct answer from that evidence alone. Good retrieval is necessary, not sufficient.

## 5. Oracle Context: Remove One Moving Part

When an end-to-end answer fails, bypass retrieval and give the same generator the labeled gold document. This is an oracle-context ablation, not a deployable feature.

![Oracle-context diagnostic and metric-fingerprint decision map](images/05-rag-evaluation-theory-02.png)

Interpret the before/after result:

- **Answer scores jump:** the generator can use good evidence; retrieval is the likely bottleneck.
- **Scores barely move:** better retrieval alone will not fix the case; inspect answer construction or the evaluator.
- **Only some queries jump:** the pipeline has multiple failure modes; inspect slices separately.

### Small worked case

Question: "How does ReAct combine reasoning and acting?"

Gold context: ReAct interleaves thought and action steps and uses tools such as Wikipedia search or a calculator.

Consider three outputs:

1. **LoRA answer from a LoRA document.** Groundedness may be respectable because the answer faithfully repeats its context. Retrieval relevance and reference correctness fall. Diagnosis: coherent wrong answer; repair retrieval first.
2. **Quantum-entanglement ReAct answer from the gold document.** Answer relevance may stay high because the response is about ReAct. Groundedness and correctness fall because the claims are invented. Diagnosis: generator failure.
3. **Fine-tuning answer from the ReAct context.** Retrieval can look healthy and copied words can make groundedness look acceptable, but answer relevance and correctness fall. Diagnosis: off-topic answer construction.

Now force the gold ReAct document into the same generator. If case 1 becomes a good ReAct answer, the score gain isolates retrieval. If the hallucination in case 2 remains, retrieval was not the controlling defect.

## 6. Judges and Evidence Have Limits

An LLM judge keeps the same bounded evaluation questions but replaces weak token or embedding proxies with semantic judgment. A groundedness judge receives facts and an answer, then returns a structured verdict plus a concise rubric-grounded rationale citing decisive evidence. It should not be asked for hidden chain-of-thought.

This upgrade detects entailment, coreference, pragmatic implication, negation, and subtle distortion better than local proxies. It also introduces judge-specific risks:

- **Verbosity bias:** longer, polished answers may look better without adding grounded facts.
- **Position bias:** pairwise verdicts may change when answer order changes.
- **Self-preference:** a judge may favor outputs resembling its own style.

The notebook demonstrates verbosity bias and names the other two as requiring a reorderable or multi-judge harness. Therefore judge outputs need audit, scorer versioning, and calibration. Evidence is also incomplete when labels are narrow: eight hand-written cases are a rehearsal, not a production claim.

## 7. Decision Rules

1. **Diagnose per query before averaging.** A mean says how much; the fingerprint says where.
2. **Run the oracle ablation on weak cases.** Large gold-context gain points first to retrieval; little gain points to generation or evaluation.
3. **Use cheap proxies for frequent regression checks.** Reserve audited semantic judges for periodic review and cases near a boundary.
4. **Treat narrow margins as review, not proof.** With eight cases, one binary example can move a mean by $1/8=0.125$. The notebook marks results within one-case influence of a threshold as `REVIEW`, not an automatic release decision.
5. **Keep critical gates separate.** Citation correctness, refusal appropriateness, and safety cannot be averaged away by strong relevance.
6. **Version the evidence.** Dataset, prompt, retriever, generator, evaluator, threshold, and report belong to the release record.
7. **Feed production failures back into offline evaluation.** Privacy-reviewed online samples reveal drift; difficult cases become labeled regression cases.

## 8. Breadth Checklist

Before calling a RAG evaluation complete, check the notebook's full landscape:

- [ ] Versioned questions, references, and gold document IDs
- [ ] Recall@k and Precision@k when multiple documents can be relevant
- [ ] Retrieval relevance by query and slice
- [ ] Groundedness tested against invention, negation, and verbatim-copy gaming
- [ ] Answer relevance tested against on-topic hallucination and off-topic drift
- [ ] Reference correctness tested against light and heavy paraphrases
- [ ] Per-query metric fingerprints, not only a composite mean
- [ ] Oracle-context ablation for weak cases
- [ ] Unsupported and unauthorized cases with expected answer, abstain, or refusal behavior
- [ ] Citations checked against retrieved, authorized, supporting passages
- [ ] Judge bias audit and evaluator versioning
- [ ] Threshold sensitivity before release decisions
- [ ] Offline regression plus privacy-reviewed online telemetry
- [ ] Safety, latency, and cost as operational gates

MRR, nDCG, BLEU, BERTScore, human annotation agreement, bootstrap intervals, synthetic golden-data generation, and A/B testing are part of the broader landscape, but the notebook does not implement them. Name that boundary explicitly. Evaluation is trustworthy only when "mentioned" is never mistaken for "measured."

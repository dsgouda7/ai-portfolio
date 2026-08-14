# Hallucination Detection and Factual Grounding

A hallucination is a fluent claim that conflicts with or lacks support from the evidence boundary. In a retrieval-augmented system, that boundary is usually the retrieved context. In open-domain fact-checking, it may include trusted external sources. The first question is therefore not "Does this sound good?" but "What evidence supports this claim?"

This distinction matters because quality is not factuality. Similarity metrics measure resemblance, LLM judges may measure usefulness, and perplexity measures how expected the wording is. None independently establishes that a claim is correct.

![Hallucination types and complementary detection signals](images/03-hallucination-detection-theory-01.png)

## 1. Grounding Is Not World Truth

A claim is **grounded** when the supplied evidence supports it. A claim is **true** when it agrees with the world. These are related but different properties. A true claim omitted by retrieval is ungrounded, while a false statement in a faulty source can be grounded to that source. A RAG guard evaluates attribution to its evidence; world-truth verification requires authoritative sources or human fact-checking.

Useful failure labels are:

- **Intrinsic:** the claim contradicts the supplied evidence.
- **Extrinsic:** the evidence neither supports nor contradicts the added claim.
- **Entity-level:** a name, date, number, title, or role is wrong or unsupported.
- **Relation-level:** real entities are connected by the wrong action or relation.

The labels overlap and are routing aids, not perfectly separate classes. Relation errors are especially difficult because nearly every word in a sentence may be supported except the decisive verb.

## 2. One Riverside Claim, Four Signals

Riverside's source says: **"Elena Marchetti was longlisted for the Booker Prize in 2019."**

The editorial assistant generates: **"Elena Marchetti won the Booker Prize in 2019 for her debut novel *The Silence of Bridges*."**

### Attribution

Natural Language Inference (NLI) compares the source as the **premise** with the generated claim as the **hypothesis**. Entailment means the evidence supports the claim; contradiction means it conflicts; neutral means the evidence is silent. Here, the source supports only "longlisted" and does not entail the stronger "won" claim; the novel title is also unsupported. "Won" becomes a contradiction only if the evidence explicitly says she did not win. Claim-level attribution therefore surfaces unsupported and contradictory content more clearly than whole-answer similarity.

Neutral does not mean false: the title could be true but missing from the passage. Likewise, partial support must not hide the wrong award relation. Review the evidence span and the unsupported or contradictory phrase, not only an aggregate score.

### Entity Gap

Entity-gap detection compares names, dates, quantities, works, and roles in the claim with those in the evidence. Elena Marchetti, Booker Prize, and 2019 appear in both; *The Silence of Bridges* does not, so it becomes a concrete review target. However, entity matching alone misses "won" versus "longlisted" because all surrounding entities are present. Entity gaps locate suspicious details; they do not verify relations.

### Consistency

When trusted context is absent or a case remains ambiguous, generate several independent answers and compare their claims. If samples alternate between "won," "longlisted," different years, or different novel titles, instability raises risk. If every sample repeats "won," consistency is high, but the claim can still be consistently false. Consistency is evidence about model stability, not truth, and extra generations add cost and latency.

### Perplexity

Perplexity asks how expected the token sequence is to a language model. The Riverside sentence is polished and conventional, so it may receive low perplexity despite being wrong. Length, domain style, and calibration also affect the value. Perplexity can diagnose fluency or domain fit; it must never serve as a factuality gate.

## 3. Proxies Versus Validated Classifiers

A tutorial score, heuristic threshold, zero-shot wrapper, entity matcher, or weighted risk formula is a **proxy**. It can rank examples and demonstrate a workflow, but its output is not a calibrated probability and its threshold is not measured performance.

A validated classifier has been evaluated on labeled, in-domain claims with the same retrieval and generation conditions used in deployment. Its label mapping is verified; thresholds are selected against the cost of false negatives and false positives; and precision, recall, calibration, and subgroup failures are measured on held-out data. Direct sequence-pair NLI is preferable to treating a context string as a zero-shot label, but even direct NLI must be validated and monitored.

Do not describe a proxy as a detector with a known catch rate unless a representative evaluation establishes that rate. Record which signals actually ran, and never replace a missing measurement with an estimated value presented as observed evidence.

## 4. Layered Guard and Triage

The composite score does not choose its own threshold. Lowering the HIGH-risk gate catches more hallucinations and sends more answers to human review; raising it saves review capacity and misses more failures. Sweep the threshold on labeled cases and choose the operating point from recall, precision, and review budget. `0.50` is a starting hypothesis, not a law.

![Layered hallucination guard and triage flow](images/03-hallucination-detection-theory-02.png)

Use complementary signals in a deliberate order:

1. Split the answer into atomic, scoreable claims.
2. Run attribution against the retrieved evidence and retain supporting or conflicting spans.
3. Extract entity gaps to show the reviewer exactly what needs checking.
4. Use consistency sampling only when context is unavailable or the measured risk remains ambiguous.
5. Combine only observed signals, using weights and decision bands fitted on labeled deployment data.
6. Serve low-risk claims, visibly qualify uncertain claims, and hold high-risk author-facing claims for review.

For the Riverside claim, the guard should surface the unsupported upgrade from "longlisted" to "won" and the unsupported title. It should not silently rewrite the sentence or declare it globally false based only on the retrieved passage. The editor needs the claim, evidence, signal provenance, and reason for escalation.

## 5. Practical Rules

1. Separate prose quality from factual support.
2. Define the evidence boundary before scoring.
3. Score atomic claims rather than only complete answers.
4. Treat contradiction as strong grounding risk and neutrality as a verification candidate.
5. Use entity gaps to focus review, not to validate relations.
6. Use consistency as a secondary stability signal, never as proof of truth.
7. Use perplexity for fluency or domain fit, never factuality.
8. Label tutorial heuristics as proxies until validated on representative data.
9. Tune thresholds to business costs and monitor them across model and retrieval changes.
10. Keep a human or authoritative-source path for consequential and relation-level claims.

A useful hallucination guard does not promise truth from one score. It makes evidence boundaries explicit, combines complementary signals, explains uncertainty, and routes unresolved claims to the right verification process.

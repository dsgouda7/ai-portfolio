# Hybrid Search for RAG - Handwritten Theory Notes

## 1. The central picture

Riverside should **fine-tune stable behavior** and **retrieve changing knowledge**. Policies belong in retrieval because they change, require citations, can be withdrawn, and have access controls.

Retrieval must place a current, authorized, supporting passage near the top. Questions may use an exact code such as `RIGHTS-17` or a paraphrase such as `support when welcoming a baby`, so one search method is rarely enough.

![Handwritten flow from authorization through hybrid retrieval, reranking, support gating, and cited evidence](images/01-hybrid-search-theory-01.png)

## 2. BM25 and dense retrieval

**BM25 asks which passages contain the query's words.** Rare matches such as `RIGHTS-17` matter most. Repetition and long passages receive limited extra credit.

For `What does RIGHTS-17 require?`, BM25 might return:

1. Copyright and Licensing (`RIGHTS-17`)
2. Procurement (`FIN-42`)
3. AI Usage

BM25 excels at identifiers, names, numbers, and approved wording. It may miss `parental leave` for `welcoming a baby` when useful words do not overlap.

**Dense retrieval asks which passages have similar meaning.** A model represents queries and passages as vectors and ranks nearby vectors. For `support when welcoming a baby`, it might return:

1. Parental Leave
2. Employee Benefits
3. Flexible Working

Dense search handles paraphrases, but may treat `RIGHTS-17` as noise or retrieve a topically similar, operationally wrong policy. Use hybrid search when labeled code and paraphrase queries show complementary results.

## 3. Fusion

Fusion reorders passages supplied by BM25 or dense retrieval. If both omit the correct passage, later stages cannot recover it, so candidate depth sets a recall ceiling.

Do not add raw BM25 and dense scores because their scales differ. Use ranks or normalize each list first.

**Reciprocal Rank Fusion (RRF)** ignores raw scores and rewards high positions across lists. For each document, it adds a small contribution from every retriever that returned it. That contribution shrinks as the document's rank gets lower, while a damping constant controls how quickly the credit falls. A document absent from a list receives no contribution from that retriever.

For a `RIGHTS-17` query, imagine these lists:

| Rank | BM25 | Dense |
| ---: | --- | --- |
| 1 | Copyright and Licensing | AI Usage |
| 2 | Procurement | Copyright and Licensing |
| 3 | AI Usage | Procurement |

Copyright and Licensing receives strong credit twice and can finish first. RRF is robust because it trusts position rather than incomparable score magnitudes. Riverside uses $k=60$ as a notebook baseline, not a universal optimum.

![Handwritten worked RRF ranking for a Riverside RIGHTS-17 query](images/01-hybrid-search-theory-02.png)

**Weighted fusion** first puts BM25 and dense scores on a comparable scale, then gives each side a chosen weight. A 70% lexical blend says exact matching matters more; a 70% dense blend favors meaning. On the same query, normalized scores might produce:

| Passage | Lexical | Dense | 70% lexical blend |
| --- | ---: | ---: | ---: |
| Copyright and Licensing | 1.00 | 0.80 | 0.94 |
| AI Usage | 0.35 | 1.00 | 0.55 |

Tune weights on labeled traffic and check held-out questions. If weights tie, report inconclusive evidence. Equal-score lists need explicit handling because they contain no score-based order.

## 4. Reranking

A production pipeline spends cheap computation broadly and expensive computation narrowly. Retrieval gathers candidates, fusion makes a shortlist, and a cross-encoder reads each `(query, passage)` pair to refine the order.

Fusion might return `[AI Usage, Copyright, Procurement]`; reranking may produce `[Copyright, AI Usage, Procurement]`. Labels must confirm that this movement helps. Reranking changes rank, not membership, and costs more as the shortlist grows.

## 5. Recall@K and MRR

**Recall@K asks whether relevant material appears within the first K results.** If relevant passages are `[Copyright, AI Usage]` and the top three are `[Copyright, Procurement, Travel]`, Recall@3 is one out of two. With one target, Recall@5 is simply yes or no.

**MRR asks how early the first relevant result appears.** Rank 1 earns 1, rank 2 one half, and rank 4 one quarter; average across queries. Systems can tie on Recall@5 while the one placing targets earlier has better MRR. Evaluate answer correctness and citation support separately.

## 6. Authorization and support

Authorization runs **before scoring**:

```text
identity + tenant + groups -> authorized candidates -> BM25/dense -> fusion -> rerank
```

A late ACL filter is unsafe: an inaccessible passage can influence ranks, enter logs, expose a snippet, or reach the model. Preserve stable chunk ID, policy ID, revision, owner, tenant, and ACL metadata through retrieval and citation. If identity or ACL evaluation is uncertain, fail closed. An empty authorized set means abstain, never retry without filters.

A ranker always chooses a winner, even for an unsupported dental-deductible question. Use a separately calibrated support gate. If no authorized passage clears it, state that the available policy collection does not support an answer.

## 7. Fallback, telemetry, and failure rules

Publish dense and sparse artifacts together under one immutable version, with checksums and a manifest. Never mix revisions; retain the previous complete version for rollback.

If one retriever times out, use the surviving authorized ranking and record degraded mode. If reranking fails, use fused order. If both retrievers fail, return an error or abstain. Authorization uncertainty never permits broader access.

Log privacy-aware telemetry: index version, retrieval mode, stage latency and errors, authorized candidate count, dense and sparse counts, overlap, empty results, fallback rate, support decisions, and rank distributions. Combine it with offline Recall@K and MRR, sampled relevance judgments, stale-version tests, and no-answer errors. Do not log confidential passage text for convenience.

Keep these rules explicit:

- Exact code lost by dense search: retain BM25.
- Paraphrase lost by BM25: retain dense search.
- Raw scores added: use RRF or normalize before weighted fusion.
- Relevant passage absent: increase or improve first-stage candidates before tuning reranking.
- Reranker changes order: verify relevant-document movement with labels.
- Unsupported query gets a plausible passage: calibrate the support gate.
- ACL filtering occurs after retrieval: move authorization before every scoring path.
- Dense and sparse versions drift: publish and roll back atomically.
- One attractive toy result: test representative traffic before generalizing.

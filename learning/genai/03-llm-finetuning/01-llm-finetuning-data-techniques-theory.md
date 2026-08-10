# LLM Fine-Tuning Data Techniques: Handwritten Theory Notes

## 1. Start with the behavior, not the trainer

Riverside House already has a fluent general model. Its problem is repeated correction. Editors keep restating who Aria Voss is, how the *Meridian's Promise*, Wren, the Lantern, and the Choir relate, how long an answer should be, and which technically valid draft is actually useful. Fine-tuning is worth exploring only when that stable behavior should become a default habit across many requests.

The first question is therefore not “Which library should I run?” It is:

> What training experience most closely resembles the behavior I want the model to practice?

- **Continued pretraining (CPT):** practice raw domain sequences.
- **Supervised fine-tuning (SFT):** practice an approved request/response contract.
- **Preference learning:** practice choosing one acceptable response over another.
- **PPO:** use fresh outcomes produced by a changing policy.
- **DPO:** use a fixed archive of chosen/rejected comparisons.

These are not a mandatory chain. Riverside demonstrates all three main data shapes because it has three different repeated costs.

![Handwritten objective-selection path from persistent behavior to CPT, SFT, DPO, or PPO](images/01-llm-finetuning-data-techniques-theory-01.png)

## 2. One decoder, different answer keys

The model remains a causal decoder. Instruction, context, and completion occupy one growing left-to-right tape. The objective changes which positions count as lessons.

For token IDs $x_1,\ldots,x_T$, ordinary causal language modeling minimizes next-token loss over selected positions:

$$
\mathcal{L}_{\text{CLM}}=-\sum_{t \in A}\log p_\theta(x_{t+1}\mid x_{\le t}),
$$

where $A$ is the set of context positions whose next-token targets remain active. The data contract defines $A$. In the stored label tensor, targets set to `-100` are ignored after the causal shift. Attention masking and loss masking are different: attention decides what a token may read; labels decide where prediction error creates a gradient.

### CPT contract: raw prose is the lesson

Record shape:

```text
{ text: "Aria traced the prime-number signal ..." }
```

Token shape:

```text
input:  [real manuscript tokens ........ padding]
labels: [same real token IDs ........... -100   ]
mask:   [1 1 1 1 1 .................... 0 0    ]
```

Every real token participates. The causal-LM implementation shifts labels internally, so each position learns the next manuscript token. CPT can make Riverside names, relationships, terminology, and prose patterns less foreign. It does **not** rehearse “one sentence and stop,” and it does not state which of two valid drafts an editor prefers.

Long text needs an explicit policy. Plain truncation silently discards tails. The notebook preserves overflow as fixed-length blocks, masks padding, and accepts that a new block cannot attend to the previous block. Production choices may include packing, document-aware boundaries, block-diagonal attention, best-fit grouping, or local overlap. Each changes efficiency or cross-boundary context; none makes the trade-off disappear.

### SFT contract: the prompt is visible, the answer is graded

Riverside harvests demonstrations from adjacent manuscript paragraphs:

```text
paragraph i     -> context in the editor request
paragraph i + 1 -> first complete sentence as approved response
```

Tiny example:

```text
Request: Continue this Aria scene in exactly one sentence and stop.
Context: Aria watched the signal repeat across node seventeen.
Answer:  She isolated the transmission and called Wren.
```

The system role and request must stay visible because the answer depends on them, but Riverside is not training the assistant to reproduce the brief.

```text
input:  [system + request + context] [assistant sentence + EOS] [padding]
labels: [-100 -100 -100 ..........] [real IDs ..............] [-100   ]
mask:   [1 1 1 1 1 ..............] [1 1 1 .................] [0 0    ]
```

The notebook budgets at most 64 rendered prompt tokens and 32 assistant-suffix tokens inside 96 positions. It tests the fully rendered native chat template, preserves as much context as fits, supervises exactly one EOS token, and checks that prompt tokenization remains stable where prompt and response meet. Those numbers describe this teaching implementation, not universal settings.

EOS supervision matters: “stop after one sentence” is partly a stopping lesson. Masking EOS would remove that direct target. Conversely, supervising prompt tokens changes the lesson toward reproducing the conversation rather than answering it.

![Handwritten loss-boundary diagram contrasting CPT, response-masked SFT, and DPO pair scoring](images/01-llm-finetuning-data-techniques-theory-02.png)

## 3. Preference data: control everything except the choice

SFT says, “Practice this response.” Preference data says, “For this same request, prefer this response to that response.” A useful record is:

```text
{
  prompt: shared editor request and context,
  chosen: complete scene-advancing sentence + EOS,
  rejected: complete coherent stalling sentence + EOS
}
```

Both alternatives should satisfy the surface contract. Otherwise the model may learn an accidental shortcut such as “prefer grammatical endings,” “prefer shorter text,” or “prefer the response with EOS.” Riverside therefore rejects fragments, bounds prompt and response lengths, and permits at most a four-token chosen/rejected length gap.

The chosen sentence is the authentic first sentence of the next manuscript paragraph. The rejected sentence comes from a small hand-authored bank of grammatical stalls. The label is assigned by construction: advance the scene over repeating the setup. This is reproducible and controls obvious shortcuts, but it is not measured editor agreement. Production data still needs a written rubric, qualified review, randomized ordering, duplicate checks, privacy and consent rules, tie handling, disagreement adjudication, automated-label audits, and provenance for prompts, candidates, raters or outcome sources.

### PPO: online experience plus cautious credit

PPO is appropriate when the changing policy must generate new experience: tool attempts, simulator trajectories, environment outcomes, or newly scored drafts.

One batch has several roles:

1. A **current writer** samples a fresh rollout.
2. A frozen **reward model** scores the completed outcome.
3. A frozen **SFT reference** charges for cumulative drift from trusted behavior.
4. A learned **value model** estimates expected regularized return for the context.
5. The return minus expectation becomes the **advantage**.
6. The current-versus-old policy probability ratio measures one-batch movement.
7. **Clipping** stops extra credit for an excessive batch move.
8. A value loss improves future expectations; an optional entropy bonus preserves exploration.

The two baselines must not be confused. The SFT reference is captured once for the run and anchors long-term behavior. The old policy is the short-lived before-photo of the policy that generated the current batch; it refreshes with each rollout batch.

PPO is only as sound as its evidence. It can fail when the judge is poorly calibrated, reward is exploitable, rollouts are unrepresentative, value estimates are noisy, or safety and retention gates are weak. Clipping makes updates more stable; it does not make a bad reward correct.

### DPO: fixed comparisons, direct update

Riverside already has fixed pairs, so the reward model, value model, fresh rollout collection, advantage estimation, and PPO clipping are unnecessary overhead. DPO starts with two copies of the accepted SFT state:

- the **reference policy** is frozen;
- the **current policy** is trainable.

For each response, sum token log-probabilities over the complete suffix. Then compare current support with reference support:

```text
chosen movement   = policy(chosen)   - reference(chosen)
rejected movement = policy(rejected) - reference(rejected)
margin            = chosen movement - rejected movement
```

Example: chosen movement `+1.2`, rejected movement `+0.2`, margin `+1.0`. The important quantity is movement from the shared SFT start, not which response had higher raw probability before training. The DPO loss penalizes a weak or negative margin and backpropagates through the current policy only.

`beta` belongs to the preference objective. In the underlying reference-regularized view, smaller `beta` lowers the price of leaving the reference and may permit farther movement; larger `beta` keeps the policy closer. It is not the optimizer learning rate, and its practical effect must be checked against preference gain and contract retention.

DPO does not let the model invent the correct preference or judge itself. Bad, ambiguous, or inconsistent labels teach the wrong ranking directly.

## 4. Provenance is not evidence

The notebook pins model IDs and revisions, the seed, package versions, training arguments, profile-specific artifact paths, and SHA-256 hashes for all 40 training chapters. That is a strong **training-lineage contract**: it records what produced an artifact and supports audit and reconstruction.

It does not establish quality. Part 1 deliberately uses the complete 40-chapter novel and only ten optimizer steps on small, compute-bounded checkpoints. Therefore every before/after generation is an **in-sample mechanism illustration**. It cannot support claims about unseen Riverside behavior, production prose quality, generalization, release readiness, or live Azure behavior.

Independent evidence must be separated from training-data construction, checkpoint selection, prompt tuning, and threshold setting. It needs a separately versioned evaluation suite, the right baseline for each objective, workload-aligned measurements, retention checks, and gates fixed before inspecting results. An artifact proves that state was saved. Part 3 owns decisions about whether that state is useful.

## 5. Choice rules and common failures

- Use **prompting first** when clear instructions or a few examples already work.
- Use **RAG** for current, citable, deletable, permission-filtered facts. Fine-tuning is a poor database.
- Use **CPT** for recurring language and domain sequence patterns. Failure: truncation, bad packing, too much exposure, or forgetting without retention evidence.
- Use **SFT** for role, format, scope, and stopping contracts. Failure: prompt leakage into labels, missing EOS supervision, target truncation, unstable chat serialization, or low-quality demonstrations.
- Use **DPO** for a stable archive of controlled comparisons. Failure: unclear rubric, disagreement, length/style shortcuts, malformed alternatives, or reference drift tuned without retention checks.
- Use **PPO** when fresh policy outcomes are essential. Failure: reward hacking, unrepresentative rollouts, weak calibration, unstable credit, or confusing old-policy clipping with SFT-reference drift control.
- Hold seed policy, effective batch, token budget, data boundary, and trainable surface fixed when comparing objectives. The loss chooses gradient direction; effective batch controls noise; learning rate scales the step; exposure repeats it; token budget controls visible evidence.

## 6. Final breadth checklist

- [ ] Desired persistent behavior is written before the objective is chosen.
- [ ] Every record has an explicit schema: raw text, demonstration, or comparison.
- [ ] Tokenizer and native chat template match the pinned checkpoint.
- [ ] Attention visibility and loss labels are inspected separately.
- [ ] CPT labels every real token and masks only padding.
- [ ] SFT masks prompt and padding, supervises the assistant suffix, and includes one EOS.
- [ ] Truncation, overflow, packing, prompt budget, and response budget are deliberate.
- [ ] Preference alternatives share one prompt, satisfy the same surface contract, and resist length or grammar shortcuts.
- [ ] Preference provenance records rubric, source, review, ties, uncertainty, and audits.
- [ ] PPO is reserved for fresh experience; DPO is selected for trustworthy fixed pairs.
- [ ] Frozen references receive no gradient and their role is documented.
- [ ] Model revision, seed, packages, arguments, source hashes, and artifact ancestry are recorded.
- [ ] In-sample illustrations are never described as held-out evidence.
- [ ] Independent behavior, retention, safety, and release gates remain outside training construction.

The short memory aid is: **raw text teaches familiarity; demonstrations teach the answer contract; comparisons teach the choice; independent evaluation decides whether any of it worked.**

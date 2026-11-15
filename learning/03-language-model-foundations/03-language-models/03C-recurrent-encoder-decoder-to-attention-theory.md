# Readers, Writers and Their Training: Handwritten Theory Notes

**Question:** how does an answer's error teach a reader that never chooses the answer itself?

Follow [03C](03C-recurrent-encoder-decoder-to-attention.ipynb) or its
[Python companion](03C-recurrent-encoder-decoder-to-attention.py).
The main notebook keeps all explanations and four short implementation cells.
[Worked details](03C1-encoder-decoder-worked-details.ipynb) preserve every separate
diagnostic and the original two-model experiment; the measured comparisons below
belong to that route, not the compact main demo.
Keep **the cat sat on the mat**. Our checkable answer is
**mat the on sat cat the**, then `<eos>`.
This reversal is a reading/writing drill, not English generation or translation quality.

## 1. Two jobs, not necessarily two layer families

First read the source, then write a separate answer. The encoder is the reader:
it turns source words into useful contextual notes. The decoder is the writer:
it uses source context and its previous answer input to make the next writing state.
A separate head converts that state into word scores.

Both components use GRUs here. That does not make them the same component.
They have separate learned weights because retaining source details and using them
while writing are different jobs. Their word table is shared; their recurrent weights
are not. The roles could also be implemented with other layer types.

**Complaint:** what exactly can the reader hand over?

## 2. A state crosses the boundary, not a word or weights

Six source words fetch six 24-number word rows. The reader makes a 64-number note
after each read. Its final note initializes the writer's 64-number state.
This is not the literal last word, a transcript, or a copied set of reader weights.

The writer also needs an input that says writing has begun: `<bos>`.
That word row and source-derived state make its first new writing state.
Only then does the head produce thirteen word scores.
The writer changes its own state as it proceeds; it does not rerun the source reader.

A fixed-width summary can work well. The restriction is that this first writer
cannot reopen earlier source notes when it needs a detail.

**Who says what the right encoding is?** Nobody gives a target note.
We give the correct answer and grade what the encoder/decoder pair produces.
The encoder is learning to write notes useful to this decoder, not to copy
some universal description of the source.

Compare sources ending in `mat` and `rug`. Both writers start at `<bos>`, but
the first required answer differs. If the encoder sends exactly the same note,
the one-card decoder must give identical scores. It cannot recover that missing distinction.

An answer error tells us, through the current decoder, which small note changes
would help. Following that feedback into the encoder's source calculations
tells us which encoder weights to change. On the next read those new weights
compute a changed note. The decoder normally learns alongside it.
No coordinate has to be assigned a human meaning; many representations can work.
Reversal rewards word identity and order. A different task can reward different details.

**Complaint:** after its first guess, what should we feed it next?

## 3. Teacher forcing keeps practice on track

Early guesses are unreliable. Feeding a wrong word into the next step can send
the remaining answer off track, like a student misreading step 1 and then being
marked wrong on later steps while following that mistaken start.

**Teacher forcing separates grading from the next input.** If the first target is
`mat` but the model prefers `cat`, grade its actual scores against `mat` normally.
Then supply the true previous word, `mat`, as the second input instead of the
guessed `cat`. The next practice question stays on its intended path.
The first error remains; neither the state nor the weights are reset by this input rule.

```text
practice inputs: <bos> mat the on  sat cat the
current targets: mat   the on sat cat the <eos>
```

The current answer is not its own input. The model receives answer data during
training, but uses it only as the shifted previous-answer history.
The full source may already contain `mat`; source access is not future-answer leakage.

Target length determines the number of practice steps; an answer need not be as
long as its source. Our GRU still computes successive states sequentially.

At generation time there is no teacher: each own guess becomes the next input.
Practising with correct history but generating with imperfect history creates
**exposure bias**. Teacher forcing alone does not teach recovery from every wrong
history, so check generation separately. One mistake can affect later words
without necessarily ruining all of them.

**Complaint:** correct practice inputs still do not guarantee correct predictions.

## 4. Every current word still pays an error

At each writing position, check how much probability the scores give the current
correct answer. Low probability means a larger error. Combine those word errors,
including the stopping answer.

We train through scores and cross-entropy, not through the spelling of an argmax guess.
Supplying `mat` as the second input does not forgive the first prediction of `cat`.
Our controlled next-input check holds weights and the first state fixed:
changing only `mat` to `cat` changes the next computation.

**Complaint:** the head receives grades. How does the source reader learn?

## 5. Feedback follows the handoff backward

The reader's summary helped produce the writer states and scores.
Follow the error backward through the head, through the writer, into its initial
state, and across that boundary into the reader's source calculations.
Shared reader weights gather contributions from the reads that helped make the summary.
That is how the reader learns which details a later writer will need.

The [worked gradient check](03C1-encoder-decoder-worked-details.ipynb#gradients)
cuts only this backward connection with `detach`.
The forward summary numbers, predictions and loss stay identical.
But encoder weight gradients disappear; decoder and head gradients remain.
A useful-looking forward value is not automatically a teachable route.

The shared word table still feeds previous-answer inputs to the writer.
Its gradient therefore need not disappear. The difference between connected and
detached table gradients isolates the reader contribution.
Signed contributions can cancel; a larger norm is not a definition of better memory.

**Complaint:** computed gradients are advice, not yet changed weights.

## 6. Backward prepares; the optimizer changes

Backward populates gradients while leaving weights and saved forward values unchanged.
Adam then changes reader, writer, word-table and head weights.
The next forward pass computes new states using those updated rules.

The [worked optimizer check](03C1-encoder-decoder-worked-details.ipynb#optimizer)
takes one update on a copy: this example's error falls from 2.5996 to 2.4178.
That verifies the wiring, not generalization. The full comparison separately trains
fresh models for 1,000 updates with matched batches and gradient clipping.
Independent sources start fresh states; learned weights persist.

To isolate the encoder's contribution, the
[encoder-only proof](03C1-encoder-decoder-worked-details.ipynb#encoder-only) freezes the word table,
decoder and head. Only encoder weights change on the mat/rug pair.
Its notes change and its first-answer error falls with the decoder fixed.
That is a directly checkable meaning of **a better encoding**: more useful for
the decoder's answer, not closer to a supplied target vector.

**Complaint:** even a trained summary is still only one handoff.

## 7. Attention changes access, not the teaching rule

Keep the reader's note after every source word. At each writing step, project the
current writer state into a query, compare it with those notes, and blend them by
the match weights. That fresh context helps the writer update and score its answer.

This source attention retrieves contextual reader states, not raw word IDs.
Its weights sum to one over source positions; they are not answer-word probabilities.
The two occurrences of `the` have different source addresses and contextual notes.

Both models remain recurrent. Either can use teacher forcing during practice and
its own guesses during generation. Gates can improve retention without providing
this direct source-access route. Attention predates Transformers.

## 8. Grade actual answers, not just plausible maps

Ten word choices create controlled source orders of lengths two through eight.
Forty unseen sources per length are held out. Orders need not be grammatical.
The comparison tests familiar lengths, not longer-length extrapolation.

The original [worked-details comparison](03C1-encoder-decoder-worked-details.ipynb#evaluation)
records complete free-running scores averaging 76.43% for one card and 97.50%
for look-back. At length eight they are 17.5% and 90%.
Parameter counts differ: 35,717 versus 52,933. Access and extra capacity arrive
together, so this is not an architecture ranking.

Whole-answer accuracy fails a row after one wrong word. Token accuracy reveals
later mistakes hidden by that unit. Practice and generation can tie on whole
answers while their later histories and token scores differ.
A bright attention square diagnoses one route; it does not prove why an answer was correct.

## 9. The trainer does not invent reading or writing

The data creates correct answers. In the compact main demo, the training loop explicitly
shifts them into previous-answer inputs before calling the model. In the original
[worked adapter](03C1-encoder-decoder-worked-details.ipynb#adapter), `ReaderWriter`
receives the answers explicitly and builds those shifted inputs itself.
A small registered wrapper passes word scores to the common
loss while leaving attention maps out. Targets already align with scores; there is
no extra beginning label to remove from grading.

Keep these distinctions: state versus weights; correct previous input versus correct
current guess; forward values versus backward routes; source access versus retention.

[Stage 04](../../04-transformers/README.md) changes the internal reading machinery,
not the need for clear source/answer roles and correct supervised alignment.

**Durable summary:** the reader learns what to preserve because the writer's errors
reach it; the writer learns what to say from source context and previous-answer inputs.

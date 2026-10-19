# One Reader Card or Looking Back: Handwritten Theory Notes

**Chapter 03D:** [Notebook](03D-recurrent-encoder-decoder-to-attention.ipynb)
· [Code-only Python](03D-recurrent-encoder-decoder-to-attention.py).
You can already continue a stream with an RNN or LSTM. Now let a reader take in a complete
source and a writer make a separate answer. Stage 04 will rebuild those jobs with Transformer blocks.

## 1. The answer is a different sequence

A running note can continue the color-clue stream. Now the whole input is available, and the required answer has its own order. Reverse `[3, 1, 4, 1]` into `[1, 4, 1, 3]`. Reversal is not translation, but every needed source address is known. That makes it an inspectable test of reading and writing.

Let one component read the input and another write the answer. The reader is the encoder. The writer is the decoder. Those names describe roles, not internal layer types. A sequence-to-sequence task asks for a separate ordered answer. Recurrence or attention can implement its components.

You met those reading and writing jobs briefly in 03A. Here we'll compare two ways to pass information between them and check their answers on new sequences. Digits let us change the task without building another tokenizer.

**But there's a catch:** the reader may understand the source order, but what exactly can it hand to the writer?

## 2. Practice supplies the previous answer

To practice writing the reversal, begin with a start marker. Ask for `1`. Next supply the correct previous answer, `1`, and ask for `4`. Continue with input `4` and expected answer `1`, then input `1` and expected answer `3`. Finally input `3` and ask for the end marker.

Giving the correct previous token during practice is teacher forcing. The current answer never appears as its own input. The start marker begins writing; the end marker teaches stopping. This is a training method, not a requirement of an encoder-decoder architecture. Models that continue a single stream also practice with known earlier tokens.

Generation cannot borrow correct history. Suppose the first guess is `4`, not `1`. The next writer input is now `4`, not `1`. The source remains unchanged, but the next writing computation differs. A wrong guess need not cause every later answer to fail.

This practice-versus-generation mismatch is called exposure bias. Name it after tracing the changed input. Otherwise the label hides the simple operational difference that you need to test.

**But there's a catch:** practice loss cannot tell you whether the writer survives its own history.

## 3. The reader really learns its summary

The fixed-summary model reads source tokens through a gated recurrent unit, or GRU. This is a running-note update with learned controls over retained and new information. Its final note initializes another recurrent writer. No earlier source state is available to that writer.

Think of the final note as a summary card. It isn't CBOW's average: the reader really read in order and learned what to keep. The catch is the card's fixed size, not that it must have forgotten all order. The writer gets that final version, not the earlier notes.

Short sources may fit very well. In the observed run, the fixed-summary model reverses every held-out source of length two and three correctly. The interface does not imply inevitable failure on every example. It means the writer cannot directly reopen an arbitrary source address if the final note is insufficient.

An LSTM improves the route through which information survives recurrent updates. It does not remove this interface. An LSTM reader can still hand a writer only one card.

## 4. Keep the reader's notes available

Instead, save the note after every source token and let the writer consult them. At each writing step it forms a query: a numerical vector playing the role of "what detail would help now?" It isn't literally asking a question in words. Compare that query with each saved note, give better matches more weight, and blend the notes. The weights are positive and add to one. The writer uses this fresh blend to update its state and score the next answer.

This is attention: a new source read for the current writing step. It does not replace recurrence here. Both readers still update sequentially, and both writers still generate token by token. Attention predates Transformers.

Both models get the same training/test split and batches in the same order. Sources have two through eight digits, with forty new test sequences at each length. None of those test sequences appears in training. We're testing new sequences at familiar lengths, not whether the models can suddenly handle longer ones. Each batch has one length, so we don't need blank source or answer slots.

The look-back model has 52,627 weights; the one-card model has 35,539. Looking back and having more weights arrive together here. One run can't separate every reason for a better score, much less rank architectures for all tasks.

## 5. Grade both ways and keep the unit clear

Sequence accuracy requires every predicted token, including end, to match. One wrong digit fails the whole row. Report this by source length under both correct-history practice and free-running generation.

In the recorded run, the mean free-running sequence accuracy over the seven length buckets is 75.0% for one card and about 94.6% for look-back. At length eight, the corresponding scores are 17.5% and 87.5%. At length four, the fixed-summary score is slightly higher: 97.5% against 95.0%. Looking back helped in this run. It didn't win at every length.

Practice and generation have identical exact-sequence scores here. That is not evidence that their histories are identical. After the first wrong prediction, the whole row already fails either exact score. Token accuracy reveals later differences: at length eight, one-card practice reaches about 85.3% while free-running reaches about 71.7%.

Generation uses the known source length only as a step budget that includes end. A correct answer must place end exactly after the reversed digits. An early end fails; later steps cannot rescue that row.

## 6. Read the picture beside the answer

For the repeated-digit probe, both models generate the correct reversal and end. The attentive model's most-read source addresses are `3, 3, 2, 1, 0`, while the expected direct digit addresses are `3, 2, 1, 0`. End has no single required source address.

That imperfect map beside a correct answer is useful. Reader states already contain ordered context, so retrieving one state can retrieve information about more than its local token. The output also depends on the writer state. A bright square tells us where the model looked most, not the whole reason it chose an answer.

Use the heatmap to ask a better diagnostic question, not to manufacture proof. Behavior remains the scorecard. If you later mix source lengths, hide blank positions from attention. Separately, exclude blank target positions from grading.

## 7. Carry the next complaint forward

The last example connects this model to the shared practice loop without changing our comparison.
A small wrapper keeps the reader/writer as a PyTorch submodule, so its weights remain included.
It passes the word scores to the inherited loss method and leaves the attention picture out.
The model still handles reading, writing, attention, and teacher forcing; the trainer just
handles practice. We test one update on a deep copy and check that the original weights haven't
changed. This wrapper is for reversal, not a general interface for translation examples.

Do not replace a learned recurrent summary with averaging and call them equivalent. Do not feed a current answer as its own input. Do not report practice as generation, or a plausible map as proof. Keep capacity, seed, and tiny-task boundaries visible.

The writer can now look back. Reading still proceeds through recurrent updates. [Stage 04](../../04-transformers/README.md) asks: what if we drop recurrence and let attention do all the reading?

**Durable summary:** attention changes the writer's access to source notes; it does not by itself remove recurrence, guarantee better scores, or solve the practice-versus-generation mismatch.

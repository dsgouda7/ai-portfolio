# Sequence Memory: Handwritten Theory Notes

**Question:** how can an early note still guide a later guess?

Follow [the main notebook](01A-sequence-memory-theory.ipynb).
[Worked details](01A1-sequence-memory-worked-details.ipynb) are optional:
full calculations, tiny nudges and the original two-opening echo, not the
main notebook's newer music comparison.

## 1. Counting loses the opening

Our made-up rule says: hear three notes, then repeat the first.
D3 A3 D4 needs D3; A3 A3 D4 needs A3. The ending alone cannot decide.
The digits name octaves, not sequence positions.

Counting also fails. D3 A3 D4 and D4 A3 D3 have identical counts but
different openings. No training can recover order already discarded.

An ordinary feed-forward network can use ordered input positions.
Recurrence instead reuses one reading rule as notes arrive.

**Complaint:** how do we carry earlier context into the next read?

## 2. Carry a notecard, not a transcript

A recurrent neural network (RNN) carries a fixed-size numerical summary
called the hidden state, h.
Read D3 with blank h, then A3 with D3's resulting h, then D4 with the
updated h. More notes mean more reads, not more memory numbers.

Shared weights are the reusable recipe. The changing h holds this
phrase's context. A separate head reads h to score possible next notes;
memory numbers are not note scores.

Each RNN read mixes the input and old h, then applies tanh, a soft
squeeze between -1 and +1. It does not erase the past, but it does
transform it again.

**Complaint:** was the wrong ending caused by poor memory or poor reading?

## 3. Let the ending teach earlier reads

Training temporarily saves the reads and follows the final D3 error
through the head and backward through those calculations. This is
backpropagation through time: repeated uses of one reader, not separate
layers. Saved cards are training records, not extra carried memory.

A gradient means: if this number increased slightly, how would error
change? Negative suggests increasing it; positive suggests decreasing it.
It is a local rate, not the error itself. Loss can improve before the
highest-scoring note changes.

The [handwritten proof](images/01a-rnn-gradient-discovery.png) keeps
`softmax(scores)` as a named block. The numerical check lowers a weight
with positive feedback and reduces D3 loss.

Backward leaves saved values unchanged; weights stay fixed throughout
the pass. Add contributions from every use of each shared weight,
update once, then read forward again.

**Complaint:** can useful guidance reach the opening after many reads?

## 4. Feedback can fade or grow

The [longer-chain proof](images/01a-rnn-gradient-chain.png) follows repeated
handoffs. Even without tanh, twenty multiplications by 0.8 leave about
1.15% of the feedback; multiplying by 1.2 amplifies it about 38-fold.
Tanh adds each read's local response; it did not create the repeated-chain
problem.

Vanishing means weak guidance reaches an early read. Exploding means
guidance amplifies along the route. Neither measures notes remembered or
prediction-error size.

Gradient clipping limits oversized feedback before the weight update.
It does not rewrite memory, restore faded guidance or preserve a clue.

**Complaint:** safer learning still does not choose what memory should keep.

## 5. Keep selectively, not just add

**Attempt 1: remix everything.** Every RNN read transforms old contents.
A clue can become harder to recover.

**Attempt 2: just add writes.** An early change survives, but every
distraction enters too. In the check, D3 A3 D4 A3 and D4 A3 D3 A3
produce the same final card. A surviving change is not a surviving
distinction.

**Attempt 3: select.** Keep useful old amounts and reject unwanted writes.
Other memory numbers can still change.

Long short-term memory (LSTM) adds a **cell state**: a separate fixed-size
running store of computed memory numbers. A controlled view of that store
supplies current context. Quieting the message should not rub out the notecard.

Now name them: **c is the stored contents; h is the exposed view**.
Both continue onward. The improvement is a controlled carry/update route
and separate readout, not extra capacity alone. Selected old contents
can pass onward without being remixed and squeezed again.

**Complaint:** how do these two roles cooperate during one read?

## 6. LSTM: store, then show

A3 arrives after D3. Processing A3 need not mean rubbing out every old clue,
and using a clue need not mean exposing all of it now.
Think of three learned volume controls, one setting per memory number:
**KEEP** admits old contents, **WRITE** admits proposed new contents,
and **SHOW** controls the working view.

The note and old h choose these settings together using shared weights;
the gates do not feed through one another. Follow the notebook's inline GIF,
whose initialized weights demonstrate mechanics, not trained musical skill:

1. Input and old h choose controls and proposed content.
2. KEEP retains old c; WRITE admits the proposal. Their sum makes new c.
3. SHOW exposes a bounded view as new h.
4. A learned Linear head maps h to note scores. Softmax gives probabilities;
   this demo picks the largest. Both c and h continue onward.

Controls are fractions per memory number, not note probabilities or
direct choices of the next note.
KEEP and WRITE are independent; the signed proposal is content, not a
fourth gate.

The SHOW=0 preview hides h without erasing c. It changes one read only;
continuation restores the recorded run. A genuinely changed h could
alter later controls.

Keeping is not guaranteed: weak KEEP fractions still compound, and the
readout can still saturate.

**Complaint:** must selective keeping require two carried lists?

## 7. GRU: keep, then blend

A gated recurrent unit (GRU) carries one h, used by both the head and the next read.
Picture a whiteboard with selected areas protected.

KEEP preserves part of old h. WRITE is its complement: keeping less
admits more proposed content. The blend has no final tanh, so kept
content avoids another squeeze.

RESET controls old context used in the proposal, not the retained-old
branch. Closing RESET need not erase a kept clue.

There is no separate c or SHOW, and keep/write cannot be chosen
independently. Old h also affects controls and proposals, so KEEP alone
does not describe the whole gradient.

**Complaint:** useful routes still need learned contents and controls.

## 8. Make all three learn the same music

The main comparison uses 384 original synthetic 32-event phrases:
307 for practice, 77 held out. Every available next-event answer teaches
the models, not just an ending echo.

All three share the split, hidden width and 320 updates; parameter
counts differ. Saved held-out losses fall from about 3.2 to 0.6-0.7.
That is useful next-event learning here, not an architecture ranking or
proof of long-range recall.

Teacher forcing supplies correct previous events. Generation feeds back
guesses, so one mistake can redirect later music. Better scoring does
not guarantee better sound.

Independent phrases start fresh state; learned weights remain.

## 9. Keep the distinction

**RNN rewrites. LSTM stores, then shows. GRU keeps, then blends.**

Watch for counting away order, confusing state with weights or scores,
carrying unrelated phrases together, and treating gates as guaranteed
recall.

Next, [01B](01B-sequence-memory-pytorch-lab.ipynb) asks whether padded
blanks can rewrite memory after the last real note.

**Memory is current context; learning shapes the shared rules that keep it useful.**

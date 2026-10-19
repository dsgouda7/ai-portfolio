# Running Notes for Text: Handwritten Theory Notes

**Chapter 03C:** [Notebook](03C-fixed-windows-to-recurrent-language-models.ipynb)
· [Code-only Python](03C-fixed-windows-to-recurrent-language-models.py).
First watch a simple RNN carry the clue. Then use an LSTM on the same text task and follow
the call into our shared training loop.

## 1. Pick up where the window failed

You left two prefixes at the window boundary: `red rest calmly now` and `blue rest calmly now`. Their last three words match, but their required answers differ. A fixed-window predictor gives them identical predictions because it receives identical inputs. The measured final-token accuracy was 50%.

Keep that world unchanged. The recurrent notebook uses the same eight sentences, the same six training rows, and the same two held-out rows. It does not solve the problem by changing the labels or supplying a secret extra input. It changes how earlier words can remain available.

The new mental model is a running note. Read `red`, write a note, then carry that note while reading `rest`, `calmly`, and `now`. For the blue sentence, begin with `blue` instead. The later words are identical, but the carried notes do not have to be.

**But there's a catch:** the old window cut off the color before the model could use it. Do we have to keep cutting it off?

## 2. Reuse one update through time

A recurrent neural network, or RNN, repeats one learned update in sequence order. At each position it reads the current token row and the previous note, then writes the next note. The note is called a hidden state. Hidden means internal to the model, not a hidden answer supplied by the dataset.

The simple version used here is a vanilla RNN. The same update weights are reused at every position. You do not create a new unrelated model for each sentence length. A longer input means more applications of the same update.

We still fetch a word's embedding row using its ID. The recurrent update reads those numbers, not the numerical size of the ID. It uses them to rewrite the running note. Then the output layer reads that note and scores the possible next words. The stored row and the running note have different jobs: one belongs to a word ID, the other changes with the sequence.

This makes recurrence useful without making it magical. The note has a fixed width. It is rewritten repeatedly. Relevant evidence can be kept, distorted, or forgotten. Having a route from the opening color to the final prediction does not prove that training will learn a useful route.

**The note gives the clue a way to survive. Did the model actually learn to keep it?** We still need to check its answers.

## 3. Keep the practice answers one step ahead

The input row is `red wait quietly now`. Its expected output row is `wait quietly now warm`. Each answer sits one position ahead of its input word. At the last position the model has read `now` and all earlier words, but it has not read the answer `warm`.

That one-word shift keeps the answer out of its own input. If we showed the network `warm` while grading it on guessing `warm`, we'd be testing copying instead of next-word prediction.

The fixed-window comparison and recurrent model use these same next-token lessons. The window receives only its last seats. The recurrent model receives the ordered prefix through its running note. Both must learn useful scores from the same small corpus.

An optimizer changes the trainable weights after the loss sends teaching signals backward. Gradient clipping caps the size of that signal before the weight update. It can prevent an excessively large update. It cannot restore evidence the note has already discarded.

## 4. What happened when we tried it?

The recurrent training loss falls from 2.3361 to 0.2749. On the two held-out final answers, the fixed-window model reaches 50% and the recurrent model reaches 100%. Swapping only the first color while keeping the later words fixed changes the largest final probability by about 0.999414.

So this trained model did use a clue from beyond the three-word window. We didn't just change some filler and get lucky: the filler words are identical in the two tests, and only the color changes.

The models do not have equal capacities. The window model has 930 parameters; the recurrent model has 1,146. Their update budgets also differ slightly. This is evidence about a documented task comparison, not a fair test that changes only the architecture.

The held-out set has only two sentences. Their filler combination is new, but their words and rule are familiar. Getting these two answers right doesn't tell us how it will handle everyday language. Keep the claim as small as the evidence.

**But there's a catch:** correct behavior on a short delay does not establish reliable memory over a long delay.

## 5. Look at the note without worshipping it

The notebook records the distance between the red and blue hidden states after each read position. Those distances remain nonzero through the final word. The static plot and short animation show the same measured values; the animation does not invent a fading curve.

A distance answers whether two internal states differ. It does not answer whether the difference matters to the output. Two notes can differ in an irrelevant direction. Two useful notes can also be close together while the output layer remains sensitive to their small difference.

That is why you inspect the state trace beside final-token accuracy and the color-swap intervention. The three views ask different questions: did the state change, did the answer change, and was the answer right? None should impersonate the others.

The pictured fading notecard is a caution about possible memory loss. It is not a claim that the measured distances in this run decrease. In fact, a trained model may amplify a useful distinction as it approaches a prediction.

## 6. Connect to gates without repeating their lesson

A long short-term memory network, or LSTM, adds learned controls around a more protected route for carried information. The existing [sequence-memory lesson](../01-sequence-memory/README.md) measures why signals can fade and how those controls help. Return there for the gate experiment rather than treating this small language task as a substitute.

A gated recurrent unit, or GRU, offers another controlled running-note design. Gates help decide what to retain and rewrite. They do not guarantee perfect recall, and they do not by themselves let a writer reopen every source position.

### Use LSTM memory inside a language model

The LSTM example follows a familiar path: fetch word rows, update the running note, score the
next word. LSTM carries both hidden and cell state, like a working note and a more protected
memory line with learned controls over what to keep. These are numerical vectors, not text.
The output layer reads the hidden output at each position. The cell state helps carry memory;
it isn't a second answer to grade.

We already paired each input with the following word, so the model doesn't need to shift
the answers again. It inherits the same practice loop as CBOW. Feedback reaches the embedding,
LSTM, and output layer through one loss. Check the measured embedding-weight change:
the word rows really did learn along with the rest of the model.

Generation feeds back the model's own choice with no answer tensor. Changing a future input
token must leave earlier logits unchanged; the notebook checks that directly. This simple
generation helper recomputes the prefix and uses a fixed step budget. It does not demonstrate
efficient inference or general language quality.

This run shows how to connect an LSTM to the shared loop, not that LSTM wins a fair contest
against RNN. The original RNN keeps its own clipping and training budget.

## 7. Failure checks and the next question

Check that future answer words never enter an earlier prediction. Read training loss beside held-out behavior. Report parameter counts. Never call two test rows a benchmark or a nonzero hidden-state distance proof of understanding.

You can now continue one stream using a running note. What if the full input is available, but the required answer is a separate sequence? The next notebook gives one component the reading job and another the writing job.

**Durable summary:** recurrence creates a route for an earlier clue; measured behavior, not the existence of a hidden state, tells you whether this trained model used it.

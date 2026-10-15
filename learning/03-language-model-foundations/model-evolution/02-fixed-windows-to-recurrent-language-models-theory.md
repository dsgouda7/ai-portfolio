# Running Notes for Text: Handwritten Theory Notes

## 1. Resume the exact complaint

You left two prefixes at the window boundary: `red rest calmly now` and `blue rest calmly now`. Their last three words match, but their required answers differ. A fixed-window predictor gives them identical predictions because it receives identical inputs. The measured final-token accuracy was 50%.

Keep that world unchanged. The recurrent notebook uses the same eight sentences, the same six training rows, and the same two held-out rows. It does not solve the problem by changing the labels or supplying a secret extra input. It changes how earlier words can remain available.

The new mental model is a running note. Read `red`, write a note, then carry that note while reading `rest`, `calmly`, and `now`. For the blue sentence, begin with `blue` instead. The later words are identical, but the carried notes do not have to be.

**Complaint:** chopping off the clue was unavoidable for the old input contract. Must every predictor have that contract?

## 2. Reuse one update through time

A recurrent neural network, or RNN, repeats one learned update in sequence order. At each position it reads the current token row and the previous note, then writes the next note. The note is called a hidden state. Hidden means internal to the model, not a hidden answer supplied by the dataset.

The simple version used here is a vanilla RNN. The same update weights are reused at every position. You do not create a new unrelated model for each sentence length. A longer input means more applications of the same update.

Text still needs a representation. Token IDs are addresses. Each ID retrieves a trainable embedding row. The recurrent update reads that row; it does not treat the ID's numerical size as word meaning. The output layer turns each current note into scores for the following token.

This makes recurrence useful without making it magical. The note has a fixed width. It is rewritten repeatedly. Relevant evidence can be kept, distorted, or forgotten. Having a route from the opening color to the final prediction does not prove that training will learn a useful route.

**Complaint:** architecture provides access over time; behavior still needs measurement.

## 3. Keep the practice answers one step ahead

The input row is `red wait quietly now`. Its expected output row is `wait quietly now warm`. Each answer sits one position ahead of its input word. At the last position the model has read `now` and all earlier words, but it has not read the answer `warm`.

This shift is a training alignment rule. It prevents the current answer from being its own input. A network that saw `warm` while being graded on predicting `warm` would answer an easier, wrong question.

The fixed-window comparison and recurrent model use these same next-token lessons. The window receives only its last seats. The recurrent model receives the ordered prefix through its running note. Both must learn useful scores from the same small corpus.

An optimizer changes the trainable weights after the loss sends teaching signals backward. Gradient clipping caps the size of that signal before the weight update. It can prevent an excessively large update. It cannot restore evidence the note has already discarded.

## 4. What the measured run actually establishes

The recurrent training loss falls from 2.3361 to 0.2749. On the two held-out final answers, the fixed-window model reaches 50% and the recurrent model reaches 100%. Swapping only the first color while keeping the later words fixed changes the largest final probability by about 0.999414.

Those measurements support a narrow conclusion: this trained recurrent model used information beyond the three-word window on this controlled task. It did not merely become sensitive to the filler words, since the filler words match between the two probes.

The models do not have equal capacities. The window model has 930 parameters; the recurrent model has 1,146. Their update budgets also differ slightly. This is evidence about a documented task comparison, not a clean isolation of every architectural, optimization, and capacity effect.

The held-out set has only two sentences. Their filler combination is new, but their words and rule are familiar. A perfect score here is not a credible estimate of performance on arbitrary language. Keep the claim as small as the evidence.

**Complaint:** correct behavior on a short delay does not establish reliable memory over a long delay.

## 5. Look at the note without worshipping it

The notebook records the distance between the red and blue hidden states after each read position. Those distances remain nonzero through the final word. The static plot and short animation show the same measured values; the animation does not invent a fading curve.

A distance answers whether two internal states differ. It does not answer whether the difference matters to the output. Two notes can differ in an irrelevant direction. Two useful notes can also be close together while the output layer remains sensitive to their small difference.

That is why you inspect the state trace beside final-token accuracy and the color-swap intervention. The three views ask different questions: did the state change, did the answer change, and was the answer right? None should impersonate the others.

The pictured fading notecard is a caution about possible memory loss. It is not a claim that the measured distances in this run decrease. In fact, a trained model may amplify a useful distinction as it approaches a prediction.

## 6. Connect to gates without repeating their lesson

A long short-term memory network, or LSTM, adds learned controls around a more protected route for carried information. The existing [sequence-memory lesson](../01-sequence-memory/README.md) measures why signals can fade and how those controls help. Return there for the gate experiment rather than treating this small language task as a substitute.

A gated recurrent unit, or GRU, offers another controlled running-note design. Gates help decide what to retain and rewrite. They do not guarantee perfect recall, and they do not by themselves let a writer reopen every source position.

## 7. Failure checks and the next question

Check that future answer words never enter an earlier prediction. Read training loss beside held-out behavior. Report parameter counts. Never call two test rows a benchmark or a nonzero hidden-state distance proof of understanding.

You can now continue one stream using a running note. What if the full input is available, but the required answer is a separate sequence? The next notebook gives one component the reading job and another the writing job.

**Durable summary:** recurrence creates a route for an earlier clue; measured behavior, not the existence of a hidden state, tells you whether this trained model used it.

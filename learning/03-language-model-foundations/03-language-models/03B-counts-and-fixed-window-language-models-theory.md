# Counts and Fixed Windows: Handwritten Theory Notes

**Chapter 03B:** [03A](03A-training-skeleton-and-word-objectives-theory.md) averaged word rows
and lost their order. Give the words separate seats instead, then notice who got left outside.
The color-clue sentences make that missing clue easy to spot and stay with us in 03C.
[Notebook](03B-counts-and-fixed-window-language-models.ipynb)
· [Code-only Python](03B-counts-and-fixed-window-language-models.py).

## 1. Ask what the predictor receives

You want the last word in `red wait quietly now warm`. Replace `red` with `blue` and the answer becomes `cool`. The middle words delay the clue. They do not decide the answer. This tiny world makes a useful question unavoidable: does the predictor receive the evidence that determines its answer?

Split each sentence at spaces. Each piece is a token, a unit you count or predict. There is no reason to introduce a complicated text splitter yet. There is also no reason to turn words into numbers before a tensor needs them. A count table can use the strings directly.

The corpus has eight sentences. Six are used for training. Two hold out the combination `rest calmly`, once with each color. The words themselves already occur in training. This is a small test of a new combination, not a test of vocabulary growth or general English.

**But there's a catch:** before choosing a clever predictor, inspect what its input can possibly reveal.

## 2. One previous word leaves two answers

A unigram count records single tokens. A bigram count records neighboring pairs. For prediction, you can turn the second table around: after this word, how often did each next word occur?

After `now`, the training corpus contains three `warm` answers and three `cool` answers. The best single constant choice therefore gets 50% of these final answers right. More precision in the counts cannot identify the missing color. The same previous word belongs to both histories.

Counts are not useless because this example defeats them. They give you a cheap, inspectable baseline. Some previous words determine their continuation perfectly in this tiny corpus. Others have several valid continuations. A higher average score can hide the exact position where the useful clue has disappeared.

Longer count contexts can include more clues, but their exact combinations occur less often. That is count sparsity: a longer table key has fewer observations behind it. Smoothing repairs estimates when counts are small or missing. It does not reveal a word that was excluded from the key.

**But there's a catch:** each count table treats its history as a separate case. Could a network learn useful features that different histories can share, without losing word order?

## 3. Give words addresses and positions seats

A tensor needs integer token IDs. An ID is an address, not a quantity. A larger ID does not mean a warmer color or a more important word. Changing the address assignment consistently should not change the task.

Each address retrieves a trainable row of numbers. That row is an embedding. The network adjusts the row while it learns the prediction task. It is not born knowing what a word means. On this corpus, it only has the evidence supplied by these few sentences.

Picture a row of seats. Put one retrieved word row in each seat, then join the rows in seat order. The next layer can react differently to the first and third seats. This is an ordered fixed-window language model: a next-token predictor whose input is a fixed number of preceding pieces.

A bag has no seats. If you average the same three rows, shuffling the pieces leaves the mathematical average unchanged. In the executed notebook, floating-point arithmetic leaves only a tiny rounding difference, about one ten-millionth. The ordered predictor's untrained scores differ by about 0.515 for the same permutation.

So the ordered model can notice the swap. That doesn't mean its untrained guess is right. Keeping a clue and learning to use it are two different things.

**But there's a catch:** giving the pieces seats repairs lost order, but the row of seats still has a left edge.

## 4. Training cannot move an input boundary

The neural model learns from every available next-token pair in the six training sentences. Its training loss falls from 2.3512 to 0.4480 in the recorded run. Loss measures how poorly the next-token scores fit the observed answers; lower is better for that objective.

Do not read the remaining loss as a mysterious failure to memorize. Some identical visible contexts require different answers. The optimizer cannot make one deterministic input demand two different top choices at once.

Now inspect the held-out prefixes. One is `red rest calmly now`; the other is `blue rest calmly now`. Trim either to its last three words and you get `rest calmly now`. The network receives the same addresses in the same seats. Its two probability rows have a maximum difference of exactly zero in the measured run.

The final-word accuracy is 50% in this run. We measured that score. The matching predictions have a simpler explanation: we gave this deterministic model identical inputs, so it had to return identical outputs. Another badly trained model could guess an unrelated word and score even worse. It would still give both inputs the same answer.

**But there's a catch:** better feature learning cannot recover a color removed before the forward pass.

## 5. A wider view is not yet a better trained model

Change the visible-word exercise from three words to four. The color reappears, and the two inspected inputs stop being identical. That brings the clue into view. It doesn't train a four-word network.

The existing input layer has seats for three retrieved rows. A fair four-word comparison would rebuild that layer and fit the model again under a documented budget. Keep those claims separate: seeing the needed clue is necessary here, but access alone does not guarantee a correct prediction.

Real fixed-context systems have larger vocabularies, wider feature rows, and far more examples. The same question survives: where does their input begin? A boundary is a boundary even when the table is expensive.

## 6. Failure checks and the next complaint

Do not mistake token addresses for meaningful numbers. Do not call an average ordered. Do not report an untrained score difference as a language capability. Do not confuse new filler combinations with a broad held-out benchmark. Most importantly, do not blame the optimizer for information that never reached the model.

You have repaired the bag's lost order and exposed the window's missing clue. Instead of extending the row of seats indefinitely, could you pass a note from one word to the next? [The recurrent comparison](03C-fixed-windows-to-recurrent-language-models.ipynb) keeps the same sentences and asks that question directly.

**Durable summary:** a better predictor can use its visible evidence better, but it cannot recover a clue removed at the input boundary.

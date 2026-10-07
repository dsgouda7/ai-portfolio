# Tokenization and Embeddings: Handwritten Theory Notes

## 1. Why text needs a boundary

A model consumes fixed-width numeric rows. A contract arrives as a string with words, punctuation, rare names, and multilingual fragments. Character tokens never become unknown, but they make sequences long. Whole-word tokens are shorter, but every unseen spelling becomes an out-of-vocabulary failure.

The useful complaint is not "strings are not numbers." It is: **which reusable pieces preserve coverage without wasting the context window?**

Token boundaries change the task itself. A long sequence consumes more compute and leaves less room for useful context. A huge vocabulary gives every row fewer learning examples and makes the output head wider. There is no universally correct split; the vocabulary is a trained compression code for a particular corpus and objective.

## 2. BPE grows reusable pieces

Start each word as characters plus an end marker. Count adjacent pairs. Merge the most frequent pair. Repeat.

The memorable model is a zipper. Each merge closes one common seam. Frequent fragments become single pieces while rare words can still fall back to smaller pieces.

A merge budget creates a trade-off:

| Too few merges | Too many merges |
|---|---|
| long sequences | large vocabulary |
| more repeated character work | more rare, brittle pieces |
| excellent coverage | weaker reuse |

The theory notebook tracks `non-disclosure` through real merge steps and measures its token count. The animation is paired with a static storyboard so the evidence survives without JavaScript.

The merge list is ordered. Later pieces depend on earlier pieces existing, so you cannot treat the learned pairs as an unordered dictionary. Encoding replays the ranked merge decisions; decoding concatenates the stored byte or text pieces. The toy notebook begins from characters because the mechanics remain visible. Production GPT-2 begins from bytes so every possible input remains representable, including unusual Unicode sequences after UTF-8 encoding.

**Complaint:** pieces now exist, but the model still cannot consume strings.

## 3. Token IDs are addresses

A vocabulary assigns each piece an integer ID. ID 500 is not five times ID 100. It is row 500 in an embedding table.

`token ID -> one selected embedding row`

Training changes useful rows because prediction errors send gradients into the rows that participated. Similar use can create similar geometry, but an embedding map is evidence about the learned objective, not a complete explanation of meaning.

`nn.Embedding` is therefore not a magic semantic database. It is a trainable table with sparse participation: IDs used in a lesson select rows, and those rows receive gradient through the downstream objective. Two words can move closer because they help predict the same targets. They can also stay far apart if the corpus or task never creates shared pressure.

**Complaint:** one lookup row gives the same vector every time. The word `bank` still begins from one static row in both "river bank" and "bank loan."

## 4. Static lookup versus contextual representation

Static embedding lookup answers: **which token type is this?**

Contextualization answers: **what role is this token playing here?**

A contextual model begins with the static row, then mixes information from neighboring token representations. The same `bank` row can therefore become two different token representations after context mixing.

Tokenization and embeddings do not perform that mixing by themselves. Transformers will own the contextual mechanism.

This boundary prevents a common overclaim. Nearest neighbors in a static table show geometric consequences of training, not sentence-level understanding. Word2Vec and GloVe are valuable optional history because they make that geometry explicit. They still assign one learned type vector to every occurrence of `bank`.

## 5. Padding creates fake positions

Batches need a rectangle, but sentences have different lengths. Padding fills the unused positions. Those pad IDs are bookkeeping, not language.

A mask marks real positions:

`1 = real token, 0 = padding`

Without a mask, a model can appear to improve by learning the easy repeated pad target. The wrong loss averages over every rectangle cell. The correct loss sums only real-token losses and divides by the count of real targets. In PyTorch, `ignore_index` implements the same contract; the lab proves it against an explicit reduction.

Three similarly shaped controls have different jobs:

1. A **padding mask** marks positions that are not real input evidence.
2. A **loss mask** removes positions that are not real prediction targets.
3. A **causal mask** prevents a real earlier position from reading a real future position.

One mask cannot be assumed to perform all three jobs. Recurrent models get causal direction from their left-to-right computation, but they still need padding and loss handling. Transformers will need padding and causality expressed explicitly in attention.

**Complaint:** masking fixes the denominator, but it does not automatically prevent future-token leakage. Causal attention will require another mask with a different job.

## 6. Production bridge

GPT-2 uses byte-level BPE. Bytes guarantee coverage, learned merges shorten common sequences, and spaces can become part of token pieces. `tiktoken` exposes the production contract:

`text -> token IDs -> decoded token bytes -> text`

Production vocabularies are pretrained artifacts. You inspect and use their merge decisions; you do not retrain them inside an application notebook.

Round-trip checks matter. Token strings may display whitespace markers or partial byte sequences that look strange in isolation. The reliable contract is that encoding followed by decoding reconstructs the original text. When debugging, inspect IDs, raw token bytes, and the decoded result rather than guessing from a prettified token label.

## 7. Practical failure modes

| Failure | Check |
|---|---|
| IDs treated as numeric features | verify IDs enter `nn.Embedding` as `torch.long` |
| padding row learns content | set `padding_idx` and inspect its gradient |
| masked loss still changes with extra pads | compare explicit real-token reduction |
| decoded pieces look odd | inspect bytes and whitespace-bearing tokens |
| static vectors described as contextual | compare lookup output before context mixing |

**Durable summary:** tokenization chooses reusable text pieces, IDs address embedding rows, and masks keep batch bookkeeping from becoming a learning target.

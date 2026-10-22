# Chapter 02: Tokenization and Embeddings

**Contents:** [directory map](CONTENTS.md)

Turn a pet-sitter's note into pieces, integer addresses, and learned rows.
Then check why blank batch slots must not count as evidence.

## Before you start

Use [Stage 02 — PyTorch Foundations](../../02-pytorch-foundations/README.md), or the
[training-loop refresher](../../02-pytorch-foundations/quick-refresher/pytorch-training-skeleton.ipynb)
if you can pass its closing check.
[Chapter 01B](../01-sequence-memory/01B-sequence-memory-pytorch-lab.ipynb) already used
IDs, shifted targets, and masks to study memory; here you explain and test those choices.

## Follow the embedding thread without backtracking

Each notebook includes a self-contained carry-in, the same pipeline map, and
experiment-state labels. Earlier links are references, not a request to recover
missing values or kernel state.

| Step | New question | What carries forward |
|---|---|---|
| [01B](../01-sequence-memory/01B-sequence-memory-pytorch-lab.ipynb) | Can fixed note IDs feed a trainable sequence reader? | Lookup supplies rows; the LSTM supplies phrase memory |
| [02A](02A-tokenization-and-embeddings-theory.ipynb) | What did lookup do, and what could teach its rows? | The **lookup microscope**, `the ___ sat`, isolates hand-picked row selection |
| [02B](02B-tokenization-and-embeddings-pytorch-lab.ipynb) | What feedback makes stored rows useful? | The **learning game**, `cat/dog -> sat/slept; bird -> flew/sang`, trains fresh rows on six supplied cards |
| [03A](../03-language-models/03A-training-skeleton-and-word-objectives.ipynb) | How can text supply the practice questions? | The same cat sentence connects lookup, pooling, and different objectives |

Hand-picked, random, and trained tables are marked separately. The three-word
microscope is an arithmetic close-up, not another training corpus; the GPT-2
tokenizer, controlled learning game, and later story models have separate vocabularies.

## Read in order

1. [02A — Tokenization and Embeddings Theory](02A-tokenization-and-embeddings-theory.ipynb) ·
   [handwritten notes](02A-tokenization-and-embeddings-theory.md) — Discover BPE, separate
   addresses from compact rows, check one-hot selection and ID renumbering, and expose a padding failure.
2. [02B — Tokenization and Embeddings PyTorch Lab](02B-tokenization-and-embeddings-pytorch-lab.ipynb) ·
   [handwritten notes](02B-tokenization-and-embeddings-pytorch-lab-theory.md) —
   Build the small BPE loop, inspect GPT-2 with `tiktoken`, train an embedding table,
   inspect probability shortlists and sampling, and take one visible optimizer step:
   stored rows change while input and answer IDs stay fixed.

02B also follows one concrete value path through two actual layers:
`cat -> ID 1 -> embedding [1, 3] -> scores [3, 1, 4]`.
The [generated illustration](images/02-07-word-id-to-embedding-to-scores.png) and
executable arithmetic use the same hand-picked values. It then fetches the actual
trained model's four-number embedding separately from its seven prediction scores.

IDs are registry numbers; embeddings are stored profiles across abstract learned
sensor channels. These are not literal astronomical measurements. The profile and
numeric-trace diagrams can be rebuilt offline with the
[sensor diagram renderer](../03-language-models/generate_sensor_analogy_diagrams.py).

## Run it

From this directory, run [`setup.ps1`](setup.ps1) on Windows or [`setup.sh`](setup.sh) on
Linux/macOS. It creates a chapter `.venv`, installs [requirements](requirements.txt),
and assigns kernel `stage03-language-model-02-tokenization`.
Restart and run each notebook independently. The lab uses GPT-2's tokenizer, not its
pretrained model weights; `tiktoken` may fetch its vocabulary on first use.

The [illustration manifest](images/02-illustration-prompts.json) records each
prompt and its accuracy checks. Regenerating an illustration uses the existing
[image generator](../03-language-models/generate_illustrations.py), Azure CLI sign-in,
and the deployment's normal charges. Running the notebook doesn't call an image service.

**Next:** [Chapter 03 — Language Models](../03-language-models/README.md).
[03A](../03-language-models/03A-training-skeleton-and-word-objectives.ipynb) ·
[theory notes](../03-language-models/03A-training-skeleton-and-word-objectives-theory.md)
asks how ordinary text can supply the practice questions that train those rows.

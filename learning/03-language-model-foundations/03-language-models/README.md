# Chapter 03: Language Models

Text models learn by predicting words. This chapter follows that idea from
learning word vectors, through memory that carries a clue along a sentence, to a
model that reads a whole input before writing a separate answer.

**Before this:** [Chapter 02 — Tokenization and embeddings](../02-tokenization-and-embeddings/README.md).
**After this:** [Stage 04 — Transformers](../../04-transformers/README.md).

## Setup

Creates a local Python environment, installs the packages and registers the
`stage03-language-model-03-models` notebook kernel. Run once, then pick that kernel.
[setup.ps1](setup.ps1) (Windows) · [setup.sh](setup.sh) (Linux/macOS) · [requirements.txt](requirements.txt)

## Main lessons, in order

### 03A — How word vectors learn

The input IDs never change, so what does training change? Trains a CBOW model to
fill in a missing word in children's stories, then compares the word rows before and
after. Ends with a short preview of a model that reads, then writes.
[Notebook](03A-training-skeleton-and-word-objectives.ipynb)

The same story in short prose, without code: IDs as addresses, rows that learn,
and why averaging clue words loses their order.
[Theory notes](03A-training-skeleton-and-word-objectives-theory.md)

All of the notebook's code in one runnable script.
[Code-only version](03A-training-skeleton-and-word-objectives.py)

### 03B — From counting words to recurrent memory

Why can't counting or a fixed window of recent words predict an ending that
depends on an earlier word? Builds both, shows the blind spot, then trains an
RNN and an LSTM that carry the clue forward.
[Notebook](03B-counts-to-recurrent-language-models.ipynb)

What ordered seats fix, what carried memory fixes, and what still breaks.
[Theory notes](03B-counts-to-recurrent-language-models-theory.md)

All of the notebook's code in one runnable script.
[Code-only version](03B-counts-to-recurrent-language-models.py)

### 03C — Reading, then writing: the encoder-decoder

How can a model answer a whole input in a different order? Splits the job into a
reader and a writer, trains them to reverse "the cat sat on the mat", and shows why
one final memory limits the writer, which motivates attention.
[Notebook](03C-recurrent-encoder-decoder-to-attention.ipynb)

How the writer's mistakes teach the reader, what teacher forcing supplies, and why
looking back at the source differs from remembering one summary.
[Theory notes](03C-recurrent-encoder-decoder-to-attention-theory.md)

All of the notebook's code in one runnable script.
[Code-only version](03C-recurrent-encoder-decoder-to-attention.py)

## Quick revision notes

Turning text into model input: tokenizing into IDs, looking up learned rows, and
padding uneven sentences into batches. Short code you can copy by hand.
[Text to batches](03C-minimal-tokens-embeddings-batches-notes.md)

A complete encoder-decoder in short, copyable code: how earlier models led here,
the one-step answer shift, an LSTM reader and writer, training, generation and
evaluation, with experiments that deliberately break it.
[LSTM encoder-decoder](03C-minimal-lstm-encoder-decoder-notes.md)

The same task with attention in place of one final memory. Lives in Stage 04.
[Transformer encoder-decoder](../../04-transformers/05b-minimal-transformer-encoder-decoder-notes.md)

## Optional labs

The larger experiment behind 03A: trains CBOW, Skip-gram and a reader/writer on
8,000 stories and compares their word maps, held-out losses and closest words.
[Notebook](optional/03A-word-objective-comparison-lab.ipynb) · [Code-only version](optional/03A-word-objective-comparison-lab.py)

Hands-on proofs behind 03C: cut the reader-to-writer link, train only the reader,
compare a one-memory writer with one that looks back, and test both on unseen inputs.
[Notebook](optional/03C1-encoder-decoder-worked-details.ipynb) · [Code-only version](optional/03C1-encoder-decoder-worked-details.py)

## Shared code

Turns word windows into input/answer pairs for each word-prediction task and
batches them with a fixed random seed.
[language_data.py](language_data.py)

The common model and training loop: word lookup, scoring and the
predict-grade-update cycle shared by every word model in the chapter.
[language_model.py](language_model.py)

Checks a model on held-out examples before and after training, without changing it.
[language_validation.py](language_validation.py)

Downloads and caches the children's-stories text, builds word windows and runs the
larger comparison used by 03A and its optional lab.
[real_text_embeddings.py](real_text_embeddings.py)

The fixed list-reversal data for 03C: word IDs, start/stop markers, shifted practice
inputs and a small generator.
[encoder_decoder_example.py](encoder_decoder_example.py)

Traces how prediction errors flow back into the word table in 03A's reader/writer
preview, and rebuilds its saved animations.
[encoder_decoder_feedback.py](encoder_decoder_feedback.py)

Records the 03C model's real reading and writing steps for the step-through player.
[encoder_decoder_walkthrough.py](encoder_decoder_walkthrough.py)

## Image builders

Rebuild the saved CBOW before/after word profiles and the training-window storyboard.
[generate_cbow_static_views.py](generate_cbow_static_views.py)

Rebuild the six sketches of the counts-to-memory-to-attention story.
[generate_model_evolution_views.py](generate_model_evolution_views.py)

Rebuild the space-probe catalogue diagrams used to explain word vectors.
[generate_sensor_analogy_diagrams.py](generate_sensor_analogy_diagrams.py)

Regenerate the decorative artwork with an Azure-hosted image model; may incur charges.
[generate_illustrations.py](generate_illustrations.py)

## Tests

Offline checks that the lessons, code-only versions and saved visuals stay correct.
Run all of them with `.\.venv\Scripts\python.exe -m unittest discover -p "test_*.py"`.

Word windows, task pairs, the CBOW lesson's training and its saved visuals.
[test_real_text_embeddings.py](test_real_text_embeddings.py)

Data loaders, model base classes and the training-step order.
[test_language_contracts.py](test_language_contracts.py)

Held-out checking counts, answer alignment and untouched model state.
[test_language_validation.py](test_language_validation.py)

Error flow back into the word table in 03A's reader/writer preview.
[test_encoder_decoder_feedback.py](test_encoder_decoder_feedback.py)

That the 03C step-through player shows the model's real steps.
[test_encoder_decoder_walkthrough.py](test_encoder_decoder_walkthrough.py)

The cat examples, reader/writer behavior and the 03C1 experiments.
[test_shared_cat_examples.py](test_shared_cat_examples.py)

That the optional comparison lab runs on its own and matches the lesson's models.
[test_word_objective_comparison_lab.py](test_word_objective_comparison_lab.py)

## Folders

Diagrams, animations, players and measured results shown in the lessons.
[images/](images/)

Cached children's stories and the fixed list-reversal data.
[data/](data/)

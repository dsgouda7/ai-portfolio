# Pretraining Data Pipelines: Handwritten Theory Notes

## 1. Data Is the Model's Curriculum

A folder of text is not yet training data. Before a model can learn from it, you must decide:

- which documents belong to training and validation;
- how repeated material is handled;
- how much practice each source receives;
- which tokenizer defines the vocabulary;
- where document boundaries appear;
- how tokens are packed into fixed training blocks;
- how every artifact is identified and verified.

Mental model: **the model practices whatever the pipeline repeats.** Data preparation is therefore curriculum design, not file conversion.

**Track position:** this pipeline builds the input to the Transformer learning spine.

```text
raw documents -> split and audit -> fit tokenizer -> token IDs
-> mark boundaries -> pack blocks -> verified training shards
-> embeddings -> Transformer -> logits -> loss -> backpropagation
```

The pipeline does not learn attention weights. It decides which examples, repetitions, boundaries, and held-out evidence will drive those later updates.

## 2. Split Complete Documents First

Training data teaches the model. Validation data measures whether that learning transfers to unseen material. Validation stops being useful if pieces of the same chapter appear on both sides.

The safe order is:

```text
inventory documents
-> assign whole documents to train or validation
-> freeze that ownership
-> fit tokenizer on training documents only
-> encode and pack each split independently
```

A random token split leaks nearby prose, character names, phrasing, and narrative continuation across the boundary. Validation can then look good because the model has already practiced adjacent text.

The durable rule is: **split at the highest meaningful unit before learning anything from the content.**

## 3. Duplicates Change the Practice Budget

An exact duplicate makes one passage count twice. A lightly edited duplicate can do nearly the same while escaping exact hashes.

Use two complementary checks:

- **Exact hash:** detects byte-identical or normalized-identical documents.
- **Near-duplicate similarity:** compares overlapping phrase shingles and flags strongly shared wording.

Near-duplicate thresholds are policy choices. Too loose joins unrelated documents; too strict misses edited copies. Keep detected groups in the manifest instead of silently deleting them.

Why this matters: duplicates can cross the train/validation boundary, inflate apparent generalization, and overrepresent one source during training.

Memory aid: **a duplicate is not just storage waste; it is an extra vote during learning.**

## 4. Source Mixture Is an Explicit Budget

Corpora contain sources of different sizes. Sampling documents equally and sampling tokens equally create different curricula.

- **Equal-document sampling:** small documents or small collections receive relatively more practice.
- **Proportional-token sampling:** large sources dominate according to their size.

Neither policy is automatically correct. The mixture should state what behavior the model is expected to practice, then record that choice so the run can be reproduced.

The durable idea is: **source percentages are behavioral priorities expressed as data.**

## 5. The Tokenizer Is Part of the Model

The tokenizer turns raw text into the token IDs the model will see forever. Its vocabulary, merge rules, special tokens, and normalization choices define the model's input language.

A byte-level BPE tokenizer starts from byte-safe pieces and learns common larger fragments. This avoids unknown-character dead ends while keeping frequent text compact.

Fit the tokenizer on training documents only. Even though tokenizer fitting has no Transformer weights, letting it inspect validation text still leaks information about validation spelling and frequency.

Once model training begins, freeze the tokenizer. Changing it later changes what every token ID means and invalidates the embedding table.

Memory aid: **the tokenizer is not preprocessing attached to the model; it is the model's alphabet and word-part dictionary.**

## 6. EOD Makes Document Boundaries Learnable

Concatenating documents without a marker creates a false transition: the last sentence of one chapter appears to lead naturally into the first sentence of another.

An end-of-document token makes the edge explicit:

```text
document A tokens -> EOD -> document B tokens -> EOD
```

The model can now learn that a document ended instead of treating unrelated prose as one continuous sequence. EOD is a real prediction target, not padding and not ignored space.

The boundary token should encode as one stable token. If it fragments, the boundary contract is ambiguous.

## 7. Packing Turns Streams Into Training Blocks

Transformers train on fixed-length blocks. Padding every short document to block width preserves isolation but wastes most positions on emptiness.

Packing takes already bounded documents, appends EOD to each, joins them into one split-specific token stream, and slices complete blocks:

```text
train documents + EOD -> training stream -> full training blocks + one tail
validation documents + EOD -> validation stream -> full validation blocks + one tail
```

Packing improves useful-token density while EOD preserves visible boundaries. Train and validation must be packed independently so no block crosses the split.

Memory aid: **padding leaves empty seats; packing fills the bus but keeps a sign between groups.**

## 8. Shards and Manifests Close the Trust Boundary

Binary shards make token loading fast, but a file that opens is not necessarily the file you intended to train on.

A trustworthy dataset artifact records:

- source document identities and exclusions;
- train/validation ownership;
- tokenizer identity and special tokens;
- vocabulary size and token dtype;
- context length and packing statistics;
- duplicate groups and threshold;
- token counts;
- hashes of tokenizer and shard bytes.

After writing, reload everything from disk. Verify hashes, ID ranges, vocabulary size, dtype, and decoded sample windows. This proves that the stored artifacts, not just in-memory objects, satisfy the contract.

The manifest is the dataset's **receipt and tamper seal**. Notebook 06 should refuse to train when the receipt no longer matches the bytes.

## 9. Practical Failure Modes

- **Splitting after tokenization:** adjacent content can leak across train and validation.
- **Fitting the tokenizer on all documents:** validation influences the model's input vocabulary.
- **Removing duplicates silently:** the final curriculum can no longer be explained or reproduced.
- **Letting the largest source dominate accidentally:** undeclared mixture becomes undeclared behavior.
- **Concatenating without EOD:** the model practices false document transitions.
- **Treating EOD as padding:** EOD is meaningful content and should participate in next-token learning.
- **Packing train and validation together:** block boundaries destroy split isolation.
- **Casting before checking token range:** large IDs can overflow a narrow integer dtype.
- **Trusting successful writes:** always hash, reload, and decode stored artifacts.
- **Changing tokenizer after training starts:** existing embedding rows no longer match token meanings.

The durable model is: **split first, audit repetition, declare the source budget, freeze the tokenizer, mark boundaries, pack each split separately, and make every output verifiable.**

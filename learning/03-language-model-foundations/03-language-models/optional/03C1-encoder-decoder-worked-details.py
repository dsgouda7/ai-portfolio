"""Code companion for 03C1-encoder-decoder-worked-details.ipynb. Run from this folder or the chapter folder."""

# %% 2. Teacher forcing keeps the training example on track
import json
import random
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
import sys
# Helpers live in the chapter folder, one level above this optional lab.
CHAPTER_DIR = next(
    folder for folder in (
        Path.cwd(), Path.cwd().parent,
        Path.cwd() / "learning" / "03-language-model-foundations" / "03-language-models",
    ) if (folder / "language_data.py").is_file()
)
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))
from language_data import CAT_SENTENCE, CAT_WORDS, CAT_FAMILY_WORDS
random.seed(17); np.random.seed(17); torch.manual_seed(17)
torch.set_num_threads(1)
started = time.perf_counter()
PAD = 0
word_to_id = {word: index + 1 for index, word in enumerate(CAT_FAMILY_WORDS)}
BOS, EOS = len(word_to_id) + 1, len(word_to_id) + 2
V = len(word_to_id) + 3
id_to_word = {index: word for word, index in word_to_id.items()}
id_to_word.update({PAD: "<pad>", BOS: "<bos>", EOS: "<eos>"})


def encode_words(words):
    return [word_to_id[word] for word in words]


def decode_ids(ids):
    return [id_to_word[int(index)] for index in ids]


def reversal_targets(source):
    return torch.cat([source.flip(1), source.new_full((len(source), 1), EOS)], dim=1)


def practice_prefix(target):
    return torch.cat([target.new_full((len(target), 1), BOS), target[:, :-1]], dim=1)


assert len({PAD, BOS, EOS}) == 3
assert {PAD, BOS, EOS}.isdisjoint(word_to_id.values())
assert decode_ids(encode_words(CAT_WORDS)) == list(CAT_WORDS)

# %% 2. Teacher forcing keeps the training example on track
first_guess = "cat"  # CHANGE THIS: a hypothetical wrong word; "mat" is correct.
correct_target = list(reversed(CAT_WORDS)) + ["<eos>"]
practice_inputs = ["<bos>"] + correct_target[:-1]
print("Canonical source:", CAT_SENTENCE)
print("Practice input -> answer:", list(zip(practice_inputs, correct_target)))
print("Hypothetical generation: <bos> -> guessed", first_guess, "-> next input", first_guess)
print("Correct previous input at step two:", practice_inputs[1])
print("Takeaway: a wrong guess changes the next input, not the stored source.")

# %% Worked details: Original split construction and frozen-corpus check
rng=random.Random(31)
train,test={},{}
word_ids = tuple(word_to_id.values())
for length in range(2,9):
    unique=set()
    count=min(1200,len(word_ids)**length-40)
    while len(unique)<count+40:
        unique.add(tuple(rng.choice(word_ids) for _ in range(length)))
    rows=list(sorted(unique)); rng.shuffle(rows)
    # One length per bucket -> ID rectangles [N_train,L] and [40,L]; no blank slots are needed.
    train[length]=torch.tensor(rows[:count]); test[length]=torch.tensor(rows[count:])

# %% Worked details: Original split construction and frozen-corpus check
from encoder_decoder_example import load_cat_corpus

frozen_corpus = load_cat_corpus()
for split_name, generated_split in (("train", train), ("test", test)):
    for length, rows in generated_split.items():
        torch.testing.assert_close(rows, frozen_corpus[split_name][length], rtol=0, atol=0)
print("Original seeded source orders match the frozen main-demo corpus exactly.")

# %% Worked details: Original split construction and frozen-corpus check
for length in train:
    training_rows = {tuple(r) for r in train[length].tolist()}
    testing_rows = {tuple(r) for r in test[length].tolist()}
    assert training_rows.isdisjoint(testing_rows)
    assert len(testing_rows) == 40
    assert len(training_rows) == min(1200, len(word_ids)**length - 40)
    assert set(train[length].flatten().tolist() + test[length].flatten().tolist()) <= set(word_ids)
    print(f"Length {length}: {len(training_rows)} train, {len(testing_rows)} held out; "
          f"{len(word_ids)**length:,} possible word orders")
print("Takeaway: both models receive the same split; no test source appears in training.")
print("Generated word orders are a controlled access test, not a corpus of English sentences.")

# %% Worked details: Original split construction and frozen-corpus check
example_length = len(CAT_WORDS)  # CHANGE THIS: 2 through 8
example = test[example_length][0].tolist()
print("source words:", " ".join(decode_ids(example)))
print("required words:", " ".join(decode_ids(reversal_targets(torch.tensor([example]))[0])))

# %% 4. Open the reader/writer before training it
class ReaderWriter(nn.Module):
    def __init__(self,look_back):
        super().__init__(); self.look_back=look_back
        self.embedding=nn.Embedding(V,24)
        self.reader=nn.GRU(24,64,batch_first=True)
        self.writer=nn.GRUCell(24+(64 if look_back else 0),64)
        self.query=nn.Linear(64,64,bias=False) if look_back else None
        self.head=nn.Linear(128 if look_back else 64,V)
    def read(self,source):
        # Source IDs [B,L] -> rows [B,L,24] -> saved notes [B,L,64].
        # Final state [1,B,64] -> [B,64] drops the one-layer axis, not a source position.
        memory,last=self.reader(self.embedding(source)); return memory,last[0]
    def step(self,previous,state,memory):
        if self.look_back:
            # Project writer state [B,64] -> query [B,64]; unsqueeze -> [B,64,1].
            # Batched dot products [B,L,64] @ [B,64,1] -> [B,L,1] -> [B,L]; divide by sqrt(64) to tame score scale.
            scores=torch.bmm(memory,self.query(state).unsqueeze(2)).squeeze(2)/8
            # Normalize over source positions: [B,L] -> [B,L], one unit-sum reading budget per example.
            weights=scores.softmax(-1)
            # [B,L] -> [B,1,L] @ notes [B,L,64] -> [B,1,64] -> [B,64], a weighted source blend.
            context=torch.bmm(weights.unsqueeze(1),memory).squeeze(1)
            # Previous IDs [B] -> [B,24]; append context features [B,64] -> [B,88] for the next writer update.
            state=self.writer(torch.cat([self.embedding(previous),context],-1),state)
            # Join state/context features [B,64] + [B,64] -> [B,128]; project 128 -> V scores [B,V].
            logits=self.head(torch.cat([state,context],-1))
        else:
            # Without look-back, [B] -> [B,24] feeds the writer; its state [B,64] alone projects -> [B,V].
            state=self.writer(self.embedding(previous),state); logits=self.head(state); weights=None
        return logits,state,weights
    def forward(self,source,target_ids=None,*,teacher=True,detach_reader=False):
        if teacher:
            if target_ids is None:
                raise ValueError("Teacher forcing needs the correct answer IDs from the data.")
            if target_ids.ndim != 2 or len(target_ids) != len(source) or target_ids.shape[1] < 1:
                raise ValueError("Answers must be a nonempty [batch, writer time] tensor.")
            if target_ids.dtype != torch.long:
                raise TypeError("Answer IDs must use torch.long.")
            inputs = practice_prefix(target_ids)
            steps = target_ids.shape[1]
        else:
            if target_ids is not None:
                raise ValueError("Generation must not receive an answer tensor.")
            inputs = None
            steps = source.shape[1] + 1
        memory,state=self.read(source); previous=source.new_full((len(source),),BOS)
        if detach_reader:
            memory, state = memory.detach(), state.detach()
        scores=[]; maps=[]
        for t in range(steps):
            if teacher: previous=inputs[:,t]
            logits,state,weights=self.step(previous,state,memory); scores.append(logits)
            if weights is not None: maps.append(weights)
            # Generation selects guesses [B,V] -> [B]; practice only reads the shifted prefix.
            if not teacher: previous=logits.argmax(-1)
        # Stack adds writer time: T score arrays [B,V] -> [B,T,V]; reading maps [B,L] -> [B,T,L].
        return torch.stack(scores,1), torch.stack(maps,1) if maps else None

# %% 4.1 Follow one source read and the first writing step
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(31)
    mechanism_model = ReaderWriter(False)
walkthrough_source = torch.tensor([encode_words(CAT_WORDS)])
walkthrough_target = reversal_targets(walkthrough_source)
with torch.no_grad():
    source_rows = mechanism_model.embedding(walkthrough_source)
    reader_notes, reader_last = mechanism_model.reader(source_rows)
    writer_start = reader_last[0]
    start_ids = walkthrough_source.new_full((1,), BOS)
    first_scores, writer_after_start, _ = mechanism_model.step(start_ids, writer_start, reader_notes)
print("Source rows:", tuple(source_rows.shape), "| reader notes:", tuple(reader_notes.shape))
print("Final reader state = initial writer state:", tuple(writer_start.shape))
for word, note in zip(CAT_WORDS, reader_notes[0]):
    print(word, "-> reader feature sample", note[:3].tolist())
role_modules = {
    "word rows": mechanism_model.embedding, "encoder": mechanism_model.reader,
    "decoder": mechanism_model.writer, "head": mechanism_model.head,
}
role_parameters = {name: sum(p.numel() for p in module.parameters())
                   for name, module in role_modules.items()}
print("Different learned jobs:", role_parameters)
print("First writer input:", decode_ids(start_ids), "| scores:", tuple(first_scores.shape))
print("Untrained first guess:", decode_ids(first_scores.argmax(-1)),
      "| required first word:", decode_ids(walkthrough_target[:, 0]))

# %% 4.2 What if that first guess were wrong?
with torch.no_grad():
    correct_previous = walkthrough_target[:, 0]
    wrong_previous = correct_previous.new_full((1,), word_to_id[first_guess])
    teacher_scores, teacher_state, _ = mechanism_model.step(
        correct_previous, writer_after_start, reader_notes,
    )
    own_guess_scores, own_guess_state, _ = mechanism_model.step(
        wrong_previous, writer_after_start, reader_notes,
    )
second_input_gap = (teacher_scores - own_guess_scores).abs().max().item()
print("Hypothetical first guess:", first_guess, "; the first scores and writer state remain unchanged.")
print("Next practice input:", decode_ids(correct_previous),
      "| next hypothetical generation input:", decode_ids(wrong_previous))
print("Largest next-score change:", second_input_gap)
assert second_input_gap == 0 if first_guess == "mat" else second_input_gap > 0
print("Takeaway: the input policy changes the next computation; it does not repair the earlier guess.")

# %% 4.3 What tells these parts how to improve?
with torch.no_grad():
    untrained_scores, _ = mechanism_model(
        walkthrough_source, target_ids=walkthrough_target, teacher=True,
    )
    answer_probabilities = untrained_scores.softmax(-1).gather(
        2, walkthrough_target.unsqueeze(-1),
    )[0, :, 0]
    answer_losses = F.cross_entropy(
        untrained_scores.flatten(0, 1), walkthrough_target.flatten(), reduction="none",
    )
for previous, answer, probability, error in zip(
    decode_ids(practice_prefix(walkthrough_target)[0]),
    decode_ids(walkthrough_target[0]), answer_probabilities.tolist(), answer_losses.tolist(),
):
    print(f"Input {previous:>5} | check {answer:>5} | probability {probability:.4f} | error {error:.4f}")
print("One example's mean error:", answer_losses.mean().item())
print("Takeaway: correct previous inputs do not excuse incorrect current predictions.")

# %% 4.4 How can an encoder learn if it never gets a word grade?
from copy import deepcopy


def gradient_norm(module):
    return sum(float(parameter.grad.square().sum())
               for parameter in module.parameters() if parameter.grad is not None) ** 0.5


training_probes, gradient_records, probe_scores, probe_losses = {}, {}, {}, {}
for detached in (False, True):
    trial = deepcopy(mechanism_model)
    scores, _ = trial(walkthrough_source, target_ids=walkthrough_target,
                      teacher=True, detach_reader=detached)
    loss = F.cross_entropy(scores.flatten(0, 1), walkthrough_target.flatten())
    loss.backward()
    modules = {"word rows": trial.embedding, "encoder": trial.reader,
               "decoder": trial.writer, "head": trial.head}
    gradient_records[detached] = {name: gradient_norm(module) for name, module in modules.items()}
    training_probes[detached], probe_scores[detached], probe_losses[detached] = (
        trial, scores.detach(), loss.item(),
    )
    print("Reader handoff detached:", detached, "| feedback norms:", gradient_records[detached])
torch.testing.assert_close(probe_scores[False], probe_scores[True], rtol=0, atol=0)
assert gradient_records[False]["encoder"] > 0 and gradient_records[True]["encoder"] == 0
assert gradient_records[True]["decoder"] > 0 and gradient_records[True]["word rows"] > 0
lookup_reader_feedback = (
    training_probes[False].embedding.weight.grad - training_probes[True].embedding.weight.grad
).norm().item()
print("Same forward scores/errors; changed backward route. Reader contribution to word-row feedback:",
      lookup_reader_feedback)
assert lookup_reader_feedback > 0
for name, parameter in mechanism_model.named_parameters():
    torch.testing.assert_close(parameter, training_probes[False].get_parameter(name), rtol=0, atol=0)

# %% 4.4 How can an encoder learn if it never gets a word grade?
fig, ax = plt.subplots(figsize=(8, 3.5))
positions = np.arange(len(role_modules))
for detached, offset, label in ((False, -.18, "Connected handoff"), (True, .18, "Handoff detached")):
    ax.bar(positions + offset, list(gradient_records[detached].values()), width=.36, label=label)
ax.set_xticks(positions, list(role_modules))
ax.set(ylabel="Weight-gradient norm", title="Same prediction, but can its error teach the encoder?")
ax.legend(); fig.tight_layout()
fig.savefig(CHAPTER_DIR / "images" / "03c-training-gradient-paths.png", dpi=160)
plt.show(); plt.close(fig)

# %% 4.5 When do the weights actually change?
learning_copy = training_probes[False]
before_step = {name: parameter.detach().clone() for name, parameter in learning_copy.named_parameters()}
one_step_optimizer = torch.optim.Adam(learning_copy.parameters(), lr=0.004)
one_step_optimizer.step()
with torch.no_grad():
    after_step_scores, _ = learning_copy(walkthrough_source, target_ids=walkthrough_target, teacher=True)
    after_step_loss = F.cross_entropy(after_step_scores.flatten(0, 1), walkthrough_target.flatten()).item()
update_sizes = {
    prefix: max((parameter.detach() - before_step[name]).abs().max().item()
                for name, parameter in learning_copy.named_parameters() if name.startswith(prefix + "."))
    for prefix in ("embedding", "reader", "writer", "head")
}
print("One copied example's error:", probe_losses[False], "->", after_step_loss)
print("Largest actual weight changes:", update_sizes)
assert after_step_loss < probe_losses[False] and all(change > 0 for change in update_sizes.values())
training_evidence = {
    "source_words": list(CAT_WORDS), "required_words": decode_ids(walkthrough_target[0]),
    "teacher_inputs": decode_ids(practice_prefix(walkthrough_target)[0]),
    "role_parameters": role_parameters, "source_rows_shape": list(source_rows.shape),
    "reader_notes_shape": list(reader_notes.shape), "writer_initial_shape": list(writer_start.shape),
    "next_input_score_gap": second_input_gap, "gradient_norms": gradient_records,
    "reader_lookup_feedback_norm": lookup_reader_feedback,
    "before_update_loss": probe_losses[False], "after_update_loss": after_step_loss,
    "weight_changes": update_sizes,
}
print("Takeaway: backward did not rewrite weights; one optimizer step trained all four learned jobs.")

# %% Is the encoder learning a better note, or is only the decoder getting better?
# %% 4.5 - Can a better message help an unchanged decoder?
encoding_copy = deepcopy(mechanism_model)
for parameter in encoding_copy.parameters():
    parameter.requires_grad_(False)
for parameter in encoding_copy.reader.parameters():
    parameter.requires_grad_(True)
contrast_words = [list(CAT_WORDS), [*CAT_WORDS[:-1], "rug"]]
contrast_sources = torch.tensor([encode_words(words) for words in contrast_words])
correct_first = contrast_sources[:, -1]
same_start = contrast_sources.new_full((2,), BOS)
frozen_before = {name: parameter.detach().clone()
                 for name, parameter in encoding_copy.named_parameters() if not parameter.requires_grad}


def first_word_outputs():
    memory, message = encoding_copy.read(contrast_sources)
    scores, _, _ = encoding_copy.step(same_start, message, memory)
    return message, scores


messages, first_scores = first_word_outputs()
with torch.no_grad():
    identical_scores, _, _ = encoding_copy.step(
        same_start, messages[:1].expand_as(messages), None,
    )
assert torch.equal(identical_scores[0], identical_scores[1])
initial_messages = messages.detach().clone()
initial_probabilities = first_scores.detach().softmax(-1).gather(1, correct_first[:, None])[:, 0]
encoding_loss = F.cross_entropy(first_scores, correct_first)
encoder_optimizer = torch.optim.SGD(encoding_copy.reader.parameters(), lr=0.1)
encoder_optimizer.zero_grad()
encoding_loss.backward()
encoder_optimizer.step()
with torch.no_grad():
    new_messages, new_scores = first_word_outputs()
    new_loss = F.cross_entropy(new_scores, correct_first).item()
    new_probabilities = new_scores.softmax(-1).gather(1, correct_first[:, None])[:, 0]
message_change = (new_messages - initial_messages).norm().item()
assert message_change > 0 and new_loss < encoding_loss.item()
assert all(torch.equal(parameter.detach(), frozen_before[name])
           for name, parameter in encoding_copy.named_parameters() if not parameter.requires_grad)
for words, before, after in zip(contrast_words, initial_probabilities, new_probabilities):
    print(" ".join(words), "| correct first word:", words[-1],
          f"| probability {before.item():.4f} -> {after.item():.4f}")
training_evidence["encoder_only"] = {
    "source_words": contrast_words, "first_answers": decode_ids(correct_first),
    "before_probabilities": initial_probabilities.tolist(), "after_probabilities": new_probabilities.tolist(),
    "before_loss": encoding_loss.item(), "after_loss": new_loss,
    "message_change_norm": message_change, "decoder_and_head_unchanged": True,
    "identical_message_score_gap": (identical_scores[0] - identical_scores[1]).abs().max().item(),
}
print("Only encoder weights changed; first-word error:", encoding_loss.item(), "->", new_loss)
print("Takeaway: better encoding means a message that helps this decoder's task, not a prescribed vector.")

# %% 4.6 That teaches a summary. Can the writer reopen a source detail?
models, histories, parameter_counts = {}, {}, {}
canonical_source = torch.tensor([encode_words(CAT_WORDS)])
canonical_target = reversal_targets(canonical_source)
assert decode_ids(canonical_target[0]) == ["mat", "the", "on", "sat", "cat", "the", "<eos>"]
assert practice_prefix(canonical_target)[0, 0].item() == BOS
assert torch.equal(practice_prefix(canonical_target)[:, 1:], canonical_target[:, :-1])
for look_back in [False, True]:
    torch.manual_seed(31)
    model = ReaderWriter(look_back)
    seen_inputs = []
    real_step = model.step

    def traced_step(previous, state, memory):
        seen_inputs.append(previous.detach().clone())
        return real_step(previous, state, memory)

    with torch.no_grad(), patch.object(model, "step", side_effect=traced_step):
        model(canonical_source, target_ids=canonical_target, teacher=True)
        assert torch.equal(torch.stack(seen_inputs, 1), practice_prefix(canonical_target))
        seen_inputs.clear()
        free_scores, _ = model(canonical_source, teacher=False)
        actual_inputs = torch.stack(seen_inputs, 1)
        assert bool((actual_inputs[:, 0] == BOS).all())
        assert torch.equal(actual_inputs[:, 1:], free_scores.argmax(-1)[:, :-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    schedule = random.Random(99)
    batch_rng = torch.Generator().manual_seed(99)
    history = []
    for update in range(1000):
        length = schedule.randint(2, 8)
        rows = train[length]
        source = rows[torch.randint(len(rows), (64,), generator=batch_rng)]
        # Reverse [64,L] and append [64,1] end IDs -> [64,L+1], including the end prediction in training.
        target = reversal_targets(source)
        logits, _ = model(source, target_ids=target, teacher=True)
        # Scores [64,L+1,V] -> [64*(L+1),V], targets -> [64*(L+1)]; merge matching batch/time axes only.
        loss = F.cross_entropy(logits.flatten(0, 1), target.flatten())
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append(loss.item())
    model.eval()
    models[look_back], histories[look_back] = model, history
    parameter_counts[look_back] = sum(p.numel() for p in model.parameters())
    print("look back:", look_back, "loss", round(history[0], 4), "->", round(history[-1], 4),
          "parameters", parameter_counts[look_back])

# %% 4.6 That teaches a summary. Can the writer reopen a source detail?
look_back_to_inspect = False  # CHANGE THIS: True
print(models[look_back_to_inspect])

# %% Source notes remain available while the writer changes
from copy import deepcopy

memory_probe_model = deepcopy(models[True]).eval()
memory_probe_source = torch.tensor([encode_words(CAT_WORDS)])
memory_probe_parameters = {
    name: parameter.detach().clone()
    for name, parameter in memory_probe_model.named_parameters()
}
memory_probe_rows, memory_probe_last_weights = [], None
with torch.no_grad():
    memory_probe_notes, memory_probe_state = memory_probe_model.read(memory_probe_source)
    memory_probe_original = memory_probe_notes.clone()
    memory_probe_previous = practice_prefix(reversal_targets(memory_probe_source))[:, :3]
    for memory_probe_index in range(memory_probe_previous.shape[1]):
        memory_probe_input = memory_probe_previous[:, memory_probe_index]
        memory_probe_old_state = memory_probe_state.clone()
        memory_probe_scores, memory_probe_state, memory_probe_weights = memory_probe_model.step(
            memory_probe_input, memory_probe_state, memory_probe_notes,
        )
        if memory_probe_weights is None:
            raise RuntimeError("The look-back model returned no source-attention reading.")
        memory_probe_rows.append({
            "step": memory_probe_index + 1,
            "previous": decode_ids(memory_probe_input)[0],
            "chosen": decode_ids(memory_probe_scores.argmax(-1))[0],
            "source_change": (memory_probe_notes - memory_probe_original).abs().max().item(),
            "writer_change": (memory_probe_state - memory_probe_old_state).abs().max().item(),
            "attention_change": None if memory_probe_last_weights is None else (
                memory_probe_weights - memory_probe_last_weights
            ).abs().max().item(),
        })
        memory_probe_last_weights = memory_probe_weights.clone()
assert torch.equal(memory_probe_notes, memory_probe_original)
assert all(torch.equal(parameter, memory_probe_parameters[name])
           for name, parameter in memory_probe_model.named_parameters())
print("step | previous | chosen | source change | writer change | attention change")
for memory_probe_row in memory_probe_rows:
    memory_probe_change = "--" if memory_probe_row["attention_change"] is None else f'{memory_probe_row["attention_change"]:.6f}'
    print(f'{memory_probe_row["step"]:>4} | {memory_probe_row["previous"]:>8} | {memory_probe_row["chosen"]:>6} | '
          f'{memory_probe_row["source_change"]:.6f} | {memory_probe_row["writer_change"]:.6f} | {memory_probe_change}')

# %% 5. Can it write without the teacher's history?
results = []
with torch.no_grad():
    for look_back, model in models.items():
        for length, source in test.items():
            # The held-out answers use the same layout: reverse [B,L], append end [B,1] -> [B,L+1].
            target = reversal_targets(source)
            tf_scores, _ = model(source, target_ids=target, teacher=True)
            free_scores, _ = model(source, teacher=False)
            # Choose IDs [B,L+1,V] -> [B,L+1]; all(time) -> [B] whole-answer checks, mean -> scalar accuracy.
            # The token scores below instead average every [B,L+1] check, so partial answers still count there.
            tf_ok = (tf_scores.argmax(-1) == target).all(1).float().mean().item()
            free_ok = (free_scores.argmax(-1) == target).all(1).float().mean().item()
            tf_token = (tf_scores.argmax(-1) == target).float().mean().item()
            free_token = (free_scores.argmax(-1) == target).float().mean().item()
            results.append((look_back, length, tf_ok, free_ok, tf_token, free_token))
print("look-back length practice-sequence free-sequence practice-token free-token")
for row in results:
    print(row[0], row[1], *[f"{value:.2%}" for value in row[2:]])
# Seven per-length scores -> one equally weighted mean; this is not a pooled token accuracy.
fixed_mean = np.mean([r[3] for r in results if not r[0]])
attention_mean = np.mean([r[3] for r in results if r[0]])
print(f"Mean over lengths, free-running: one card {fixed_mean:.1%}; look-back {attention_mean:.1%}")
aggregate_scores = {}
for mode in models:
    rows = [row for row in results if row[0] == mode]
    token_counts = [(row[1] + 1) * len(test[row[1]]) for row in rows]
    aggregate_scores[mode] = {
        "practice_sequence": float(np.mean([row[2] for row in rows])),
        "free_sequence": float(np.mean([row[3] for row in rows])),
        "practice_token_pooled": float(np.average([row[4] for row in rows], weights=token_counts)),
        "free_token_pooled": float(np.average([row[5] for row in rows], weights=token_counts)),
    }
    print("Aggregate look-back", mode, {key: f"{value:.2%}" for key, value in aggregate_scores[mode].items()})
if attention_mean > fixed_mean:
    print("Takeaway: look-back improved this held-out run; this is not a universal ranking.")
else:
    print("Takeaway: look-back did not improve this run; access exists but payoff was not established.")
fig, ax = plt.subplots(figsize=(6, 3))
for mode, label, color in [(False, "One learned card", "#ef8a23"), (True, "Writer looks back", "#2c7fb8")]:
    rows = [r for r in results if r[0] == mode]
    ax.plot([r[1] for r in rows], [r[3] for r in rows], "o-", label=label, color=color)
ax.set(xlabel="Source length", ylabel="Complete free-running accuracy", ylim=(0, 1.05))
ax.set_title("Word-order reversal: cat-family controlled access test")
ax.legend(); fig.tight_layout()
fig.savefig(CHAPTER_DIR / "images" / "03c-cat-accuracy-by-length.png", dpi=160)
plt.show(); plt.close(fig)

# %% 5. Can it write without the teacher's history?
source_to_reverse = list(CAT_WORDS)  # CHANGE THIS: first word only, from CAT_FAMILY_WORDS.
source = torch.tensor([encode_words(source_to_reverse)])
print("Source words:", " ".join(source_to_reverse))
probe_answers = {}
with torch.no_grad():
    for mode, model in models.items():
        scores, _ = model(source, teacher=False)
        probe_answers[mode] = decode_ids(scores.argmax(-1)[0])
        print("look back", mode, "generated words:", " ".join(probe_answers[mode]))
print("Required words:", " ".join(decode_ids(reversal_targets(source)[0])))
source_key = tuple(source[0].tolist())
print("Probe membership: training", source_key in {tuple(row) for row in train[len(source_to_reverse)].tolist()},
      "; held out", source_key in {tuple(row) for row in test[len(source_to_reverse)].tolist()})

# %% 6. Inspect the route without declaring victory from a picture
with torch.no_grad():
    probe = source
    logits, weights = models[True](probe, teacher=False)
    generated = logits.argmax(-1)[0].tolist()
    # Choose one example: [1,L+1,L] -> [L+1,L], writer steps as rows and source positions as columns.
    heat = weights[0].numpy()
source_labels = [f"{word} @ {position}" for position, word in enumerate(source_to_reverse)]
generated_words = decode_ids(generated)
expected_addresses = list(reversed(range(probe.shape[1])))
assert heat.shape == (len(generated_words), len(source_labels))
assert np.allclose(heat.sum(1), 1.0)
fig, ax = plt.subplots(figsize=(9, 4.5))
image = ax.imshow(heat, vmin=0, vmax=1, cmap="Blues", aspect="auto")
ax.set_xticks(range(len(source_labels)), source_labels)
ax.set_yticks(range(len(generated_words)), [f"step {i}: chose {word}" for i, word in enumerate(generated_words)])
ax.set(xlabel="Reader source word and position", ylabel="Writer step",
       title="Measured free-running weights: diagnostic, not proof")
fig.suptitle("Source: " + " ".join(source_to_reverse))
fig.colorbar(image, ax=ax, label="Read weight"); fig.tight_layout()
fig.savefig(CHAPTER_DIR / "images" / "03c-cat-attention-map.png", dpi=160)
plt.show(); plt.close(fig)
# Sum only source columns [L+1,L] -> [L+1], one reading budget at each writer step.
print("Row sums:", heat.sum(1).round(6).tolist())
print("Most-read source positions:", heat.argmax(1).tolist())
print("Required word addresses:", [source_labels[address] for address in expected_addresses],
      "; <eos> has no single source address")
print("Most-read words and addresses:", [source_labels[address] for address in heat.argmax(1)])
print("Takeaway: inspect routing beside generated answers; the picture alone cannot prove correctness.")

# %% 6. Inspect the route without declaring victory from a picture
writing_step = 0  # CHANGE THIS: any step from 0 through len(generated_words) - 1.
assert 0 <= writing_step < len(generated_words)
expected_address = source_labels[expected_addresses[writing_step]] if writing_step < len(expected_addresses) else "<eos>: no single source address"
print("Chose:", generated_words[writing_step], "; most-read:", source_labels[int(heat[writing_step].argmax())],
      "; required address:", expected_address)

# %% 7. What can you now explain and prove?
evidence = {"fixed_summary_mean_free_sequence_accuracy": round(float(fixed_mean), 6),
       "attention_mean_free_sequence_accuracy": round(float(attention_mean), 6),
       "held_out_sequences_per_length": len(test[2]), "source_lengths": list(test),
       "word_vocabulary_size": len(word_to_id), "total_vocabulary_with_controls": V,
       "parameters": parameter_counts, "aggregate_scores": aggregate_scores,
       "per_length_scores": results, "probe_generated_words": probe_answers,
       "source_words": source_to_reverse, "source_addresses": source_labels,
       "required_words": decode_ids(reversal_targets(probe)[0]),
       "most_read_addresses": heat.argmax(1).tolist(), "required_addresses": expected_addresses,
       "training_walkthrough": training_evidence}
(CHAPTER_DIR / "images" / "03c-cat-measured-evidence.json").write_text(
    json.dumps(evidence, indent=2) + "\n", encoding="utf-8",
)
print(evidence)
print(f"Elapsed CPU wall time: {time.perf_counter() - started:.1f}s")

# %% 8. Keep the reading/writing job inside the model, not the trainer
from copy import deepcopy
from language_model import TrainableLanguageModel
from torch.utils.data import DataLoader, TensorDataset


class ReaderWriterTrainingModel(TrainableLanguageModel):
    def __init__(self, reader_writer):
        super().__init__()
        self.reader_writer = reader_writer

    def forward(self, source_ids, target_ids=None):
        # Keep scores [B,L+1,V] for the shared loss; reading maps are a diagnostic, not a second prediction target.
        logits, _ = self.reader_writer(source_ids, target_ids=target_ids, teacher=True)
        return logits


adapter = ReaderWriterTrainingModel(deepcopy(models[False]))
adapter_source = train[len(CAT_WORDS)][:8]
# Reverse [8,L] and append end IDs [8,1] -> [8,L+1], aligned with the writer's output steps.
adapter_target = reversal_targets(adapter_source)
adapter_loader = DataLoader(TensorDataset(adapter_source, adapter_target), batch_size=8)
adapter_optimizer = torch.optim.Adam(adapter.parameters(), lr=0.001)
original_embedding = models[False].embedding.weight.detach().clone()
original_parameters = {name: parameter.detach().clone()
                       for name, parameter in models[False].named_parameters()}
adapter_before = adapter.reader_writer.embedding.weight.detach().clone()

assert ReaderWriterTrainingModel.fit_language_model is TrainableLanguageModel.fit_language_model
adapter_losses = adapter.fit_language_model(
    adapter_loader, nn.CrossEntropyLoss(), adapter_optimizer, epochs=1,
)
assert torch.equal(models[False].embedding.weight.detach(), original_embedding)
assert all(torch.equal(parameter.detach(), original_parameters[name])
           for name, parameter in models[False].named_parameters())
adapter_change = (adapter.reader_writer.embedding.weight.detach() - adapter_before).abs().max().item()
assert adapter_change > 0
print("Inherited encoder-decoder training loss:", adapter_losses[0])
print("Embedding updated through reader and writer; original experiment unchanged:", adapter_change)

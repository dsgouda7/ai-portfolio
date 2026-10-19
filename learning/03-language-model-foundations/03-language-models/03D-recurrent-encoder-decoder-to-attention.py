"""Code companion for 03D-recurrent-encoder-decoder-to-attention.ipynb. Run from this directory."""

# %% 2. Practice and generation give different help
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
random.seed(17); np.random.seed(17); torch.manual_seed(17)
torch.set_num_threads(1)
started = time.perf_counter()

# %% 2. Practice and generation give different help
first_guess = 4  # CHANGE THIS: 1 is the correct first answer
correct_target = [1, 4, 1, 3, "end"]
practice_inputs = ["start"] + correct_target[:-1]
print("Practice input -> answer:", list(zip(practice_inputs, correct_target)))
print("Generation: start -> guessed", first_guess, "-> next input", first_guess)
print("Correct previous input at step two:", practice_inputs[1])
print("Takeaway: a wrong guess changes the next input, not the stored source.")

# %% 3. Give both models exactly the same examples
PAD,BOS,EOS,V=0,9,10,11
rng=random.Random(31)
train,test={},{}
for length in range(2,9):
    unique=set()
    count=min(1200,8**length-40)
    while len(unique)<count+40:
        unique.add(tuple(rng.randint(1,8) for _ in range(length)))
    rows=list(sorted(unique)); rng.shuffle(rows)
    train[length]=torch.tensor(rows[:count]); test[length]=torch.tensor(rows[count:])

# %% 3. Give both models exactly the same examples
for length in train:
    training_rows = {tuple(r) for r in train[length].tolist()}
    testing_rows = {tuple(r) for r in test[length].tolist()}
    assert training_rows.isdisjoint(testing_rows)
    print(f"Length {length}: {len(training_rows)} train, {len(testing_rows)} held out")
print("Takeaway: both models receive the same split; no test source appears in training.")

# %% 3. Give both models exactly the same examples
example_length = 4  # CHANGE THIS: 2 through 8
example = test[example_length][0].tolist()
print("source:", example, "target:", list(reversed(example)) + [EOS])

# %% 4. Build a learned recurrent reader and writer
class ReaderWriter(nn.Module):
    def __init__(self,look_back):
        super().__init__(); self.look_back=look_back
        self.embedding=nn.Embedding(V,24)
        self.reader=nn.GRU(24,64,batch_first=True)
        self.writer=nn.GRUCell(24+(64 if look_back else 0),64)
        self.query=nn.Linear(64,64,bias=False) if look_back else None
        self.head=nn.Linear(128 if look_back else 64,V)
    def read(self,source):
        memory,last=self.reader(self.embedding(source)); return memory,last[0]
    def step(self,previous,state,memory):
        if self.look_back:
            scores=torch.bmm(memory,self.query(state).unsqueeze(2)).squeeze(2)/8
            weights=scores.softmax(-1)
            context=torch.bmm(weights.unsqueeze(1),memory).squeeze(1)
            state=self.writer(torch.cat([self.embedding(previous),context],-1),state)
            logits=self.head(torch.cat([state,context],-1))
        else:
            state=self.writer(self.embedding(previous),state); logits=self.head(state); weights=None
        return logits,state,weights
    def forward(self,source,teacher=True):
        memory,state=self.read(source); previous=torch.full((len(source),),BOS)
        target=torch.cat([source.flip(1),torch.full((len(source),1),EOS)],1) if teacher else None
        scores=[]; maps=[]
        for t in range(source.shape[1] + 1):
            logits,state,weights=self.step(previous,state,memory); scores.append(logits)
            if weights is not None: maps.append(weights)
            previous=target[:,t] if teacher else logits.argmax(-1)
        return torch.stack(scores,1), torch.stack(maps,1) if maps else None

# %% 4. Build a learned recurrent reader and writer
models, histories = {}, {}
for look_back in [False, True]:
    torch.manual_seed(31)
    model = ReaderWriter(look_back)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    schedule = random.Random(99)
    batch_rng = torch.Generator().manual_seed(99)
    history = []
    for update in range(1000):
        length = schedule.randint(2, 8)
        rows = train[length]
        source = rows[torch.randint(len(rows), (64,), generator=batch_rng)]
        target = torch.cat([source.flip(1), torch.full((64, 1), EOS)], 1)
        logits, _ = model(source, teacher=True)
        loss = F.cross_entropy(logits.flatten(0, 1), target.flatten())
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append(loss.item())
    model.eval()
    models[look_back], histories[look_back] = model, history
    print("look back:", look_back, "loss", round(history[0], 4), "->", round(history[-1], 4),
          "parameters", sum(p.numel() for p in model.parameters()))

# %% 4. Build a learned recurrent reader and writer
look_back_to_inspect = False  # CHANGE THIS: True
print(models[look_back_to_inspect])

# %% 5. Grade complete answers, by source length
results = []
with torch.no_grad():
    for look_back, model in models.items():
        for length, source in test.items():
            target = torch.cat([source.flip(1), torch.full((len(source), 1), EOS)], 1)
            tf_scores, _ = model(source, teacher=True)
            free_scores, _ = model(source, teacher=False)
            tf_ok = (tf_scores.argmax(-1) == target).all(1).float().mean().item()
            free_ok = (free_scores.argmax(-1) == target).all(1).float().mean().item()
            tf_token = (tf_scores.argmax(-1) == target).float().mean().item()
            free_token = (free_scores.argmax(-1) == target).float().mean().item()
            results.append((look_back, length, tf_ok, free_ok, tf_token, free_token))
print("look-back length practice-sequence free-sequence practice-token free-token")
for row in results:
    print(row[0], row[1], *[f"{value:.1%}" for value in row[2:]])
fixed_mean = np.mean([r[3] for r in results if not r[0]])
attention_mean = np.mean([r[3] for r in results if r[0]])
print(f"Mean over lengths, free-running: one card {fixed_mean:.1%}; look-back {attention_mean:.1%}")
if attention_mean > fixed_mean:
    print("Takeaway: look-back improved this held-out run; this is not a universal ranking.")
else:
    print("Takeaway: look-back did not improve this run; access exists but payoff was not established.")
fig, ax = plt.subplots(figsize=(6, 3))
for mode, label, color in [(False, "One learned card", "#ef8a23"), (True, "Writer looks back", "#2c7fb8")]:
    rows = [r for r in results if r[0] == mode]
    ax.plot([r[1] for r in rows], [r[3] for r in rows], "o-", label=label, color=color)
ax.set(xlabel="Source length", ylabel="Complete free-running accuracy", ylim=(0, 1.05))
ax.legend(); plt.show()

# %% 5. Grade complete answers, by source length
source_to_reverse = [3, 1, 4, 1]  # CHANGE THIS: first digit only, within 1..8
source = torch.tensor([source_to_reverse])
with torch.no_grad():
    for mode, model in models.items():
        scores, _ = model(source, teacher=False)
        print("look back", mode, "generated IDs:", scores.argmax(-1)[0].tolist())
print("Expected:", list(reversed(source_to_reverse)) + [EOS])

# %% 6. Inspect the route without declaring victory from a picture
with torch.no_grad():
    probe = torch.tensor([[3, 1, 4, 1]])
    logits, weights = models[True](probe, teacher=False)
    generated = logits.argmax(-1)[0].tolist()
    heat = weights[0].numpy()
fig, ax = plt.subplots(figsize=(6, 3))
image = ax.imshow(heat, vmin=0, vmax=1, cmap="Blues", aspect="auto")
ax.set_xticks(range(4), ["3 @ 0", "1 @ 1", "4 @ 2", "1 @ 3"])
ax.set_yticks(range(5), [f"step {i}: chose {digit}" for i, digit in enumerate(generated)])
ax.set(xlabel="Reader source position", ylabel="Writer step", title="Measured weights: diagnostic, not proof")
fig.colorbar(image, ax=ax, label="Read weight"); plt.show()
print("Row sums:", heat.sum(1).round(6).tolist())
print("Most-read source positions:", heat.argmax(1).tolist())
print("Expected digit addresses:", [3, 2, 1, 0], "; end has no single source address")
print("Takeaway: inspect routing beside generated answers; the picture alone cannot prove correctness.")

# %% 6. Inspect the route without declaring victory from a picture
writing_step = 0  # CHANGE THIS: 1, 2, or 3
print("Most-read:", int(heat[writing_step].argmax()), "expected:", 3 - writing_step)

# %% 7. Evidence scorecard and the next complaint
print({"fixed_summary_mean_free_sequence_accuracy": round(float(fixed_mean), 6),
       "attention_mean_free_sequence_accuracy": round(float(attention_mean), 6),
       "held_out_sequences_per_length": len(test[2]), "source_lengths": list(test)})
print(f"Elapsed CPU wall time: {time.perf_counter() - started:.1f}s")

# %% 8. Where does this encoder-decoder reference the common trainer?
from copy import deepcopy
from language_model import TrainableLanguageModel
from torch.utils.data import DataLoader, TensorDataset


class ReaderWriterTrainingModel(TrainableLanguageModel):
    def __init__(self, reader_writer):
        super().__init__()
        self.reader_writer = reader_writer

    def forward(self, source_ids, target_ids=None):
        logits, _ = self.reader_writer(source_ids, teacher=True)
        return logits


adapter = ReaderWriterTrainingModel(deepcopy(models[False]))
adapter_source = train[4][:8]
adapter_target = torch.cat(
    [adapter_source.flip(1), torch.full((len(adapter_source), 1), EOS)], dim=1,
)
adapter_loader = DataLoader(TensorDataset(adapter_source, adapter_target), batch_size=8)
adapter_optimizer = torch.optim.Adam(adapter.parameters(), lr=0.001)
original_embedding = models[False].embedding.weight.detach().clone()
adapter_before = adapter.reader_writer.embedding.weight.detach().clone()

assert ReaderWriterTrainingModel.fit_language_model is TrainableLanguageModel.fit_language_model
adapter_losses = adapter.fit_language_model(
    adapter_loader, nn.CrossEntropyLoss(), adapter_optimizer, epochs=1,
)
assert torch.equal(models[False].embedding.weight.detach(), original_embedding)
adapter_change = (adapter.reader_writer.embedding.weight.detach() - adapter_before).abs().max().item()
assert adapter_change > 0
print("Inherited encoder-decoder training loss:", adapter_losses[0])
print("Embedding updated through reader and writer; original experiment unchanged:", adapter_change)

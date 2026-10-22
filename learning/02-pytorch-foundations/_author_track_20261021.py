"""Source-aware authoring workspace; removed after successful verification."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SESSION = Path(r"..\..\Users\deepaksha\.copilot\session-state\78a59e14-e8bd-471f-9577-f35a620524b8\files\pytorch-track-20261021")
PROSE = {}
EDITS = {}


def lesson(stem, prose):
    PROSE[stem] = prose


def change(stem, cell_id, old, new):
    EDITS.setdefault(stem, []).append((cell_id, old, new))


lesson("02-tensors-and-autograd", {
    "3e97e3aa": """### Before we open the containers

The sensor lesson already gave you useful gradients. Now imagine handing that
same loop a whole sentence instead of one reading. A row is no longer enough:
you need to know which example, which position, and which feature a value belongs
to. Think of the tensor's shape as an address label on a parcel, not its meaning.

We'll use two four-word strips, **“the cat sat on”** and **“the dog sat on”**.
Their IDs are made-up addresses into ten rows; nothing here has learned English.
Six random features per position make the arithmetic inspectable. A fixed seed
repeats the starting numbers, and a printed device tells you where they live.
The setup cell only prepares tools; it does not train anything.

**Predict:** Will selecting a GPU change what the three axes mean? No: hardware
placement and the address contract are separate responsibilities.""",
    "03a85653": """### Count roles, not just numbers

A single sensor value is a scalar. A list is a vector. Our sentence batch is a
table of word addresses: two examples with four positions each. Fetching or
inventing six feature values at each position adds a third axis; it does not
create more sentences.

The dtype is part of this story. An ID is an integer address; adding half an ID
would not describe a useful lookup. Features are floating-point values because
the model will combine and adjust them. Device is the storage location. A
matching shape does not make a CPU tensor compatible with a GPU-only operation.

The next printout audits these contracts side by side. Read each shape aloud,
then check the dtype rather than trusting a tensor that merely looks plausible.
**Next complaint:** we can name a batch, but selecting one example may erase its
batch axis.""",
    "494404ac": """Imagine taking one folder from a filing cabinet. You can hand somebody the
folder's contents, or hand them a smaller cabinet that still contains one folder.
Integer indexing does the first; a one-item slice does the second.

Both results refer to the same sentence's features. But a later function may
expect three coordinates—example, position, feature—even when there is just one
example. Removing that coordinate is a contract change, not a numerical change.
The explicit squeeze below removes the singleton axis only for the equality
comparison. It is not a recommendation to squeeze every small dimension.

**Predict:** Will the values agree after that deliberate squeeze, while the
printed shapes differ before it? The equality check and shape trace answer both
questions. **Next complaint:** keeping all the values still does not guarantee
we have grouped them under the right roles.""",
    "a4c5ccc7": """Suppose you put each position's six features into two little bundles of three.
That is regrouping: reshape changes the address layout without changing the
logical value order. Now put the bundle label before the position label so one
bundle's positions are read together. That is moving axes: transpose.

Those actions are not interchangeable. A direct reshape of the head-first
layout can keep the value count while assigning values to different positions.
We therefore undo the axis move before joining the bundles back together.

Here “head” only names a feature bundle. We have not implemented attention,
routing, or a reason for multiple heads. **Predict:** Will undoing both actions
recover every original value exactly? The round-trip check is the proof.
**Next complaint:** sometimes we want to reuse an offset, not regroup anything.""",
    "7d349365": """An offset for “feature number three” should be reused at every word position
in both sentences. Broadcasting lets us state that reuse without writing out
copies. PyTorch aligns dimensions from the right, so the six feature offsets
line up naturally with the final six-feature axis.

Position offsets have a different job: one offset for “the”, another for “cat”
or “dog”, and so on. A flat four-item list still meets the final feature axis
first. Four does not match six. That failure is welcome: it prevents an
accidental interpretation.

The repair `(1, 4, 1)` names the missing roles: reuse across examples and
features, vary across positions. **Predict:** Which version raises an error?
We catch only that expected mismatch, then print the repaired shape.
**Next complaint:** reusing offsets cannot mix features into new answer scores.""",
    "2794f37a": """A gate is like a separate volume knob on each feature: turn one feature up
or down, but do not combine it with its neighbors. Element-wise multiplication
keeps six features at each word position.

A vocabulary projection asks a different question. For each possible answer,
combine contributions from all six features into one score. That is matrix
multiplication. The feature axis is summed away, and ten candidate-score
columns take its place. Example and position remain untouched.

**Predict:** Which output has six final entries and which has ten? The next
cell prints both contracts. The random matrix is a wiring demonstration, not a
trained word guesser; a score is not a probability or an explanation of a word.
**Next complaint:** the scores run forward, but how can we audit the backward
calculation without taking it on faith?""",
    "6c084803": """Before trusting a complicated graph, give autograd a calculation small enough
to check on paper. Start at three, multiply by the adjustable weight two, then
add the adjustable offset minus one. The intermediate result is five. Squaring
it makes one scalar objective.

If that intermediate value increases a little, the square changes at twice
its current value. A weight change first passes through the input three; an
offset change passes through unchanged. So the expected feedback is thirty for
the weight and ten for the offset.

The code compares those independently calculated sensitivities with `.grad`.
This is local feedback, not an optimizer update and not proof that any chosen
objective is meaningful. **Predict:** Do both routes agree? After that audit,
the remaining complaint is what happens when feedback is requested twice.""",
    "6b65aee8": """Think of `.grad` as a notepad beside a parameter. Backward adds a contribution
to the notepad; it does not automatically throw the old page away. That makes
deliberate accumulation possible, but it also explains a very quiet training bug.

The next experiment builds a fresh tiny calculation for each backward call.
We are not reusing a graph whose saved intermediates have already been released.
One call writes three; the second adds another three. Clearing the buffer
returns it to zero.

Detach and inference mode address a different boundary: whether later results
participate in gradient tracking. Detach is not a storage copy, and neither
operation clears a parameter's existing gradient. **Predict:** Will both
non-training results report `requires_grad=False`? The printout checks that
separately from the accumulation measurement.""",
    "2781b23f": """Now put the address story and the feedback story together. Each word ID fetches
one six-feature row from an embedding table. The output layer combines those
features into ten candidate scores at the same position.

There is deliberately no correct next-word target here. Squaring and averaging
the scores gives one convenient scalar that can send feedback through both
parameter tables. Reducing every score to that scalar is useful for this wiring
audit; reducing scores toward zero would not teach the model to predict “cat”.

**Predict:** Will the embedding gradient have the table's shape, and the output
gradient have the output layer's stored shape? Matching shapes show where
influence reached, not useful language knowledge. Later word-objective lessons
replace this artificial objective with actual checking answers.
**Next complaint:** can you regroup features without losing a single value?""",
    "5320d90e": """### Your turn: change only the number of bundles

Keep the two examples, four positions, and six features exactly as they are.
Change only how many equal bundles share those six features. Two bundles had
three features apiece; three bundles should have two. This is a shape drill,
not a newly trained attention mechanism.

The divisibility assertion is part of the contract. Four equal bundles cannot
hold six features without changing the problem; a helpful failure is better
than silently dropping or padding values.

**Predict:** With the supplied setting, will the final two axes be `(3, 2)` or
`(2, 3)`? Read the output as “bundle count, width”, then change the setting back
to two. **What remains:** even perfect tensor bookkeeping leaves trainable
weights scattered around. Part 03 gives those weights an explicit home.""",
})

lesson("03-models-and-training-loops", {
    "3fd09ce2": """### Keep the numbers small enough to audit

The last notebook borrowed word positions to expose axis mistakes. Now remove
that extra axis so we can isolate a new problem: who owns the adjustable
numbers? We return to numeric sensors, but this time each row has **engine
load** and **ambient temperature**, checked against one made-up fuel-use value.

Think of the rule as two influences plus a baseline. More load raises the
target; temperature contributes with the opposite sign. A little synthetic
noise keeps the task from being exact equation copying. These are not measured
fuel units or a physical engine model.

We stack the two sensor lists as columns, never as extra examples. **Predict:**
Should 96 rows produce 96 targets or 192? The shape printout establishes that
contract before we create a single trainable weight.""",
    "34f87fc0": """Each sensor gets one adjustable influence, and the model gets one baseline
offset. To make a guess, multiply a row's load and temperature by their
influences, add those contributions, then add the baseline. Every row uses the
same adjustable numbers.

Nothing in that recipe requires a model class. Ordinary floating-point tensors
with gradient tracking can already express it. Starting them at zero also gives
us a controlled initial state for the module comparison later.

**Predict:** Will a zero-weight, zero-offset model return one zero guess per
row? The next cell checks the forward shape and initial values. Keep the
parameter list in mind: right now you, not a container, must remember which
tensors need saving, moving, or updating. **Next complaint:** a valid guess
shape says nothing about whether those guesses improve.""",
    "1d5022df": """### Attempt 1: turn the knobs ourselves

Compare the fuel guesses with their matching targets, square each miss, and
average. That gives one mean-squared-error score. Backward writes feedback into
the weight and offset notepads. We then subtract a small multiple of that
feedback ourselves.

The update happens without recording another graph: it changes the adjustable
state for the next pass rather than becoming part of the prediction formula.
Clear the gradient notepads afterward so the next pass starts fresh.

**Predict:** Will this completely class-free implementation lower its loss?
We record 180 checks, each just before its corresponding update. The last
printed check therefore precedes the last update; it is not a fresh evaluation
of the final state. **Next complaint:** the math works, but every responsibility
still depends on remembering a loose collection of tensors.""",
    "e403dbb6": """Put the same knobs in a labeled equipment case. That is the useful mental
model for `nn.Module`: a home for parameters and the recipe that uses them.
Registering a child `nn.Linear` lets PyTorch discover its weight and bias through
`parameters()` and `state_dict()`.

`__init__` builds the case once. `forward` describes one prediction. Calling
`model(X)` uses the current contents; it does not clear gradients, calculate a
task loss, or decide an update. Keeping those jobs outside makes the model
reusable.

For a fair comparison, overwrite the layer's ordinary random initialization
with the same zeros used by the raw tensors. **Predict:** Should the named
parameter list contain the layer's weight and bias? We inspect it rather than
assuming that assigning any tensor attribute would register a parameter.""",
    "f72ce811": """### A good container can still compute the wrong comparison

The module should return one fuel value in a column for every sensor row.
The target must have that same row-and-column layout. If you drop the target's
last axis, broadcasting can compare each prediction against many targets
instead of its matching row. The code may run while asking the wrong question.

Think of prediction and target as two transparent sheets laid on top of one
another. Their coordinates should line up deliberately before error is reduced
to a single score.

**Predict:** Will prediction and target both be `(96, 1)`, while the mean loss
has no axes? The assertion checks the stronger condition before training.
**Next complaint:** parameter registration made discovery automatic, but we
still need a visible policy for acting on the gradients.""",
    "da4e19fb": """### Watch one update before trusting the loop

An optimizer is a separate worker holding the parameter list and an update
policy. Here SGD performs the same gradient-scaled subtraction we wrote by
hand. We deliberately choose ordinary SGD without momentum so the equivalence
test has no extra moving state.

Clear, predict, measure, backward, step: the order is the important part.
Copying the weight before backward gives us an independent reference. Backward
should fill a nonzero gradient without changing that reference or the live
weight; only the step should move the live weight.

**Predict:** Which event changes the knobs? We assert the no-change boundary
and print the gradient magnitude separately from the update magnitude.
They measure different things. **Next complaint:** one inspected update is not
the same budget as the raw model's 180 updates.""",
    "e5af3ac9": """The module has already taken one step. Repeating another 180 would make a
pleasant-looking but unfair comparison. Run 179 more, so both implementations
finish after exactly 180 updates from the same zeros.

Keep the input rows, targets, loss, and learning rate unchanged. The only
intended change is who stores the parameters and applies the otherwise identical
update. Store the first inspected loss with the remaining history so the
plot and numerical comparison describe the same checkpoints.

**Predict:** Will the final printed loss be close to the raw implementation's
last check? A similar loss is encouraging but not sufficient: two different
parameter sets can sometimes give similar averages. **Next complaint:** let's
compare the actual weights, predictions, and recorded history before claiming
equivalence.""",
    "e18475a6": """The two cases expose their weights in opposite orientations. Our raw tensor
has one row per sensor feature and one output column. `nn.Linear` stores one
row per output and one column per input feature. Transpose only at that storage
boundary; do not reorder the examples.

We compare the largest absolute parameter difference and the largest prediction
difference. Those reductions turn many small comparisons into conservative
worst-case checks. A history comparison also catches a mismatch in update order
that the final state alone might conceal.

**Predict:** Under identical conditions, should all three comparisons fall
within the declared tolerance? The assertions make “same math” falsifiable.
This proves parity for this affine model and update policy, not every possible
framework implementation. **Next complaint:** tiny numbers are precise but
harder to scan than two overlaid histories.""",
    "ed248f0d": """Plot the two real histories on the same axes. If ownership was the only change,
the paths should overlap, not merely finish near one another. The dashed line
keeps the second trace visible when it sits directly on the first.

The horizontal coordinate counts loss checks before an update. The vertical
axis is logarithmic so later small differences remain visible; this does not
change the values or turn error into a percentage. We already asserted history
parity, because overlap by eye is weaker evidence than comparing the arrays.

**Predict:** Will you see two distinguishable training trajectories? Read the
plot alongside the numerical gaps rather than declaring success from a smooth
curve. **Next complaint:** a trained object in this kernel cannot be used by
the next fresh kernel unless we save the right contract.""",
    "a15035ed": """### Give the next process enough information

A state dictionary is the equipment case's learned contents, not the complete
case design. Part 04 will reconstruct the same class, read the saved tensors,
and ask for predictions without borrowing this kernel's memory.

We save two artifacts: weights and a small configuration describing the data
seed, row count, input/output widths, and training rate. Those facts make the
handoff auditable. Feature order matters too: a load weight applied to
temperature would preserve shapes while changing meaning.

**Predict:** Will saving change predictions or perform another update? It
should do neither; it serializes state. The next notebook tests prediction
parity after reloading. These files contain no optimizer trajectory and make
no exact-training-resume claim. **Next complaint:** can a changed learning rate
be tested without accidentally changing the initial model too?""",
    "c0d63dc9": """### Your turn: one knob, one controlled comparison

You have a saved model for the next notebook; leave it alone. Create fresh
trial models instead. Each trial starts at the same zero weights and bias and
gets the same ten full-batch updates on the same sensor rows.

Change only `YOUR_LEARNING_RATE`. The reference rate is the one used above;
the candidate uses your setting. Compare the final recorded errors, not just
whether either run improved. A slower rate can improve yet still travel less
far in the same update budget.

**Predict:** Will the smaller supplied rate make more or less progress here?
The printed comparison resolves that local question without claiming an
optimal rate. Model selection on held-out evidence belongs to Part 05.
**Next complaint:** every update still reads all rows at once. Part 04 asks
how examples arrive in smaller groups and what survives a restart.""",
})

lesson("04-data-checkpoints-and-hyperparameters", {
    "b855a304": """### Start with a real handoff, not a replacement model

Part 03 left you two files, not a shared Python object. This fresh kernel must
rebuild the fuel model's equipment case before loading its contents. The class
therefore repeats the same two-input, one-output structure.

We locate this lesson relative to the notebook or repository, not whichever
folder an editor happened to use. Then we locate Part 03 beside it. If either
artifact is missing, stop with an actionable message. Quietly training a
replacement would conceal a broken handoff.

**Predict:** Is knowing the filename enough to reconstruct an arbitrary model?
No: the class and its feature order remain part of the contract. This setup
only checks prerequisites; it does not prove that the saved weights load
correctly. **Next complaint:** a file on disk is not yet usable evidence.""",
    "e7b07933": """Loading means assigning saved learned tensors to a newly constructed model
with compatible names and shapes. `map_location` chooses where those tensors
land. `weights_only=True` uses PyTorch's restricted loading path for the tensor
state we wrote ourselves; arbitrary untrusted checkpoint files are not part
of this exercise.

Read the configuration too. The input width, output width, seed, and row count
explain what the saved model expects. Evaluation mode declares prediction-time
behavior, even though this affine model has no dropout to switch off.

**Predict:** Should the reconstructed state contain the layer's weight and
bias keys? We inspect and validate the configuration, then load. A successful
load proves compatibility, not prediction parity. **Next complaint:** before
comparing predictions, we need to recreate the exact sensor-column contract.""",
    "630aec8a": """The saved seed lets us rebuild the same synthetic practice table. That is a
convenience of this lesson, not a general replacement for preserving real data
and its provenance.

Each row again pairs engine load with temperature, in that order. The target
uses the same made-up fuel rule and noise recipe. We keep a target column so
the loss will compare row partners rather than broadcast across rows.

**Predict:** If we stack the two sensor lists as rows rather than columns,
would the saved layer still see two input features per example? No. The shape
and configuration checks guard that boundary. **Next complaint:** having 96
correct pairs does not say how to deliver them. We will first write that policy
by hand so the later DataLoader cannot hide it.""",
    "82a96cd1": """### Attempt 1: shuffle addresses, then fill boxes

Imagine putting each input row and its target into one envelope. Shuffle the
envelope addresses, never the input and target contents independently. Slice
the address list into groups of twenty.

Ninety-six is not divisible by twenty. Four full groups leave sixteen
envelopes. Keeping that remainder is a deliberate policy; dropping it would
discard evidence this pass. Concatenating and sorting the selected addresses
lets us compare against the complete original address list, catching both
omissions and duplicates.

**Predict:** Will the five groups cover every original row once despite the
shuffled order? The equality check answers that, not just the group sizes.
**Next complaint:** repeating index slicing beside every training loop would
mix delivery policy with the model's learning responsibilities.""",
    "d1ba0f09": """A Dataset is the filing cabinet: ask for row `i`, receive that row's features
and target together. A DataLoader is the delivery trolley: choose an order,
group size, and remainder policy, then bring groups to the learner.

`TensorDataset` is enough because our entire table already fits in memory.
The loader does not normalize measurements, choose a loss, or update weights.
Its dedicated random generator makes the shuffle policy repeatable independently
of model initialization.

**Predict:** Should one item have shapes `(2,)` and `(1,)`, while a batch has
`(20, 2)` and `(20, 1)`? The printout distinguishes item and batch contracts.
Taking a peek consumes a loader iteration, so do not treat this as a saved
continuation point. **Next complaint:** a plausible first batch still does not
prove the whole pass retained every pair.""",
    "fe59002f": """Turn shuffling off for an audit. If the loader is only grouping the original
paired rows, joining its feature batches along the example axis should
reconstruct the feature table exactly. Joining target batches should do the
same for the answer column.

This is why `cat`, not `stack`, is appropriate: the groups already have an
example axis. We want to extend that axis, not add a “batch number” axis.
The last group's sixteen rows must survive both joins.

**Predict:** Will features and targets each match, with a smaller final batch?
We assert all three conditions. Ordered parity proves this delivery contract;
the earlier address check proved the shuffled manual contract.
**Next complaint:** the examples now arrive correctly, but a saved model and
a saved training process still answer different questions.""",
    "cf746eb2": """A reload test needs two independent objects. Use the model reconstructed at
the opening as the reference, build another fresh case, and load the same
weights into it. Ask both for fuel guesses on seven identical input rows.

No training happens between these predictions. Evaluation mode declares the
behavior, and inference mode avoids recording a graph. The maximum absolute
gap asks whether even the worst probe prediction changed.

**Predict:** Will the gap be zero, within the stated tolerance? That is a
testable inference claim. It is not an interrupted-training experiment: there
is no next batch, no optimizer restoration, and no matching-next-update
comparison. **Next complaint:** Adam also remembers previous gradients, so
preserving predictions alone cannot preserve all the state that determines
future updates.""",
    "9eba2ce4": """### Pack selected training state without overselling the bag

Now train a fresh fuel model with mini-batches. Each batch gets its own clear,
guess, check, backward, and step. An epoch means one complete loader pass, not
one update; with our remainder policy, an epoch contains five updates.

Adam carries running summaries that influence later steps. Save those alongside
weights, the zero-based epoch index, and the selected batch-size/rate settings.
Those fields are useful records of the process.

**Predict:** Does writing that bag prove the same next update after a crash?
No. We do not restore and compare an interrupted run here, and the loader's
random state is not in the bag. The historical filename remains for compatibility,
not as a certification. **Next complaint:** saved settings are visible, but
how do we decide which setting helped?""",
    "62dae55a": """A fast-looking loss curve is not a controlled comparison if the starting
weights, data, or update budget changed at the same time. For each learning
rate, rebuild the same initial fuel model from the same seed and run thirty
full-batch updates.

Only the step scale changes. A small rate may barely move; a larger one may
make rapid progress or overshoot. The last printed loss is the check before
the final update, so compare like with like rather than relabeling it as a
fresh post-training score.

**Predict:** Which candidate improves most within this budget? The numbers
decide; there is no universal ranking promised in advance. These are training
comparisons, not held-out model selection. **Next complaint:** changing batch
size changes how many updates fit inside an epoch.""",
    "b9e7a884": """### Your turn: count deliveries before training again

Change only `YOUR_BATCH_SIZE`. Leave the original 96 envelopes, their pairing,
and the remainder policy unchanged. This drill inspects delivery; it does not
retrain and pretend that different update counts are a fair optimizer test.

With sixteen rows per group, all six groups should be full. A size that does
not divide 96 should instead reveal a smaller final group. Summing group sizes
checks the number delivered; the ordered parity audit above checked the actual
contents as well.

**Predict:** How many groups will the supplied value produce? Compare your
count with the printed list, then try twenty. **What remains:** correct
delivery and saved fields do not protect an exam from accidentally becoming
practice. Part 05 separates learning from validation and chooses which model
to keep.""",
})

lesson("05-modern-training-lifecycle", {
    "41cb5d93": """### Keep the fuel task; change the experiment's responsibilities

You can already deliver pairs and update a model. Now imagine reporting a
beautiful training curve while the model quietly gets worse on rows it did
not practise on. A working update loop does not prevent that mistake.

This notebook keeps load and temperature as inputs and fuel as the target.
A curved interaction now makes a small nonlinear model useful. We will keep
training, validation, and final testing separate rather than changing domains
again.

Configuration is a visible checklist of choices, not learned state: seed,
batch size, learning rate, epoch budget, and patience. **Predict:** Will changing
patience alter the forward formula? No; it changes when we stop practising.
The setup prints that checklist before any data or model state can obscure it.""",
    "b966fb71": """First make a practice world whose answer rule is known to us but not supplied
to the model. Load raises fuel use, temperature changes it, and their interaction
adds a bend. Synthetic noise prevents every target from lying exactly on the
same clean surface.

Each of the 180 rows still contains two measurements. The target list receives
a final singleton column so one output can line up with one target. We have
changed the target rule from the affine Part 03 example, not silently reused its
checkpoint in an incompatible task.

**Predict:** Should adding an interaction create more feature columns or more
targets? Neither: the shapes remain `(180, 2)` and `(180, 1)`.
**Next complaint:** if all these rows help set preprocessing, the supposed exam
has already influenced practice.""",
    "a9b23399": """### Attempt 1: use every row to choose a measuring ruler

Normalization chooses a center and scale for each sensor column. Using all
rows seems harmless—no labels are involved—but it lets future evaluation
inputs influence a training choice. That is information leakage.

Split the raw row addresses first. Only the 108 training rows choose the
ruler; 36 validation rows guide later decisions; 36 test rows wait until those
decisions are finished. The random split can produce different feature means
without an artificially imposed distribution shift.

**Predict:** Will the train-only center equal the all-row center exactly?
Measure the difference. A nonzero gap shows precisely what information the
all-row ruler would import; it does not quantify an accuracy penalty.
**Next complaint:** the training ruler must be reused consistently, including
for the target scale.""",
    "0e46a309": """Subtract the training mean and divide by the training standard deviation for
each sensor column. The resulting values say “how far from a typical training
measurement?” rather than mixing the original measurement scales.

Fit the fuel target's center and scale on training rows too. That makes the
optimizer's numerical task convenient, but predictions later need to be put
back in fuel units. Validation and test rows must use these same training
rulers; centering each split separately would change what a coordinate means.

**Predict:** Should training feature means be near zero while validation means
can remain different? The reductions print one value per feature, not one
value per row. A validation shift is evidence, not something to erase.
**Next complaint:** the split tables are honest, but their batch delivery still
needs an explicit policy.""",
    "48e03227": """Shuffle practice cards, not exam cards. The training loader varies which
rows share an update; validation and test loaders keep their order stable
because they only measure. All three keep input-target pairs together.

The training split is not divisible by the batch size. Keeping the smaller
final batch preserves evidence, but it means an epoch score cannot simply
average batch means equally. Each batch mean must contribute in proportion
to its number of examples.

**Predict:** Will the sample batch still have two feature columns and one
target column? The shape check confirms that delivery has not changed the
model contract. The later loops will weight their averages by row count.
**Next complaint:** even perfect delivery is not enough when the model itself
behaves differently during practice and evaluation.""",
    "cc4b70c7": """A straight guesser cannot express our new interaction by stacking only
straight layers. Give it sixteen hidden features and a ReLU bend, then mix them
back into one fuel prediction.

We also add dropout: during training it randomly masks some hidden activations
and rescales the retained ones. This encourages the model not to depend on one
fixed hidden route. It does not create new examples or remove feature axes.
During evaluation that masking stops.

**Predict:** Does counting parameters tell us whether the model learned
anything? No, but it audits the architecture before training. The model keeps
prediction separate from loss, normalization, and optimization.
**Next complaint:** saying that train and eval modes differ is weaker than
giving the same input twice and measuring the difference.""",
    "0a1e51a7": """Give the untrained model the exact same six sensor rows twice. In train mode,
dropout can choose a different mask each time. In eval mode, it uses the full
hidden representation deterministically.

Notice the two controls: `eval()` chooses layer behavior, while inference mode
chooses whether autograd records a graph. Neither is a substitute for the
other. This probe makes no update and no claim about final prediction quality.

**Predict:** Which repeat gap should be exactly zero? We assert the evaluation
gap, print the training gap honestly, and do not require every imaginable pair
of stochastic masks to differ. **Next complaint:** once training begins, an
optimizer, a scheduler, and optional accelerator arithmetic each own different
parts of the update process. Put those responsibilities in separate places.""",
    "885dba00": """AdamW chooses parameter updates and also applies decoupled weight decay.
The learning rate sets their overall scale. A plateau scheduler observes
validation loss and reduces that scale after repeated non-improvement; it does
not backpropagate through the validation examples.

Mixed precision is a separate numerical option. On supported CUDA hardware,
selected operations can use fewer digits, while a scaler protects small
backward signals and compensates before updating. CPU execution keeps this
path disabled rather than implying that a CPU run measured accelerator speed.

**Predict:** In our required CPU validation, will the enabled flag be true or
false? The printout declares the branch actually used. **Next complaint:**
owning these tools is not enough; the train function must use them in the right
order and aggregate examples fairly.""",
    "7eb77589": """### One function is allowed to practise

The training function selects train mode, receives each batch, and clears old
gradients. Inside the optional autocast region it predicts and measures error.
Then the scaler-aware backward and step perform the update. With scaling
disabled on CPU, these calls preserve the ordinary update responsibilities.

The loss function returns a mean over the current batch. Multiply by that
batch's row count to recover its contribution, accumulate those totals, then
divide by all training rows. Otherwise the smaller remainder batch receives
too much voting power.

**Predict:** Does defining this function train a model yet? No: execution waits
for a call. The printout marks that boundary. **Next complaint:** the exam
function should share the averaging contract, but must not share permission
to write gradients or update parameters.""",
    "aa634347": """### The other function only checks

Validation selects eval mode before it sees a row. Inference mode prevents
new gradient graphs, and there is no backward or optimizer step in the
function. The same example-weighted averaging keeps train and validation scores
comparable even when their final batches differ.

Do not read an existing parameter gradient as proof that validation created
it: training may have left a value there. The meaningful audit copies both
weights and any existing gradients, calls evaluation, and checks that neither
changed.

**Predict:** Should all copied state still match afterward? We run that
side-effect check below, rather than relying on a reassuring function name.
This protects the current implementation; a future edit must keep passing it.
**Next complaint:** honest validation still needs a rule for deciding when
progress has stalled.""",
    "08b53a92": """An early-stopping observer remembers two things: its best validation error
and how many consecutive checks failed to beat that error. An improvement resets
the counter; a non-improvement increments it. Reaching patience says to stop.

Strictly lower counts as improvement here. Equal scores therefore consume
patience. That is a declared policy, not a universal definition of progress.
Saving a best model is related but separate: the observer says when to quit,
while the best checkpoint says which earlier state to keep.

**Predict:** Does constructing this observer train, save, or evaluate anything?
No; it creates decision state. The later synthetic sequence will isolate the
stopping rule from optimizer noise. **Next complaint:** we need two file
purposes, because “best for inference” and “current process state” are not the
same object.""",
    "316964cf": """Give the two checkpoint jobs different paths before the loop begins.
`best_model.pt` will hold a selected model plus its preprocessing facts.
`resume_training.pt` will record selected state from the latest completed
training process, not necessarily the best epoch.

The history starts empty because the plot must come from actual measurements.
We retain separate lists for train loss, validation loss, and the rate chosen
after the scheduler observes validation. Those lists describe different events
within each epoch.

**Predict:** Does preparing a path prove a useful checkpoint exists? No; the
save and reload experiments must establish that. The directory printout makes
the artifact location visible even when the kernel started at repository root.
**Next complaint:** now assemble the pieces without hiding their ordering.""",
    "4bcd6a63": """For each epoch, practise first and validate second. Let the scheduler see the
validation result, then ask the stopping observer whether it improved and
whether patience is exhausted. Record those actual values.

When validation improves, save the current model with the training-derived
feature and target rulers. A later worse epoch must not overwrite that best
file. Stopping and saving are separate decisions, so the final in-memory model
may differ from the best model on disk.

**Predict:** Must training use all eighty allowed epochs? Not if patience runs
out. We print the actual count and best epoch rather than promising early
stopping will trigger on every run. **Next complaint:** a selected inference
file does not record all the state of the latest training process.""",
    "3a7752ed": """Now write a different bag: the latest model, optimizer, scheduler, epoch,
early-stopping counters, and configuration. That records how the process was
moving, not just which model validation preferred.

The filename is historical. This teaching snapshot deliberately omits several
recovery inputs: dropout and loader random states, CUDA scaler state, and
preprocessing statistics. Merely saving the listed fields cannot certify
reconstruction of the same next update.

**Predict:** Should the latest-state file always contain the best weights?
No; an older epoch may have won selection. We keep the two paths separate and
later check the declared snapshot keys, not a continuation we did not run.
**Next complaint:** the printed minima hide when improvements, plateaus, and
rate changes occurred. Plot the recorded sequence before drawing conclusions.""",
    "a94f2519": """A history plot is a timeline, not a manufactured success curve. Training loss
uses noisy training-time behavior across changing weights; validation uses
evaluation behavior at the end of each epoch. Their values therefore have
different measurement conditions, even though both are normalized MSE.

The second panel shows the rate after the scheduler's decision. A flat line
means no reduction occurred; it is not a plotting failure. We report the number
of actual reductions rather than asserting a plateau happened.

**Predict:** Does the last epoch necessarily have the lowest validation point?
Locate the minimum and compare it with the separately recorded best epoch.
This picture summarizes evidence; the saved-file reload is still needed.
**Next complaint:** normalized scores are convenient for training but not what
a fuel prediction consumer wants to read.""",
    "45921844": """Open the selected best-model file into a fresh module. Reconstruct the test
features using the **saved** training mean and scale, not a convenient live
normalized variable from this kernel. That tests the inference artifact's
preprocessing contract too.

The model emits normalized fuel values. Multiply by the saved target scale
and add its saved mean to return to the original synthetic units. Mean absolute
error then averages the absolute fuel misses across untouched test rows.

**Predict:** Should prediction and target still have one column per row after
undoing normalization? Check the shapes and report the measured MAE.
The result describes this generated task, not real fuel-system performance.
**Next complaint:** successful inference reload says nothing yet about the
separate training snapshot's declared fields.""",
    "a953bc54": """A key check answers a narrow question: did the written training snapshot
contain the fields we promised? Load that file using the restricted tensor
loader, compare its key set with the declared set, and report anything missing.

This does not restore an optimizer into a live training loop, reconstruct
the next batch, or compare against uninterrupted learning. A dictionary with
the right labels can still lack enough information for exact continuation.
The missing random and preprocessing state remains missing even when this
assertion passes.

**Predict:** Should the missing-key list be empty for this writer? Yes; that
is the whole assertion. Keep that small success separate from the stronger
recovery claim we intentionally do not make. **Next complaint:** test stopping
policy on known inputs rather than guessing why a training run stopped.""",
    "acf4941a": """### Your turn: give the observer a known sequence

Training curves mix optimization noise, dropout, and data order. To isolate
patience, use an explicitly synthetic list of validation values. It improves
once, then gets slightly worse on successive checks. These are decision-test
inputs, not fabricated training measurements.

Change only `YOUR_PATIENCE`. Count consecutive non-improving checks from the
last improvement, not from the first epoch. The supplied value should stop on
the sixth check; a larger value may not stop within this short sequence, which
is a valid result rather than an assertion failure.

**Predict:** Where should your chosen value stop, or is the sequence too
short? The printout and derived expectation close that loop. **What remains:**
this whole trainer still treats fuel rows independently. Sequence memory adds
ordered history while keeping these evidence boundaries.""",
})

lesson("pytorch-training-skeleton", {
    "00r-03": """### Give the shortcut an honest little task

Think of two standardized sensor readings and a two-state alarm. Negative
readings belong to the calm cluster; positive readings belong to the alert
cluster. These eight hand-written points are practice cards, not a real safety
system or a held-out dataset.

Classification changes the checking answer from a fuel number to an integer
category address. The model will eventually emit two raw scores per row, but
the target is just the correct class ID. Keep those labels attached while the
loader shuffles row addresses.

**Predict:** Does a four-example batch contain four input rows and four class
IDs, or four one-hot target vectors? The printout checks the former. **Next
complaint:** delivery works, but the loader does not own the adjustable guesser
or decide how its weights are organized.""",
    "00r-05": """Put the guesser's adjustable numbers in a module, not in the delivery
mechanism. Its first layer mixes the two sensor readings into eight hidden
features, ReLU adds a bend, and the last layer produces calm/alert scores.

Those final columns are not probabilities. Leaving them raw lets cross-entropy
perform its stable checking calculation. A softmax is only needed later if a
person wants probabilities; choosing the largest score does not require it.

`__init__` creates layers once; `forward` uses the current layers for one guess.
The registered-parameter printout lets you verify what the optimizer will see.
**Predict:** Should a four-row input batch return four rows with two scores
each? The assertion answers before training. **Next complaint:** a callable
model is still only a guesser, not a practice routine.""",
    "00r-07": """### Keep the routine reusable without making it mysterious

Imagine a coach with four independent items: a guesser, a stack of practice
cards, a checking rule, and a worker allowed to turn the knobs. `fit` coordinates
those items; it does not secretly choose new ones.

Within every batch, clear old feedback, make guesses, calculate one loss,
backpropagate, then update. The optimizer owns its own running state, while
the model owns parameters. Multiplying a batch's mean loss by its row count
before the final average keeps a small remainder batch from receiving an
oversized vote.

**Predict:** Will defining `fit` already change any weights? No, a function
definition only stores instructions. The next cell actually calls it.
**Next complaint:** the independent pieces still need wiring, and training
success must be kept separate from broad prediction quality.""",
    "00r-09": """Now choose cross-entropy for the integer class targets and Adam for updates.
Call the same routine for eighty passes over our eight practice cards. Each
epoch contains two updates at the supplied batch size.

After practice, switch to evaluation behavior and stop recording a gradient
graph. The forward calculation still runs. Taking the largest final score
chooses one alarm class per example; averaging the correct/incorrect flags
gives a training accuracy.

**Predict:** Can this tiny model fit all eight supplied points? The assertions
require decreasing recorded loss, correct labels, and no inference graph.
That is deliberately a memorably small smoke test, not evidence about new
sensor measurements. **What remains:** no validation split, checkpoint
recovery, scheduler, or device branch was tested by this shortcut. Carry the
loop to the word lessons; use the full route for those missing responsibilities.""",
})

lesson("01-pytorch-cnns", {
    "12ebf280": """### Why images are a useful detour

A sensor table has named columns; an image has nearby pixels whose relative
positions matter. Flattening those pixels would hide the locality we want
to inspect. This optional branch therefore changes domains deliberately:
vertical and horizontal bars make a shared spatial detector visible.

We generate every image locally, with small shifts and noise. A shift should
not change the orientation label. One grayscale channel is explicit because
`Conv2d` expects examples, channels, height, and width—even when channels is
one. The labels remain a separate one-dimensional class list.

**Predict:** Does stacking 320 images create 320 channels or 320 examples?
The shape printout checks the latter. An always-vertical baseline also exposes
the balanced task before learning. **Next complaint:** a separate weight for
every pixel position would repeatedly relearn the same local edge.""",
    "f47b7316": """### Start with one output value you can reconstruct

A convolutional filter is a reusable local rule. At one position, multiply
each patch pixel by its matching filter weight and sum the products. Move the
same weights to the next position, then repeat.

Our hand-written Sobel filter detects changes across the horizontal direction,
so it responds near vertical edges. It is not learned in this experiment.
We compare one manually calculated response with the corresponding PyTorch
output. No padding means the filter only visits positions where its whole
three-by-three patch fits inside the image.

**Predict:** Should both calculations return the same single number?
The assertion checks arithmetic, while the printed output shape checks the
smaller valid spatial grid. **Next complaint:** agreement at one position
does not show what the detector responds to across a whole picture.""",
    "b3e03744": """Apply that same fixed detector to one vertical bar and one horizontal bar.
The weights do not change between the images. The response maps tell you
where the local rule found matching edge evidence.

Take absolute responses for display so edge direction does not cancel when
we summarize magnitude. Summing a response map gives one total over all its
spatial positions; it is not a class probability or a learned decision rule.
Noise and the ends of the bars can create responses in either orientation.

**Predict:** Which generated bar gives the larger total in this run? Read the
printed values and the paired image/map panels together, rather than assuming
one map must be blank. **Next complaint:** visiting every patch keeps detail,
but it leaves us carrying a fairly large response map.""",
    "0893f8c6": """There are two different ways to shrink that map. Stride changes where a
convolution evaluates its patch: taking two-pixel steps skips intermediate
locations. Pooling first receives an existing map, then summarizes each local
block without learning another filter.

Both can halve height and width here, but equal output shapes do not make
their values or jobs identical. Padding lets the stride-one convolution
retain the input's spatial size before either reduction.

**Predict:** Will stride-two convolution and two-by-two max-pooling both
produce eight-by-eight maps from this setup? The printout checks the geometry.
The random filter bank is only a shape experiment, not a detector trained
to solve the task. **Next complaint:** a hand-designed edge filter is useful,
but who decides which local clues the classifier should actually learn?""",
    "b75c3cc9": """### Let checking answers teach the shared filters

The first learned convolution produces six local feature maps. Pooling
shrinks their spatial grid, and a second convolution mixes nearby evidence
into ten maps. Averaging each final map over space makes ten summaries for
one image. A final linear layer turns those into two raw orientation scores.

The loss checks the integer orientation, so feedback can adjust both
convolutions and the final decision layer. Record the untrained validation
score before practice, then train with the explicit loop you already know.
The recorded epoch loss is the final batch's loss, not an epoch-wide mean.

**Predict:** Will the trained classifier beat the always-vertical baseline
on new locally generated examples? The measured validation accuracy and
assertion answer that bounded question. **Next complaint:** useful accuracy
still hides what the first filters and their maps look like.""",
    "85e1c830": """A filter is a tiny learned weight table; its feature map is the much larger
set of responses when that table visits an input image. Put each filter above
its response map so those two roles cannot be confused.

All six first-layer filters see the same validation image. Different-looking
weights and responses are evidence of different local calculations, but they
are not automatically six human-named concepts. One channel may combine with
others later rather than serve as a standalone detector.

**Predict:** Do the filter pictures have the same spatial size as the maps?
No: the first are local rules and the second are location-indexed responses.
We use no-gradient inspection because these plots do not train anything.
**Next complaint:** a second-layer value reads earlier maps, so which original
pixels can actually influence that later response?""",
    "498efaf9": """Trace backward from one second-layer pre-activation. Earlier convolution,
ReLU, pooling, and the second convolution connect it to a region of the original
image. That region is larger than one original filter patch.

Choose the strongest channel at the central position to avoid probing a
conveniently dead ReLU output. The gradient mask marks input pixels with
measurable local influence for this image and current weights. Taking its
row/column extrema describes a bounding box, not a guarantee that every
enclosed pixel contributes equally.

**Predict:** Will that measured box span more than three pixels in each
direction? The assertion checks this active path. Zero local gradient still
does not prove a structural connection could never activate on another
input. **Next complaint:** deeper chains enlarge reach but can weaken
backward signals along the way.""",
    "e6799796": """### Give the correction a route around the work

To isolate a shortcut, temporarily leave images for twelve-feature vectors.
The two chains receive identical inputs and identical small linear transforms.
Each transform is followed by tanh. Only the residual chain adds an unchanged
copy of its current input around each transform.

Repeated small transforms can shrink the plain chain's local sensitivity.
The added identity routes give backward another way through the same-depth
calculation. Summing the final features creates a common scalar probe; the
input-gradient norms measure the two resulting routes.

**Predict:** Which input gradient is larger in this controlled construction?
The printed norms and assertion resolve that consequence, not a claim about
every deep network. **What remains:** classification keeps enough evidence
to choose orientation, but not necessarily enough to rebuild the image.
The autoencoder branch changes that checking job next.""",
})

lesson("02-pytorch-autoencoders", {
    "b5c22c96": """### Keep the local image idea; change the checking answer

The CNN only needed to decide vertical or horizontal. Reconstruction needs
more detail: location, thickness, and the rest of the pixels. Keep the same
sixteen-by-sixteen grayscale setting, then add diagonals and square rings
so a compact code has more variation to preserve.

Each generated example also carries a shape label, but we will not use that
label as the reconstruction answer. It is only an audit tag for comparing
groups of latent codes later. The input image itself supplies the checking
pixels.

**Predict:** Should the encoder receive the label, or only the image? Only
the image. Stacking and adding the channel axis establishes the input shape.
**Next complaint:** copying pixels directly would solve reconstruction
without teaching a useful compressed intermediate representation.""",
    "d65fd038": """An encoder is a funnel: two stride-two convolutions shrink each spatial
axis, while the last channel count sets how many feature maps survive.
With four latent channels, each image becomes a four-by-four-by-four code,
not four class predictions.

The decoder is learned upsampling, not an exact inverse. Its transposed
convolutions expand the code back into one sixteen-by-sixteen image.
Sigmoid bounds its output values to the same zero-to-one range as the inputs,
without guaranteeing that they are correct pixels.

**Predict:** Can a randomly initialized encoder/decoder have matching input
and reconstruction shapes while rebuilding badly? Yes; this cell checks
geometry only. The code walkthrough is “shrink, mix, shrink; expand, mix,
expand”. **Next complaint:** a correctly sized reconstruction still needs
pixel-level feedback to preserve anything useful.""",
    "f021f820": """### First ask whether rebuilding improves at all

Compare each reconstructed pixel with its matching input pixel, square the
misses, and average over images, channel, height, and width. That one error
can teach both the encoder's compression and the decoder's rebuilding.
The loader's shape labels are deliberately ignored.

An all-black reconstruction is a useful crude baseline because these sparse
images contain much dark background. Measure it rather than declaring any
small-looking MSE impressive. Then train and compare on the held-out local
images, without promising a particular error in advance.

**Predict:** Will practice reduce training MSE, and does it beat the black
baseline on validation? We assert learning and report the baseline comparison
honestly. These equally sized training batches make their epoch average
well-defined. **Next complaint:** one mean can conceal a missing diagonal
or a blurred square-ring hole.""",
    "c6ec0022": """Put untouched validation images directly above their reconstructions.
Keep the same grayscale range in both rows; automatic rescaling could make
a weak reconstruction look artificially sharp.

Look for what survived: orientation, location, stroke width, and the open
center of the ring. A low average error can coexist with a visible mistake,
especially when most pixels are background. The row labels and per-image
errors make this an audit of actual outputs, not a decorative gallery.

**Predict:** Will every reconstructed shape look equally good? There is no
guarantee. Compare the visible errors with the printed per-image MSE rather
than converting one successful picture into a general quality claim.
**Next complaint:** the images show what survived, but how large was the
code, and do its coordinates have a reliable human interpretation?""",
    "acc02dd3": """Count values per image, not examples in the batch. The input contains
256 grayscale values. The four-channel four-by-four code contains 64.
That is a fourfold reduction in stored value count, not necessarily a
fourfold compressed file size: precision and model weights also matter.

Flattening joins each image's code coordinates into one row for comparison.
Average rows within each audit label to get one group center, then measure
distances between those centers. Labels group the inspection; they never
become encoder inputs or training targets.

**Predict:** Does a separation between group averages prove that one channel
means “diagonal”? No. Coordinates can jointly encode information without
human names, and distances depend on scale. **Next complaint:** can we change
the code budget while keeping the rest of the comparison controlled?""",
    "8929bbd1": """### Your turn: pack into a smaller or larger bag

Train fresh models with two and four latent channels. Reset the same seed,
use the same training rows, loss, optimizer settings, and eighteen-epoch
budget for each. Changing width necessarily changes parameter shapes and
initial draws; this is a controlled recipe, not element-for-element identical
weights across different architectures.

The earlier main model trained longer. Do not compare its final error
directly with these trials and call the difference a width effect.
Within this pair, record validation MSE and the exact stored-value ratio
together.

**Predict:** Must the wider code win on error? No; optimization and the budget
matter. The result must decide. Add one width only after reading the default
comparison. **What remains:** a useful reconstruction code is not a
distributional sampling model or a memory of an ordered text stream.""",
})

lesson("01-keras-to-pytorch-antarctic-field-guide", {
    "25e20206": """### Prepare the branch without changing its theory

Use this detour when Keras has made a responsibility feel automatic.
The required sensor labs used made-up numbers to isolate mechanisms. Real
penguin measurements now expose missing rows, uneven class counts, and
different units without requiring an image or sequence model.

The dataset is bundled in a CC0 package, so no network data download is
needed during the lesson. The first cell checks dependencies and stops
with a useful missing-package message rather than modifying a shared kernel.
Use this folder's setup only if the environment needs it.

**Predict:** Do reference Keras snippets require TensorFlow to execute this
notebook? No: only the PyTorch cells run. We are translating responsibilities,
not benchmarking two installed frameworks. **Next complaint:** reproducibility
and placement still need explicit owners.""",
    "d9cb954f": """The same notebook can draw random values from Python, NumPy, and PyTorch.
Each owns a separate random generator, so seeding only one is an incomplete
promise. We seed all three and print the selected device before the experiment.

These settings make this recipe repeatable in the tested environment; they
do not certify bit-identical arithmetic on every accelerator or library
version. A device is where tensors live, not a model-quality choice.
Inputs and parameters must meet on that same device.

**Predict:** Will explicitly selecting CPU change a species label into a
float? No: placement and dtype are separate contracts. The next cell reads
the bundled field records; before modelling, we must see what evidence
actually survived the missing-value policy.""",
    "403f3702": """Each penguin row contributes four physical clues: bill length, bill depth,
flipper length, and body mass. Species is the separate checking answer, never
an input feature. Dropping records missing a required clue is an explicit
scope choice; it does not explain away real missing-data problems.

Sort the species names before assigning integer IDs so class columns have a
stable meaning. Convert measurements to floating point and labels to integer
addresses, but keep the original table for audits. The package-size guard
detects a surprising source change rather than silently continuing.

**Predict:** Are the retained classes balanced exactly? The printed counts
answer from the actual CC0 records. **Next complaint:** a table shape can
look healthy while measurement scales and class imbalance remain invisible.""",
    "b80fd4a4": """Before scaling anything, look at the evidence. The class-count panel checks
whether one species dominates collection. The distribution panel deliberately
overlays a bill measurement and body mass in their original units: its awkward
scale is the complaint, not a polished comparison of like quantities.

Millimeters and grams are not interchangeable meanings. Normalization can make
their numerical scales more convenient for training, but it cannot turn a
gram into a millimeter or repair biased collection.

**Predict:** Will the raw measurement ranges occupy similar portions of this
shared axis? Read their legends and units before interpreting the histograms.
The assertions also check four feature columns and three species.
**Next complaint:** now fit a ruler without letting the held-out birds help
choose its center or scale.""",
    "6d07eb26": """Split each species separately so both sides retain representation. Then
fit each measurement's mean and scale on the training birds only.
Applying the same ruler to held-out birds keeps evaluation evidence outside
the preprocessing choice.

The kept row axis on the statistics is intentional: one row of four centers
is reused across birds, with each center aligned to its physical measurement.
Class IDs remain a flat integer list because cross-entropy chooses one
correct class for each score row.

**Predict:** Should the feature tensor be floating point while the label
tensor is `long`? The checks make that distinction executable. The variable
names say “test”, but later ablations reuse these birds to choose settings;
then they serve as validation. **Next complaint:** equivalent dense-layer
math can still expose its stored weights in a different orientation.""",
    "fe964a23": """### Inspect storage with a deliberately smaller probe

The penguin model will have four inputs and three outputs. For this isolated
storage proof, temporarily use three inputs and two outputs so every displayed
coefficient remains easy to inspect. This is not a smaller penguin dataset.

Keras-style row-vector math stores a feature-by-output kernel.
`nn.Linear` exposes the output-by-feature transpose. Copying that transpose
changes the storage convention, not which input contributions reach an output.
The bias remains one value per output.

**Predict:** Will fixed coefficients produce the same two outputs in both
written conventions? We compare actual tensors within tolerance. This proves
the affine calculation, not a TensorFlow runtime parity benchmark.
**Next complaint:** knowing where weights live still does not show where
PyTorch writes gradient feedback.""",
    "5902f876": """Temporarily shrink the problem again to one adjustable number: square three.
Its output is nine, and a tiny input increase changes the square at twice
the current input, giving sensitivity six.

This isolates the translation from a recorded `GradientTape` calculation to
PyTorch's recorded graph and leaf `.grad`. `requires_grad=True` requests
tracking; backward follows the recorded operations and writes feedback.
It does not move the number three toward an optimum.

**Predict:** Will `.grad` contain the squared output nine or the local
sensitivity six? The assertion checks six. These small probes remove
framework ambiguity before we return to the real penguin rows.
**Next complaint:** feedback is useful only when some explicit operation
uses it to change adjustable state.""",
    "993237a5": """### Make the feedback act on a knob

Choose a position on a number line and a known target at five. Squared
distance is smallest when the position reaches that target. This is not a
penguin feature or a species probability; it is an optimizer microscope.

Backward gives the position's local error sensitivity. A no-gradient update
moves the adjustable position against that feedback, then clears the
notepad before the next pass. The plotted trajectory records actual
positions after those updates.

**Predict:** Will repeated small corrections approach five rather than merely
calculate another derivative? The final-distance assertion closes the loop.
Eighty scalar steps establish the mechanical distinction, not a general
convergence guarantee. **Next complaint:** a classifier has several possible
answers, and its loss needs a different output contract.""",
    "bc44d6d8": """Return to the four penguin measurements. Sixteen hidden features give the
model learned combinations; ReLU supplies a bend; three final scores line up
with the stable species order. These hidden features do not need biological
names to be useful.

The last layer stays raw. Cross-entropy will handle the stable score-to-loss
calculation with the separate correct-species ID. Adding a softmax here
would change what that loss receives even though shapes still look fine.

**Predict:** Should four birds produce four rows of three scores, or one
three-score summary of the whole batch? The shape assertion checks one row
per bird. Parameter registration is the same equipment-case idea as Part 03.
**Next complaint:** can an incorrect probability conversion run without
raising a helpful shape error?""",
    "3b143ffb": """Use two fixed class-score rows to isolate the loss contract. This is a
two-class arithmetic probe, not a change to the three-species project.
In one route feed raw scores directly to cross-entropy. In the other,
convert them to probabilities first, then incorrectly pass those as new scores.

Both routes have acceptable tensor shapes. The error is semantic:
cross-entropy interprets its inputs as logits, so the second route is not
the intended objective. Comparing gradient norms follows the mistake into
its update feedback, not just its loss value.

**Predict:** Will the two routes produce identical loss and gradient
magnitude? The numbers answer. **Next complaint:** an appropriate loss still
needs the explicit clear/guess/check/backward/step routine that high-level
`fit()` usually hides.""",
    "25e3b7ab": """Before training a whole classifier, inspect one practice step. Build a
fresh penguin model from the same seed, give Adam its registered parameters,
clear old feedback, calculate species scores, and compare them with the
integer labels.

Backward leaves one gradient tensor beside each parameter tensor. The norm
combines their squared entries into one magnitude for inspection; it does not
replace the per-parameter feedback used by Adam. Only the following step
changes the weights.

**Predict:** Can a loss and a nonzero gradient exist before the update?
Yes; we print the pre-update loss and feedback magnitude separately.
The helper model does not consume a step from the main classifier's budget.
**Next complaint:** one correct step cannot show whether this explicit routine
actually learns the field-data task.""",
    "80bbfc67": """Repeat that full-batch routine with a fresh seeded model. The helper returns
the trained classifier, real loss history, and gradient-magnitude history so
later comparisons reuse the same recipe.

Every epoch sees all training birds and makes one update. Clearing gradients
is an explicit switch because the next experiment intentionally removes
that behavior. Losses are recorded before each update, so their horizontal
axis should not be confused with a post-update evaluation curve.

**Predict:** Will loss decrease while the gradient trace changes? Read the
recorded curves and final values; do not interpret a smaller gradient as
proof of biological understanding. The function's walkthrough is initialize,
guess, check, trace, update, record. **Next complaint:** is clearing only a
cleanup preference, or does it change the algorithm?""",
    "2dabdcff": """Run two models from the same seed with the same birds, widths, Adam rate,
and eighty-update budget. One clears its notepad each time; the other
deliberately adds new feedback to all previous feedback.

That is one changed behavior, not a broken graph reuse test. Each forward
pass builds a fresh graph. The difference is what remains in the leaf
gradient buffers when backward writes its next contribution.

**Predict:** Will the paths overlap, diverge numerically, or immediately
raise an exception? The plots and final errors answer from this experiment.
Accumulation is not inherently wrong; it needs an intentional scheme and
appropriate scaling. Here it is an uncontrolled history of old gradients.
**Next complaint:** a completed training curve still is not a prediction
audit on birds the model did not practise on.""",
    "80b5bf56": """Turn the classifier into a measuring instrument. Evaluation mode selects
prediction-time layer behavior, while no-gradient mode avoids recording
an unused backward graph. This model has no dropout or batch normalization,
so switching mode is a contract rather than a numerical transformation today.

Keep raw scores for class selection, and calculate probabilities separately
for human-facing inspection. Choosing the largest score consumes the class
axis and leaves one ID per bird. Averaging correct decisions gives one
held-out accuracy.

**Predict:** Can a high average hide confusion between two species? The
confusion matrix keeps actual and predicted species roles separate.
Those rows later guide ablations, so they cannot remain an untouched final
exam. **Next complaint:** totals hide individual confident mistakes; inspect
specific birds before trusting the aggregate.""",
    "2a4973cc": """Read several actual decisions as an audit trail. Each row names the source
record, the true species, the chosen species, and the model's assigned
probability for that chosen class. The original row address is useful for
tracing a surprise back to measurements.

A large softmax value means a strong relative preference among the model's
three scores. It does not establish calibrated uncertainty, data quality,
or a biological verdict. A classifier can be confidently wrong, especially
outside its measured-data setting.

**Predict:** Do the shown decisions agree with the corresponding confusion
matrix entries? Check row identities rather than assuming every strong
probability is correct. **Next complaint:** perhaps another width or feature
set helps—but once those held-out decisions guide a choice, they are
validation evidence.""",
    "2a7ad130": """### Your turn: give hidden features a different budget

Change only the list of hidden widths. Every trial uses the same birds,
split, seed, optimizer settings, and 160-update budget. A different width
necessarily changes parameter shapes and random initialization draws; it is
the same recipe, not an elementwise weight-copy experiment.

Compare both final recorded training loss and held-out decisions. More
capacity can improve training fit without reliably improving unseen-row
decisions. Width is therefore not a synonym for quality.

**Predict:** Must the widest model have the highest held-out accuracy?
The paired bars and printed values decide. Because these held-out rows
guide this decision, call them validation despite the historical `X_test`
name. **Next complaint:** perhaps the important change is which physical
clues the model sees, not how wide the hidden layer is.""",
    "47a45ea4": """Restrict one model to bill length and bill depth, then let another use all
four measurements. Keep the split, seed, hidden width, optimizer, and update
budget fixed. The changed feature set changes the input layer's width too;
that is part of the declared comparison.

The columns keep their training-fitted normalization. We are not allowing
held-out birds to redefine the measuring ruler. Again, the held-out score
guides an ablation and is validation, not an independent final quality claim.

**Predict:** Must two extra measurements improve the score? Report the
observed difference without forcing a winner. Even a tie is useful evidence
for this small task and budget. **What remains:** deployment, checkpoint
recovery, mini-batch loading, and a genuinely untouched final test need
their own experiments outside this framework translation.""",
})

# Every replacement below targets a verified cell ID and an exact source fragment.
change("02-tensors-and-autograd", "03a85653",
       'token_ids = torch.tensor([[4, 1, 9, 2], [3, 8, 0, 7]], dtype=torch.long, device=device)',
       '''vocabulary = ["the", "cat", "sat", "on", "dog", "mat", "slept", "a", "bird", "flew"]
word_to_id = {word: index for index, word in enumerate(vocabulary)}
phrases = ["the cat sat on", "the dog sat on"]
token_ids = torch.tensor(
    [[word_to_id[word] for word in phrase.split()] for phrase in phrases],
    dtype=torch.long, device=device,
)
print("Readable rows:", phrases)''')
change("02-tensors-and-autograd", "494404ac",
       'print(f"Values match after explicit squeeze: {same_values}")',
       'print(f"Values match after explicit squeeze: {same_values}")\nassert same_values')
change("02-tensors-and-autograd", "a4c5ccc7",
       'print(f"Round-trip values preserved: {torch.equal(states, restored)}")',
       'print(f"Round-trip values preserved: {torch.equal(states, restored)}")\nassert torch.equal(states, restored)')
change("02-tensors-and-autograd", "2781b23f",
       'toy_loss = lm_logits.square().mean()',
       '# (2, 4, 10) -> scalar: average all squared scores only to audit gradient wiring.\ntoy_loss = lm_logits.square().mean()')
change("03-models-and-training-loops", "da4e19fb",
       'loss.backward()\ngradient_norm', 'first_module_loss = loss.item()\nloss.backward()\nassert torch.equal(model.linear.weight.detach(), before)\n# Weight gradient (1, 2) -> scalar norm: inspect feedback magnitude, not an update.\ngradient_norm')
change("03-models-and-training-loops", "da4e19fb",
       'update_norm =', '# Weight delta (1, 2) -> scalar norm: inspect how far SGD actually moved.\nupdate_norm =')
change("03-models-and-training-loops", "e5af3ac9",
       'module_history = []', 'module_history = [first_module_loss]')
change("03-models-and-training-loops", "e18475a6",
       'max_parameter_gap = max(', '# Parameter/prediction differences -> scalar maxima: test the worst coordinate.\nmax_parameter_gap = max(')
change("03-models-and-training-loops", "e18475a6",
       'assert max_parameter_gap < 1e-5 and max_prediction_gap < 1e-5',
       '''assert max_parameter_gap < 1e-5 and max_prediction_gap < 1e-5
history_gap = max(abs(raw - module) for raw, module in zip(raw_history, module_history))
print(f"Maximum pre-update history gap: {history_gap:.2e}")
assert len(raw_history) == len(module_history) == 180 and history_gap < 1e-5''')
change("03-models-and-training-loops", "ed248f0d",
       '[raw_history[0]] + module_history', 'module_history')
change("03-models-and-training-loops", "ed248f0d",
       'xlabel="update"', 'xlabel="Loss check before update"')
change("03-models-and-training-loops", "c0d63dc9",
       '''torch.manual_seed(23)
trial = FuelLinearModel().to(device)
trial_optimizer = torch.optim.SGD(trial.parameters(), lr=YOUR_LEARNING_RATE)
trial_losses = []
for _ in range(10):
    trial_optimizer.zero_grad()
    trial_loss = loss_function(trial(X), y)
    trial_loss.backward()
    trial_optimizer.step()
    trial_losses.append(trial_loss.item())
print(f"Ten-update loss at lr={YOUR_LEARNING_RATE}: {trial_losses[0]:.4f} -> {trial_losses[-1]:.4f}")
assert YOUR_LEARNING_RATE > 0''',
       '''assert YOUR_LEARNING_RATE > 0
trial_results = {}
for rate in (learning_rate, YOUR_LEARNING_RATE):
    trial = FuelLinearModel().to(device)
    with torch.no_grad():
        trial.linear.weight.zero_()
        trial.linear.bias.zero_()
    trial_optimizer = torch.optim.SGD(trial.parameters(), lr=rate)
    trial_losses = []
    for _ in range(10):
        trial_optimizer.zero_grad()
        trial_loss = loss_function(trial(X), y)
        trial_loss.backward()
        trial_optimizer.step()
        trial_losses.append(trial_loss.item())
    trial_results[rate] = trial_losses[-1]
    print(f"Ten-update pre-step loss at lr={rate}: {trial_losses[0]:.4f} -> {trial_losses[-1]:.4f}")
print("Lower final recorded error:", min(trial_results, key=trial_results.get))''')
change("04-data-checkpoints-and-hyperparameters", "e7b07933",
       'reloaded_model = FuelLinearModel()', 'assert config["input_features"] == 2 and config["output_features"] == 1\nreloaded_model = FuelLinearModel()')
change("04-data-checkpoints-and-hyperparameters", "82a96cd1",
       'print("The smaller final batch is retained deliberately.")',
       'print("The smaller final batch is retained deliberately.")\nassert torch.equal(covered, torch.arange(n_examples))')
change("04-data-checkpoints-and-hyperparameters", "fe59002f",
       'print(f"Last batch size: {len(list(ordered_loader)[-1][0])}")',
       'print(f"Last batch size: {len(list(ordered_loader)[-1][0])}")\nassert torch.equal(loaded_X, X) and torch.equal(loaded_y, y)\nassert len(list(ordered_loader)[-1][0]) == 16')
change("04-data-checkpoints-and-hyperparameters", "cf746eb2",
       '\nmax_reload_gap =', '\n# Probe differences (7, 1) -> scalar maximum: inspect the worst prediction gap.\nmax_reload_gap =')
change("05-modern-training-lifecycle", "a9b23399",
       'train_mean =', '# (108, 2) -> (2,): reduce training rows, keeping one center/scale per sensor.\ntrain_mean =')
change("05-modern-training-lifecycle", "a9b23399",
       'all_mean =', '# (180, 2) -> (2,) is a deliberately leaked ruler for comparison only.\nall_mean =')
change("05-modern-training-lifecycle", "a9b23399",
       'mean_gap =', '# Two feature-center gaps -> scalar maximum: expose any changed sensor ruler.\nmean_gap =')
change("05-modern-training-lifecycle", "0e46a309",
       'y_mean =', '# (108, 1) -> (1,): fit one fuel center/scale on training targets only.\ny_mean =')
change("05-modern-training-lifecycle", "0e46a309",
       'print(f"Train feature mean', '# (N, 2) -> (2,): inspect one normalized average per feature, not per row.\nprint(f"Train feature mean')
change("05-modern-training-lifecycle", "0a1e51a7",
       'print(f"Train-mode', '# Each (6, 1) repeat difference -> scalar maximum: test the worst repeated guess.\nprint(f"Train-mode')
change("05-modern-training-lifecycle", "aa634347",
       'print("Defined evaluate: eval mode plus inference_mode prevents learning.")',
       '''# Copy live state to prove evaluation neither updates weights nor writes new gradients.
weights_before_eval = [p.detach().clone() for p in model.parameters()]
grads_before_eval = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
evaluation_probe_loss = evaluate(model, val_loader)
assert all(torch.equal(before, after) for before, after in zip(weights_before_eval, model.parameters()))
assert all(
    (before is None and after.grad is None)
    or (before is not None and after.grad is not None and torch.equal(before, after.grad))
    for before, after in zip(grads_before_eval, model.parameters())
)
print(f"Validation probe loss: {evaluation_probe_loss:.4f}; weights and gradient buffers unchanged.")''')
change("05-modern-training-lifecycle", "a94f2519",
       'print("The plots contain actual lifecycle measurements, not illustrative curves.")',
       '''rate_reductions = sum(
    later < earlier
    for earlier, later in zip(history["learning_rate"], history["learning_rate"][1:])
)
print(f"Recorded learning-rate reductions: {rate_reductions}")
print("These are measured curves; a flat rate trace means no reduction occurred.")''')
change("05-modern-training-lifecycle", "45921844",
       'test_prediction_n = best_model(X_test.to(device)).cpu()',
       '''# Raw (36, 2) with saved (2,) statistics -> (36, 2): reconstruct each sensor ruler.
    saved_X_test = (
        X_test_raw.to(device) - best_checkpoint["train_mean"]
    ) / best_checkpoint["train_std"]
    test_prediction_n = best_model(saved_X_test).cpu()''')
change("05-modern-training-lifecycle", "45921844",
       'test_prediction = test_prediction_n * best_checkpoint["target_std"] + best_checkpoint["target_mean"]',
       'test_prediction = test_prediction_n * best_checkpoint["target_std"].cpu() + best_checkpoint["target_mean"].cpu()')
change("05-modern-training-lifecycle", "45921844",
       'test_mae =', '# Absolute fuel misses (36, 1) -> scalar mean: report original-unit MAE.\ntest_mae =')
change("05-modern-training-lifecycle", "acf4941a",
       'simulated = EarlyStopping', 'assert YOUR_PATIENCE > 0\nsimulated = EarlyStopping')
change("05-modern-training-lifecycle", "acf4941a",
       'assert stop_at is not None', 'expected_stop = 2 + YOUR_PATIENCE if 2 + YOUR_PATIENCE <= len(sequence) else None\nassert stop_at == expected_stop')
change("pytorch-training-skeleton", "00r-09",
       '\naccuracy =', '\n# Eight correctness flags -> scalar mean: this measures fit on practice rows only.\naccuracy =')
change("01-pytorch-cnns", "12ebf280",
       'plt.rcParams.update({"figure.facecolor": "#1a1a2e", "axes.facecolor": "#1a1a2e"})',
       'plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})')
change("01-pytorch-cnns", "12ebf280",
       'print(f"Training tensor: {tuple(train_images.shape)}")',
       '''print(f"Training tensor: {tuple(train_images.shape)}")
# (80,) correctness flags -> scalar: an always-vertical guess ignores image evidence.
baseline_accuracy = (valid_labels == 0).float().mean().item()
print(f"Always-vertical baseline: {baseline_accuracy:.1%}")''')
change("01-pytorch-cnns", "f47b7316",
       'manual_pixel =', '# Products (1 channel, 3, 3) -> scalar: combine all local pixel contributions.\nmanual_pixel =')
change("01-pytorch-cnns", "b3e03744",
       'print(f"Vertical-shape', '# Each (1, 1, 14, 14) response -> scalar sum: total magnitude, not class probability.\nprint(f"Vertical-shape')
change("01-pytorch-cnns", "0893f8c6",
       'print("Takeaway: both shrink space; pooling summarizes an existing map.")',
       'assert stride_two.shape == pooled.shape == (1, 4, 8, 8)\nprint("Takeaway: both shrink space; pooling summarizes an existing map.")')
change("01-pytorch-cnns", "b75c3cc9",
       'losses = []', '''with torch.inference_mode():
    # (80, 2) -> (80,) class IDs -> scalar mean: measure the untrained decision baseline.
    initial_accuracy = (model(valid_images).argmax(dim=-1) == valid_labels).float().mean().item()
print(f"Untrained validation accuracy: {initial_accuracy:.1%}")
losses = []''')
change("01-pytorch-cnns", "b75c3cc9",
       '\n    accuracy = (model', '\n    # (80, 2) -> (80,) IDs -> scalar mean: one orientation decision per validation image.\n    accuracy = (model')
change("01-pytorch-cnns", "b75c3cc9",
       'print(f"Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")',
       'print(f"Final-batch loss by epoch: {losses[0]:.4f} -> {losses[-1]:.4f}")')
change("01-pytorch-cnns", "85e1c830",
       'first_maps, second_maps = model.features(valid_images[:1])',
       'with torch.inference_mode():\n    first_maps, second_maps = model.features(valid_images[:1])')
change("01-pytorch-cnns", "498efaf9",
       'strongest_channel =', '# Ten channel magnitudes -> scalar index: choose one informative center response.\nstrongest_channel =')
change("01-pytorch-cnns", "498efaf9",
       'height =', '# Active row/column coordinate vectors -> scalar extrema: bound this local influence mask.\nheight =')
change("01-pytorch-cnns", "e6799796",
       '''for plain, residual in zip(plain_layers, residual_layers):
    residual.weight.data.copy_(plain.weight.data)
    plain.weight.data.mul_(0.25)
    residual.weight.data.mul_(0.25)''',
       '''with torch.no_grad():
    for plain, residual in zip(plain_layers, residual_layers):
        residual.weight.copy_(plain.weight)
        plain.weight.mul_(0.25)
        residual.weight.mul_(0.25)''')
change("01-pytorch-cnns", "e6799796",
       'plain.sum().backward()', '# Each (1, 12) final feature vector -> scalar sum: use the same backward probe.\nplain.sum().backward()')
change("01-pytorch-cnns", "e6799796",
       'plain_norm =', '# Each input gradient (1, 12) -> scalar norm: compare route sensitivity magnitudes.\nplain_norm =')
change("02-pytorch-autoencoders", "b5c22c96",
       'plt.rcParams.update({"figure.facecolor": "#1a1a2e", "axes.facecolor": "#1a1a2e"})',
       'plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})')
change("02-pytorch-autoencoders", "f021f820",
       'optimizer =', '''# (80, 1, 16, 16) pixel errors -> scalar mean: can a blank reconstruction already look good?
black_baseline_mse = nn.functional.mse_loss(torch.zeros_like(valid_images), valid_images).item()
print(f"All-black validation baseline MSE: {black_baseline_mse:.6f}")
optimizer =''')
change("02-pytorch-autoencoders", "f021f820",
       'history.append(np.mean(epoch_losses))',
       '# Ten equally sized batch means -> scalar epoch mean; no remainder bias here.\n    history.append(np.mean(epoch_losses))')
change("02-pytorch-autoencoders", "f021f820",
       'assert history[-1] < history[0]',
       'print("Beats all-black baseline:", valid_mse < black_baseline_mse)\nassert history[-1] < history[0]')
change("02-pytorch-autoencoders", "c6ec0022",
       'plt.show()', '''plt.show()
# Pixel misses (80, 1, 16, 16) -> (80,): average within each image, not across examples.
per_image_mse = (valid_reconstruction - valid_images).square().mean(dim=(1, 2, 3))
print("First eight reconstruction MSEs:", per_image_mse[:8].tolist())''')
change("02-pytorch-autoencoders", "acc02dd3",
       'class_centers.append(flattened_latent[valid_labels == label].mean(dim=0))',
       '# (20 audit images, 64 coordinates) -> (64,): average examples, preserve code coordinates.\n    class_centers.append(flattened_latent[valid_labels == label].mean(dim=0))')
change("02-pytorch-autoencoders", "acc02dd3",
       'input_values =', '# Non-batch shape tuples -> scalar products: count values for ONE image/code.\ninput_values =')
change("02-pytorch-autoencoders", "8929bbd1",
       'return (\n        nn.functional.mse_loss', '# Reconstruction errors -> scalar MSE; latent shape product -> values per image.\n    return (\n        nn.functional.mse_loss')
change("01-keras-to-pytorch-antarctic-field-guide", "25e20206",
       '''import subprocess
import sys''', 'import sys')
change("01-keras-to-pytorch-antarctic-field-guide", "25e20206",
       'subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])',
       'raise ModuleNotFoundError(f"Missing {package}. Run this folder\\\'s setup in the intended environment.")')
change("01-keras-to-pytorch-antarctic-field-guide", "25e20206",
       '#  Setup — install only packages absent from the active kernel',
       '# Setup: verify dependencies; do not install into a possibly shared notebook kernel.')
change("01-keras-to-pytorch-antarctic-field-guide", "25e20206",
       "# Only install packages that aren't already importable in this kernel",
       '# Fail explicitly if a required package is missing; setup owns installation.')
change("01-keras-to-pytorch-antarctic-field-guide", "25e3b7ab",
       "# Aggregate the L2 norm across every parameter's gradient into one scalar",
       '# Each parameter gradient -> squared scalar sum; sum parameters and take sqrt for one L2 norm.')
change("01-keras-to-pytorch-antarctic-field-guide", "80bbfc67",
       'total_grad = torch.sqrt', '# Parameter-gradient entries -> scalar squared sums -> global scalar L2 norm.\n        total_grad = torch.sqrt')
change("01-keras-to-pytorch-antarctic-field-guide", "80b5bf56",
       'test_accuracy =', '# Held-out correctness flags (N,) -> scalar mean: average birds, not species columns.\ntest_accuracy =')
change("01-keras-to-pytorch-antarctic-field-guide", "2a7ad130",
       'accuracy = (candidate', '# (N birds, 3 scores) -> (N,) IDs -> scalar correctness mean.\n        accuracy = (candidate')
change("01-keras-to-pytorch-antarctic-field-guide", "47a45ea4",
       'return (candidate', '# (N birds, 3 scores) -> (N,) IDs -> scalar mean accuracy for this feature set.\n        return (candidate')

DIAGRAMS = {
    "02-tensors-and-autograd": [
        "02-word-addresses-and-axes.png", "02-broadcast-by-role.png", "02-word-addresses-and-axes.png",
    ],
    "03-models-and-training-loops": ["03-parameters-need-a-home.png", "existing-loop"],
    "04-data-checkpoints-and-hyperparameters": ["04-deliver-every-row.png", "04-snapshot-is-not-recovery.png"],
    "05-modern-training-lifecycle": ["05-learning-and-exam-rooms.png", "05-two-checkpoint-jobs.png"],
    "01-pytorch-cnns": ["vision-sliding-detector.png", "vision-shape-cnn.png", "vision-residual-route.png"],
    "02-pytorch-autoencoders": ["vision-rebuild-through-a-code.png", "vision-code-budget.png"],
}

SCOPE = {
    "02-tensors-and-autograd": ("index, regroup, move, broadcast, project, and audit gradients",
                              "word-shaped axes without a word-learning claim",
                              "attention and token preparation belong to later language chapters"),
    "03-models-and-training-loops": ("raw/module parity, explicit steps, and saved weights",
                                   "parameter ownership versus update policy",
                                   "batch delivery and validation are the next two chapters"),
    "04-data-checkpoints-and-hyperparameters": ("batch coverage, weight reload, snapshot writing, and rate comparisons",
                                             "why process state differs from inference state",
                                             "interrupted recovery needs a separate same-next-update comparison"),
    "05-modern-training-lifecycle": ("split integrity, modes, validation, selection, and inference reload",
                                   "CUDA AMP boundary and incomplete snapshot state",
                                   "Lightning/compile translations are not executed; distributed training needs another lab"),
    "pytorch-training-skeleton": ("one paired dataset, module, update loop, and tiny training fit",
                                 "how the same loop maps to word objectives",
                                 "validation and recovery are absent; use Parts 04–05 for their contracts"),
    "01-pytorch-cnns": ("local convolution, pooling, learned maps, influence, and shortcut gradients",
                       "shared spatial computation and its limits",
                       "pretrained backbones and detection need real image tasks, not these local bars"),
    "02-pytorch-autoencoders": ("pixel reconstruction, code size, group-center inspection, and width trials",
                               "value-count compression is not file compression or named concepts",
                               "a VAE needs a distributional objective and sampling tests, not just an encoder"),
    "01-keras-to-pytorch-antarctic-field-guide": ("typed tensors, storage parity, gradients, species decisions, and ablations",
                                               "Keras references and validation reuse",
                                               "TensorFlow execution and deployment are separate jobs; mini-batches/checkpoints stay in the core route"),
}


def set_source(cell, source):
    cell["source"] = source.splitlines(keepends=True)


def replace_source(cells, cell_id, old, new):
    cell = next(c for c in cells if c["id"] == cell_id)
    source = "".join(cell["source"])
    assert source.count(old) == 1, (cell_id, old, source.count(old))
    set_source(cell, source.replace(old, new, 1))


def author():
    inventory = json.loads((SESSION / "baseline.json").read_text())
    prompts = json.loads((ROOT / "images" / "pytorch-illustration-prompts.json").read_text())
    by_filename = {p["filename"]: p for p in prompts}
    for item in inventory:
        path = ROOT / item["path"]
        stem = path.stem
        if stem == "01-deep-learning-map":
            assert path.read_bytes() == (SESSION / f"{stem}.backup").read_bytes()
            continue
        book = json.loads((SESSION / f"{stem}.backup").read_bytes())
        cells = book["cells"]
        code_ids = {c["id"] for c in cells if c["cell_type"] == "code"}
        assert code_ids == set(PROSE[stem]), (stem, code_ids.symmetric_difference(PROSE[stem]))
        for cell_id, old, new in EDITS.get(stem, []):
            replace_source(cells, cell_id, old, new)
        for cell in cells:
            if cell["cell_type"] == "code":
                continue
            source = "".join(cell["source"])
            if "```mermaid" in source:
                filename = DIAGRAMS[stem].pop(0)
                if filename == "existing-loop":
                    image = ROOT / "optional" / "framework-translation" / "images" / "3.png"
                    alt = "Clear gradients, predict and measure, backpropagate, then update; skipping clearing accumulates feedback."
                else:
                    image = ROOT / "images" / filename
                    alt = by_filename[filename]["alt_text"]
                assert image.is_file(), image
                import os
                link = Path(os.path.relpath(image, path.parent)).as_posix()
                source, count = re.subn(r"```mermaid\n.*?```", f"![{alt}]({link})", source, flags=re.S)
                assert count == 1
                set_source(cell, source)
        if stem in {"02-tensors-and-autograd", "01-pytorch-cnns", "02-pytorch-autoencoders",
                    "01-keras-to-pytorch-antarctic-field-guide"}:
            # Move the existing title first without deleting or replacing any raw cell identifier.
            cells[0], cells[1] = cells[1], cells[0]
        title_cell = cells[0]
        built, explained, deferred = SCOPE[stem]
        scope = f"""

### What we will and will not prove

| Coverage | Boundary |
|---|---|
| Built and measured | {built} |
| Explained | {explained} |
| Named, not executed | {deferred} |

[Handwritten theory]({stem}-theory.md) keeps the same complaint chain without code.
"""
        set_source(title_cell, "".join(title_cell["source"]).rstrip() + scope)
        if stem == "01-keras-to-pytorch-antarctic-field-guide":
            replace_source(cells, "c55a8c8d", "images/4.png", "../../images/penguin-framework-contracts.png")
            replace_source(cells, "c55a8c8d",
                           "Seven-stage Keras-to-PyTorch workflow from Palmer Penguins data through typed tensors, model construction, logits and loss, explicit training, evaluation, and controlled ablations.",
                           by_filename["penguin-framework-contracts.png"]["alt_text"])
        if stem == "02-tensors-and-autograd":
            # Replace the old up-front glossary with the domain bridge; mechanisms are explained at point of use.
            set_source(next(c for c in cells if c["id"] == "557655c3"),
                       """## Why borrow words for an axis lesson?

The sensor correction had examples and features. “The cat sat on” has positions
inside each example too. That extra role makes a lost or swapped axis visible.
The second phrase changes only cat to dog, so our nouns stay familiar.

IDs are lookup addresses, not little measurements of meaning. The six-feature
rows and ten-score projections below are randomly initialized. We will check
their shapes and gradient routes, not pretend they learned a language.
Part 03 returns to two sensor columns so parameter ownership can be the only
new problem. Later tokenization and word-objective chapters make this shape
preview into a meaningful text task.
""")
        if stem == "pytorch-training-skeleton":
            target = next(c for c in cells if c["id"] == "a1893f11")
            source = "".join(target["source"])
            source = source.replace("The current full route is Parts 01–05; older 00–04 labels above describe the earlier numbering.", "The full route remains Parts 01–05.")
            set_source(target, source)
            intro = next(c for c in cells if c["id"] == "00r-02")
            set_source(intro, "".join(intro["source"]) + """

![The explicit loop clears, predicts and measures, backpropagates, then updates.](../optional/framework-translation/images/3.png)
*A reusable static memory card; the next experiments give these jobs a tiny
two-sensor calm/alert task, not a real alarm system.*
""")
        if stem == "01-pytorch-cnns":
            replace_source(cells, "e1adc879",
                           "Both can; their evidence is different.",
                           "Both can shrink the map. The convolution uses filter weights; max-pooling adds no learned weights.")
        if stem == "02-pytorch-autoencoders":
            replace_source(cells, "ba5b08c8",
                           "how small can a visual code become before reconstruction evidence visibly fails?",
                           "what survives a smaller visual code, and how do we check the rebuilding?")
        # Locate artifact-producing lessons reliably, including VS Code root-started kernels.
        first_code = next(c for c in cells if c["cell_type"] == "code")
        setup = """import sys
print(f"Kernel interpreter: {sys.executable}")
"""
        if stem in {"03-models-and-training-loops", "04-data-checkpoints-and-hyperparameters",
                    "05-modern-training-lifecycle"}:
            relative = path.parent.relative_to(ROOT.parent.parent)
            parts = ", ".join(repr(p) for p in relative.parts)
            setup += f"""from pathlib import Path
LESSON_DIR = next(
    folder for folder in (
        Path(globals().get("__vsc_ipynb_file__", "")).resolve().parent,
        Path.cwd(),
        Path.cwd().joinpath({parts}),
    )
    if (folder / "{path.name}").is_file()
)
"""
        set_source(first_code, setup + "\n" + "".join(first_code["source"]))
        if stem == "03-models-and-training-loops":
            replace_source(cells, "a15035ed", 'artifact_dir = Path("artifacts")', 'artifact_dir = LESSON_DIR / "artifacts"')
        elif stem == "04-data-checkpoints-and-hyperparameters":
            replace_source(cells, "b855a304", 'Path("../03-models-and-training-loops/artifacts")',
                           'LESSON_DIR.parent / "03-models-and-training-loops" / "artifacts"')
            replace_source(cells, "9eba2ce4", 'artifact_dir = Path("artifacts")', 'artifact_dir = LESSON_DIR / "artifacts"')
        elif stem == "05-modern-training-lifecycle":
            replace_source(cells, "316964cf", 'artifact_dir = Path("artifacts")', 'artifact_dir = LESSON_DIR / "artifacts"')
        assembled = []
        for cell in cells:
            if cell["cell_type"] == "code":
                prose = PROSE[stem][cell["id"]]
                if assembled[-1]["cell_type"] == "markdown":
                    set_source(assembled[-1], "".join(assembled[-1]["source"]).rstrip() + "\n\n" + prose + "\n")
                else:
                    assembled.append({
                        "cell_type": "markdown",
                        "id": f"track26-before-{cell['id']}",
                        "metadata": {},
                        "source": (prose + "\n").splitlines(keepends=True),
                    })
            assembled.append(cell)
        book["cells"] = assembled
        style = item["serialization"]
        text = json.dumps(book, ensure_ascii=style["ensure_ascii"], indent=style["indent"]) + style["newline"]
        path.write_bytes(text.replace("\n", style["eol"]).encode())
        persisted = json.loads(path.read_bytes())
        assert persisted == book
        assert all(c["id"] in {x["id"] for x in assembled} for c in cells)
        print(f"Authored {path.relative_to(ROOT)}: {len(assembled)} cells, {len(code_ids)} code experiments")


def baseline():
    SESSION.mkdir(parents=True, exist_ok=True)
    inventory = []
    for path in sorted(ROOT.rglob("*.ipynb")):
        raw = path.read_bytes()
        book = json.loads(raw)
        styles = [
            {"ensure_ascii": ascii_only, "indent": indent, "newline": newline, "eol": eol}
            for ascii_only in (False, True)
            for indent in (1, 2)
            for newline in ("\n", "")
            for eol in ("\n", "\r\n")
        ]
        style = next(
            (s for s in styles if (
                json.dumps(book, ensure_ascii=s["ensure_ascii"], indent=s["indent"])
                + s["newline"]
            ).replace("\n", s["eol"]).encode() == raw),
            None,
        )
        assert style is not None, f"Round-trip mismatch: {path}"
        stem = path.stem
        (SESSION / f"{stem}.backup").write_bytes(raw)
        text = "\n\n".join(
            f"--- {i} {cell['cell_type']} {cell['id']} ---\n{''.join(cell['source'])}"
            for i, cell in enumerate(book["cells"])
        )
        (SESSION / f"{stem}-source.txt").write_text(text, encoding="utf-8")
        inventory.append({
            "path": str(path.relative_to(ROOT)),
            "cells": len(book["cells"]),
            "code": sum(c["cell_type"] == "code" for c in book["cells"]),
            "mermaid": sum("```mermaid" in "".join(c["source"]) for c in book["cells"]),
            "outputs": sum(bool(c.get("outputs")) for c in book["cells"]),
            "serialization": style,
        })
    (SESSION / "baseline.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "author":
        author()
    else:
        baseline()

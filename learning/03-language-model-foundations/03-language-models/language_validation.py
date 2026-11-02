"""Read-only validation and before/after reporting for word-prediction objectives."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol, TypedDict

import torch
from torch import nn

from language_data import CBOWDataLoader, SkipGramDataLoader, prepare_sequence_data


class WordProbeResult(TypedDict):
    # A typed mapping records one expected word's probability, rank, and candidate shortlist.
    probability: float
    rank: int
    top5: list[tuple[str, float]]


def word_probe(
    logits: torch.Tensor, expected_word: str, words: Sequence[str],
) -> WordProbeResult:
    """Keep probability, rank, and the actual shortlist; never force the answer."""
    if logits.ndim != 1 or logits.numel() != len(words):
        raise ValueError("A word probe needs one vocabulary score per supplied word.")
    chances = logits.detach().softmax(dim=-1)
    expected_id = words.index(expected_word)
    rank = int((chances > chances[expected_id]).sum()) + 1
    top = chances.topk(5)
    return {
        "probability": chances[expected_id].item(),
        "rank": rank,
        "top5": [(words[index], round(chance, 4))
                 for index, chance in zip(top.indices.tolist(), top.values.tolist())],
    }


def report_probe(label: str, before: WordProbeResult, after: WordProbeResult) -> None:
    print(f"{label}: correct-word share {before['probability']:.3%} -> "
          f"{after['probability']:.3%}; rank {before['rank']} -> {after['rank']}")
    print("  Before top 5:", before["top5"])
    print("  After top 5: ", after["top5"])
    probability_change = "increased" if after["probability"] > before["probability"] else "decreased"
    rank_change = "improved" if after["rank"] < before["rank"] else (
        "worsened" if after["rank"] > before["rank"] else "stayed the same")
    print(f"  Probability {probability_change}; rank {rank_change}. "
          + ("The expected word is top-ranked." if after["rank"] == 1 else "Not the first guess."))


class ValidationModel(Protocol):
    # A structural protocol accepts compatible models defined inline or imported from modules.
    def __call__(
        self, input_ids: torch.Tensor, target_ids: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def targets_for_loss(self, target_ids: torch.Tensor) -> torch.Tensor: ...

    def modules(self) -> Iterator[nn.Module]: ...

    def eval(self) -> object: ...


class ProbeModel(ValidationModel, Protocol):
    # Extends the validation contract with the embedding table needed for before/after snapshots.
    embedding: nn.Embedding


@dataclass(frozen=True)
class ModelProbeSnapshot:
    # A snapshot keeps detached embedding values, probe inputs/logits, and named prediction readouts.
    embeddings: torch.Tensor
    input_ids: torch.Tensor
    logits: torch.Tensor
    probes: dict[str, WordProbeResult]


@dataclass(frozen=True)
class ModelProbeComparison:
    # A before/after pair carries the measured row changes without owning a training loop.
    before: ModelProbeSnapshot
    after: ModelProbeSnapshot
    row_changes: torch.Tensor


class ModelProbeRecorder:
    # Records and reports fixed probe questions; training and per-epoch callbacks remain separate.
    def __init__(
        self,
        model: ProbeModel,
        input_ids: torch.Tensor,
        *,
        words: Sequence[str],
        expected_words: Sequence[str],
        target_ids: torch.Tensor | None = None,
        positions: Sequence[tuple[int, ...]] | None = None,
        name: str = "Model",
        labels: Sequence[str] | None = None,
    ):
        if not expected_words or len(set(expected_words)) != len(expected_words):
            raise ValueError("Named probes need distinct expected words.")
        if labels is not None and len(labels) != len(expected_words):
            raise ValueError("Each named probe needs one reporting label.")
        self.model = model
        self.input_ids = input_ids.detach().clone()
        self.target_ids = target_ids.detach().clone() if target_ids is not None else None
        self.words = tuple(words)
        self.expected_words = tuple(expected_words)
        self.positions = tuple(positions) if positions is not None else tuple(
            (row,) for row in range(len(expected_words))
        )
        if len(self.positions) != len(self.expected_words):
            raise ValueError("Each named probe needs one score position.")
        self.labels = tuple(labels) if labels is not None else tuple(
            f"{name} missing {word!r}" for word in self.expected_words
        )
        self.before: ModelProbeSnapshot | None = None
        self.comparison: ModelProbeComparison | None = None

    def _snapshot(self) -> ModelProbeSnapshot:
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            self.model.eval()
            with torch.inference_mode():
                logits = self.model(self.input_ids, self.target_ids).detach().clone()
                embeddings = self.model.embedding.weight.detach().clone()
                probes = {
                    word: word_probe(logits[position], word, self.words)
                    for word, position in zip(self.expected_words, self.positions)
                }
        finally:
            for module, training in modes:
                module.training = training
        return ModelProbeSnapshot(embeddings, self.input_ids.clone(), logits, probes)

    def record_before(self, *, show: bool = False) -> ModelProbeSnapshot:
        self.before = self._snapshot()
        self.comparison = None
        if show:
            for word, position in zip(self.expected_words, self.positions):
                print("Clues:", [self.words[index] for index in self.input_ids[position[0]].reshape(-1).tolist()],
                      "-> expected:", word)
                print("Before training:", self.before.probes[word])
        return self.before

    def record_after(self) -> ModelProbeComparison:
        if self.before is None:
            raise RuntimeError("Record the before-training probe snapshot first.")
        after = self._snapshot()
        row_changes = (after.embeddings - self.before.embeddings).norm(dim=1)
        self.comparison = ModelProbeComparison(self.before, after, row_changes)
        return self.comparison

    def report_probes(self) -> None:
        if self.comparison is None:
            raise RuntimeError("Record the after-training snapshot before reporting probes.")
        for word, label in zip(self.expected_words, self.labels):
            report_probe(label, self.comparison.before.probes[word], self.comparison.after.probes[word])

    def report_embeddings(
        self, feature_words: Sequence[str], *, features: int = 4, include_row_change: bool = False,
    ) -> None:
        if self.comparison is None:
            raise RuntimeError("Record the after-training snapshot before reporting embeddings.")
        if features < 1 or features > self.comparison.after.embeddings.shape[1]:
            raise ValueError("Feature count must fit the recorded embedding width.")
        if not include_row_change:
            print("Embedding rows changed:", (self.comparison.row_changes > 0).sum().item())
        count = {3: "three", 4: "four"}.get(features, str(features))
        for word in feature_words:
            index = self.words.index(word)
            print(f"{word} (fixed ID {index}), first {count} "
                  f"{'shared features' if include_row_change else 'features'}: "
                  f"{self.comparison.before.embeddings[index, :features].tolist()} -> "
                  f"{self.comparison.after.embeddings[index, :features].tolist()}"
                  + (f"; row change {self.comparison.row_changes[index].item():.4f}"
                     if include_row_change else ""))


@dataclass(frozen=True)
class ValidationResult:
    # An immutable record stores one evaluation's scores, answer count, and example predictions.
    loss: float
    accuracy: float
    checked_items: int
    example_predictions: tuple[int, ...]
    example_answers: tuple[int, ...]


@dataclass(frozen=True)
class ValidationComparison:
    # An immutable record pairs before/after evaluations with the actual training epoch count.
    before: ValidationResult
    after: ValidationResult
    epochs: int


class ModelValidator:
    # The shared validation base evaluates fixed questions without updating model parameters.
    def __init__(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        words: Sequence[str] | None = None,
        name: str = "Language model",
        description: str = "Inputs are checked against their aligned answers.",
        batch_size: int = 512,
    ):
        if input_ids.ndim < 1 or target_ids.ndim < 1:
            raise ValueError("Validation inputs and answers need an example dimension.")
        if len(input_ids) == 0 or len(input_ids) != len(target_ids):
            raise ValueError("Validation needs the same nonzero number of inputs and answers.")
        if input_ids.dtype != torch.long or target_ids.dtype != torch.long:
            raise TypeError("Validation inputs and answers must use torch.long IDs.")
        if batch_size < 1:
            raise ValueError("Validation batch size must be positive.")
        self.input_ids = input_ids.detach().clone()
        self.target_ids = target_ids.detach().clone()
        self.words = tuple(words) if words is not None else None
        self.name = name
        self.description = description
        self.batch_size = batch_size
        self.before: ValidationResult | None = None

    def evaluate(self, model: ValidationModel) -> ValidationResult:
        modes = [(module, module.training) for module in model.modules()]
        total_loss, correct, checked = 0.0, 0, 0
        example_predictions: tuple[int, ...] = ()
        example_answers: tuple[int, ...] = ()
        try:
            model.eval()
            with torch.inference_mode():
                for start in range(0, len(self.input_ids), self.batch_size):
                    inputs = self.input_ids[start:start + self.batch_size]
                    targets = self.target_ids[start:start + self.batch_size]
                    logits = model(inputs, targets)
                    expected = model.targets_for_loss(targets)
                    if logits.ndim < 2 or logits.shape[:-1] != expected.shape:
                        raise ValueError("Validation scores must align with every checking answer.")
                    total_loss += nn.functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), expected.reshape(-1),
                        reduction="sum",
                    ).item()
                    predicted = logits.argmax(dim=-1)
                    correct += int((predicted == expected).sum().item())
                    checked += expected.numel()
                    if start == 0:
                        example_predictions = tuple(predicted[0].reshape(-1).tolist())
                        example_answers = tuple(expected[0].reshape(-1).tolist())
        finally:
            for module, training in modes:
                module.training = training
        if checked == 0:
            raise ValueError("Validation needs at least one checking answer.")
        return ValidationResult(
            total_loss / checked, correct / checked, checked,
            example_predictions, example_answers,
        )

    def record_before(self, model: ValidationModel) -> ValidationResult:
        self.before = self.evaluate(model)
        return self.before

    def _describe_ids(self, ids: Sequence[int]) -> str:
        if self.words is None:
            return " ".join(str(index) for index in ids)
        return " ".join(self.words[index] for index in ids)

    def compare_after(self, model: ValidationModel, *, epochs: int) -> ValidationComparison:
        if self.before is None:
            raise RuntimeError("Record validation before training before requesting a comparison.")
        if epochs < 1:
            raise ValueError("A training comparison needs a positive epoch count.")
        after = self.evaluate(model)
        print(f"{self.name}: validation on the same reserved questions")
        print(self.description)
        print("Stage | loss | answer accuracy | checked answers")
        print(f"Before this run | {self.before.loss:.4f} | "
              f"{self.before.accuracy:.1%} | {self.before.checked_items}")
        print(f"After {epochs} epochs | {after.loss:.4f} | "
              f"{after.accuracy:.1%} | {after.checked_items}")
        print("Example input:", self._describe_ids(self.input_ids[0].reshape(-1).tolist()))
        print("Checking answers:", self._describe_ids(after.example_answers))
        print("Before prediction:", self._describe_ids(self.before.example_predictions))
        print("After prediction:", self._describe_ids(after.example_predictions))
        print("Validation loss improved." if after.loss < self.before.loss else
              "Validation loss did not improve; training progress alone is not enough.")
        return ValidationComparison(self.before, after, epochs)


class CBOWValidator(ModelValidator):
    # Inherits read-only evaluation and reporting; supplies CBOW center-word checking pairs.
    def __init__(self, windows: torch.Tensor, *, words: Sequence[str], batch_size: int = 512):
        super().__init__(
            *CBOWDataLoader(windows).prepare_data(), words=words, name="CBOW",
            description="Four clue words predict one missing center; the answer is not an input.",
            batch_size=batch_size,
        )


class SkipGramValidator(ModelValidator):
    # Inherits read-only evaluation and reporting; supplies individual neighbour-checking pairs.
    def __init__(self, windows: torch.Tensor, *, words: Sequence[str], batch_size: int = 512):
        super().__init__(
            *SkipGramDataLoader(windows).prepare_data(), words=words, name="Skip-gram",
            description="One center predicts one observed neighbour per pair, not a position.",
            batch_size=batch_size,
        )


class EncoderDecoderValidator(ModelValidator):
    # Inherits read-only evaluation and reporting; supplies teacher-forced next-token answers.
    def __init__(self, windows: torch.Tensor, *, words: Sequence[str], batch_size: int = 512):
        if "<bos>" not in words or "<eos>" not in words:
            raise ValueError("Reader/writer validation needs start and stop markers.")
        super().__init__(
            *prepare_sequence_data(windows, words.index("<bos>"), words.index("<eos>")),
            words=words, name="Encoder-decoder",
            description="Teacher-forced next-token validation, including stop. "
                        "Correct previous words are supplied; this is not free-running generation.",
            batch_size=batch_size,
        )

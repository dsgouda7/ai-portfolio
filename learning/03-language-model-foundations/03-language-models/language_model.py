"""Shared prediction contracts, word layers, and optimizer training order."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
import time

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


@dataclass
class TrainingRun:
    # A plain result record keeps measured losses and time separate from the model's weights.
    losses: list[float]
    seconds: float


class TrainableLanguageModel(nn.Module, ABC):
    # Inherits PyTorch parameters/modes and requires forward; the base owns training-step order.
    """Share target alignment and optimization across language objectives."""

    @abstractmethod
    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError("A language model must define its prediction route.")

    def targets_for_loss(self, target_ids):
        # Which answers should the guesses be marked against?
        # CBOW and Skip-gram: the loader's answers already line up one-to-one
        # with the guesses, so hand them back unchanged.
        # A subclass overrides this when guesses and answers are offset
        # (the encoder-decoder drops its <bos> start marker).
        return target_ids

    def training_loss(self, input_ids, target_ids, loss_function):
        # 1. Make the guesses. self(...) runs the subclass's forward().
        #    target_ids rides along because the encoder-decoder needs it for
        #    teacher forcing; CBOW and Skip-gram simply ignore it.
        #      CBOW / Skip-gram scores: [B, V]     one guess per lesson
        #      Encoder-decoder scores:  [B, T, V]  one guess per output step
        logits = self(input_ids, target_ids)

        # 2. Pick the answers those guesses should be checked against.
        expected_ids = self.targets_for_loss(target_ids)

        # 3. Calculate the flat versions of logits and targets for the loss function.
        # logits.reshape(-1, logits.shape[-1]) flattens the batch and time dimensions
        # into one dimension, keeping the vocabulary dimension intact. This is needed
        # for the loss function which expects [N, V] for logits and [N] for targets.
        # for example, if logits is [2, 3, 5] (B=2, T=3, V=5), flat_logits
        # will be [6, 5] and flat_targets will be [6]
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = expected_ids.reshape(-1)

        # 4. One number: on average, how surprised was the model by the right answers?
        return loss_function(flat_logits, flat_targets)

    def configure_loss_function(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def configure_optimizer(self, learning_rate: float) -> Optimizer:
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def train_model(
        self,
        dataloader: DataLoader,
        epochs: int,
        learning_rate: float = 0.008,
        *,
        on_epoch_end: Callable[[int, float], None] | None = None,
    ) -> TrainingRun:
        if epochs < 1:
            raise ValueError("Training requires at least one epoch.")
        loss_function = self.configure_loss_function()
        optimizer = self.configure_optimizer(learning_rate)
        started = time.perf_counter()
        losses = self.fit_language_model(
            dataloader, loss_function, optimizer, epochs, on_epoch_end=on_epoch_end,
        )
        return TrainingRun(losses, time.perf_counter() - started)

    def fit_language_model(
        self, dataloader, loss_function, optimizer, epochs,
        *, on_epoch_end: Callable[[int, float], None] | None = None,
    ) -> list[float]:
        if epochs < 1:
            raise ValueError("Training requires at least one epoch.")
        history = []

        for epoch in range(1, epochs + 1):
            # set the model to training mode
            self.train()
            total_loss = 0.0
            example_count = 0

            for input_batch, target_batch in dataloader:
                # initialize the gradients for this batch
                optimizer.zero_grad()
                # compute the loss for this batch
                loss = self.training_loss(input_batch, target_batch, loss_function)
                # compute the gradients for this batch
                loss.backward()
                # update the model parameters based on the gradients
                # (subtract the gradients scaled by the learning rate)
                optimizer.step()

                # loss is the AVERAGE over this batch. Times the batch size turns it
                # back into a batch TOTAL, so a short last batch (say 2 lessons
                # instead of 5) counts for 2 lessons, not as much as a full batch.
                # this math allows us to accumulate the total loss correctly even if the
                # last batch is smaller than the others.
                total_loss += loss.item() * len(input_batch)
                example_count += len(input_batch)

            if example_count == 0:
                raise ValueError("Training requires at least one example per epoch.")
            epoch_loss = total_loss / example_count
            history.append(epoch_loss)
            if on_epoch_end is not None:
                on_epoch_end(epoch, epoch_loss)

        return history


class WordEmbeddingLanguageModel(TrainableLanguageModel):
    # Inherits training and shares word layers; subclasses supply the embedding-combination rule.
    """Share lookup and prediction layers; subclasses decide whether to pool words."""

    def __init__(self, vocabulary_size: int, embedding_width: int):
        super().__init__()
        # One trainable [V, D] table belongs to this model, not to each input.
        self.embedding = nn.Embedding(vocabulary_size, embedding_width)
        self.hidden = nn.Linear(embedding_width, embedding_width // 2)
        self.activation = nn.ReLU()
        self.output = nn.Linear(embedding_width // 2, vocabulary_size)

    @abstractmethod
    def combine_embeddings(self, token_vectors: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("A word model must define how it combines its input rows.")

    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor | None = None) -> torch.Tensor:
        # IDs select rows; only CBOW combines several rows into one context.
        token_vectors = self.embedding(input_ids)
        features = self.combine_embeddings(token_vectors)
        hidden_features = self.activation(self.hidden(features))
        return self.output(hidden_features)

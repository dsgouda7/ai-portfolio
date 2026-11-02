"""Objective-specific word pairs with one shared, seeded batching path."""

from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader, TensorDataset


class WindowDataLoader(ABC):
    # Inherits ABC to require pairing hooks while owning window checks and seeded batching.
    """Split windows once; subclasses choose which words are inputs and answers."""

    @staticmethod
    def validate_window_ids(window_ids: torch.Tensor) -> None:
        if not isinstance(window_ids, torch.Tensor):
            raise TypeError("Word windows must be a tensor of IDs.")
        if window_ids.ndim != 2 or window_ids.shape[1] < 3 or window_ids.shape[1] % 2 == 0:
            raise ValueError("Word windows must have an odd width of at least three.")
        if window_ids.dtype != torch.long:
            raise TypeError("Word windows must use torch.long IDs.")
        if len(window_ids) == 0 or bool((window_ids < 0).any()):
            raise ValueError("Word windows must be nonempty and contain nonnegative IDs.")

    @staticmethod
    def batch_tensors(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        batch_size: int = 512,
        *,
        seed: int = 17,
    ) -> DataLoader:
        if not isinstance(input_ids, torch.Tensor) or not isinstance(target_ids, torch.Tensor):
            raise TypeError("Inputs and targets must be tensors of word IDs.")
        if input_ids.ndim < 1 or target_ids.ndim < 1:
            raise ValueError("Inputs and targets need an example dimension.")
        if len(input_ids) != len(target_ids) or len(input_ids) == 0:
            raise ValueError("Inputs and targets must contain the same nonzero number of examples.")
        if input_ids.dtype != torch.long or target_ids.dtype != torch.long:
            raise TypeError("Word inputs and targets must use torch.long IDs.")
        return DataLoader(
            TensorDataset(input_ids, target_ids),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

    def __init__(self, window_ids: torch.Tensor, *, batch_size: int = 512, seed: int = 17):
        self.validate_window_ids(window_ids)
        self.batch_size = batch_size
        self.seed = seed
        center = window_ids.shape[1] // 2
        self.center_ids = window_ids[:, center]
        self.neighbour_ids = torch.cat(
            [window_ids[:, :center], window_ids[:, center + 1:]], dim=1,
        )
        self._examples = self._prepare_examples(self.center_ids, self.neighbour_ids)

    @staticmethod
    @abstractmethod
    def _prepare_examples(
        center_ids: torch.Tensor, neighbour_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("A word objective must define its input/target pairs.")

    def prepare_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._examples

    def make_dataloader(self) -> DataLoader:
        return self.batch_tensors(
            *self.prepare_data(), batch_size=self.batch_size, seed=self.seed,
        )


class CBOWDataLoader(WindowDataLoader):
    # Inherits shared checks and batching; uses neighbours as inputs and the center as the answer.
    @staticmethod
    def _prepare_examples(
        center_ids: torch.Tensor, neighbour_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return neighbour_ids, center_ids


class SkipGramDataLoader(WindowDataLoader):
    # Inherits shared checks and batching; expands each center into its neighbour-training pairs.
    @staticmethod
    def _prepare_examples(
        center_ids: torch.Tensor, neighbour_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Flatten neighbours row by row; repeat each center beside its own answers.
        # [30] and [10, 20, 40, 50] become four aligned center/answer pairs.
        inputs = center_ids.repeat_interleave(neighbour_ids.shape[1])
        return inputs, neighbour_ids.flatten()


def prepare_sequence_data(
    window_ids: torch.Tensor, bos_id: int, eos_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    WindowDataLoader.validate_window_ids(window_ids)
    if bos_id < 0 or eos_id < 0 or bos_id == eos_id:
        raise ValueError("Start and stop markers need distinct nonnegative IDs.")
    source_width = window_ids.shape[1] // 2 + 1
    sources = window_ids[:, :source_width]
    continuations = window_ids[:, source_width:]
    targets = torch.cat([
        torch.full((len(window_ids), 1), bos_id, dtype=torch.long, device=window_ids.device),
        continuations,
        torch.full((len(window_ids), 1), eos_id, dtype=torch.long, device=window_ids.device),
    ], dim=1)
    return sources, targets

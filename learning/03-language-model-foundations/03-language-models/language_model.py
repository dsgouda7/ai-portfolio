"""Common optimization and loss alignment; model layers stay in subclasses."""

from torch import nn


class TrainableLanguageModel(nn.Module):
    """Share target alignment and optimization across language objectives."""

    def targets_for_loss(self, target_ids):
        return target_ids

    def training_loss(self, input_ids, target_ids, loss_function):
        logits = self(input_ids, target_ids)
        expected_ids = self.targets_for_loss(target_ids)
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = expected_ids.reshape(-1)
        return loss_function(flat_logits, flat_targets)

    def fit_language_model(self, dataloader, loss_function, optimizer, epochs):
        history = []

        for _ in range(epochs):
            self.train()
            total_loss = 0.0

            for input_batch, target_batch in dataloader:
                optimizer.zero_grad()
                loss = self.training_loss(input_batch, target_batch, loss_function)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(input_batch)

            history.append(total_loss / len(dataloader.dataset))

        return history

"""Common optimization and loss alignment; model layers stay in subclasses."""

from torch import nn


class TrainableLanguageModel(nn.Module):
    """Share target alignment and optimization across language objectives."""

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

    def fit_language_model(self, dataloader, loss_function, optimizer, epochs):
        history = []

        for _ in range(epochs):
            # set the model to training mode
            self.train()
            total_loss = 0.0

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

            # Grand total / number of lessons = true average loss per lesson this epoch.
            history.append(total_loss / len(dataloader.dataset))

        return history

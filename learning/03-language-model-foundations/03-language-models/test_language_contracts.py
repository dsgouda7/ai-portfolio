"""Offline contracts for shared word-data and training templates."""

import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from language_data import CBOWDataLoader, SkipGramDataLoader, WindowDataLoader
from language_model import TrainableLanguageModel, TrainingRun, WordEmbeddingLanguageModel


class WordDataLoaderTests(unittest.TestCase):
    def test_objective_pairs_preserve_order_repeated_ids_and_unknown_words(self):
        windows = torch.tensor([[1, 2, 3, 4, 5], [0, 3, 2, 3, 0]])
        before = windows.clone()
        cbow = CBOWDataLoader(windows)
        skipgram = SkipGramDataLoader(windows)
        cbow_inputs, cbow_targets = cbow.prepare_data()
        skip_inputs, skip_targets = skipgram.prepare_data()
        self.assertEqual(cbow_inputs.tolist(), [[1, 2, 4, 5], [0, 3, 3, 0]])
        self.assertEqual(cbow_targets.tolist(), [3, 2])
        self.assertEqual(skip_inputs.tolist(), [3, 3, 3, 3, 2, 2, 2, 2])
        self.assertEqual(skip_targets.tolist(), [1, 2, 4, 5, 0, 3, 3, 0])
        for factory in (cbow, skipgram):
            inputs, targets = factory.prepare_data()
            self.assertIs(inputs, factory.prepare_data()[0])
            self.assertIs(targets, factory.prepare_data()[1])
            loader = factory.make_dataloader()
            self.assertIs(loader.dataset.tensors[0], inputs)
            self.assertIs(loader.dataset.tensors[1], targets)
            self.assertEqual(inputs.dtype, torch.long)
            self.assertEqual(targets.device, windows.device)
        self.assertTrue(torch.equal(windows, before))

    def test_window_width_controls_the_number_of_neighbours(self):
        for width in (3, 5, 7):
            with self.subTest(width=width):
                windows = torch.arange(2 * width).reshape(2, width)
                center = width // 2
                expected_contexts = torch.cat([windows[:, :center], windows[:, center + 1:]], dim=1)
                cbow_inputs, cbow_targets = CBOWDataLoader(windows).prepare_data()
                skip_inputs, skip_targets = SkipGramDataLoader(windows).prepare_data()
                torch.testing.assert_close(cbow_inputs, expected_contexts)
                torch.testing.assert_close(cbow_targets, windows[:, center])
                torch.testing.assert_close(skip_inputs, windows[:, center].repeat_interleave(width - 1))
                torch.testing.assert_close(skip_targets, expected_contexts.flatten())

    def test_common_batching_is_seeded_without_changing_the_global_rng(self):
        windows = torch.arange(25).reshape(5, 5)
        rng_before = torch.random.get_rng_state().clone()
        for factory_type in (CBOWDataLoader, SkipGramDataLoader):
            first = factory_type(windows, batch_size=3, seed=5).make_dataloader()
            second = WindowDataLoader.batch_tensors(
                *factory_type(windows).prepare_data(), batch_size=3, seed=5,
            )
            for actual, expected in zip(first, second):
                torch.testing.assert_close(actual[0], expected[0])
                torch.testing.assert_close(actual[1], expected[1])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))

    def test_invalid_windows_and_unimplemented_objectives_fail_explicitly(self):
        with self.assertRaises(TypeError):
            WindowDataLoader(torch.tensor([[1, 2, 3]]))
        for windows in (
            torch.tensor([1, 2, 3]), torch.ones((2, 4), dtype=torch.long),
            torch.ones((2, 1), dtype=torch.long), torch.empty((0, 5), dtype=torch.long),
            torch.tensor([[1, -1, 3]]),
        ):
            for factory in (CBOWDataLoader, SkipGramDataLoader):
                with self.subTest(shape=windows.shape, factory=factory), self.assertRaises(ValueError):
                    factory(windows)
        with self.assertRaisesRegex(TypeError, "torch.long"):
            CBOWDataLoader(torch.ones((2, 5)))
        self.assertRaisesRegex(TypeError, "tensor", CBOWDataLoader, [[1, 2, 3]])
        self.assertRaisesRegex(
            TypeError, "tensors", WindowDataLoader.batch_tensors, [1, 2], torch.tensor([1, 2]),
        )
        with self.assertRaisesRegex(ValueError, "same nonzero"):
            WindowDataLoader.batch_tensors(torch.tensor([1, 2]), torch.tensor([1]))

    def test_operations_are_owned_by_the_base_with_the_correct_method_types(self):
        self.assertIsInstance(WindowDataLoader.__dict__["validate_window_ids"], staticmethod)
        self.assertIsInstance(WindowDataLoader.__dict__["batch_tensors"], staticmethod)
        self.assertNotIsInstance(
            WindowDataLoader.__dict__["make_dataloader"], (staticmethod, classmethod),
        )
        WindowDataLoader.validate_window_ids(torch.tensor([[1, 2, 3]]))
        loader = CBOWDataLoader(torch.tensor([[1, 2, 3]]), batch_size=1, seed=5)
        self.assertEqual(loader.make_dataloader().batch_size, 1)


class LoggingOptimizer(torch.optim.SGD):
    def __init__(self, parameters, learning_rate, events):
        super().__init__(parameters, lr=learning_rate)
        self.events = events

    def zero_grad(self, set_to_none=True):
        self.events.append("zero_grad")
        super().zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        self.events.append("step")
        return super().step(closure)


class LoggingLoss(nn.Module):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def forward(self, logits, targets):
        self.events.append("loss")
        return nn.functional.cross_entropy(logits, targets)


class LoggingModel(TrainableLanguageModel):
    def __init__(self, events):
        super().__init__()
        self.events = events
        self.weight = nn.Parameter(torch.tensor([0.2, -0.2]))
        self.weight.register_hook(self.record_backward)

    def record_backward(self, gradient):
        self.events.append("backward")
        return gradient

    def configure_loss_function(self):
        self.events.append("configure_loss")
        return LoggingLoss(self.events)

    def configure_optimizer(self, learning_rate):
        self.events.append("configure_optimizer")
        return LoggingOptimizer(self.parameters(), learning_rate, self.events)

    def forward(self, input_ids, target_ids=None):
        if not self.training:
            raise AssertionError("The shared loop did not restore training mode.")
        self.events.append("forward")
        return input_ids.float().unsqueeze(1) * self.weight

    def targets_for_loss(self, target_ids):
        self.events.append("align_targets")
        return target_ids


class TrainingTemplateTests(unittest.TestCase):
    def test_template_configures_once_and_calls_every_step_in_order(self):
        events = []
        model = LoggingModel(events)
        loader = WindowDataLoader.batch_tensors(torch.tensor([1, 2]), torch.tensor([0, 1]))
        checkpoints = []

        def observe(epoch, loss):
            checkpoints.append((epoch, loss))
            events.append("epoch_end")
            model.eval()

        result = model.train_model(loader, epochs=2, on_epoch_end=observe)
        self.assertIsInstance(result, TrainingRun)
        self.assertEqual(result.losses, [loss for _, loss in checkpoints])
        self.assertEqual([epoch for epoch, _ in checkpoints], [1, 2])
        self.assertGreaterEqual(result.seconds, 0)
        self.assertEqual(events, ["configure_loss", "configure_optimizer"] + [
            "zero_grad", "forward", "align_targets", "loss", "backward", "step", "epoch_end",
        ] * 2)

    def test_injectable_loop_weights_partial_batches_by_example_count(self):
        model = LoggingModel([])
        inputs = torch.tensor([1, 2, 3, 4, 5])
        targets = torch.tensor([0, 1, 0, 1, 0])
        loader = WindowDataLoader.batch_tensors(inputs, targets, batch_size=3)
        expected = nn.functional.cross_entropy(model(inputs), targets).item()
        losses = model.fit_language_model(
            loader, nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr=0), epochs=1,
        )
        self.assertAlmostEqual(losses[0], expected, places=6)

    def test_missing_hooks_and_invalid_runs_fail_explicitly(self):
        with self.assertRaises(TypeError):
            TrainableLanguageModel()
        with self.assertRaises(TypeError):
            WordEmbeddingLanguageModel(7, 4)
        model = LoggingModel([])
        loader = WindowDataLoader.batch_tensors(torch.tensor([1]), torch.tensor([0]))
        with self.assertRaisesRegex(ValueError, "epoch"):
            model.train_model(loader, epochs=0)
        self.assertEqual(model.events, [])
        empty = DataLoader(TensorDataset(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
        with self.assertRaisesRegex(ValueError, "example"):
            model.train_model(empty, epochs=1)

    def test_epoch_observer_errors_are_not_hidden(self):
        model = LoggingModel([])
        loader = WindowDataLoader.batch_tensors(torch.tensor([1]), torch.tensor([0]))

        def failing_observer(epoch, loss):
            raise RuntimeError("observer failed")

        with self.assertRaisesRegex(RuntimeError, "observer failed"):
            model.train_model(loader, epochs=1, on_epoch_end=failing_observer)


if __name__ == "__main__":
    unittest.main()

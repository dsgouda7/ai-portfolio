"""Offline checks for read-only, objective-aligned validation comparisons."""

from contextlib import redirect_stdout
from io import StringIO
import math
import unittest
from unittest.mock import patch

import torch
from torch import nn

from language_data import prepare_sequence_data
from language_model import TrainableLanguageModel
from language_validation import (
    CBOWValidator, EncoderDecoderValidator, ModelValidator, SkipGramValidator,
    ModelProbeRecorder, report_probe, word_probe,
)
from real_text_embeddings import prepare_validation_windows


class FixedWordModel(TrainableLanguageModel):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.scores = nn.Parameter(torch.zeros(vocabulary_size))
        self.dropout = nn.Dropout()

    def forward(self, input_ids, target_ids=None):
        return self.scores.expand(len(input_ids), -1)


class FixedSequenceModel(FixedWordModel):
    def forward(self, input_ids, target_ids=None):
        return self.scores.expand(len(input_ids), target_ids.shape[1] - 1, -1)

    def targets_for_loss(self, target_ids):
        return target_ids[:, 1:]


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.words = ["<unk>", "the", "sleepy", "cat", "sat", "on"]
        self.windows = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 1]])

    def test_each_objective_counts_its_actual_checking_answers(self):
        cases = [
            (CBOWValidator(self.windows, words=self.words, batch_size=1), FixedWordModel(6), 2),
            (SkipGramValidator(self.windows, words=self.words, batch_size=3), FixedWordModel(6), 8),
            (EncoderDecoderValidator(
                self.windows, words=self.words + ["<bos>", "<eos>"], batch_size=1,
             ), FixedSequenceModel(8), 6),
        ]
        for validator, model, count in cases:
            with self.subTest(objective=validator.name):
                result = validator.evaluate(model)
                self.assertEqual(result.checked_items, count)
                self.assertAlmostEqual(result.loss, math.log(len(model.scores)), places=6)
                expected = model.targets_for_loss(validator.target_ids)
                self.assertAlmostEqual(result.accuracy, expected.eq(0).float().mean().item())
                self.assertEqual(result.example_answers, tuple(expected[0].reshape(-1).tolist()))

    def test_evaluation_preserves_parameters_gradients_rng_and_mixed_module_modes(self):
        model = FixedWordModel(6)
        model.train()
        model.dropout.eval()
        model.scores.grad = torch.arange(6, dtype=torch.float)
        before = model.scores.detach().clone()
        gradient = model.scores.grad.clone()
        rng_before = torch.random.get_rng_state().clone()
        validator = CBOWValidator(self.windows, words=self.words, batch_size=1)
        validator.evaluate(model)
        self.assertTrue(model.training)
        self.assertFalse(model.dropout.training)
        torch.testing.assert_close(model.scores, before)
        torch.testing.assert_close(model.scores.grad, gradient)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))

    def test_before_after_report_uses_the_same_fixed_questions(self):
        validator = CBOWValidator(self.windows[:1], words=self.words)
        model = FixedWordModel(6)
        before = validator.record_before(model)
        self.windows.fill_(0)
        with torch.no_grad():
            model.scores[3] = 3
        output = StringIO()
        with redirect_stdout(output):
            comparison = validator.compare_after(model, epochs=4)
        self.assertIs(comparison.before, before)
        self.assertEqual(comparison.epochs, 4)
        self.assertEqual(comparison.after.accuracy, 1)
        self.assertLess(comparison.after.loss, before.loss)
        self.assertEqual(comparison.before.checked_items, comparison.after.checked_items)
        self.assertIn("After 4 epochs", output.getvalue())
        self.assertIn("Checking answers: cat", output.getvalue())
        self.assertIn("After prediction: cat", output.getvalue())

    def test_sequence_training_and_validation_share_the_same_preparation(self):
        sources, targets = prepare_sequence_data(self.windows, 6, 7)
        validator = EncoderDecoderValidator(
            self.windows, words=self.words + ["<bos>", "<eos>"],
        )
        torch.testing.assert_close(validator.input_ids, sources)
        torch.testing.assert_close(validator.target_ids, targets)
        self.assertEqual(sources.tolist(), [[1, 2, 3], [2, 3, 4]])
        self.assertEqual(targets.tolist(), [[6, 4, 5, 7], [6, 5, 1, 7]])
        self.assertIn("not free-running generation", validator.description)

    def test_invalid_checks_fail_explicitly_and_restore_mode(self):
        validator = CBOWValidator(self.windows, words=self.words)
        model = FixedWordModel(6)
        with self.assertRaisesRegex(RuntimeError, "before training"):
            validator.compare_after(model, epochs=1)
        validator.record_before(model)
        with self.assertRaisesRegex(ValueError, "epoch"):
            validator.compare_after(model, epochs=0)
        with patch.object(model, "forward", side_effect=RuntimeError("prediction failed")):
            with self.assertRaisesRegex(RuntimeError, "prediction failed"):
                validator.evaluate(model)
        self.assertTrue(model.training)
        with self.assertRaisesRegex(ValueError, "nonzero"):
            ModelValidator(torch.empty((0, 4), dtype=torch.long), torch.empty(0, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "start and stop"):
            EncoderDecoderValidator(self.windows, words=self.words)

    def test_reserved_stories_keep_the_training_vocabulary_and_window_boundaries(self):
        stoi = {word: index for index, word in enumerate(self.words)}
        stories = ["not used for validation"] * 3 + ["the sleepy cat sat on", "the new cat sat on"]
        with patch("real_text_embeddings.read_stories", return_value=stories) as read, redirect_stdout(StringIO()):
            windows = prepare_validation_windows(stoi, 3, reserved_stories=2)
        read.assert_called_once_with(5)
        self.assertEqual(windows.tolist(), [[1, 2, 3, 4, 5], [1, 0, 3, 4, 5]])
        self.assertNotIn("new", stoi)
        with patch("real_text_embeddings.read_stories", return_value=stories[:4]), self.assertRaisesRegex(
            ValueError, "requested reserved",
        ):
            prepare_validation_windows(stoi, 3, reserved_stories=2)


class PredictionProbeTests(unittest.TestCase):
    def test_probe_uses_explicit_vocabulary_without_changing_logits_or_gradients(self):
        words = ["a", "b", "c", "d", "e"]
        logits = torch.arange(5, dtype=torch.float, requires_grad=True)
        logits.grad = torch.ones_like(logits)
        before = logits.detach().clone()
        result = word_probe(logits, "c", words)
        self.assertEqual(set(result), {"probability", "rank", "top5"})
        self.assertEqual(result["rank"], 3)
        self.assertEqual(result["probability"], logits.detach().softmax(-1)[2].item())
        self.assertEqual([word for word, _ in result["top5"]], list(reversed(words)))
        torch.testing.assert_close(logits, before)
        torch.testing.assert_close(logits.grad, torch.ones_like(logits))

    def test_report_preserves_the_before_after_prediction_readout(self):
        words = ["a", "b", "c", "d", "e"]
        before = word_probe(torch.arange(5, dtype=torch.float), "c", words)
        after = word_probe(torch.tensor([0., 0., 5., 0., 0.]), "c", words)
        output = StringIO()
        with redirect_stdout(output):
            report_probe("Missing c", before, after)
        self.assertIn("Missing c: correct-word share", output.getvalue())
        self.assertIn("Before top 5:", output.getvalue())
        self.assertIn("After top 5:", output.getvalue())
        self.assertIn("Probability increased; rank improved.", output.getvalue())
        self.assertIn("The expected word is top-ranked.", output.getvalue())

    def test_invalid_probe_inputs_fail_explicitly(self):
        words = ["a", "b", "c", "d", "e"]
        with self.assertRaisesRegex(ValueError, "one vocabulary score"):
            word_probe(torch.zeros((2, 5)), "c", words)
        with self.assertRaisesRegex(ValueError, "one vocabulary score"):
            word_probe(torch.zeros(5), "c", words[:4])
        with self.assertRaises(ValueError):
            word_probe(torch.zeros(5), "missing", words)


class RecordedWordModel(TrainableLanguageModel):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(7, 4)
        self.head = nn.Linear(4, 7)
        self.dropout = nn.Dropout()

    def forward(self, input_ids, target_ids=None):
        return self.head(self.embedding(input_ids).mean(dim=1))


class BeforeAfterRecordingTests(unittest.TestCase):
    def test_snapshots_are_fixed_read_only_and_measure_actual_row_changes(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            model = RecordedWordModel()
        model.train()
        model.dropout.eval()
        inputs = torch.tensor([[1, 2], [2, 3]])
        words = ["<unk>", "the", "cat", "dog", "sat", "on", "mat"]
        recorder = ModelProbeRecorder(
            model, inputs, words=words, expected_words=["cat", "dog"], name="CBOW",
        )
        gradients = torch.ones_like(model.embedding.weight)
        model.embedding.weight.grad = gradients.clone()
        rng_before = torch.random.get_rng_state().clone()
        before = recorder.record_before()
        inputs.zero_()
        with torch.no_grad():
            model.embedding.weight[2] += 1
        comparison = recorder.record_after()
        self.assertIs(comparison.before, before)
        self.assertEqual(comparison.after.input_ids.tolist(), [[1, 2], [2, 3]])
        self.assertEqual(comparison.row_changes[2].item(), 2)
        self.assertEqual(comparison.row_changes.ne(0).sum().item(), 1)
        self.assertFalse(before.embeddings.requires_grad)
        self.assertTrue(model.training)
        self.assertFalse(model.dropout.training)
        torch.testing.assert_close(model.embedding.weight.grad, gradients)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        output = StringIO()
        with redirect_stdout(output):
            recorder.report_probes()
            recorder.report_embeddings(["cat"], features=4)
        self.assertIn("CBOW missing 'cat'", output.getvalue())
        self.assertIn("Embedding rows changed: 1", output.getvalue())
        self.assertIn("cat (fixed ID 2), first four features:", output.getvalue())

    def test_reports_require_completed_snapshots_and_valid_feature_counts(self):
        recorder = ModelProbeRecorder(
            RecordedWordModel(), torch.tensor([[1, 2]]),
            words=["<unk>", "the", "cat", "dog", "sat", "on", "mat"], expected_words=["cat"],
        )
        with self.assertRaisesRegex(RuntimeError, "before-training"):
            recorder.record_after()
        with self.assertRaisesRegex(RuntimeError, "after-training"):
            recorder.report_probes()
        recorder.record_before()
        recorder.record_after()
        with self.assertRaisesRegex(ValueError, "Feature count"):
            recorder.report_embeddings(["cat"], features=5)


if __name__ == "__main__":
    unittest.main()

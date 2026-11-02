"""Offline contracts for the shared windows and their objective-specific pairs."""

import ast
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
import math
from pathlib import Path
import re
import time
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import torch

from language_data import CBOWDataLoader
from language_model import TrainableLanguageModel, WordEmbeddingLanguageModel
from language_validation import CBOWValidator, ModelProbeRecorder, ModelValidator, word_probe
from real_text_embeddings import (
    animate_embedding_routing,
    draw_embedding_routing,
    embedding_routing_steps,
    make_document_windows,
    objective_pairs,
    plot_embedding_routing,
    prepare_validation_windows,
    record_embedding_routing,
)


class DocumentWindowTests(unittest.TestCase):
    def test_sliding_preserves_order_repeats_and_unknown_slots(self):
        actual = make_document_windows([[1, 0, 1, 3, 4, 5, 6]], 5)
        np.testing.assert_array_equal(
            actual, [[1, 0, 1, 3, 4], [0, 1, 3, 4, 5], [1, 3, 4, 5, 6]],
        )
        self.assertEqual(actual.dtype, np.int64)

    def test_windows_never_cross_document_boundaries(self):
        actual = make_document_windows([[1, 2, 3], [], [9], [4, 5, 6, 7]], 3)
        np.testing.assert_array_equal(actual, [[1, 2, 3], [4, 5, 6], [5, 6, 7]])

    def test_empty_and_short_documents_keep_two_axes(self):
        for documents in ([], [[]], [[1, 2], [3, 4]]):
            with self.subTest(documents=documents):
                actual = make_document_windows(documents, 5)
                self.assertEqual(actual.shape, (0, 5))
                self.assertEqual(actual.dtype, np.int64)

    def test_width_one_keeps_every_token(self):
        np.testing.assert_array_equal(
            make_document_windows([[3, 0], [7]], 1), [[3], [0], [7]],
        )

    def test_nonpositive_width_is_an_error(self):
        for width in (0, -1):
            with self.subTest(width=width), self.assertRaisesRegex(ValueError, "at least one"):
                make_document_windows([[1, 2, 3]], width)

    def test_matches_previous_sentence_local_window_order(self):
        documents = [[1, 2, 3, 4, 5, 6, 7], [9, 8, 7, 6, 5], [0, 1]]
        previous_rows = [
            row for ids in documents
            for row in zip(*(ids[offset:] for offset in range(5)))
        ]
        np.testing.assert_array_equal(make_document_windows(documents, 5), previous_rows)

    def test_inline_lesson_builders_use_numpy_and_match_shared_helper(self):
        chapter = Path(__file__).resolve().parent
        stem = "03A-training-skeleton-and-word-objectives"
        notebook = json.loads((chapter / f"{stem}.ipynb").read_text(encoding="utf-8"))
        sources = {
            "notebook": "\n\n".join(
                "".join(cell["source"]) for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            ),
            "companion": (chapter / f"{stem}.py").read_text(encoding="utf-8"),
        }
        documents = [[1, 0, 1, 3, 4, 5, 6], [], [9], [4, 5, 6]]
        definitions = []
        for name, source in sources.items():
            with self.subTest(source=name):
                functions = [
                    node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "make_document_windows"
                ]
                self.assertEqual(len(functions), 1)
                definitions.append(ast.dump(functions[0]))
                namespace = {"np": np}
                module = ast.Module(body=functions, type_ignores=[])
                exec(compile(module, name, "exec"), namespace)
                builder = namespace["make_document_windows"]
                with patch(
                    "numpy.lib.stride_tricks.sliding_window_view",
                    wraps=np.lib.stride_tricks.sliding_window_view,
                ) as sliding_windows:
                    builder(documents, 3)
                self.assertEqual(sliding_windows.call_count, 2)
                self.assertEqual(
                    [call.args[0].tolist() for call in sliding_windows.call_args_list],
                    [documents[0], documents[-1]],
                )
                for width in (1, 3, 5, 8):
                    for inputs in (documents, [], [[]], [[1, 2]]):
                        actual = builder(inputs, width)
                        expected = make_document_windows(inputs, width)
                        np.testing.assert_array_equal(actual, expected)
                        self.assertEqual(actual.shape, expected.shape)
                        self.assertEqual(actual.dtype, np.int64)
                for width in (0, -1):
                    with self.assertRaisesRegex(ValueError, "at least one"):
                        builder(documents, width)
        self.assertEqual(definitions[0], definitions[1])

    def test_each_objective_uses_the_same_raw_windows_differently(self):
        windows = torch.tensor(make_document_windows([[1, 2, 3, 4, 5, 6]], 5))
        cbow_inputs, cbow_answers = objective_pairs(windows, "CBOW", bos_id=9)
        self.assertEqual(cbow_inputs.tolist(), [[1, 2, 4, 5], [2, 3, 5, 6]])
        self.assertEqual(cbow_answers.tolist(), [3, 4])
        skip_inputs, skip_answers = objective_pairs(windows, "Skip-gram", bos_id=9)
        self.assertEqual(skip_inputs.tolist(), [3, 3, 3, 3, 4, 4, 4, 4])
        self.assertEqual(skip_answers.tolist(), [1, 2, 4, 5, 2, 3, 5, 6])
        sources, targets = objective_pairs(windows, "Encoder-decoder", bos_id=9)
        self.assertEqual(sources.tolist(), [[1, 2, 3], [2, 3, 4]])
        self.assertEqual(targets.tolist(), [[9, 4, 5], [9, 5, 6]])


class CBOWEmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        chapter = Path(__file__).resolve().parent
        stem = "03A-training-skeleton-and-word-objectives"
        notebook = json.loads((chapter / f"{stem}.ipynb").read_text(encoding="utf-8"))
        source = "\n\n".join(
            "".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        definitions = [
            node for node in ast.parse(source).body
            if isinstance(node, ast.ClassDef) and node.name == "CBOW"
        ]
        if len(definitions) != 1:
            raise AssertionError("The lesson must define exactly one real CBOW model.")
        namespace = {"nn": torch.nn, "TrainableLanguageModel": TrainableLanguageModel,
                     "WordEmbeddingLanguageModel": WordEmbeddingLanguageModel}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "CBOW", "exec"), namespace)
        cls.model_type = namespace["CBOW"]

    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            self.model = self.model_type(7, 4)
        self.ids = torch.tensor([[0, 2, 4, 6], [6, 4, 2, 0]], dtype=torch.long)

    def _activate_features(self):
        # Keep routing tests on an active ReLU path, rather than testing dead units.
        with torch.no_grad():
            rows = self.model.embedding.weight
            rows.copy_(torch.arange(rows.numel()).reshape_as(rows) / rows.numel())
            self.model.hidden.weight.fill_(0.25)
            self.model.hidden.bias.fill_(1.0)

    def test_real_model_allocates_one_trainable_vocabulary_table(self):
        self.assertEqual(tuple(self.model.embedding.weight.shape), (7, 4))
        self.assertTrue(self.model.embedding.weight.requires_grad)
        self.assertIs(
            dict(self.model.named_parameters())["embedding.weight"],
            self.model.embedding.weight,
        )

    def test_forward_selects_rows_and_scores_without_rewriting_ids(self):
        before_ids = self.ids.clone()
        before_table = self.model.embedding.weight.detach().clone()
        with torch.no_grad():
            rows = self.model.embedding(self.ids)
            expected = self.model.output(
                self.model.activation(self.model.hidden(rows.mean(dim=1))),
            )
            actual = self.model(self.ids)
        self.assertEqual(tuple(rows.shape), (2, 4, 4))
        self.assertEqual(tuple(actual.shape), (2, 7))
        torch.testing.assert_close(rows, before_table[self.ids])
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual[0], actual[1])
        self.assertEqual(actual[0].argmax().item(), actual[1].argmax().item())
        self.assertTrue(torch.equal(self.ids, before_ids))
        self.assertTrue(torch.equal(self.model.embedding.weight.detach(), before_table))

    def test_shared_training_updates_embeddings_and_predictor_not_input_ids(self):
        self._activate_features()
        before_rows = self.model.embedding.weight.detach().clone()
        before_head = self.model.output.weight.detach().clone()
        before_ids = self.ids.clone()
        answers = torch.tensor([1, 1], dtype=torch.long)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.ids, answers), batch_size=2,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05)
        losses = self.model.fit_language_model(loader, torch.nn.CrossEntropyLoss(), optimizer, 20)
        self.assertLess(losses[-1], losses[0])
        self.assertFalse(torch.equal(self.model.embedding.weight.detach(), before_rows))
        self.assertFalse(torch.equal(self.model.output.weight.detach(), before_head))
        self.assertTrue(torch.equal(self.ids, before_ids))

    def test_recorded_step_routes_rows_without_touching_the_main_model_or_rng(self):
        self._activate_features()
        before = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        before_ids = self.ids.clone()
        rng_before = torch.random.get_rng_state().clone()
        trace = record_embedding_routing(self.model, self.ids[:1], torch.tensor([1]))
        gradient_rows = trace["gradient"].ne(0).any(dim=1).nonzero().flatten().tolist()
        changed_rows = trace["weights"][3].ne(trace["weights"][0]).any(dim=1).nonzero().flatten().tolist()
        self.assertEqual(set(gradient_rows), {0, 2, 4, 6})
        self.assertEqual(set(changed_rows), {0, 2, 4, 6})
        self.assertEqual(trace["gradient"].layout, torch.strided)
        self.assertTrue(torch.equal(trace["weights"][0], trace["weights"][1]))
        self.assertTrue(torch.equal(trace["weights"][0], trace["weights"][2]))
        torch.testing.assert_close(
            trace["weights"][3], trace["weights"][0] - 0.1 * trace["gradient"],
        )
        for name, value in self.model.named_parameters():
            self.assertTrue(torch.equal(value.detach(), before[name]), name)
            self.assertIsNone(value.grad)
        self.assertTrue(torch.equal(self.ids, before_ids))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))

    def test_checking_answer_does_not_fetch_its_input_row_but_head_receives_feedback(self):
        self._activate_features()
        trace = record_embedding_routing(self.model, self.ids[:1], torch.tensor([1]))
        self.assertEqual(trace["targets"].tolist(), [1])
        self.assertNotIn(1, trace["saved_ids"].flatten().tolist())
        self.assertEqual(trace["gradient"][1].count_nonzero().item(), 0)
        self.assertNotEqual(trace["head_gradient"][1].item(), 0)
        self.assertEqual(trace["head_gradient"].count_nonzero().item(), 7)

    def test_repeated_input_ids_accumulate_occurrence_gradients_automatically(self):
        self._activate_features()
        contexts = torch.tensor([[0, 0, 2, 2]])
        activations = []

        def retain_lookup(module, inputs, output):
            output.retain_grad()
            activations.append(output)

        hook = self.model.embedding.register_forward_hook(retain_lookup)
        try:
            loss = self.model.training_loss(contexts, torch.tensor([1]), torch.nn.CrossEntropyLoss())
            loss.backward()
        finally:
            hook.remove()
        occurrence_gradients = activations[0].grad
        torch.testing.assert_close(
            self.model.embedding.weight.grad[0],
            occurrence_gradients[0, 0] + occurrence_gradients[0, 1],
        )
        torch.testing.assert_close(
            self.model.embedding.weight.grad[2],
            occurrence_gradients[0, 2] + occurrence_gradients[0, 3],
        )
        self.assertEqual(self.model.embedding.weight.grad[[1, 3, 4, 5, 6]].count_nonzero().item(), 0)
        self.assertGreater(self.model.embedding.weight.grad[[0, 2]].norm().item(), 0)

    def test_saved_count_splits_by_occurrences_not_unique_ids(self):
        self._activate_features()
        contexts = torch.tensor([[0, 0, 2, 2]])
        trace = record_embedding_routing(self.model, contexts, torch.tensor([1]))
        gradients = trace["activation_gradients"]
        self.assertEqual(trace["mean_count"], 4)
        self.assertEqual(trace["saved_ids"].tolist(), [[0, 0, 2, 2]])
        torch.testing.assert_close(
            gradients["mean"], gradients["linear"] @ self.model.hidden.weight.detach(),
        )
        torch.testing.assert_close(
            gradients["lookup"],
            gradients["mean"][:, None, :].expand(1, 4, 4) / 4,
        )
        torch.testing.assert_close(trace["gradient"][0], gradients["mean"][0] / 2)
        torch.testing.assert_close(trace["gradient"][2], gradients["mean"][0] / 2)
        torch.testing.assert_close(trace["gradient"], trace["reconstructed_gradient"])
        contexts.fill_(6)
        self.assertEqual(trace["saved_ids"].tolist(), [[0, 0, 2, 2]])
        self.assertGreater(trace["gradient"].norm().item(), 0)

    def test_four_ids_reach_exactly_four_rows_of_a_hundred_by_three_table(self):
        with torch.random.fork_rng(devices=[]):
            self.model = self.model_type(100, 3)
        self._activate_features()
        trace = record_embedding_routing(
            self.model, torch.tensor([[7, 21, 64, 90]]), torch.tensor([1]),
        )
        gradient = trace["gradient"]
        nonzero_rows = set(gradient.ne(0).any(dim=1).nonzero().flatten().tolist())
        self.assertEqual(tuple(gradient.shape), (100, 3))
        self.assertEqual(nonzero_rows, {7, 21, 64, 90})
        self.assertEqual(100 - len(nonzero_rows), 96)
        self.assertEqual(tuple(trace["activations"]["mean"].shape), (1, 3))
        self.assertEqual(tuple(trace["activations"]["lookup"].shape), (1, 4, 3))
        self.assertEqual(tuple(trace["activation_gradients"]["logits"].shape), (1, 100))

    def test_recorder_rejects_empty_or_multiwindow_inputs_and_invalid_step_size(self):
        for contexts in (torch.empty((1, 0), dtype=torch.long), self.ids, self.ids[0]):
            with self.subTest(shape=contexts.shape), self.assertRaisesRegex(ValueError, "one nonempty"):
                record_embedding_routing(self.model, contexts, torch.tensor([1]))
        with self.assertRaisesRegex(ValueError, "one checking answer"):
            record_embedding_routing(self.model, self.ids[:1], torch.tensor([1, 2]))
        with self.assertRaisesRegex(ValueError, "positive"):
            record_embedding_routing(self.model, self.ids[:1], torch.tensor([1]), learning_rate=0)

    def test_adam_history_can_move_an_unread_row_with_zero_current_gradient(self):
        self._activate_features()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer.zero_grad()
        self.model.training_loss(torch.tensor([[0, 2]]), torch.tensor([1]), criterion).backward()
        optimizer.step()
        before = self.model.embedding.weight.detach().clone()
        optimizer.zero_grad()
        self.model.training_loss(torch.tensor([[4, 6]]), torch.tensor([1]), criterion).backward()
        self.assertEqual(self.model.embedding.weight.grad[[0, 2]].count_nonzero().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(self.model.embedding.weight.detach()[[0, 2]], before[[0, 2]]))

    def test_flow_stages_follow_actual_forward_values_and_reverse_gradients(self):
        self._activate_features()
        trace = record_embedding_routing(self.model, self.ids[:1], torch.tensor([1]))
        steps = embedding_routing_steps(trace)
        self.assertEqual(len(steps), 13)
        self.assertEqual([step[1] for step in steps], [0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0, 0])
        self.assertEqual([step[2] for step in steps], ["forward"] * 6 + ["backward"] * 6 + ["update"])
        torch.testing.assert_close(steps[0][3], trace["saved_ids"])
        torch.testing.assert_close(steps[1][4], trace["activations"]["mean"])
        torch.testing.assert_close(steps[3][4], trace["activations"]["hidden"])
        torch.testing.assert_close(steps[4][4], trace["activations"]["logits"])
        self.assertAlmostEqual(steps[5][4].item(), trace["loss"])
        torch.testing.assert_close(steps[6][4], trace["activation_gradients"]["logits"])
        torch.testing.assert_close(steps[9][4], trace["activation_gradients"]["mean"])
        torch.testing.assert_close(steps[10][4], trace["activation_gradients"]["lookup"])
        torch.testing.assert_close(steps[11][4], trace["gradient"])
        torch.testing.assert_close(steps[12][4], trace["weights"][3])

    def test_cross_entropy_starts_the_measured_backward_flow_and_relu_blocks_negative_inputs(self):
        self._activate_features()
        with torch.no_grad():
            self.model.hidden.bias[0] = -3
        trace = record_embedding_routing(self.model, self.ids[:1], torch.tensor([1]))
        probabilities = trace["activations"]["logits"].softmax(dim=-1)
        expected = probabilities.clone()
        expected[0, trace["targets"][0]] -= 1
        torch.testing.assert_close(trace["probabilities"], probabilities)
        torch.testing.assert_close(trace["activation_gradients"]["logits"], expected)
        self.assertAlmostEqual(trace["loss"], -probabilities[0, trace["targets"][0]].log().item(), places=6)
        self.assertLess(trace["activations"]["linear"][0, 0].item(), 0)
        self.assertEqual(trace["activations"]["hidden"][0, 0].item(), 0)
        self.assertEqual(trace["activation_gradients"]["linear"][0, 0].item(), 0)
        torch.testing.assert_close(
            trace["activation_gradients"]["linear"],
            trace["activation_gradients"]["hidden"] * trace["activations"]["linear"].gt(0),
        )

    def test_single_pipeline_enlarges_active_layer_and_displays_measured_values(self):
        with torch.random.fork_rng(devices=[]):
            self.model = self.model_type(7, 8)
        self._activate_features()
        trace = record_embedding_routing(self.model, self.ids[:1], torch.tensor([1]))
        words = ["the", "cat", "sleepy", "dog", "sat", "mat", "on"]
        figure = plot_embedding_routing(trace, words)
        try:
            self.assertEqual(len(figure.axes), 1)
            axis = figure.axes[0]
            draw_embedding_routing(axis, trace, words, stage=5, progress=0.5)
            texts = [artist.get_text() for artist in axis.texts]
            self.assertIn(f"{trace['loss']:.4f}", texts)
            widths = [patch.get_width() for patch in axis.patches[:6]]
            self.assertEqual(widths.index(max(widths)), 5)
            draw_embedding_routing(axis, trace, words, stage=10)
            texts = [artist.get_text() for artist in axis.texts]
            values = trace["activation_gradients"]["lookup"][0, 0, :3]
            expected = "[" + ", ".join(f"{value:.3g}" for value in values.tolist()) + "]"
            self.assertIn(expected, texts)
            self.assertIn("each occurrence receives gradient / 4", texts)
            draw_embedding_routing(axis, trace, words, stage=12)
            texts = [artist.get_text() for artist in axis.texts]
            values = trace["weights"][3][trace["saved_ids"][0, 0], :3]
            expected = "[" + ", ".join(f"{value:.6g}" for value in values.tolist()) + "]"
            self.assertIn(expected, texts)
        finally:
            plt.close(figure)

    def test_single_cat_animation_contains_forward_backward_and_optimizer_phases(self):
        self._activate_features()
        trace = record_embedding_routing(self.model, self.ids[:1], torch.tensor([1]))
        animation = animate_embedding_routing(
            trace, ["the", "cat", "sleepy", "dog", "sat", "mat", "on"],
        )
        self.assertEqual(list(animation.new_frame_seq()), list(range(65)))
        self.assertIn("frames[64]", animation.to_jshtml())


class VisibleLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        chapter = Path(__file__).resolve().parent
        notebook = json.loads(
            (chapter / "03A-training-skeleton-and-word-objectives.ipynb").read_text(encoding="utf-8")
        )
        cls.code = {
            cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        }
        definitions = [
            node for node in ast.parse("\n\n".join(cls.code.values())).body
            if isinstance(node, ast.ClassDef) and node.name in {"CBOW", "EncoderDecoderLM"}
        ]
        namespace = {"nn": torch.nn, "TrainableLanguageModel": TrainableLanguageModel,
                     "WordEmbeddingLanguageModel": WordEmbeddingLanguageModel}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "models", "exec"), namespace)
        cls.cbow_type = namespace["CBOW"]
        cls.sequence_type = namespace["EncoderDecoderLM"]

    def namespace(self):
        words = ["<unk>", "the", "sleepy", "cat", "sat", "on", "mat"]
        windows = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 1], [3, 4, 5, 1, 6]])
        namespace = {
            "torch": torch, "nn": torch.nn, "time": time, "math": math, "SEED": 17,
            "itos": words, "stoi": {word: index for index, word in enumerate(words)},
            "vocabulary_size": 7, "EMBEDDING_WIDTH": 4, "windows": windows,
            "CBOWDataLoader": CBOWDataLoader,
            "cat_contexts": windows[:, [0, 1, 3, 4]],
            "cat_contexts_before": windows[:, [0, 1, 3, 4]].clone(),
        }
        exec(compile(self.code["ch03a-probe-helpers"], "probes", "exec"), namespace)
        return namespace

    def loader(self, inputs, targets):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(inputs, targets), batch_size=3, shuffle=True,
            generator=torch.Generator().manual_seed(17),
        )

    def test_cbow_epoch_monitoring_preserves_the_original_ten_epoch_training(self):
        state = self.namespace()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            model = self.cbow_type(7, 4)
        with torch.no_grad():
            model.embedding.weight.copy_(torch.arange(28).reshape(7, 4) / 28)
            model.hidden.weight.fill_(0.25)
            model.hidden.bias.fill_(1)
        reference = deepcopy(model)
        contexts, answers = state["cat_contexts"], state["windows"][:, 2]

        state.update({
            "model": model, "embedding_before": model.embedding.weight.detach().clone(),
            "cbow_dataloader": self.loader(contexts, answers),
            "cbow_validator": CBOWValidator(state["windows"], words=state["itos"], batch_size=3),
        })
        state["cbow_recorder"] = ModelProbeRecorder(
            model, contexts, words=state["itos"], expected_words=("cat", "sat", "on"), name="CBOW",
        )
        snapshot = state["cbow_recorder"].record_before()
        state["embedding_before"] = snapshot.embeddings
        state["cbow_before"] = snapshot.probes
        with patch.object(model, "fit_language_model", wraps=model.fit_language_model) as fit:
            with redirect_stdout(StringIO()):
                exec(compile(self.code["06c-21"], "cbow-learning", "exec"), state)
            self.assertEqual(fit.call_count, 1)
            for call in fit.call_args_list:
                self.assertIs(call.args[0], state["cbow_dataloader"])
                self.assertIsInstance(call.args[1], torch.nn.CrossEntropyLoss)
                self.assertIsInstance(call.args[2], torch.optim.Adam)
                self.assertEqual(call.args[3], 10)
                self.assertEqual(call.kwargs, {"on_epoch_end": state["record_cbow_epoch"]})
        history = state["cbow_learning_history"]
        self.assertEqual([row["epoch"] for row in history], list(range(11)))
        self.assertIsNone(history[0]["train_loss"])
        self.assertEqual([row["train_loss"] for row in history[1:]], state["losses"])
        self.assertGreater(history[-1]["cat_row_shift"], 0)
        reference.fit_language_model(
            self.loader(contexts, answers), torch.nn.CrossEntropyLoss(),
            torch.optim.Adam(reference.parameters(), lr=0.008), epochs=10,
        )
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_reader_writer_has_its_own_six_epoch_run_and_measured_shared_rows(self):
        state = self.namespace()
        windows = state["windows"]
        sources = windows[:, :3]
        targets = torch.cat([
            torch.full((3, 1), 7), windows[:, 3:], torch.full((3, 1), 8),
        ], dim=1)
        state.update({
            "EncoderDecoderLM": self.sequence_type, "TrainableLanguageModel": TrainableLanguageModel,
            "sequence_vocabulary_size": 9, "sequence_itos": state["itos"] + ["<bos>", "<eos>"],
            "source_sequences": sources, "sequence_targets": targets,
            "sequence_dataloader": self.loader(sources, targets), "BOS_ID": 7, "EOS_ID": 8,
            "validation_windows": windows,
        })
        for prerequisite in ("criterion", "cbow_seconds", "skipgram_seconds"):
            self.assertNotIn(prerequisite, state)
        original_fit = TrainableLanguageModel.fit_language_model
        with torch.random.fork_rng(devices=[]), patch.object(
            TrainableLanguageModel, "fit_language_model", autospec=True, side_effect=original_fit,
        ) as fit:
            with redirect_stdout(StringIO()):
                exec(compile(self.code["ch03a-sequence-training"], "sequence-learning", "exec"), state)
            self.assertEqual(fit.call_count, 1)
            for call in fit.call_args_list:
                self.assertIs(call.args[0], state["sequence_model"])
                self.assertIsInstance(call.args[2], torch.nn.CrossEntropyLoss)
                self.assertIsInstance(call.args[3], torch.optim.Adam)
                self.assertEqual(call.args[4], 6)
                self.assertEqual(call.kwargs, {"on_epoch_end": state["record_sequence_epoch"]})
        history = state["sequence_learning_history"]
        self.assertEqual([row["epoch"] for row in history], list(range(7)))
        self.assertGreater(history[-1]["embedding_shift"], 0)
        self.assertLess(state["sequence_losses"][-1], state["sequence_losses"][0])
        self.assertTrue(torch.equal(state["sequence_embedding_before"][8], state["sequence_embedding_after"][8]))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            reference = self.sequence_type(9, 4, 48)
        reference.fit_language_model(
            self.loader(sources, targets), torch.nn.CrossEntropyLoss(),
            torch.optim.Adam(reference.parameters(), lr=0.008), epochs=6,
        )
        for actual, expected in zip(state["sequence_model"].parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_teacher_forcing_routes_both_lookup_paths_into_the_same_table(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            model = self.sequence_type(9, 4, 6)
        sources = torch.tensor([[1, 2, 3]])
        targets = torch.tensor([[7, 4, 5, 8]])
        occurrences = []

        def remember(module, inputs, output):
            output.retain_grad()
            occurrences.append((inputs[0].clone(), output))

        hook = model.embedding.register_forward_hook(remember)
        try:
            logits = model(sources, targets)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 9), model.targets_for_loss(targets).reshape(-1),
            )
            loss.backward()
        finally:
            hook.remove()
        self.assertEqual(len(occurrences), 2)
        self.assertTrue(torch.equal(occurrences[0][0], sources))
        self.assertTrue(torch.equal(occurrences[1][0], targets[:, :-1]))
        reconstructed = torch.zeros_like(model.embedding.weight)
        for ids, vectors in occurrences:
            reconstructed.index_add_(0, ids.flatten(), vectors.grad.reshape(-1, 4))
        torch.testing.assert_close(model.embedding.weight.grad, reconstructed)
        self.assertGreater(model.embedding.weight.grad[3].norm().item(), 0)
        self.assertGreater(model.embedding.weight.grad[7].norm().item(), 0)
        self.assertEqual(model.embedding.weight.grad[8].count_nonzero().item(), 0)
        self.assertNotEqual(model.output.bias.grad[8].item(), 0)
        changed_future = targets.clone()
        changed_future[0, 1] = 6
        with torch.inference_mode():
            torch.testing.assert_close(model(sources, targets)[:, 0], model(sources, changed_future)[:, 0])

    def test_a_wrong_guess_does_not_replace_the_correct_previous_training_word(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            model = self.sequence_type(9, 4, 6)
        sources = torch.tensor([[1, 2, 3]])
        targets = torch.tensor([[7, 4, 5, 8]])
        with torch.no_grad():
            model.output.weight.zero_()
            model.output.bias.zero_()
            model.output.bias[6] = 8
        seen = []

        def capture_inputs(module, inputs, output):
            seen.append(inputs[0].clone())

        hook = model.embedding.register_forward_hook(capture_inputs)
        try:
            with torch.inference_mode():
                logits = model(sources, targets)
        finally:
            hook.remove()
        self.assertTrue(bool((logits.argmax(-1) == 6).all()))
        torch.testing.assert_close(seen[-1], targets[:, :-1], rtol=0, atol=0)
        self.assertNotIn(6, seen[-1].flatten().tolist())
        torch.testing.assert_close(model.targets_for_loss(targets), torch.tensor([[4, 5, 8]]))

    def test_each_prediction_cannot_see_its_own_answer_or_later_answers(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            model = self.sequence_type(9, 4, 6)
        sources = torch.tensor([[1, 2, 3]])
        targets = torch.tensor([[7, 4, 5, 8]])
        with torch.inference_mode():
            baseline = model(sources, targets)
            for position in range(3):
                changed = targets.clone()
                changed[0, position + 1] = 6
                torch.testing.assert_close(
                    model(sources, changed)[:, :position + 1], baseline[:, :position + 1], rtol=0, atol=0,
                )
            changed_previous = targets.clone()
            changed_previous[0, 1] = 6
            self.assertFalse(torch.allclose(model(sources, changed_previous)[:, 1], baseline[:, 1]))


class CoreLessonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        chapter = Path(__file__).resolve().parent
        stem = "03A-training-skeleton-and-word-objectives"
        cls.notebook = json.loads((chapter / f"{stem}.ipynb").read_text(encoding="utf-8"))
        cls.companion = (chapter / f"{stem}.py").read_text(encoding="utf-8")
        cls.code = {
            cell["id"]: "".join(cell["source"]) for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        }

    def test_catalogue_scenario_and_mapping_precede_the_training_task(self):
        opening = next(
            "".join(cell["source"]) for cell in self.notebook["cells"]
            if cell["cell_type"] == "markdown"
            and "".join(cell["source"]).startswith("## 1 - ")
        )
        landmarks = [
            "Imagine you are building a catalogue of objects discovered by space probes.",
            "#### Sensors turn an identity into a descriptive profile",
            "#### Words can have descriptive profiles too",
            "#### A separate question: how can a prediction task teach the profiles?",
            "1. **Prepare** integer IDs",
        ]
        positions = [opening.index(landmark) for landmark in landmarks]
        self.assertEqual(positions, sorted(positions))
        for explanation in (
            "Object-417", "Object-892", "Earth", "size, temperature",
            "units and scales", "| Registry identifier | Token ID:",
            "| Sensor profile for one object | Learned embedding",
            "| Catalogue of profiles | Embedding matrix",
            "words do not come with physical sensors",
            "one training exercise, not the definition of an embedding",
        ):
            with self.subTest(explanation=explanation):
                self.assertIn(explanation, " ".join(opening.split()))
        self.assertIn("images/03a-object-sensor-signatures.png", opening)

    def test_analogy_reaches_relatedness_and_distinguishes_word_scores(self):
        markdown = "\n\n".join(
            "".join(cell["source"]) for cell in self.notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        for explanation in (
            "Earth is your **query profile**",
            "searching for semantically related words",
            "uses the available context to score candidate answers",
            "it does not simply choose the nearest word embedding",
            "decoding strategies such as sampling, which belong with text generation",
            "images/03a-celestial-neighborhood-principle.png",
            "images/03a-01-shared-word-features.png",
        ):
            with self.subTest(explanation=explanation):
                self.assertIn(explanation, " ".join(markdown.split()))

    def test_core_keeps_embedding_evidence_without_the_sampling_detour(self):
        source = "\n\n".join(self.code.values())
        self.assertNotIn("ch03a-similarity-and-sampling", self.code)
        self.assertNotIn("torch.multinomial", source)
        self.assertIsNone(re.search(r"\bbridge_\w+", source))
        self.assertIn("def most_similar", self.code["06c-27"])
        self.assertIn("cbow_before", self.code["ch03a-probe-baseline"])
        self.assertIn("cbow_recorder.report_probes()", self.code["06c-21"])
        self.assertIn("cbow_validator.evaluate(model)", self.code["06c-21"])
        self.assertIn("embedding_row_shifts", self.code["06c-21"])

    def test_compact_skipgram_setup_preserves_pairs_without_consuming_a_batch(self):
        windows = torch.tensor([[1, 2, 3, 4, 5], [0, 3, 2, 3, 0]])
        namespace = {
            "torch": torch, "SEED": 17, "CONTEXT_RADIUS": 2, "window_ids": windows,
            "DataLoader": torch.utils.data.DataLoader,
            "TensorDataset": torch.utils.data.TensorDataset,
        }
        exec(compile(self.code["06c-11"], "dataloader", "exec"), namespace)
        exec(compile(self.code["06c-sg-02"], "skipgram-pairs", "exec"), namespace)
        self.assertEqual(namespace["skipgram_inputs"].tolist(), [3, 3, 3, 3, 2, 2, 2, 2])
        self.assertEqual(namespace["skipgram_targets"].tolist(), [1, 2, 4, 5, 0, 3, 3, 0])
        loader = namespace["skipgram_dataloader"]
        self.assertEqual(len(loader.dataset), 8)
        torch.testing.assert_close(loader.dataset.tensors[0], namespace["skipgram_inputs"])
        torch.testing.assert_close(loader.dataset.tensors[1], namespace["skipgram_targets"])
        self.assertTrue(torch.equal(
            loader.generator.get_state(), torch.Generator().manual_seed(17).get_state(),
        ))

    def test_skipgram_top_four_comes_from_one_cat_prediction(self):
        tree = ast.parse(self.code["06c-sg-04"])
        definition = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SkipGram"
        )
        namespace = {"nn": torch.nn, "TrainableLanguageModel": TrainableLanguageModel,
                     "WordEmbeddingLanguageModel": WordEmbeddingLanguageModel}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), "SkipGram", "exec"), namespace)
        with torch.random.fork_rng(devices=[]):
            model = namespace["SkipGram"](7, 4)
        with torch.no_grad():
            model.output.weight.zero_()
            model.output.bias.copy_(torch.arange(7))
        namespace.update({"torch": torch, "skipgram_model": model, "stoi": {"cat": 2}})
        predictions = [node for node in tree.body if isinstance(node, ast.With)]
        self.assertEqual(len(predictions), 1)
        with patch.object(model, "forward", wraps=model.forward) as forward:
            exec(compile(ast.Module(body=predictions, type_ignores=[]), "top-four", "exec"), namespace)
        forward.assert_called_once()
        torch.testing.assert_close(forward.call_args.args[0], torch.tensor([2]))
        self.assertEqual(tuple(namespace["neighbour_logits"].shape), (1, 7))
        self.assertEqual(namespace["predicted_ids"].tolist(), [6, 5, 4, 3])

    def test_prediction_driven_word_profiles_finish_the_training_section(self):
        explanation = next(
            "".join(cell["source"]) for cell in self.notebook["cells"]
            if cell["id"] == "06c-26"
        )
        self.assertTrue(explanation.startswith("### 3.4 - "))
        for idea in (
            "word that actually appeared in the text",
            "cat sat ___ the mat", "dog sat ___ the mat",
            "Both `cat` and `dog` are clues here",
            "similar kinds of correction",
            "related profiles can be a useful side effect",
            "no guarantee every related pair moves closer",
            "`hot` and `cold`",
            "not which word wins a particular gap",
        ):
            with self.subTest(idea=idea):
                self.assertIn(idea, " ".join(explanation.split()))
        ids = [cell["id"] for cell in self.notebook["cells"]]
        self.assertLess(ids.index("06c-21"), ids.index("06c-26"))
        self.assertLess(ids.index("06c-26"), ids.index("06c-27"))
        self.assertLess(ids.index("06c-27"), ids.index("06c-30"))
        self.assertNotIn("ch03a-sampling-bridge-intro", ids)
        self.assertTrue(self.code["06c-27"].startswith("# %% 3.4 - "))

    def test_order_limit_closes_cbow_in_plain_english_before_optional_experiments(self):
        explanation = next(
            "".join(cell["source"]) for cell in self.notebook["cells"]
            if cell["id"] == "06c-30"
        )
        self.assertLess(len(explanation.split()), 220)
        self.assertIn("the cat ___ the dog", explanation)
        self.assertIn("the dog ___ the cat", explanation)
        self.assertIn("same word scores and probabilities", explanation)
        self.assertIn("learn useful word relationships", explanation)
        self.assertTrue(explanation.startswith("#### "))
        self.assertIn("The core lesson ends here", explanation)
        self.assertIn("03B-counts-and-fixed-window-language-models.ipynb", explanation)
        self.assertNotIn("ch03a-next", [cell["id"] for cell in self.notebook["cells"]])
        self.assertNotIn("06c-31", self.code)
        self.assertNotIn("permuted_context", self.companion)

    def test_section_four_keeps_only_optional_skipgram(self):
        ids = [cell["id"] for cell in self.notebook["cells"]]
        ordered_ids = [
            "06c-30", "ch03a-extensions", "06c-sg-01", "06c-sg-02",
            "06c-sg-03", "06c-sg-04", "06c-s2s-01",
        ]
        positions = [ids.index(cell_id) for cell_id in ordered_ids]
        self.assertEqual(positions, sorted(positions))
        for cell_id, section in (
            ("06c-sg-02", "4.1"), ("06c-sg-04", "4.2"),
        ):
            with self.subTest(section=section):
                self.assertTrue(self.code[cell_id].startswith(f"# %% {section} - "))
        markdown = "\n\n".join(
            "".join(cell["source"]) for cell in self.notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        self.assertNotRegex(markdown, r"(?m)^#{2,3} [AB](?:\.|\s+-)")
        self.assertNotIn("A/B", markdown)
        core_closing = ids.index("06c-30")
        prior_headings = [
            heading[1] for cell in self.notebook["cells"][:core_closing]
            if cell["cell_type"] == "markdown"
            for heading in re.finditer(r"(?m)^## ([1-5]) - ", "".join(cell["source"]))
        ]
        self.assertEqual(prior_headings[-1], "3")

    def test_static_captures_are_beside_their_core_explanations_without_rendering_code(self):
        from generate_cbow_static_views import source_fingerprint
        from PIL import Image

        ids = [cell["id"] for cell in self.notebook["cells"]]
        self.assertLess(ids.index("06c-07"), ids.index("ch03a-window-animation-intro"))
        self.assertLess(ids.index("ch03a-window-animation-intro"), ids.index("ch03a-window-storyboard-intro"))
        self.assertLess(ids.index("06c-27"), ids.index("06c-28"))
        self.assertLess(ids.index("06c-28"), ids.index("06c-30"))
        for removed in ("06c-14", "06c-15", "06c-29", "ch03a-window-animation-code"):
            self.assertNotIn(removed, ids)
        markdown = {
            cell["id"]: "".join(cell["source"]) for cell in self.notebook["cells"]
            if cell["cell_type"] == "markdown"
        }
        for cell_id, filename in (
            ("06c-28", "03a-cbow-word-profiles-before-after.png"),
            ("ch03a-window-animation-intro", "03a-cbow-window-storyboard.png"),
        ):
            self.assertIn(f"](images/{filename})", markdown[cell_id])
            self.assertTrue(markdown[cell_id].startswith("#### "))
            with Image.open(Path(__file__).parent / "images" / filename) as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreaterEqual(image.width, 1500)
                self.assertGreaterEqual(image.height, 700)
                self.assertEqual(image.info["SourceFingerprint"], source_fingerprint())
                if cell_id == "06c-28":
                    self.assertGreaterEqual(float(image.info["CosineBefore"]), -1)
                    self.assertLessEqual(float(image.info["CosineAfter"]), 1)
                else:
                    self.assertEqual(image.info["WindowCount"], "3")
                    self.assertEqual(image.info["Answers"], "cat, sat, on")
        code = "\n\n".join(self.code.values())
        self.assertNotIn("draw_cbow_window", code)
        self.assertNotIn("EmbeddingBag", code)
        self.assertNotIn("window_animation.to_jshtml", code)

    def test_preview_jump_links_distinguish_the_preview_from_the_next_chapter(self):
        markdown = {
            cell["id"]: "".join(cell["source"]) for cell in self.notebook["cells"]
            if cell["cell_type"] == "markdown"
        }
        all_markdown = "\n\n".join(markdown.values())
        anchors = re.findall(r'<a id="([^"]+)"></a>', all_markdown)
        self.assertEqual(len(anchors), len(set(anchors)))
        self.assertTrue({
            "encoder-decoder-preview", "shared-model-definitions",
            "prediction-diagnostics", "skipgram-training",
        }.issubset(anchors))
        for fragment in re.findall(r"\]\(#([^)]+)\)", all_markdown):
            self.assertIn(fragment, anchors)
        opening = markdown["ch03a-extensions"]
        self.assertLess(
            opening.index("#encoder-decoder-preview"),
            opening.index("03B-counts-and-fixed-window-language-models.ipynb"),
        )
        self.assertIn("CBOW training and section 4 are not prerequisites", markdown["06c-s2s-01"])
        self.assertIn("#skipgram-training", markdown["ch03a-real-opening"])
        for cell_id in ("06c-sg-04", "ch03a-sequence-training"):
            loads = {
                node.id for node in ast.walk(ast.parse(self.code[cell_id]))
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            }
            self.assertNotIn("criterion", loads)
            if cell_id == "ch03a-sequence-training":
                self.assertTrue({"cbow_seconds", "skipgram_seconds"}.isdisjoint(loads))

    def test_heading_numbers_follow_the_reading_order(self):
        sections = []
        subsections = {}
        section = None
        for cell in self.notebook["cells"]:
            if cell["cell_type"] != "markdown":
                continue
            for line in "".join(cell["source"]).splitlines():
                heading = re.match(r"## ([1-5]) - ", line)
                if heading:
                    section = heading[1]
                    sections.append(section)
                    subsections[section] = 0
                elif line.startswith("### "):
                    heading = re.match(r"### ([1-5])\.(\d+) - ", line)
                    self.assertIsNotNone(heading, line)
                    self.assertEqual(heading[1], section, line)
                    subsections[section] += 1
                    self.assertEqual(int(heading[2]), subsections[section], line)
        self.assertEqual(sections, list("12345"))
        self.assertEqual(subsections["3"], 4)
        self.assertEqual(subsections["4"], 2)
        self.assertEqual(subsections["5"], 12)

    def test_code_labels_match_headings_and_the_python_companion(self):
        section = None
        notebook_labels = []
        for cell in self.notebook["cells"]:
            source = "".join(cell["source"])
            if cell["cell_type"] == "markdown":
                for line in source.splitlines():
                    heading = re.match(r"#{2,3} ([1-5](?:\.\d+)?) - ", line)
                    if heading:
                        section = heading[1]
            elif cell["cell_type"] == "code":
                label = source.splitlines()[0]
                notebook_labels.append(label)
                if section is not None:
                    self.assertTrue(label.startswith(f"# %% {section} - "), label)
        self.assertEqual(
            notebook_labels,
            [line for line in self.companion.splitlines() if line.startswith("# %% ")],
        )

    def test_one_real_model_replaces_the_standalone_toy(self):
        source = "\n\n".join(self.code.values())
        self.assertNotIn("ToyCBOW", source)
        self.assertIsNone(re.search(r"\btoy_\w+", source))
        self.assertNotIn("ch03a-optional-arithmetic", [
            cell["id"] for cell in self.notebook["cells"]
        ])
        training_flow = next(
            "".join(cell["source"]) for cell in self.notebook["cells"]
            if cell["id"] == "06c-12"
        )
        self.assertEqual(re.findall(r"^([1-6])\. ", training_flow, re.MULTILINE), list("123456"))
        self.assertIn("embedding_before", self.code["ch03a-probe-baseline"])
        self.assertIn("embedding_row_shifts", self.code["06c-21"])

    def test_data_and_model_subclasses_supply_hooks_not_duplicate_pipelines(self):
        self.assertIn("CBOWDataLoader(window_ids", self.code["06c-09"])
        self.assertIn("cbow_data.prepare_data()", self.code["06c-09"])
        self.assertIn("cbow_data.make_dataloader()", self.code["06c-09"])
        self.assertIn("SkipGramDataLoader(window_ids", self.code["06c-sg-02"])
        self.assertIn("skipgram_data.prepare_data()", self.code["06c-sg-02"])
        self.assertIn("skipgram_data.make_dataloader()", self.code["06c-sg-02"])
        self.assertNotIn("torch.cat", self.code["06c-09"])
        skipgram_workflow = ast.Module(body=[
            node for node in ast.parse(self.code["06c-sg-02"]).body
            if not isinstance(node, ast.ClassDef)
        ], type_ignores=[])
        self.assertNotIn(".repeat_interleave(", ast.unparse(skipgram_workflow))
        self.assertIn("CBOWDataLoader(cat_window_ids)", self.code["ch03a-probe-baseline"])
        self.assertIn("CBOWValidator(validation_windows", self.code["ch03a-core-validation"])
        for cell_id, name in (("06c-17", "CBOW"), ("06c-sg-04", "SkipGram")):
            definition = next(
                node for node in ast.parse(self.code[cell_id]).body
                if isinstance(node, ast.ClassDef) and node.name == name
            )
            self.assertEqual(ast.unparse(definition.bases[0]), "WordEmbeddingLanguageModel")
            self.assertEqual([node.name for node in definition.body], ["combine_embeddings"])
        for cell_id in ("06c-21", "06c-sg-04", "ch03a-sequence-training"):
            self.assertIn(".train_model(", self.code[cell_id])
            self.assertNotIn(".fit_language_model(", self.code[cell_id])

    def test_text_precedes_training_and_extensions_follow_core(self):
        ids = [cell["id"] for cell in self.notebook["cells"]]
        self.assertLess(ids.index("06c-03"), ids.index("06c-17"))
        self.assertLess(ids.index("06c-17"), ids.index("06c-21"))
        self.assertLess(ids.index("ch03a-core-validation"), ids.index("06c-21"))
        self.assertLess(ids.index("06c-30"), ids.index("ch03a-extensions"))
        self.assertLess(ids.index("ch03a-extensions"), ids.index("06c-sg-04"))
        self.assertLess(ids.index("ch03a-extensions"), ids.index("06c-s2s-04"))
        self.assertNotIn("06c-13", ids)
        self.assertNotIn("read_story", self.code["ch03a-core-setup"])

    def test_section_five_preserves_the_reader_writer_and_independent_comparison(self):
        labels = [source.splitlines()[0] for source in self.code.values()]
        self.assertFalse(any(re.match(r"# %% [CD]\.", label) for label in labels))
        for section in range(1, 13):
            self.assertTrue(any(label.startswith(f"# %% 5.{section} - ") for label in labels))
        ids = [cell["id"] for cell in self.notebook["cells"]]
        self.assertLess(ids.index("06c-s2s-04"), ids.index("ch03a-sequence-training"))
        self.assertLess(ids.index("ch03a-sequence-training"), ids.index("06c-s2s-06"))
        self.assertLess(ids.index("06c-s2s-06"), ids.index("ch03a-real-training"))
        self.assertIn("cbow_learning_history", self.code["06c-21"])
        self.assertIn("sequence_learning_history", self.code["ch03a-sequence-training"])
        self.assertIn("sequence_embedding_before", self.code["ch03a-sequence-training"])
        self.assertIn("train_comparison(", self.code["ch03a-real-training"])

    def test_reader_writer_preparation_is_setup_not_a_separate_subsection(self):
        introduction = next(
            "".join(cell["source"]) for cell in self.notebook["cells"]
            if cell["id"] == "06c-s2s-01"
        )
        first_subsection = next(
            "".join(cell["source"]) for cell in self.notebook["cells"]
            if cell["id"] == "ch03a-preview-boundary"
        )
        self.assertTrue(introduction.startswith("## 5 - "))
        self.assertNotIn("### ", introduction)
        self.assertNotIn("Align the source and answer", introduction)
        self.assertTrue(first_subsection.startswith("### 5.1 - GRU reader and writer"))
        self.assertTrue(self.code["06c-s2s-02"].startswith("# %% 5 - "))
        for setup in ("source_sequences, sequence_targets =", "prepare_sequence_data(", "sequence_dataloader ="):
            self.assertIn(setup, self.code["06c-s2s-02"])
        ids = [cell["id"] for cell in self.notebook["cells"]]
        self.assertLess(ids.index("06c-s2s-02"), ids.index("ch03a-preview-boundary"))
        self.assertLess(ids.index("ch03a-preview-boundary"), ids.index("06c-s2s-04"))

    def test_teacher_forcing_is_explained_before_training_and_metrics_are_not_conflated(self):
        markdown = {cell["id"]: "".join(cell["source"]) for cell in self.notebook["cells"]
                    if cell["cell_type"] == "markdown"}
        intro = markdown["ch03a-sequence-training-intro"]
        for idea in (
            "correct previous word", "teacher forcing", "not the word currently being guessed",
            "<bos>", "<eos>", "sat", "on", "and", "hidden state", "six epochs",
            "03a-correct-previous-word-vs-own-guess.png",
            "03a-correct-previous-word-vs-own-guess.gif",
        ):
            self.assertIn(idea, " ".join(intro.split()))
        self.assertIn("one word", intro)
        readout = markdown["ch03a-sequence-learning-readout"]
        for idea in ("Training accuracy", "Reserved-story validation", "correct previous words",
                     "not free-running generation", "stop", "row change"):
            self.assertIn(idea, " ".join(readout.split()))
        self.assertIn("same trained weights", markdown["06c-s2s-05"])

    def test_setup_diagnostics_and_familiar_questions_have_separate_cells(self):
        ids = [cell["id"] for cell in self.notebook["cells"]]
        ordered_ids = [
            "06c-19", "ch03a-probe-helpers", "ch03a-probe-baseline",
            "ch03a-routing-intro", "06c-21",
        ]
        positions = [ids.index(cell_id) for cell_id in ordered_ids]
        self.assertEqual(positions, sorted(positions))
        setup = ast.parse(self.code["06c-19"])
        helpers = ast.parse(self.code["ch03a-probe-helpers"])
        baseline = ast.parse(self.code["ch03a-probe-baseline"])
        self.assertEqual(len(helpers.body), 1)
        self.assertIsInstance(helpers.body[0], ast.ImportFrom)
        self.assertEqual(helpers.body[0].module, "language_validation")
        self.assertEqual([alias.name for alias in helpers.body[0].names], ["ModelProbeRecorder", "word_probe"])
        self.assertNotIn("cat_contexts", self.code["06c-19"])
        self.assertNotIn("model = CBOW(", self.code["ch03a-probe-baseline"])
        for module in (setup, helpers, baseline):
            training_calls = [
                node.func.attr for node in ast.walk(module)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"backward", "step", "fit_language_model"}
            ]
            self.assertEqual(training_calls, [])

    def test_routing_animation_is_embedded_without_regeneration_cells(self):
        markdown = "\n\n".join(
            "".join(cell["source"]) for cell in self.notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        self.assertRegex(
            markdown, r"!\[[^\]]+\]\(images/03a-embedding-forward-backward\.gif\)",
        )
        self.assertTrue(
            (Path(__file__).parent / "images" / "03a-embedding-forward-backward.gif").is_file(),
        )
        self.assertIn("images/03a-embedding-forward-backward.png", markdown)
        self.assertNotIn("routing_question_index", markdown)
        self.assertNotIn("ch03a-routing-record", self.code)
        self.assertNotIn("ch03a-routing-visuals", self.code)
        for source in ("\n\n".join(self.code.values()), self.companion):
            with self.subTest(source="notebook" if source != self.companion else "companion"):
                for helper in (
                    "record_embedding_routing", "plot_embedding_routing", "animate_embedding_routing",
                ):
                    self.assertNotIn(helper, source)
                self.assertIsNone(re.search(r"\brouting_\w+", source))

    def test_familiar_baseline_records_three_questions_without_training(self):
        definitions = [
            node for node in ast.parse(self.code["06c-17"]).body
            if isinstance(node, ast.ClassDef) and node.name == "CBOW"
        ]
        namespace = {"nn": torch.nn, "TrainableLanguageModel": TrainableLanguageModel,
                     "WordEmbeddingLanguageModel": WordEmbeddingLanguageModel,
                     "CBOWDataLoader": CBOWDataLoader}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "CBOW", "exec"), namespace)
        expected_contexts = torch.tensor([[1, 2, 4, 5], [2, 3, 5, 1], [3, 4, 1, 6]])
        namespace.update({
            "torch": torch, "SEED": 17, "EMBEDDING_WIDTH": 4,
            "vocabulary_size": 7, "itos": ["<unk>", "the", "sleepy", "cat", "sat", "on", "mat"],
            "cat_window_ids": torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 1], [3, 4, 5, 1, 6]]),
            "cbow_batch_contexts": expected_contexts[:2],
        })
        with torch.random.fork_rng(devices=[]), redirect_stdout(StringIO()):
            exec(compile(self.code["06c-19"], "setup", "exec"), namespace)
            parameters_before = {
                name: value.detach().clone() for name, value in namespace["model"].named_parameters()
            }
            rng_before = torch.random.get_rng_state().clone()
            exec(compile(self.code["ch03a-probe-helpers"], "helpers", "exec"), namespace)
            exec(compile(self.code["ch03a-probe-baseline"], "baseline", "exec"), namespace)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        self.assertTrue(torch.equal(namespace["cat_contexts"], expected_contexts))
        self.assertTrue(torch.equal(namespace["cat_contexts_before"], expected_contexts))
        self.assertEqual(tuple(namespace["cat_before_logits"].shape), (3, 7))
        self.assertEqual(tuple(namespace["untrained_logits"].shape), (2, 7))
        self.assertEqual(list(namespace["cbow_before"]), ["cat", "sat", "on"])
        for row, word in enumerate(("cat", "sat", "on")):
            chances = namespace["cat_before_logits"][row].softmax(dim=-1)
            word_id = namespace["itos"].index(word)
            self.assertEqual(namespace["cbow_before"][word]["probability"], chances[word_id].item())
            self.assertEqual(
                namespace["cbow_before"][word]["rank"], int((chances > chances[word_id]).sum()) + 1,
            )
        for name, value in namespace["model"].named_parameters():
            self.assertTrue(torch.equal(value.detach(), parameters_before[name]), name)
            self.assertIsNone(value.grad)

    def test_notebook_and_companion_share_all_executable_code(self):
        notebook_source = "\n\n".join(self.code.values())
        companion = ast.parse(self.companion)
        self.assertIsInstance(companion.body[0], ast.Expr)
        companion.body = companion.body[1:]
        self.assertEqual(ast.dump(ast.parse(notebook_source)), ast.dump(companion))

    def test_notebook_and_companion_leave_assertions_to_offline_tests(self):
        for source in ("\n\n".join(self.code.values()), self.companion):
            self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(ast.parse(source))))

    def test_core_helper_definitions_are_inline_and_match_their_python_modules(self):
        notebook_nodes = {
            node.name: node for node in ast.parse("\n\n".join(self.code.values())).body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        }
        for filename, names in (
            ("language_data.py", [
                "WindowDataLoader", "CBOWDataLoader",
                "SkipGramDataLoader", "prepare_sequence_data",
            ]),
            ("language_model.py", [
                "TrainingRun", "TrainableLanguageModel", "WordEmbeddingLanguageModel",
            ]),
        ):
            module_nodes = {
                node.name: node for node in ast.parse(
                    (Path(__file__).parent / filename).read_text(encoding="utf-8")
                ).body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            }
            for name in names:
                with self.subTest(helper=name):
                    self.assertIn(name, notebook_nodes)
                    self.assertEqual(ast.dump(notebook_nodes[name]), ast.dump(module_nodes[name]))
        self.assertNotIn("heldout_cbow_loss", notebook_nodes)
        self.assertNotIn("ModelValidator", notebook_nodes)
        self.assertNotIn("make_dataloader", notebook_nodes)
        self.assertNotIn("validate_window_ids", notebook_nodes)
        self.assertNotIn("word_probe", notebook_nodes)
        self.assertNotIn("report_probe", notebook_nodes)
        self.assertNotIn("ModelProbeRecorder", notebook_nodes)

    def test_prediction_probes_are_imported_directly_without_notebook_globals(self):
        namespace = {}
        exec(compile(self.code["ch03a-probe-helpers"], "probe-imports", "exec"), namespace)
        self.assertIs(namespace["word_probe"], word_probe)
        self.assertIs(namespace["ModelProbeRecorder"], ModelProbeRecorder)
        for cell in self.notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            for node in ast.walk(ast.parse("".join(cell["source"]))):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "word_probe":
                    self.assertTrue(len(node.args) >= 3 or any(
                        keyword.arg == "words" for keyword in node.keywords
                    ))

    def test_before_after_recording_is_external_while_epoch_callbacks_stay_inline(self):
        self.assertIn("cbow_recorder.record_before(", self.code["ch03a-probe-baseline"])
        self.assertNotIn("with torch.inference_mode", self.code["ch03a-probe-baseline"])
        for cell_id, recorder, callback in (
            ("06c-21", "cbow_recorder", "record_cbow_epoch"),
            ("ch03a-sequence-training", "sequence_recorder", "record_sequence_epoch"),
        ):
            self.assertIn(f"{recorder}.record_after()", self.code[cell_id])
            self.assertIn(f"{recorder}.report_probes()", self.code[cell_id])
            self.assertIn(f"{recorder}.report_embeddings(", self.code[cell_id])
            self.assertTrue(any(
                isinstance(node, ast.FunctionDef) and node.name == callback
                for node in ast.parse(self.code[cell_id]).body
            ))

    def test_each_training_section_is_followed_by_its_validation_comparison(self):
        ids = [cell["id"] for cell in self.notebook["cells"]]
        for training, comparison, validator in (
            ("06c-21", "ch03a-cbow-validation-comparison", "cbow_validator"),
            ("06c-sg-04", "ch03a-skipgram-validation-comparison", "skipgram_validator"),
            ("ch03a-sequence-training", "ch03a-sequence-validation-comparison", "sequence_validator"),
        ):
            self.assertEqual(ids.index(comparison), ids.index(training) + 1)
            self.assertIn(f"{validator}.record_before(", self.code[training])
            self.assertIn(f"{validator}.compare_after(", self.code[comparison])
            self.assertIn("epochs=len(", self.code[comparison])

    def test_reader_writer_bridge_explains_the_internals_needed_for_embedding_learning(self):
        explanation = next(
            "".join(cell["source"]) for cell in self.notebook["cells"]
            if cell["id"] == "ch03a-preview-boundary"
        )
        for idea in (
            "GRU mechanics", "live cat probe", "Source word profile",
            "Previous output word profile", "E3 initializes decoder D0",
            "temporary values", "shared embedding rows", "cat",
            "03a-encoder-decoder-embedding-feedback.png",
            "03a-encoder-decoder-state-flow.gif", "03a-encoder-decoder-state-flow.html",
            "model's own guess during generation",
            "same values", "Linear", "one GRU layer", "03a-encoder-decoder-token-layers.png",
        ):
            with self.subTest(idea=idea):
                self.assertIn(idea, " ".join(explanation.split()))
        self.assertIn("compare_feedback_routes(", self.code["ch03a-sequence-training"])
        self.assertRegex(explanation, r"!\[[^\]]+\]\(images/03a-encoder-decoder-state-flow\.gif\)")
        self.assertIn("<details>", explanation)
        self.assertLessEqual(len(explanation.split()), 350)
        self.assertNotIn("reset gate r", explanation)

    def test_prior_memory_references_resolve_to_the_sections_actually_taught(self):
        explanation = next(
            "".join(cell["source"]) for cell in self.notebook["cells"]
            if cell["id"] == "ch03a-preview-boundary"
        )
        earlier_path = Path(__file__).parent.parent / "01-sequence-memory" / "01A-sequence-memory-theory.ipynb"
        earlier = json.loads(earlier_path.read_text(encoding="utf-8"))
        headings = {
            re.sub(r"[^\w\s-]", "", line.lstrip("#").strip().lower()).replace(" ", "-")
            for cell in earlier["cells"] if cell["cell_type"] == "markdown"
            for line in "".join(cell["source"]).splitlines() if line.startswith("#")
        }
        headings.update(
            anchor for cell in earlier["cells"] if cell["cell_type"] == "markdown"
            for anchor in re.findall(r'<a id="([^"]+)"></a>', "".join(cell["source"]))
        )
        fragments = re.findall(r"01A-sequence-memory-theory\.ipynb#([^)]+)", explanation)
        self.assertEqual(set(fragments), {"gru-hidden-state-mechanics"})
        self.assertIn("gru-hidden-state-mechanics", fragments)
        for fragment in fragments:
            self.assertIn(fragment, headings)
        prior_code = "\n\n".join(
            "".join(cell["source"]) for cell in earlier["cells"] if cell["cell_type"] == "code"
        )
        self.assertIn("state = torch.tanh(W_x @ x_t + W_h @ state)", prior_code)
        self.assertIn("GRU mechanics", explanation)
        self.assertNotIn("not GRU mechanics", explanation)

    def test_reserved_stories_exclude_training_and_reuse_the_training_vocabulary(self):
        stories = ["training words must not appear"] * 3 + ["the sleepy cat sat on"] * 200
        with patch("builtins.print"), patch(
            "real_text_embeddings.read_stories", return_value=stories,
        ) as read_stories:
            namespace = {
                "INTRO_STORIES": 3,
                "stoi": {"<unk>": 0, "the": 1, "sleepy": 2, "cat": 3, "sat": 4, "on": 5},
                "itos": ["<unk>", "the", "sleepy", "cat", "sat", "on"],
                "window_width": 5, "prepare_validation_windows": prepare_validation_windows,
            }
            setup = [
                node for node in ast.parse(self.code["06c-09"]).body
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id in {"VALIDATION_STORIES", "validation_windows"}
                    for target in node.targets
                )
            ]
            exec(compile(ast.Module(body=setup, type_ignores=[]), "reserved-windows", "exec"), namespace)
            exec(compile(self.code["ch03a-core-validation"], "validation", "exec"), namespace)
        read_stories.assert_called_once_with(203)
        expected = torch.tensor([[1, 2, 3, 4, 5]]).repeat(200, 1)
        self.assertTrue(torch.equal(namespace["validation_windows"], expected))
        self.assertTrue(torch.equal(namespace["cbow_validator"].input_ids, expected[:, [0, 1, 3, 4]]))
        self.assertTrue(torch.equal(namespace["cbow_validator"].target_ids, expected[:, 2]))
        self.assertNotIn("training", namespace["stoi"])

    def test_heldout_loss_weights_the_partial_batch_and_updates_no_parameters(self):
        class FixedScores(TrainableLanguageModel):
            def __init__(self):
                super().__init__()
                self.scores = torch.nn.Parameter(torch.tensor([0.1, 0.3, -0.5, 0.8, 0.0, -0.2]))

            def forward(self, inputs, target_ids=None):
                return self.scores.expand(len(inputs), -1)

        model = FixedScores()
        contexts = torch.zeros((767, 4), dtype=torch.long)
        targets = torch.arange(767) % 6
        validator = ModelValidator(contexts, targets)
        before = model.scores.detach().clone()
        expected = torch.nn.functional.cross_entropy(
            model(contexts).detach(), targets,
        ).item()
        actual = validator.evaluate(model).loss
        self.assertAlmostEqual(actual, expected, places=6)
        self.assertTrue(torch.equal(model.scores.detach(), before))
        self.assertIsNone(model.scores.grad)

    def test_heldout_loss_rejects_an_empty_evaluation_set(self):
        with self.assertRaisesRegex(ValueError, "nonzero"):
            ModelValidator(torch.empty((0, 4), dtype=torch.long), torch.empty(0, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()

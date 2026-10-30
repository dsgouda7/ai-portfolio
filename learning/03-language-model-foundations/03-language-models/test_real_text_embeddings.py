"""Offline contracts for the shared windows and their objective-specific pairs."""

import ast
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
import math
from pathlib import Path
import re
import sys
from types import ModuleType
import time
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import torch

from language_model import TrainableLanguageModel
from real_text_embeddings import (
    animate_embedding_routing,
    draw_embedding_routing,
    embedding_routing_steps,
    make_document_windows,
    objective_pairs,
    plot_embedding_routing,
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
        namespace = {"nn": torch.nn, "TrainableLanguageModel": TrainableLanguageModel}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "CBOW", "exec"), namespace)
        cls.model_type = namespace["CBOW"]
        cls.routing_source = next(
            "".join(cell["source"]) for cell in notebook["cells"]
            if cell["id"] == "ch03a-routing-record"
        )

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

    def _record_routing(self):
        namespace = {
            "torch": torch, "nn": torch.nn, "model": self.model,
            "cat_contexts": self.ids,
            "cat_window_ids": torch.tensor([[0, 2, 1, 4, 6], [6, 4, 1, 2, 0]]),
            "CONTEXT_RADIUS": 2,
            "itos": ["the", "cat", "sleepy", "dog", "sat", "mat", "on"],
        }
        with redirect_stdout(StringIO()):
            exec(compile(self.routing_source, "routing", "exec"), namespace)
        return namespace

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
        state = self._record_routing()
        trace = state["routing_trace"]
        self.assertEqual(state["routing_gradient_rows"], {0, 2, 4, 6})
        self.assertEqual(state["routing_changed_rows"], {0, 2, 4, 6})
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

    def test_recorder_refreshes_helpers_cached_before_the_notebook_was_updated(self):
        self._activate_features()
        for missing_helper in (True, False):
            with self.subTest(missing_helper=missing_helper):
                cached = ModuleType("real_text_embeddings")
                cached.__file__ = str(Path(__file__).with_name("real_text_embeddings.py"))
                if not missing_helper:
                    cached.record_embedding_routing = lambda: None
                original_module = sys.modules["real_text_embeddings"]
                try:
                    sys.modules["real_text_embeddings"] = cached
                    if missing_helper:
                        with self.assertRaisesRegex(ImportError, "record_embedding_routing"):
                            exec("from real_text_embeddings import record_embedding_routing", {})
                    state = self._record_routing()
                    self.assertTrue(callable(cached.animate_embedding_routing))
                finally:
                    sys.modules["real_text_embeddings"] = original_module
                self.assertEqual(state["routing_gradient_rows"], {0, 2, 4, 6})
                self.assertEqual(state["routing_changed_rows"], {0, 2, 4, 6})

    def test_checking_answer_does_not_fetch_its_input_row_but_head_receives_feedback(self):
        self._activate_features()
        state = self._record_routing()
        self.assertEqual(state["routing_targets"].tolist(), [1])
        self.assertNotIn(1, state["routing_selected"])
        self.assertEqual(state["routing_gradient"][1].count_nonzero().item(), 0)
        self.assertNotEqual(state["routing_head_gradient"][1].item(), 0)
        self.assertEqual(state["routing_head_gradient"].count_nonzero().item(), 7)

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
        trace = self._record_routing()["routing_trace"]
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
        trace = self._record_routing()["routing_trace"]
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
        state = self._record_routing()
        trace = state["routing_trace"]
        figure = plot_embedding_routing(trace, state["itos"])
        try:
            self.assertEqual(len(figure.axes), 1)
            axis = figure.axes[0]
            draw_embedding_routing(axis, trace, state["itos"], stage=5, progress=0.5)
            texts = [artist.get_text() for artist in axis.texts]
            self.assertIn(f"{trace['loss']:.4f}", texts)
            widths = [patch.get_width() for patch in axis.patches[:6]]
            self.assertEqual(widths.index(max(widths)), 5)
            draw_embedding_routing(axis, trace, state["itos"], stage=10)
            texts = [artist.get_text() for artist in axis.texts]
            values = trace["activation_gradients"]["lookup"][0, 0, :3]
            expected = "[" + ", ".join(f"{value:.3g}" for value in values.tolist()) + "]"
            self.assertIn(expected, texts)
            self.assertIn("each occurrence receives gradient / 4", texts)
            draw_embedding_routing(axis, trace, state["itos"], stage=12)
            texts = [artist.get_text() for artist in axis.texts]
            values = trace["weights"][3][state["routing_contexts"][0, 0], :3]
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
        namespace = {"nn": torch.nn, "TrainableLanguageModel": TrainableLanguageModel}
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

        def heldout(candidate):
            candidate.eval()
            with torch.inference_mode():
                return torch.nn.functional.cross_entropy(candidate(contexts), answers).item()

        state.update({
            "model": model, "embedding_before": model.embedding.weight.detach().clone(),
            "cbow_dataloader": self.loader(contexts, answers), "heldout_cbow_loss": heldout,
        })
        with torch.inference_mode():
            logits = model(contexts)
            state["cbow_before"] = {
                word: state["word_probe"](logits[row], word)
                for row, word in enumerate(("cat", "sat", "on"))
            }
        with patch.object(model, "fit_language_model", wraps=model.fit_language_model) as fit:
            with redirect_stdout(StringIO()):
                exec(compile(self.code["06c-21"], "cbow-learning", "exec"), state)
            self.assertEqual(fit.call_count, 10)
            for call in fit.call_args_list:
                self.assertIs(call.args[0], state["cbow_dataloader"])
                self.assertIs(call.args[2], state["optimizer"])
                self.assertEqual(call.kwargs, {"epochs": 1})
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
            "criterion": torch.nn.CrossEntropyLoss(), "cbow_seconds": 0.0, "skipgram_seconds": 0.0,
        })
        original_fit = TrainableLanguageModel.fit_language_model
        with torch.random.fork_rng(devices=[]), patch.object(
            TrainableLanguageModel, "fit_language_model", autospec=True, side_effect=original_fit,
        ) as fit:
            with redirect_stdout(StringIO()):
                exec(compile(self.code["ch03a-sequence-training"], "sequence-learning", "exec"), state)
            self.assertEqual(fit.call_count, 6)
            for call in fit.call_args_list:
                self.assertIs(call.args[0], state["sequence_model"])
                self.assertIs(call.args[3], state["sequence_optimizer"])
                self.assertEqual(call.kwargs, {"epochs": 1})
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
            "The model produces vocabulary scores.",
            "Choosing the highest score gives its best guess",
            "sampling is an optional decoding strategy for later text-generation lessons",
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
        self.assertIn("report_probe", self.code["06c-21"])
        self.assertIn("heldout_cbow_loss(model)", self.code["06c-21"])
        self.assertIn("embedding_row_shifts", self.code["06c-21"])

    def test_order_limit_is_plain_english_and_optional_a_sections_stay_available(self):
        explanation = next(
            "".join(cell["source"]) for cell in self.notebook["cells"]
            if cell["id"] == "06c-30"
        )
        self.assertLess(len(explanation.split()), 220)
        self.assertIn("the cat ___ the dog", explanation)
        self.assertIn("the dog ___ the cat", explanation)
        self.assertIn("same word scores and probabilities", explanation)
        self.assertIn("learn useful word relationships", explanation)
        self.assertNotIn("06c-31", self.code)
        self.assertNotIn("permuted_context", self.companion)
        self.assertIn("nn.EmbeddingBag", self.code["06c-15"])
        self.assertIn("torch.linalg.svd", self.code["06c-29"])
        self.assertIn("FuncAnimation", self.code["ch03a-window-animation-code"])

    def test_heading_numbers_follow_the_reading_order(self):
        sections = []
        subsections = {}
        section = None
        for cell in self.notebook["cells"]:
            if cell["cell_type"] != "markdown":
                continue
            for line in "".join(cell["source"]).splitlines():
                heading = re.match(r"## ([1-6AB]) - ", line)
                if heading:
                    section = heading[1]
                    sections.append(section)
                    subsections[section] = 0
                elif line.startswith("### "):
                    heading = re.match(r"### ([1-6AB])\.(\d+) - ", line)
                    self.assertIsNotNone(heading, line)
                    self.assertEqual(heading[1], section, line)
                    subsections[section] += 1
                    self.assertEqual(int(heading[2]), subsections[section], line)
        self.assertEqual(sections, list("12345AB6"))

    def test_code_labels_match_headings_and_the_python_companion(self):
        section = None
        notebook_labels = []
        for cell in self.notebook["cells"]:
            source = "".join(cell["source"])
            if cell["cell_type"] == "markdown":
                for line in source.splitlines():
                    heading = re.match(r"#{2,3} ([1-6AB](?:\.\d+)?) - ", line)
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
        self.assertIn("embedding_before", self.code["06c-19"])
        self.assertIn("embedding_row_shifts", self.code["06c-21"])

    def test_text_precedes_training_and_extensions_follow_core(self):
        ids = [cell["id"] for cell in self.notebook["cells"]]
        self.assertLess(ids.index("06c-03"), ids.index("06c-17"))
        self.assertLess(ids.index("06c-17"), ids.index("06c-21"))
        self.assertLess(ids.index("ch03a-core-validation"), ids.index("06c-21"))
        self.assertLess(ids.index("ch03a-next"), ids.index("ch03a-extensions"))
        self.assertLess(ids.index("ch03a-extensions"), ids.index("06c-sg-04"))
        self.assertLess(ids.index("ch03a-extensions"), ids.index("06c-s2s-04"))
        self.assertLess(ids.index("ch03a-extensions"), ids.index("06c-15"))
        self.assertNotIn("06c-13", ids)
        self.assertNotIn("read_story", self.code["ch03a-core-setup"])

    def test_section_six_preserves_the_reader_writer_and_independent_comparison(self):
        labels = [source.splitlines()[0] for source in self.code.values()]
        self.assertFalse(any(re.match(r"# %% [CD]\.", label) for label in labels))
        for section in range(1, 14):
            self.assertTrue(any(label.startswith(f"# %% 6.{section} - ") for label in labels))
        ids = [cell["id"] for cell in self.notebook["cells"]]
        self.assertLess(ids.index("06c-s2s-04"), ids.index("ch03a-sequence-training"))
        self.assertLess(ids.index("ch03a-sequence-training"), ids.index("06c-s2s-06"))
        self.assertLess(ids.index("06c-s2s-06"), ids.index("ch03a-real-training"))
        self.assertIn("cbow_learning_history", self.code["06c-21"])
        self.assertIn("sequence_learning_history", self.code["ch03a-sequence-training"])
        self.assertIn("sequence_embedding_before", self.code["ch03a-sequence-training"])
        self.assertIn("train_comparison(", self.code["ch03a-real-training"])

    def test_setup_diagnostics_and_familiar_questions_have_separate_cells(self):
        ids = [cell["id"] for cell in self.notebook["cells"]]
        ordered_ids = [
            "06c-19", "ch03a-probe-helpers", "ch03a-probe-baseline",
            "ch03a-routing-record", "06c-21",
        ]
        positions = [ids.index(cell_id) for cell_id in ordered_ids]
        self.assertEqual(positions, sorted(positions))
        setup = ast.parse(self.code["06c-19"])
        helpers = ast.parse(self.code["ch03a-probe-helpers"])
        baseline = ast.parse(self.code["ch03a-probe-baseline"])
        self.assertEqual([node.name for node in helpers.body], ["word_probe", "report_probe"])
        self.assertNotIn("cat_contexts", self.code["06c-19"])
        self.assertNotIn("model = CBOW(", self.code["ch03a-probe-baseline"])
        for module in (setup, helpers, baseline):
            training_calls = [
                node.func.attr for node in ast.walk(module)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"backward", "step", "fit_language_model"}
            ]
            self.assertEqual(training_calls, [])

    def test_familiar_baseline_records_three_questions_without_training(self):
        definitions = [
            node for node in ast.parse(self.code["06c-17"]).body
            if isinstance(node, ast.ClassDef) and node.name == "CBOW"
        ]
        namespace = {"nn": torch.nn, "TrainableLanguageModel": TrainableLanguageModel}
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

    def test_reserved_stories_exclude_training_and_reuse_the_training_vocabulary(self):
        stories = ["training words must not appear"] * 3 + ["the sleepy cat sat on"] * 200
        with patch("builtins.print"), patch(
            "real_text_embeddings.read_stories", return_value=stories,
        ) as read_stories:
            namespace = {
                "torch": torch, "nn": torch.nn, "INTRO_STORIES": 3,
                "read_stories": read_stories,
                "story_sentences": lambda story: [story.split()],
                "stoi": {"<unk>": 0, "the": 1, "sleepy": 2, "cat": 3, "sat": 4, "on": 5},
                "make_document_windows": make_document_windows,
                "window_width": 5, "CONTEXT_RADIUS": 2,
            }
            exec(compile(self.code["ch03a-core-validation"], "validation", "exec"), namespace)
        read_stories.assert_called_once_with(203)
        expected = torch.tensor([[1, 2, 3, 4, 5]]).repeat(200, 1)
        self.assertTrue(torch.equal(namespace["validation_windows"], expected))
        self.assertTrue(torch.equal(namespace["validation_contexts"], expected[:, [0, 1, 3, 4]]))
        self.assertTrue(torch.equal(namespace["validation_targets"], expected[:, 2]))
        self.assertNotIn("training", namespace["stoi"])

    def test_heldout_loss_weights_the_partial_batch_and_updates_no_parameters(self):
        class FixedScores(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scores = torch.nn.Parameter(torch.tensor([0.1, 0.3, -0.5, 0.8, 0.0, -0.2]))

            def forward(self, inputs):
                return self.scores.expand(len(inputs), -1)

        model = FixedScores()
        contexts = torch.zeros((767, 4), dtype=torch.long)
        targets = torch.arange(767) % 6
        namespace = self.evaluation_namespace(contexts, targets)
        before = model.scores.detach().clone()
        expected = torch.nn.functional.cross_entropy(
            model(contexts).detach(), targets,
        ).item()
        actual = namespace["heldout_cbow_loss"](model)
        self.assertAlmostEqual(actual, expected, places=6)
        self.assertTrue(torch.equal(model.scores.detach(), before))
        self.assertIsNone(model.scores.grad)

    def test_heldout_loss_rejects_an_empty_evaluation_set(self):
        namespace = self.evaluation_namespace(
            torch.empty((0, 4), dtype=torch.long), torch.empty(0, dtype=torch.long),
        )
        with self.assertRaisesRegex(ValueError, "at least one target"):
            namespace["heldout_cbow_loss"](torch.nn.Identity())

    def evaluation_namespace(self, contexts, targets):
        functions = [
            node for node in ast.parse(self.code["ch03a-core-validation"]).body
            if isinstance(node, ast.FunctionDef) and node.name == "heldout_cbow_loss"
        ]
        self.assertEqual(len(functions), 1)
        namespace = {
            "torch": torch, "nn": torch.nn,
            "validation_contexts": contexts, "validation_targets": targets,
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), "heldout", "exec"), namespace)
        return namespace


if __name__ == "__main__":
    unittest.main()

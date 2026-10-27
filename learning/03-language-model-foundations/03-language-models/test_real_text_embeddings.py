"""Offline contracts for the shared windows and their objective-specific pairs."""

import ast
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from real_text_embeddings import make_document_windows, objective_pairs


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


class ToyEmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        chapter = Path(__file__).resolve().parent
        stem = "03A-training-skeleton-and-word-objectives"
        notebook = json.loads((chapter / f"{stem}.ipynb").read_text(encoding="utf-8"))
        companion = (chapter / f"{stem}.py").read_text(encoding="utf-8")
        start = companion.index("# %% Toy 1 - ")
        stop = companion.index("# %% CBOW: give every context word an equal vote", start)
        cls.sources = {
            "notebook": "\n\n".join(
                "".join(cell["source"]) for cell in notebook["cells"]
                if cell["cell_type"] == "code" and (
                    "".join(cell["source"]).startswith("# %% Toy ")
                    or "".join(cell["source"]).startswith("# %% Your turn - look up a different token")
                )
            ),
            "companion": companion[start:stop],
        }
        cls.states = {}
        cls.rng_unchanged = {}
        for name, source in cls.sources.items():
            namespace = {"torch": torch, "nn": torch.nn, "SEED": 17}
            before_rng = torch.random.get_rng_state().clone()
            with redirect_stdout(StringIO()):
                exec(compile(source, name, "exec"), namespace)
            cls.states[name] = namespace
            cls.rng_unchanged[name] = torch.equal(before_rng, torch.random.get_rng_state())

    def test_notebook_and_companion_have_identical_toy_computation(self):
        self.assertEqual(
            ast.dump(ast.parse(self.sources["notebook"])),
            ast.dump(ast.parse(self.sources["companion"])),
        )

    def test_dimensions_create_a_random_trainable_parameter(self):
        for name, state in self.states.items():
            with self.subTest(source=name):
                embedding = state["toy_embedding"]
                self.assertIsInstance(embedding, torch.nn.Embedding)
                self.assertIsInstance(embedding.weight, torch.nn.Parameter)
                self.assertTrue(embedding.weight.requires_grad)
                self.assertEqual(tuple(embedding.weight.shape), (3, 2))
                torch.testing.assert_close(
                    state["toy_initial_weights"], state["toy_manual_initialization"],
                    rtol=0, atol=0,
                )
                self.assertTrue(self.rng_unchanged[name])

    def test_lookup_preserves_axes_and_mean_erases_only_word_order(self):
        for name, state in self.states.items():
            with self.subTest(source=name):
                self.assertEqual(tuple(state["toy_rows"].shape), (2, 2, 2))
                self.assertEqual(tuple(state["toy_averages"].shape), (2, 2))
                self.assertFalse(state["toy_ids"].requires_grad)
                torch.testing.assert_close(
                    state["toy_rows"].detach(),
                    state["toy_initial_weights"][state["toy_ids"]],
                    rtol=0, atol=0,
                )
                torch.testing.assert_close(
                    state["toy_averages"][0], state["toy_averages"][1], rtol=0, atol=0,
                )
                torch.testing.assert_close(
                    state["toy_final_rows"], state["toy_final_weights"][state["toy_ids"]],
                    rtol=0, atol=0,
                )

    def test_model_owns_the_same_embedding_and_forward_accepts_only_ids(self):
        for name, state in self.states.items():
            with self.subTest(source=name):
                model = state["toy_model"]
                self.assertIsInstance(model, torch.nn.Module)
                self.assertIs(model.embedding, state["toy_embedding"])
                self.assertEqual(
                    set(state["toy_registered_parameters"]), {"embedding.weight", "head.weight"},
                )
                self.assertIs(
                    state["toy_registered_parameters"]["embedding.weight"],
                    state["toy_embedding"].weight,
                )
                self.assertIs(model._modules["embedding"], state["toy_embedding"])
                self.assertIs(
                    state["toy_embedding"]._parameters["weight"], state["toy_embedding"].weight,
                )
                self.assertEqual(list(model.named_parameters(recurse=False)), [])
                self.assertNotIn("lookup_vectors", dict(model.named_parameters()))
                with torch.no_grad():
                    torch.testing.assert_close(
                        model(state["toy_ids"]),
                        model.head(model.embedding(state["toy_ids"]).mean(dim=1)),
                        rtol=0, atol=0,
                    )

    def test_backward_accumulates_occurrence_gradients_into_the_selected_rows(self):
        for name, state in self.states.items():
            with self.subTest(source=name):
                gradients = state["toy_first_vector_gradient"]
                self.assertEqual(tuple(gradients.shape), (2, 2, 2))
                torch.testing.assert_close(
                    state["toy_first_gradient"][0], gradients[0, 0] + gradients[1, 1],
                )
                torch.testing.assert_close(
                    state["toy_first_gradient"][2], gradients[0, 1] + gradients[1, 0],
                )
                torch.testing.assert_close(
                    state["toy_scatter_gradient"], state["toy_first_gradient"],
                )
                torch.testing.assert_close(
                    state["toy_first_forward_vectors"],
                    state["toy_initial_weights"][state["toy_ids"]],
                    rtol=0, atol=0,
                )

    def test_plain_sgd_applies_the_recorded_gradient_to_the_same_coordinates(self):
        for name, state in self.states.items():
            with self.subTest(source=name):
                gradient = state["toy_first_gradient"]
                self.assertEqual(tuple(gradient.shape), (3, 2))
                self.assertEqual(torch.count_nonzero(gradient[1]).item(), 0)
                self.assertTrue(torch.all(gradient[[0, 2]].norm(dim=1) > 0).item())
                torch.testing.assert_close(
                    state["toy_after_first_step"],
                    state["toy_initial_weights"] - state["toy_learning_rate"] * gradient,
                )
                self.assertEqual(
                    {id(parameter) for group in state["toy_optimizer"].param_groups
                     for parameter in group["params"]},
                    {id(state["toy_embedding"].weight), id(state["toy_model"].head.weight)},
                )

    def test_training_changes_used_rows_and_head_without_changing_ids(self):
        for name, state in self.states.items():
            with self.subTest(source=name):
                self.assertEqual(state["toy_training_steps"], 40)
                self.assertLess(state["toy_loss_after"], state["toy_loss_before"])
                self.assertGreater(state["toy_final_logits"][0].softmax(dim=-1)[1].item(), 0.9)
                torch.testing.assert_close(
                    state["toy_final_weights"][1], state["toy_initial_weights"][1],
                    rtol=0, atol=0,
                )
                self.assertTrue(
                    torch.all(state["toy_weight_shift"][[0, 2]].norm(dim=1) > 0).item(),
                )
                self.assertFalse(torch.equal(
                    state["toy_model"].head.weight.detach(), state["toy_initial_head"],
                ))
                self.assertTrue(torch.equal(state["toy_ids"], state["toy_initial_ids"]))
                self.assertFalse(torch.equal(state["toy_final_rows"], state["toy_rows"].detach()))


if __name__ == "__main__":
    unittest.main()

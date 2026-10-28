"""Offline contracts for the shared windows and their objective-specific pairs."""

import ast
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import torch

from language_model import TrainableLanguageModel
from real_text_embeddings import make_document_windows, objective_pairs, plot_embedding_routing


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

    def test_static_storyboard_plots_full_row_measurements_not_only_displayed_features(self):
        with torch.random.fork_rng(devices=[]):
            self.model = self.model_type(7, 8)
        self._activate_features()
        state = self._record_routing()
        trace = state["routing_trace"]
        figure = plot_embedding_routing(trace, state["itos"])
        try:
            for axis, measurements in (
                (figure.axes[1], trace["gradient"].norm(dim=1)),
                (figure.axes[2], (trace["weights"][3] - trace["weights"][0]).norm(dim=1)),
            ):
                for bar, label in zip(axis.patches, axis.get_yticklabels()):
                    index = state["itos"].index(label.get_text())
                    self.assertAlmostEqual(bar.get_width(), measurements[index].item())
            self.assertEqual(figure.axes[0].images[0].get_array().shape[1], 4)
        finally:
            plt.close(figure)


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

    def test_analogy_reaches_relatedness_and_distinguishes_sampling(self):
        markdown = "\n\n".join(
            "".join(cell["source"]) for cell in self.notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        for explanation in (
            "Earth is your **query profile**",
            "searching for semantically related words",
            "Not in the embedding lookup.",
            "Sampling draws from that distribution",
            "Softmax computes probabilities; it does not itself make a random choice.",
            "not a generated sentence",
            "images/03a-celestial-neighborhood-principle.png",
            "images/03a-01-shared-word-features.png",
        ):
            with self.subTest(explanation=explanation):
                self.assertIn(explanation, " ".join(markdown.split()))

    def test_sampling_changes_choices_without_updating_weights_or_global_rng(self):
        with torch.random.fork_rng(devices=[]):
            model = torch.nn.Embedding(1, 5)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0]]))
        before = model.weight.detach().clone()
        rng_before = torch.random.get_rng_state().clone()
        namespace = {
            "torch": torch, "model": model, "cat_contexts": torch.tensor([0]),
            "SEED": 17, "vocabulary_size": 5,
            "itos": ["cat", "dog", "sat", "slept", "<unk>"],
            "most_similar": lambda word: [("dog", 0.9)],
        }
        source = self.code["ch03a-similarity-and-sampling"]
        with redirect_stdout(StringIO()):
            exec(compile(source, "sampling", "exec"), namespace)
        expected = before[0].softmax(dim=-1)
        torch.testing.assert_close(namespace["bridge_probabilities"], expected)
        self.assertEqual(namespace["bridge_greedy_id"], expected.argmax().item())
        self.assertEqual(namespace["bridge_counts"].sum().item(), 2000)
        self.assertTrue(torch.equal(namespace["bridge_samples"], namespace["bridge_replay"]))
        self.assertGreater(namespace["bridge_samples"].unique().numel(), 1)
        other_seed = dict(namespace, SEED=23)
        with redirect_stdout(StringIO()):
            exec(compile(source, "sampling", "exec"), other_seed)
        torch.testing.assert_close(other_seed["bridge_probabilities"], expected)
        self.assertEqual(other_seed["bridge_greedy_id"], namespace["bridge_greedy_id"])
        self.assertFalse(torch.equal(other_seed["bridge_samples"], namespace["bridge_samples"]))
        self.assertTrue(torch.equal(model.weight.detach(), before))
        self.assertIsNone(model.weight.grad)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))

    def test_heading_numbers_follow_the_reading_order(self):
        sections = []
        subsections = {}
        section = None
        for cell in self.notebook["cells"]:
            if cell["cell_type"] != "markdown":
                continue
            for line in "".join(cell["source"]).splitlines():
                heading = re.match(r"## ([1-5A-D]) - ", line)
                if heading:
                    section = heading[1]
                    sections.append(section)
                    subsections[section] = 0
                elif line.startswith("### "):
                    heading = re.match(r"### ([1-5A-D])\.(\d+) - ", line)
                    self.assertIsNotNone(heading, line)
                    self.assertEqual(heading[1], section, line)
                    subsections[section] += 1
                    self.assertEqual(int(heading[2]), subsections[section], line)
        self.assertEqual(sections, list("12345ABCD"))

    def test_code_labels_match_headings_and_the_python_companion(self):
        section = None
        notebook_labels = []
        for cell in self.notebook["cells"]:
            source = "".join(cell["source"])
            if cell["cell_type"] == "markdown":
                for line in source.splitlines():
                    heading = re.match(r"#{2,3} ([1-5A-D](?:\.\d+)?) - ", line)
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

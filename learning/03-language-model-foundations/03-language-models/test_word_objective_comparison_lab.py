"""The optional comparison remains independent and uses the same learning implementations."""

import ast
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn

from language_model import CBOW, EncoderDecoderLM, SkipGram, TrainableLanguageModel, WordEmbeddingLanguageModel
from real_text_embeddings import WORD_GROUPS, load_real_text, make_document_windows, train_comparison


HERE = Path(__file__).parent


class ComparisonLabTests(unittest.TestCase):
    def test_encoder_samples_six_word_windows_from_the_same_sentence_local_split(self):
        groups = " ".join(word for group in WORD_GROUPS.values() for word in group)
        stories = [f"the cat sat on the mat. {groups}."] * 10
        with patch("real_text_embeddings.read_stories", return_value=stories), redirect_stdout(StringIO()):
            corpus = load_real_text(stories=10, train_windows=12, valid_windows=6)
        ids = corpus["word_ids"]
        self.assertTrue(all(word in ids for word in ("the", "cat", "sat", "on", "mat")))
        allowed = make_document_windows([
            [ids[word] for word in "the cat sat on the mat".split()],
            [ids[word] for word in groups.split()],
        ], 6).tolist()
        for split in ("encoder_train", "encoder_valid"):
            self.assertEqual(corpus[split].shape[1], 6)
            self.assertTrue(all(row in allowed for row in corpus[split].tolist()))
        self.assertEqual(corpus["train"].shape[1], 5)
        self.assertEqual(corpus["valid"].shape[1], 5)

    @classmethod
    def setUpClass(cls):
        cls.lab = json.loads((HERE / "optional" / "03A-word-objective-comparison-lab.ipynb").read_text(encoding="utf-8"))
        cls.main = json.loads((HERE / "03A-training-skeleton-and-word-objectives.ipynb").read_text(encoding="utf-8"))

    def test_lab_companion_matches_and_has_no_live_main_notebook_dependencies(self):
        source = "\n\n".join("".join(cell["source"]) for cell in self.lab["cells"] if cell["cell_type"] == "code")
        companion = ast.parse((HERE / "optional" / "03A-word-objective-comparison-lab.py").read_text(encoding="utf-8"))
        companion.body.pop(0)
        self.assertEqual(ast.dump(ast.parse(source)), ast.dump(companion))
        self.assertIn("from language_model import CBOW, SkipGram, EncoderDecoderLM", source)
        self.assertIn("REAL_TEXT_EPOCHS = 16", source)
        self.assertIn("RUN_SEED_CHECK = False", source)
        self.assertNotIn("sequence_model", source)
        self.assertNotIn("cbow_dataloader", source)
        self.assertNotIn("skipgram_dataloader", source)
        self.assertEqual(len({cell["id"] for cell in self.lab["cells"]}), len(self.lab["cells"]))

    def test_comparison_run_keeps_common_initial_rows_and_matches_inline_model_definitions(self):
        source = "\n".join("".join(cell["source"]) for cell in self.main["cells"] if cell["cell_type"] == "code")
        definitions = [node for node in ast.parse(source).body
                       if isinstance(node, ast.ClassDef) and node.name in {"CBOW", "SkipGram", "EncoderDecoderLM"}]
        namespace = {"nn": nn, "TrainableLanguageModel": TrainableLanguageModel,
                     "WordEmbeddingLanguageModel": WordEmbeddingLanguageModel}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "inline-objectives", "exec"), namespace)
        words = ["<unk>"] + [word for group in WORD_GROUPS.values() for word in group]
        windows = torch.tensor([[(row + offset) % len(words) for offset in range(5)] for row in range(24)])
        corpus = {"words": words, "word_ids": {word: index for index, word in enumerate(words)},
                  "train": windows, "valid": windows[:12]}
        with redirect_stdout(StringIO()):
            actual = train_comparison(corpus, CBOW, SkipGram, EncoderDecoderLM, epochs=3, verbose=False)
            reference = train_comparison(corpus, namespace["CBOW"], namespace["SkipGram"],
                                         namespace["EncoderDecoderLM"], epochs=3, verbose=False)
        for objective in actual:
            for key in ("train_loss", "valid_loss", "within", "across", "purity", "epoch_fit_loss"):
                self.assertEqual(actual[objective][key], reference[objective][key])
            for rows, expected in zip(actual[objective]["snapshots"], reference[objective]["snapshots"]):
                torch.testing.assert_close(rows, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual[objective]["snapshots"][0], actual["CBOW"]["snapshots"][0],
                                       rtol=0, atol=0)

    def test_main_no_longer_runs_the_larger_comparison(self):
        code = "\n".join("".join(cell["source"]) for cell in self.main["cells"] if cell["cell_type"] == "code")
        self.assertNotIn("train_comparison", code)
        self.assertNotIn("REAL_TEXT_EPOCHS", code)
        self.assertNotIn("real_results", code)
        markdown = "\n".join("".join(cell["source"]) for cell in self.main["cells"]
                             if cell["cell_type"] == "markdown")
        self.assertIn("03A-word-objective-comparison-lab.ipynb", markdown)
        self.assertIn("Attention does **not** create order", markdown)


if __name__ == "__main__":
    unittest.main()

"""Focused evaluation-lesson contracts; no downloads or model generation."""

import ast
import base64
from copy import deepcopy
from contextlib import redirect_stdout
import json
import math
from io import BytesIO, StringIO
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageDraw
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
import torch
from torch.nn import functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from evaluation_metrics import (
    ENCODER_ID, ENCODER_REVISION, MODEL_ID, MODEL_REVISION, HERE, RESULT_PATH,
    LessonTokenizer, file_sha256, fingerprint, load_eval_model, local_checkpoint_inventory,
    reference_perplexity,
)
from evaluation_shared import RIVERSIDE_CASES, normalized_terms, tokenize


def notebook_cell(filename, raw_id):
    book = json.loads((HERE / filename).read_bytes())
    return next("".join(cell["source"]) for cell in book["cells"] if cell["id"] == raw_id)


def policy_selector():
    tree = ast.parse(notebook_cell("04-calibration-and-confidence.ipynb", "eval04-012"))
    definitions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)],
                             type_ignores=[])
    namespace = {}
    exec(compile(definitions, "<operating point selector>", "exec"), namespace)
    return namespace["choose_operating_point"]


class EvaluationMetricsTests(unittest.TestCase):
    def test_loss_probability_example_separates_cost_from_chance_and_accuracy(self):
        namespace = {"torch": torch, "pd": pd, "display": lambda *args: None}
        exec(notebook_cell("01-llm-evaluation-metrics-and-benchmarks.ipynb",
                           "eval01-loss-probability-probe"), namespace)
        table = namespace["loss_probe_table"]
        np.testing.assert_allclose(table["single_token_perplexity"],
                                   1 / table["correct_token_probability"], rtol=1e-12)
        self.assertEqual(table["top_choice_correct"].tolist(), [True, True, False, False])
        self.assertTrue((table.loc[2:, "loss"] > 1).all())
        self.assertGreater(table.loc[3, "loss"], table.loc[2, "loss"])

    def test_multiple_reference_probe_keeps_the_answer_and_original_report_fixed(self):
        report_bytes = RESULT_PATH.read_bytes()
        report = json.loads(report_bytes)
        answers = deepcopy(report["items"])
        cases_before, answers_before = deepcopy(RIVERSIDE_CASES), deepcopy(answers)
        namespace = {"pd": pd, "tokenize": tokenize, "RIVERSIDE_CASES": RIVERSIDE_CASES,
                     "measured_answers": answers, "display": lambda *args: None}
        with redirect_stdout(StringIO()):
            exec(notebook_cell("01-llm-evaluation-metrics-and-benchmarks.ipynb",
                               "eval01-multiple-reference-probe"), namespace)
        rows = namespace["reference_probe_rows"]
        self.assertEqual([row["accepted_wordings"] for row in rows], [1, 2])
        original = next(row for row in report["items"] if row["id"] == "pendant")
        self.assertEqual(namespace["reference_probe_answer"], original["answer"])
        self.assertAlmostEqual(rows[0]["bleu_0_100"], original["bleu_0_100"])
        self.assertAlmostEqual(rows[1]["bleu_0_100"], 15.415064977510756)
        self.assertEqual(RIVERSIDE_CASES, cases_before)
        self.assertEqual(answers, answers_before)
        self.assertEqual(RESULT_PATH.read_bytes(), report_bytes)

    def test_stronger_controlled_bonus_exposes_a_swap_the_default_misses(self):
        namespace = {"np": np, "pd": pd, "normalized_terms": normalized_terms,
                     "RIVERSIDE_CASES": RIVERSIDE_CASES, "display": lambda *args: None}
        filename = "02-llm-as-judge-safety-and-pipeline.ipynb"
        for raw_id in ("eval02-005", "eval02-007", "eval02-strong-bias-probe"):
            exec(notebook_cell(filename, raw_id), namespace)
        self.assertTrue(namespace["swaps"]["stable"].all())
        probe = namespace["probe_swaps"].set_index("case")
        self.assertFalse(probe.loc["award", "stable"])
        tree = ast.parse(notebook_cell(filename, "eval02-strong-bias-probe"))
        setting = next(node for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "probe_bonus" for target in node.targets))
        setting.value = ast.Constant(value=0.0)
        exec(compile(ast.fix_missing_locations(tree), "<zero bonus exercise>", "exec"), namespace)
        self.assertTrue(namespace["probe_swaps"]["stable"].all())

    def test_empty_returned_set_has_undefined_risk_and_no_approved_policy(self):
        namespace = {"np": np, "pd": pd, "display": lambda *args: None,
                     "calibrated": np.array([.2, .3]), "correct": np.array([0, 1]), "split": 0}
        exec(notebook_cell("04-calibration-and-confidence.ipynb", "eval04-011"), namespace)
        empty_rows = namespace["risk_coverage"].query("served == 0")
        self.assertFalse(empty_rows.empty)
        self.assertTrue(empty_rows["selective_risk"].isna().all())
        output = StringIO()
        with redirect_stdout(output):
            chosen = policy_selector()(empty_rows, max_risk=.2)
        self.assertIsNone(chosen)
        self.assertIn("hold the policy", output.getvalue())

    def test_policy_selector_uses_only_nonempty_qualifying_points(self):
        table = pd.DataFrame([
            {"threshold": .9, "served": 0, "coverage": 0., "selective_risk": 0.},
            {"threshold": .7, "served": 2, "coverage": .2, "selective_risk": .05},
            {"threshold": .5, "served": 5, "coverage": .5, "selective_risk": .15},
            {"threshold": .1, "served": 9, "coverage": .9, "selective_risk": .4},
        ])
        choose = policy_selector()
        selected = choose(table, max_risk=.2)
        self.assertEqual(selected["threshold"], .5)
        self.assertEqual(selected["coverage"], .5)
        with self.assertRaises(ValueError):
            choose(table, max_risk=1.5)

    def test_metric_visuals_embed_final_images_and_preserve_measured_panels(self):
        manifest = json.loads((HERE / "images" / "01-metric-illustration-prompts.json").read_bytes())
        book = json.loads((HERE / "01-llm-evaluation-metrics-and-benchmarks.ipynb").read_bytes())
        sources = "\n".join("".join(cell["source"]) for cell in book["cells"])
        for entry in manifest:
            filename = entry.get("notebook_filename", entry["filename"])
            cell = next(cell for cell in book["cells"] if f"](attachment:{filename})" in "".join(cell["source"]))
            raw = base64.b64decode(cell["attachments"][filename]["image/png"], validate=True)
            self.assertEqual(raw, (HERE / "images" / filename).read_bytes())
            with Image.open(BytesIO(raw)) as image:
                self.assertEqual(image.size, (entry["width"], entry["height"]))
                self.assertEqual(image.format, "PNG")
                if "numerical_overlay" in entry:
                    with Image.open(HERE / "images" / entry["filename"]) as original:
                        top = entry["numerical_overlay"]["bounds"][1]
                        actual_top = image.convert("RGB").crop((0, 0, image.width, top))
                        original_top = original.convert("RGB").crop((0, 0, original.width, top))
                        for bounds in entry["numerical_overlay"].get("proofreading_bounds", []):
                            ImageDraw.Draw(actual_top).rectangle(bounds, fill="white")
                            ImageDraw.Draw(original_top).rectangle(bounds, fill="white")
                        self.assertIsNone(ImageChops.difference(
                            actual_top, original_top,
                        ).getbbox())
                        self.assertIn("recorded model values", image.info["NumericalPanel"])
                        self.assertNotIn(f"attachment:{entry['filename']}", sources)

    def test_visual_evidence_agrees_with_word_matches_and_saved_reference_loss(self):
        data = json.loads((HERE / "images" / "01-metric-visual-evidence.json").read_bytes())
        scorer = rouge_scorer.RougeScorer(["rougeL"], tokenizer=LessonTokenizer())
        expected = scorer.score(data["reference"], data["rouge"]["inserted"])["rougeL"]
        self.assertAlmostEqual(expected.fmeasure, data["rouge"]["f1"])
        for name in ("ordered", "reversed"):
            item = data["meteor"]["cases"][name]
            self.assertAlmostEqual(
                item["score"], 1 - .5 * (item["chunks"] / len(item["matches"])) ** 3,
            )
        bert = data["bertscore"]
        values = np.asarray(bert["cosine_similarities"])
        self.assertEqual(values.shape, (len(bert["candidate_tokens"]), len(bert["reference_tokens"])))
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((abs(values) <= 1.00001).all())
        self.assertEqual(bert["encoder_revision"], ENCODER_REVISION)
        ppl = data["perplexity"]
        losses, probabilities = np.asarray(ppl["token_losses"]), np.asarray(ppl["token_probabilities"])
        self.assertEqual(len(losses), len(ppl["token_ids"]))
        np.testing.assert_allclose(np.exp(-losses), probabilities, rtol=1e-6)
        saved = next(row for row in json.loads(RESULT_PATH.read_bytes())["items"] if row["id"] == ppl["case"])
        self.assertAlmostEqual(losses.sum(), saved["reference_nll_sum"], delta=1e-4)
        self.assertAlmostEqual(math.exp(losses.mean()), saved["reference_ppl"], delta=1e-5)

    def test_concept_recap_code_sketches_parse_and_balance_matches(self):
        recap = notebook_cell("01-llm-evaluation-metrics-and-benchmarks.ipynb", "eval01-concept-recap")
        blocks = re.findall(r"```python\n(.*?)\n```", recap, re.DOTALL)
        for block in blocks:
            ast.parse(block)
        self.assertNotIn("handwrit", recap.lower())
        matching = next(block for block in blocks if "max(produced, 1)" in block)
        for counts, expected in (((5, 6, 5), 10 / 11), ((0, 0, 0), 0)):
            namespace = dict(zip(("matches", "produced", "required"), counts))
            exec(matching, namespace)
            self.assertAlmostEqual(namespace["f1"], expected)
        for filename, raw_id in (
            ("02-llm-as-judge-safety-and-pipeline.ipynb", "eval02-concept-recap"),
            ("03-hallucination-detection.ipynb", "eval03-concept-recap"),
            ("04-calibration-and-confidence.ipynb", "eval04-concept-recap"),
        ):
            recap = notebook_cell(filename, raw_id)
            self.assertNotIn("handwrit", recap.lower())
            for block in re.findall(r"```python\n(.*?)\n```", recap, re.DOTALL):
                ast.parse(block)

    def test_manual_rouge_matches_library_with_the_same_tokenization(self):
        namespace = {"np": np, "tokenize": tokenize}
        exec(notebook_cell("01-llm-evaluation-metrics-and-benchmarks.ipynb", "eval01-005"), namespace)
        scorer = rouge_scorer.RougeScorer(["rougeL"], tokenizer=LessonTokenizer())
        for answer, reference in (
            ("the pendant hides a small letter", "the pendant hides a letter"),
            ("letter pendant", "pendant letter"),
            ("letter letter letter", "letter"),
            ("", "pendant"),
        ):
            with self.subTest(answer=answer, reference=reference):
                expected = scorer.score(reference, answer)["rougeL"].fmeasure
                self.assertAlmostEqual(namespace["rouge_l"](answer, reference), expected)

    def test_checkpoint_inventory_distinguishes_adapter_full_and_incomplete(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("adapter", "full", "incomplete"):
                (root / name).mkdir()
            (root / "adapter" / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "example/base"}), encoding="utf-8",
            )
            (root / "adapter" / "adapter_model.safetensors").touch()
            for name in ("full", "incomplete"):
                (root / name / "config.json").write_text("{}", encoding="utf-8")
                (root / name / "model.safetensors").touch()
            (root / "full" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            rows = {row["checkpoint"]: row for row in local_checkpoint_inventory(root)}
            self.assertTrue(rows["adapter"]["ready"])
            self.assertEqual(rows["adapter"]["kind"], "adapter")
            self.assertTrue(rows["full"]["ready"])
            self.assertFalse(rows["incomplete"]["ready"])

    def test_invalid_selected_local_source_does_not_download_an_hf_fallback(self):
        with TemporaryDirectory() as directory, patch("evaluation_metrics.download_snapshot") as download:
            with self.assertRaises(FileNotFoundError):
                load_eval_model("local", Path(directory) / "missing")
            with self.assertRaises(ValueError):
                load_eval_model("local", Path(directory))
            download.assert_not_called()

    def test_reference_loss_excludes_prompt_and_includes_first_answer_token(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(9)
            model = GPT2LMHeadModel(GPT2Config(
                vocab_size=13, n_layer=1, n_head=1, n_embd=8, n_positions=16,
                bos_token_id=1, eos_token_id=2,
            )).eval()
        prompt = torch.tensor([[1, 2, 3]])
        reference = torch.tensor([[4, 5]])
        actual = reference_perplexity(model, prompt, reference)
        with torch.inference_mode():
            logits = model(torch.cat((prompt, reference), dim=1)).logits
        expected = F.cross_entropy(logits[0, 2:4], reference[0], reduction="sum").item()
        self.assertEqual(actual["reference_tokens"], 2)
        self.assertAlmostEqual(actual["reference_nll_sum"], expected, places=6)
        self.assertAlmostEqual(actual["reference_ppl"], math.exp(expected / 2), places=5)

    def test_saved_values_reproduce_lexical_scores_and_weighted_aggregates(self):
        report = json.loads(RESULT_PATH.read_bytes())
        self.assertEqual(report["dataset_sha256"], fingerprint(RIVERSIDE_CASES))
        self.assertEqual(report["test_ids"], [case["id"] for case in RIVERSIDE_CASES])
        self.assertEqual(report["model"], {"source": "hf", "model_id": MODEL_ID, "revision": MODEL_REVISION})
        self.assertEqual(report["scorers"]["bertscore"]["model_id"], ENCODER_ID)
        self.assertEqual(report["scorers"]["bertscore"]["revision"], ENCODER_REVISION)
        self.assertEqual(report["code_sha256"], file_sha256(HERE / "evaluation_metrics.py"))
        self.assertEqual(report["fixture_code_sha256"], file_sha256(HERE / "evaluation_shared.py"))
        self.assertEqual(report["runtime"]["device"], "cpu")
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=LessonTokenizer())
        for row, case in zip(report["items"], RIVERSIDE_CASES, strict=True):
            self.assertEqual(row["reference"], case["reference"])
            self.assertTrue(row["answer"])
            for name, value in scorer.score(row["reference"], row["answer"]).items():
                self.assertAlmostEqual(row[f"{name}_fmeasure"], value.fmeasure)
            self.assertTrue(all(math.isfinite(row[key]) for key in ("meteor", "bertscore_f1", "reference_ppl")))
        items = report["items"]
        expected_ppl = math.exp(sum(row["reference_nll_sum"] for row in items) /
                                sum(row["reference_tokens"] for row in items))
        self.assertAlmostEqual(report["aggregate"]["reference_ppl"], expected_ppl)
        bleu = BLEU(tokenize="none", smooth_method="exp", effective_order=True)
        expected_bleu = bleu.corpus_score(
            [" ".join(tokenize(row["answer"])) for row in items],
            [[" ".join(tokenize(row["reference"])) for row in items]],
        ).score
        self.assertAlmostEqual(report["aggregate"]["bleu_0_100"], expected_bleu)
        notebook = notebook_cell("01-llm-evaluation-metrics-and-benchmarks.ipynb", "eval01-measured-snapshot")
        keys = (
            "rouge1_fmeasure", "rouge2_fmeasure", "rougeL_fmeasure", "bleu_0_100",
            "meteor", "bertscore_f1", "reference_ppl",
        )
        for row in items + [{"id": "Aggregate", **report["aggregate"]}]:
            expected_row = f"| {row['id']} | " + " | ".join(f"{row[key]:.3f}" for key in keys) + " |"
            self.assertIn(expected_row, notebook)

    def test_longlisted_alone_does_not_establish_a_win_or_a_contradiction(self):
        namespace = {"normalized_terms": normalized_terms}
        tree = ast.parse(notebook_cell("03-hallucination-detection.ipynb", "eval03-006"))
        definitions = ast.Module(body=[
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            or isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "CONTRADICTIONS" for target in node.targets
            )
        ], type_ignores=[])
        exec(compile(definitions, "<notebook claim rules>", "exec"), namespace)
        classify = namespace["claim_outcome"]
        self.assertEqual(classify(
            "Elena Marchetti won the Booker Prize in 2019.",
            "Elena Marchetti was longlisted for the Booker Prize in 2019.",
        ), "unknown")
        self.assertEqual(classify("The rule allows disclosure.", "Disclosure is prohibited."), "contradicted")

    def test_notebook_sources_parse_without_executing_or_replacing_outputs(self):
        for filename in (
            "01-llm-evaluation-metrics-and-benchmarks.ipynb",
            "02-llm-as-judge-safety-and-pipeline.ipynb",
            "03-hallucination-detection.ipynb",
            "04-calibration-and-confidence.ipynb",
        ):
            book = json.loads((HERE / filename).read_bytes())
            snapshot = deepcopy(book)
            for cell in book["cells"]:
                if cell["cell_type"] == "code":
                    ast.parse("".join(cell["source"]))
                    self.assertEqual(cell["outputs"], [])
                    self.assertIsNone(cell["execution_count"])
            self.assertEqual(book, snapshot)


if __name__ == "__main__":
    unittest.main()

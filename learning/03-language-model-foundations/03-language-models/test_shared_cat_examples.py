"""Check that the recorded memory/access experiments use one word example."""

import ast
from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

from language_data import CAT_FAMILY_WORDS, CAT_WORDS
from encoder_decoder_example import BOS, EOS, VOCABULARY, decode_ids, encode_words, generate, teacher_inputs


HERE = Path(__file__).parent


class SharedCatExampleTests(unittest.TestCase):
    def test_memory_probe_keeps_the_animal_outside_identical_windows(self):
        evidence = json.loads((HERE / "images" / "03b-cat-measured-evidence.json").read_bytes())
        cat, dog = evidence["held_out_rows"]
        self.assertEqual(tuple(cat[:-1]), CAT_WORDS)
        self.assertEqual(cat[-1], "purring")
        self.assertEqual(dog[1], "dog")
        self.assertEqual(dog[-1], "panting")
        self.assertEqual(cat[:1] + cat[2:-1], dog[:1] + dog[2:-1])
        self.assertEqual(cat[-4:-1], dog[-4:-1])
        self.assertEqual(cat[-5:-1], dog[-5:-1])
        self.assertNotEqual(cat[-6:-1], dog[-6:-1])
        distances = evidence["state_trace"]["distances"]
        self.assertEqual(len(distances), len(CAT_WORDS))
        self.assertEqual(distances[0], 0.0)
        self.assertTrue(all(value > 0 for value in distances[1:]))
        self.assertGreater(evidence["scorecard"]["swapped_clue_probability_effect"], 0.5)
        self.assertEqual(evidence["generated_rows"], [cat, dog])
        self.assertEqual(evidence["scorecard"]["future_token_earlier_logit_gap"], 0.0)

    def test_source_access_probe_reverses_words_and_keeps_repeated_addresses_distinct(self):
        evidence = json.loads((HERE / "images" / "03c-cat-measured-evidence.json").read_bytes())
        self.assertEqual(tuple(evidence["source_words"]), CAT_WORDS)
        self.assertEqual(evidence["required_words"], list(reversed(CAT_WORDS)) + ["<eos>"])
        self.assertEqual(evidence["required_addresses"], list(reversed(range(len(CAT_WORDS)))))
        addresses = evidence["source_addresses"]
        self.assertEqual(len(addresses), len(CAT_WORDS))
        self.assertEqual(len(set(addresses)), len(CAT_WORDS))
        self.assertTrue(addresses[0].startswith("the"))
        self.assertTrue(addresses[4].startswith("the"))
        allowed = set(CAT_FAMILY_WORDS) | {"<pad>", "<bos>", "<eos>"}
        for output in evidence["probe_generated_words"].values():
            self.assertEqual(len(output), len(CAT_WORDS) + 1)
            self.assertTrue(all(isinstance(word, str) and word in allowed for word in output))
        self.assertTrue(all(0 <= index < len(CAT_WORDS) for index in evidence["most_read_addresses"]))


class ReaderWriterTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespace = {
            "torch": torch, "nn": nn, "BOS": len(CAT_FAMILY_WORDS) + 1,
            "EOS": len(CAT_FAMILY_WORDS) + 2, "V": len(CAT_FAMILY_WORDS) + 3,
        }
        tree = ast.parse((HERE / "optional" / "03C1-encoder-decoder-worked-details.py").read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in {
                "practice_prefix", "reversal_targets", "ReaderWriter",
            }:
                exec(compile(ast.Module(body=[node], type_ignores=[]), "<reader-writer>", "exec"), namespace)
        cls.model_type = namespace["ReaderWriter"]
        cls.namespace = namespace
        cls.bos, cls.eos = len(CAT_FAMILY_WORDS) + 1, len(CAT_FAMILY_WORDS) + 2
        word_ids = {word: index + 1 for index, word in enumerate(CAT_FAMILY_WORDS)}
        cls.source = torch.tensor([[word_ids[word] for word in CAT_WORDS]])
        cls.target = torch.cat([cls.source.flip(1), torch.tensor([[cls.eos]])], dim=1)
        cls.cat_id = word_ids["cat"]

    def model(self, look_back):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31)
            return self.model_type(look_back)

    def test_teacher_inputs_are_previous_answers_and_generation_never_receives_targets(self):
        for look_back in (False, True):
            with self.subTest(look_back=look_back):
                model = self.model(look_back)
                seen = []
                real_step = model.step

                def record(previous, state, memory):
                    seen.append(previous.clone())
                    return real_step(previous, state, memory)

                with torch.no_grad(), patch.object(model, "step", side_effect=record):
                    model(self.source, target_ids=self.target, teacher=True)
                    expected = torch.cat([torch.tensor([[self.bos]]), self.target[:, :-1]], dim=1)
                    torch.testing.assert_close(torch.stack(seen, 1), expected, rtol=0, atol=0)
                    seen.clear()
                    scores, _ = model(self.source, teacher=False)
                    self.assertEqual(seen[0].item(), self.bos)
                    torch.testing.assert_close(torch.stack(seen, 1)[:, 1:], scores.argmax(-1)[:, :-1],
                                               rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, "correct answer"):
                    model(self.source)
                with self.assertRaisesRegex(ValueError, "must not receive"):
                    model(self.source, target_ids=self.target, teacher=False)
                shorter, _ = model(self.source, target_ids=self.target[:, :3], teacher=True)
                self.assertEqual(shorter.shape[1], 3)

    def test_changing_an_answer_cannot_change_its_own_or_earlier_scores(self):
        for look_back in (False, True):
            with self.subTest(look_back=look_back), torch.no_grad():
                model = self.model(look_back)
                changed = self.target.clone()
                changed[:, 1] = self.cat_id
                original, _ = model(self.source, target_ids=self.target, teacher=True)
                altered, _ = model(self.source, target_ids=changed, teacher=True)
                torch.testing.assert_close(original[:, :2], altered[:, :2], rtol=0, atol=0)
                self.assertGreater((original[:, 2:] - altered[:, 2:]).abs().max().item(), 0)

    def test_source_memory_probe_reuses_notes_without_updating_the_trained_model(self):
        book = json.loads((HERE / "optional" / "03C1-encoder-decoder-worked-details.ipynb").read_bytes())
        cell = next(cell for cell in book["cells"] if cell["id"] == "ch03c1-source-memory-probe")
        original = self.model(True)
        parameters = {name: parameter.detach().clone() for name, parameter in original.named_parameters()}
        namespace = dict(self.namespace, models={True: original}, CAT_WORDS=CAT_WORDS,
                         encode_words=encode_words, decode_ids=decode_ids)
        with torch.random.fork_rng(devices=[]), redirect_stdout(StringIO()):
            exec(compile("".join(cell["source"]), "<read-only-source-memory>", "exec"), namespace)
        rows = namespace["memory_probe_rows"]
        self.assertEqual([row["previous"] for row in rows], ["<bos>", "mat", "the"])
        self.assertTrue(all(row["source_change"] == 0 for row in rows))
        self.assertTrue(any(row["writer_change"] > 0 for row in rows))
        self.assertIsNone(rows[0]["attention_change"])
        self.assertTrue(any(row["attention_change"] > 0 for row in rows[1:]))
        self.assertTrue(original.training)
        for name, parameter in original.named_parameters():
            torch.testing.assert_close(parameter, parameters[name], rtol=0, atol=0)
            self.assertIsNone(parameter.grad)
        notebook_code = "\n\n".join("".join(cell["source"]) for cell in book["cells"]
                                   if cell["cell_type"] == "code")
        companion = ast.parse((HERE / "optional" / "03C1-encoder-decoder-worked-details.py").read_text(encoding="utf-8"))
        companion.body.pop(0)
        self.assertEqual(ast.dump(ast.parse(notebook_code)), ast.dump(companion))

    def test_detaching_the_handoff_blocks_reader_learning_without_changing_predictions(self):
        for look_back in (False, True):
            with self.subTest(look_back=look_back):
                connected = self.model(look_back)
                detached = deepcopy(connected)
                scores, _ = connected(self.source, target_ids=self.target, teacher=True)
                blocked, _ = detached(self.source, target_ids=self.target, teacher=True, detach_reader=True)
                torch.testing.assert_close(scores, blocked, rtol=0, atol=0)
                nn.functional.cross_entropy(scores.flatten(0, 1), self.target.flatten()).backward()
                nn.functional.cross_entropy(blocked.flatten(0, 1), self.target.flatten()).backward()
                self.assertGreater(sum(p.grad.abs().sum().item() for p in connected.reader.parameters()), 0)
                self.assertTrue(all(p.grad is None for p in detached.reader.parameters()))
                for left, right in zip(connected.writer.parameters(), detached.writer.parameters(), strict=True):
                    torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
                self.assertGreater(detached.embedding.weight.grad.abs().sum().item(), 0)
                self.assertGreater((connected.embedding.weight.grad - detached.embedding.weight.grad).norm().item(), 0)

    def test_encoder_only_update_changes_the_message_not_the_decoder(self):
        book = json.loads((HERE / "optional" / "03C1-encoder-decoder-worked-details.ipynb").read_bytes())
        cell = next(cell for cell in book["cells"] if cell["id"] == "ch03c-encoder-only-probe")
        word_ids = {word: index + 1 for index, word in enumerate(CAT_FAMILY_WORDS)}
        id_to_word = {value: word for word, value in word_ids.items()}
        namespace = {
            "torch": torch, "F": nn.functional, "deepcopy": deepcopy,
            "mechanism_model": self.model(False), "CAT_WORDS": CAT_WORDS, "BOS": self.bos,
            "encode_words": lambda words: [word_ids[word] for word in words],
            "decode_ids": lambda ids: [id_to_word[int(index)] for index in ids],
            "training_evidence": {},
        }
        with redirect_stdout(StringIO()):
            exec(compile("".join(cell["source"]), "<encoder-only-probe>", "exec"), namespace)
        evidence = namespace["training_evidence"]["encoder_only"]
        self.assertEqual(evidence["first_answers"], ["mat", "rug"])
        self.assertEqual(evidence["identical_message_score_gap"], 0)
        self.assertGreater(evidence["message_change_norm"], 0)
        self.assertLess(evidence["after_loss"], evidence["before_loss"])
        self.assertTrue(evidence["decoder_and_head_unchanged"])
        original = namespace["mechanism_model"]
        updated = namespace["encoding_copy"]
        for name, parameter in original.named_parameters():
            if not name.startswith("reader."):
                torch.testing.assert_close(parameter, updated.get_parameter(name), rtol=0, atol=0)


class MainEncoderDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespace = {
            "torch": torch, "nn": nn, "BOS": BOS, "EOS": EOS,
            "V": len(VOCABULARY), "CAT_WORDS": CAT_WORDS,
            "encode_words": encode_words, "decode_ids": decode_ids, "generate": generate,
        }
        source = (HERE / "03C-recurrent-encoder-decoder-to-attention.py").read_text(encoding="utf-8")
        for node in ast.parse(source).body:
            if isinstance(node, ast.ClassDef) and node.name == "EncoderDecoder":
                exec(compile(ast.Module(body=[node], type_ignores=[]), "<main-encoder-decoder>", "exec"),
                     namespace)
        cls.namespace = namespace
        book = json.loads((HERE / "03C-recurrent-encoder-decoder-to-attention.ipynb").read_bytes())
        cls.training_code = "".join(next(
            cell["source"] for cell in book["cells"] if cell["id"] == "ch03c-main-training"
        ))

    def test_notebook_cells_run_in_order_from_repository_root_and_chapter(self):
        notebook = (HERE / "03C-recurrent-encoder-decoder-to-attention.ipynb").resolve()
        script = f"""
import json
from pathlib import Path

book = json.loads(Path({str(notebook)!r}).read_bytes())
cells = {{cell["id"]: "".join(cell["source"]) for cell in book["cells"]}}
namespace = {{"__name__": "__main__"}}
for cell_id in ("ch03c-main-data", "ch03c-main-model"):
    exec(compile(cells[cell_id], cell_id, "exec"), namespace)
batches = namespace["training_batches"]
namespace["training_batches"] = lambda corpus: batches(corpus, updates=1)
exec(compile(cells["ch03c-main-training"], "ch03c-main-training", "exec"), namespace)
assert namespace["torch"].isfinite(namespace["loss"]).item()
"""
        for directory in (HERE.resolve().parents[2], HERE.resolve()):
            with self.subTest(directory=directory):
                result = subprocess.run(
                    [sys.executable, "-c", script], cwd=directory,
                    text=True, capture_output=True, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_training_shifts_explicit_answers_and_teaches_the_separate_source_reader(self):
        source = torch.tensor([encode_words(CAT_WORDS)])
        target = torch.cat([source.flip(1), source.new_full((1, 1), EOS)], dim=1)
        seen = []
        model_type = self.namespace["EncoderDecoder"]

        class RecordingEncoderDecoder(model_type):
            def forward(self, source, previous):
                seen.append((source.clone(), previous.clone()))
                return super().forward(source, previous)

        namespace = dict(self.namespace, EncoderDecoder=RecordingEncoderDecoder, corpus={},
                         training_batches=lambda _: iter([(source, torch.full_like(target, EOS), target)]))
        with torch.random.fork_rng(devices=[]), redirect_stdout(StringIO()):
            exec(compile(self.training_code, "<main-training>", "exec"), namespace)
        trained = namespace["model"]
        torch.testing.assert_close(seen[0][0], source, rtol=0, atol=0)
        torch.testing.assert_close(seen[0][1], teacher_inputs(target), rtol=0, atol=0)
        self.assertEqual(seen[0][1].tolist(), [[BOS, *target[0, :-1].tolist()]])
        self.assertGreater(sum(p.grad.abs().sum().item() for p in trained.encoder.parameters()), 0)
        with torch.no_grad():
            state = trained.encode(source)
            self.assertEqual(state.shape, (1, 1, 64))
            logits, next_state = trained.decode(seen[0][1], state)
            torch.testing.assert_close(logits, trained(source, seen[0][1]), rtol=0, atol=0)
            self.assertEqual(next_state.shape, state.shape)
            changed = target.clone()
            changed[:, 1] = encode_words(["cat"])[0]
            altered = trained(source, teacher_inputs(changed))
            torch.testing.assert_close(logits[:, :2], altered[:, :2], rtol=0, atol=0)
            self.assertGreater((logits[:, 2:] - altered[:, 2:]).abs().max().item(), 0)
        self.assertEqual(teacher_inputs(torch.tensor([[EOS]])).tolist(), [[BOS]])

    def test_generation_reads_once_carries_state_and_stops_at_eos_or_budget(self):
        model_type, generate = self.namespace["EncoderDecoder"], self.namespace["generate"]
        cat = encode_words(["cat"])[0]
        words = CAT_WORDS[:2]
        for answers in ([cat, EOS], [cat] * (len(words) + 1)):
            with self.subTest(answers=answers), torch.random.fork_rng(devices=[]):
                torch.manual_seed(31)
                model = model_type(len(VOCABULARY))
                real_decode = model.decode
                inputs, returned_states = [], []

                def record(previous, state):
                    if returned_states:
                        self.assertIs(state, returned_states[-1])
                    logits, new_state = real_decode(previous, state)
                    logits = torch.full_like(logits, -100)
                    logits[:, -1, answers[len(inputs)]] = 100
                    inputs.append(previous.clone())
                    returned_states.append(new_state)
                    return logits, new_state

                with patch.object(model, "encode", wraps=model.encode) as read, \
                        patch.object(model, "decode", side_effect=record) as write:
                    actual = generate(model, words)
                read.assert_called_once()
                torch.testing.assert_close(read.call_args.args[0],
                                           torch.tensor([encode_words(words)]), rtol=0, atol=0)
                self.assertEqual(actual, decode_ids(answers))
                self.assertEqual(write.call_count, len(answers))
                self.assertEqual(inputs[0].tolist(), [[BOS]])
                self.assertEqual([row.item() for row in inputs[1:]], answers[:-1])


if __name__ == "__main__":
    unittest.main()

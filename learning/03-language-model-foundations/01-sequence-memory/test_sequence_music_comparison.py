"""Focused contracts for the original offline RNN/LSTM/GRU music comparison."""

import base64
from dataclasses import replace
from io import BytesIO
import json
import random
import re
import unittest
import wave

import torch
from torch import nn

from sequence_music_comparison import (
    GRUSequenceModel,
    LSTMSequenceModel,
    RNNSequenceModel,
    VOCABULARY,
    build_music_dataset,
    make_next_note_batch,
    melody_wav,
    music_comparison_html,
    run_music_experiment,
)


MODELS = (("RNN", RNNSequenceModel), ("LSTM", LSTMSequenceModel), ("GRU", GRUSequenceModel))


class MusicDatasetTests(unittest.TestCase):
    def test_phrase_split_is_disjoint_varied_fixed_and_reproducible(self):
        state = random.getstate()
        data = build_music_dataset()
        self.assertEqual(state, random.getstate())
        self.assertEqual(data, build_music_dataset())
        self.assertEqual(data.fingerprint, build_music_dataset().fingerprint)
        train, heldout = set(data.train_phrases), set(data.heldout_phrases)
        self.assertFalse(train & heldout)
        self.assertEqual(len(train), len(data.train_phrases))
        self.assertEqual(len(heldout), len(data.heldout_phrases))
        self.assertGreaterEqual(len(train | heldout), 300)
        self.assertEqual({len(p) for p in train | heldout}, {32})
        self.assertGreater(len({p[:8] for p in train | heldout}), 50)
        self.assertEqual(data.vocabulary, VOCABULARY)
        other_split = build_music_dataset(split_seed=19)
        self.assertEqual(other_split.vocabulary, data.vocabulary)
        self.assertEqual(set(other_split.train_phrases + other_split.heldout_phrases), train | heldout)
        self.assertNotEqual(data.fingerprint, other_split.fingerprint)

    def test_causal_input_target_alignment_does_not_cross_phrase_boundaries(self):
        phrases = ((1, 3, 2, 0), (3, 0, 1, 2))
        x, y = make_next_note_batch(phrases, vocabulary_size=4)
        self.assertEqual(tuple(x.shape), (2, 3, 4))
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.long)
        self.assertEqual(x.argmax(-1).tolist(), [[1, 3, 2], [3, 0, 1]])
        self.assertEqual(y.tolist(), [[3, 2, 0], [0, 1, 2]])
        self.assertTrue(torch.equal(x.sum(-1), torch.ones(2, 3)))
        with self.assertRaisesRegex(ValueError, "outside"):
            make_next_note_batch(((0, 4),), 4)


class GenericSequenceModelTests(unittest.TestCase):
    def test_shapes_carried_state_and_causality_for_all_three_generic_models(self):
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.random.fork_rng(devices=[]), torch.inference_mode():
                torch.random.default_generator.manual_seed(8)
                x = torch.randn(2, 7, 5)
                for kind, factory in MODELS:
                    with self.subTest(kind=kind):
                        model = factory(input_size=5, hidden_size=9, output_size=3).eval()
                        self.assertTrue(model.recurrent.batch_first)
                        self.assertIsInstance(model.head, nn.Linear)
                        full, state = model(x)
                        first, carried = model(x[:, :3])
                        rest, streamed_state = model(x[:, 3:], carried)
                        self.assertEqual(tuple(full.shape), (2, 7, 3))
                        torch.testing.assert_close(torch.cat((first, rest), 1), full)
                        states = state if isinstance(state, tuple) else (state,)
                        streamed = streamed_state if isinstance(streamed_state, tuple) else (streamed_state,)
                        self.assertEqual(len(states), 2 if kind == "LSTM" else 1)
                        for final, incremental in zip(states, streamed):
                            self.assertEqual(tuple(final.shape), (1, 2, 9))
                            torch.testing.assert_close(final, incremental)
                        changed = x.clone()
                        changed[:, 3:] += 10
                        changed_scores, _ = model(changed)
                        torch.testing.assert_close(changed_scores[:, :3], first)
        finally:
            torch.set_num_threads(previous_threads)


class MusicExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        full = build_music_dataset()
        cls.data = replace(
            full, train_phrases=full.train_phrases[:16],
            heldout_phrases=full.heldout_phrases[:4],
        )
        cls.reports = {}
        for kind, factory in MODELS:
            cls.reports[kind] = run_music_experiment(
                factory, cls.data, kind=kind, updates=32, hidden_size=12, batch_size=8,
            )

    def test_minimal_training_improves_each_model_without_hiding_actual_guesses(self):
        for kind, report in self.reports.items():
            with self.subTest(kind=kind):
                json.dumps(report, allow_nan=False)
                self.assertLess(report["after"]["train"]["loss"], report["before"]["train"]["loss"])
                self.assertGreater(report["after"]["train"]["accuracy"], report["before"]["train"]["accuracy"])
                self.assertLess(report["after"]["heldout"]["loss"], report["before"]["heldout"]["loss"])
                self.assertEqual(len(report["training_losses"]), 32)
                self.assertEqual(report["dataset"]["fingerprint"], self.data.fingerprint)
                example = report["example"]
                self.assertEqual(example["reference_tokens"], list(self.data.heldout_phrases[0]))
                self.assertEqual(example["prompt_tokens"], example["reference_tokens"][:16])
                self.assertEqual(example["reference_continuation"], example["reference_tokens"][16:])
                for stage in ("before", "after"):
                    self.assertEqual(len(report[stage]["teacher_fed_next_tokens"]), 31)
                    continuation = report[stage]["generated_continuation"]
                    self.assertEqual(len(continuation), 16)
                    self.assertTrue(all(0 <= token < len(VOCABULARY) for token in continuation))
        self.assertEqual(len({r["dataset"]["fingerprint"] for r in self.reports.values()}), 1)
        self.assertEqual(len({tuple(r["example"]["prompt_tokens"]) for r in self.reports.values()}), 1)

    def test_repeated_seed_and_inline_factory_are_used_and_settings_are_restored(self):
        calls = []

        def inline_factory(**kwargs):
            calls.append(kwargs)
            return GRUSequenceModel(**kwargs)

        state = torch.random.get_rng_state().clone()
        python_state = random.getstate()
        threads = torch.get_num_threads()
        repeated = run_music_experiment(
            inline_factory, self.data, kind="GRU", updates=32, hidden_size=12, batch_size=8,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], {"input_size": len(VOCABULARY), "hidden_size": 12, "output_size": len(VOCABULARY)})
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(python_state, random.getstate())
        self.assertEqual(threads, torch.get_num_threads())
        for key in ("before", "after", "training_losses", "example", "config"):
            self.assertEqual(repeated[key], self.reports["GRU"][key])

    def test_generation_is_free_running_not_a_second_teacher_fed_pass(self):
        seen = []

        class AlwaysNext(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.bias = nn.Parameter(torch.zeros(output_size))
                self.output_size = output_size

            def forward(self, x, state=None):
                seen.append((x.argmax(-1).tolist(), state))
                scores = torch.zeros(*x.shape[:2], self.output_size) + self.bias
                scores[..., 2] += 10
                return scores, 7

        report = run_music_experiment(
            AlwaysNext, self.data, kind="RNN", updates=1, hidden_size=2, batch_size=2,
        )
        self.assertEqual(report["before"]["generated_continuation"], [2] * 16)
        prompt = [report["example"]["prompt_tokens"]]
        prompt_calls = [index for index, (tokens, state) in enumerate(seen) if tokens == prompt and state is None]
        self.assertEqual(len(prompt_calls), 2)
        for start in prompt_calls:
            for tokens, state in seen[start + 1:start + 16]:
                self.assertEqual(tokens, [[2]])
                self.assertEqual(state, 7)

    def test_audio_panel_is_offline_and_encodes_reference_and_actual_model_notes(self):
        report = self.reports["GRU"]
        panel = music_comparison_html(report)
        self.assertNotRegex(panel, r"https?://|<script|<iframe|autoplay")
        self.assertIn("Teacher-fed", panel)
        self.assertIn("different parameter counts", panel)
        encoded = re.findall(r'src="data:audio/wav;base64,([^"]+)"', panel)
        self.assertEqual(len(encoded), 3)
        prompt = report["example"]["prompt_tokens"]
        clips = [
            report["example"]["reference_tokens"],
            prompt + report["before"]["generated_continuation"],
            prompt + report["after"]["generated_continuation"],
        ]
        for uri, tokens in zip(encoded, clips):
            actual = base64.b64decode(uri)
            self.assertEqual(actual, melody_wav([VOCABULARY[token] for token in tokens]))
            with wave.open(BytesIO(actual), "rb") as audio:
                self.assertEqual((audio.getnchannels(), audio.getsampwidth(), audio.getframerate()), (1, 2, 16000))
                self.assertEqual(audio.getnframes(), 32 * 2560)
                pcm = torch.frombuffer(bytearray(audio.readframes(audio.getnframes())), dtype=torch.int16)
                self.assertLess(int(pcm.abs().max()), 32767)
                self.assertGreater(int(pcm.abs().max()), 1000)
        with wave.open(BytesIO(melody_wav([None])), "rb") as silent:
            self.assertEqual(set(silent.readframes(silent.getnframes())), {0})

    def test_rejects_split_leakage_and_restores_settings_when_factory_fails(self):
        leaked = replace(self.data, heldout_phrases=(self.data.train_phrases[0],))
        with self.assertRaisesRegex(ValueError, "disjoint"):
            run_music_experiment(GRUSequenceModel, leaked, kind="GRU", updates=1)
        state = torch.random.get_rng_state().clone()
        threads = torch.get_num_threads()

        def broken_factory(**kwargs):
            torch.rand(3)
            raise RuntimeError("inline edit failed")

        with self.assertRaisesRegex(RuntimeError, "inline edit"):
            run_music_experiment(broken_factory, self.data, kind="GRU", updates=1)
        self.assertEqual(threads, torch.get_num_threads())
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_synthesized_pitch_matches_the_actual_midi_note(self):
        with wave.open(BytesIO(melody_wav([62], seconds_per_note=0.5)), "rb") as audio:
            samples = torch.frombuffer(
                bytearray(audio.readframes(audio.getnframes())), dtype=torch.int16,
            ).float()
            spectrum = torch.fft.rfft(samples * torch.hann_window(len(samples))).abs()
            measured_hz = int(spectrum.argmax()) * audio.getframerate() / len(samples)
        self.assertAlmostEqual(measured_hz, 293.665, delta=3)


if __name__ == "__main__":
    unittest.main()

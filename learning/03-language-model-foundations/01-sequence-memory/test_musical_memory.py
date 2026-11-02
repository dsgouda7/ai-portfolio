"""Verify the learning claim, recorded GRU states, musical pitches, and offline displays."""

import ast
import base64
from io import BytesIO
import inspect
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import wave

import numpy as np
from PIL import Image
import torch

from memory_gate_flows import refresh_image_attachments
from musical_memory import (
    EchoGRU, MIDI, MODES, MusicFrame, fit_echo_model, make_echo_questions, music_frames,
    replay_html, run_echo_experiment, source_fingerprint, synthesize_timeline, wav_bytes,
)


HERE = Path(__file__).parent


class MusicalMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comparison = run_echo_experiment()
        cls.frames = music_frames(cls.comparison)
        cls.notebook = json.loads((HERE / "01A-sequence-memory-theory.ipynb").read_text(encoding="utf-8"))

    def test_questions_require_the_opening_not_the_counts_or_last_note(self):
        prompts, practice, answers = make_echo_questions()
        self.assertEqual(prompts.shape, (2, 24))
        self.assertTrue(torch.equal(prompts[0, 3:], prompts[1, 3:]))
        self.assertTrue(torch.equal(prompts[0, :3].sort().values, prompts[1, :3].sort().values))
        self.assertTrue(torch.equal(practice[:, -2:], answers[:, :2]))
        self.assertTrue(torch.equal(practice[:, :-2], prompts))
        self.assertEqual(prompts[0, -1].item(), 3)

    def test_real_free_running_guesses_establish_before_after_and_state_intervention(self):
        traces = self.comparison.traces
        self.assertEqual([trace.generated for trace in traces["Before learning"]], [(1, 1, 1)] * 2)
        self.assertEqual([trace.generated for trace in traces["After learning"]], [(0, 1, 2), (2, 1, 0)])
        self.assertEqual([trace.generated for trace in traces["Opening erased"]], [(2, 1, 0)] * 2)
        self.assertEqual([trace.generated for trace in traces["No carry"]], [(0, 1, 0)] * 2)
        self.assertLess(self.comparison.losses[-1], 0.002)
        self.assertGreater(self.comparison.losses[0], 1)
        for mode, cases in traces.items():
            for trace in cases:
                for index in range(1, 3):
                    self.assertEqual(trace.outputs[index].input_id, trace.generated[index - 1], mode)

    def test_recorded_gates_reconstruct_actual_cells_and_distinguish_erasing_from_detaching(self):
        for cases in self.comparison.traces.values():
            for trace in cases:
                for step in (*trace.prompt, *trace.outputs):
                    torch.testing.assert_close(step.retained + step.written, step.hidden)
                    torch.testing.assert_close(step.retained, step.keep * step.old)
                    torch.testing.assert_close(step.written, (1 - step.keep) * step.candidate)
                    for gate in (step.reset, step.keep):
                        self.assertTrue(bool(((gate >= 0) & (gate <= 1)).all()))
        after = self.comparison.traces["After learning"]
        erased = self.comparison.traces["Opening erased"]
        for case in range(2):
            for index in range(3):
                torch.testing.assert_close(after[case].prompt[index].hidden, erased[case].prompt[index].hidden)
            self.assertGreater(after[case].prompt[3].old.norm().item(), 0)
            self.assertEqual(erased[case].prompt[3].old.norm().item(), 0)
        torch.testing.assert_close(erased[0].prompt[-1].logits, erased[1].prompt[-1].logits)

    def test_experiment_preserves_rng_and_thread_settings(self):
        state = torch.random.get_rng_state().clone()
        threads = torch.get_num_threads()
        run_echo_experiment()
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(threads, torch.get_num_threads())

    def test_keyboard_sound_uses_the_named_notes_and_timing_without_clipping(self):
        for note, midi in MIDI.items():
            frames = [MusicFrame("pitch test", "", note, 0.55)]
            samples = synthesize_timeline(frames)
            self.assertEqual(len(samples), 8800)
            self.assertLess(np.abs(samples).max(), 1)
            spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
            measured = np.fft.rfftfreq(len(samples), 1 / 16000)[spectrum.argmax()]
            self.assertAlmostEqual(measured, 440 * 2 ** ((midi - 69) / 12), delta=2)
        silence = synthesize_timeline([MusicFrame("silent instruction", "", None, 0.38)])
        self.assertTrue(np.all(silence == 0))
        with wave.open(BytesIO(wav_bytes(silence)), "rb") as audio:
            self.assertEqual((audio.getnchannels(), audio.getsampwidth(), audio.getframerate()), (1, 2, 16000))
            self.assertEqual(audio.getnframes(), len(silence))

    def test_all_replayed_notes_come_from_the_actual_input_or_guess(self):
        self.assertLessEqual(len(self.frames), 60)
        self.assertEqual(sum(frame.title.startswith("Read ") for frame in self.frames), 24)
        self.assertLess(sum(frame.duration for frame in self.frames), 40)
        for frame in self.frames:
            if frame.title.startswith("Read "):
                expected = ("D3", "A3", "D4")[frame.step.input_id] if frame.step.input_id < 3 else None
                self.assertEqual(frame.note, expected)
            elif frame.mode:
                index = int(re.search(r"note (\d)/3", frame.title)[1]) - 1
                trace = self.comparison.traces[frame.mode][frame.case]
                self.assertEqual(frame.note, ("D3", "A3", "D4")[trace.generated[index]])

    def test_notebook_learning_code_matches_the_reusable_implementation(self):
        code = "\n".join("".join(cell["source"]) for cell in self.notebook["cells"] if cell["cell_type"] == "code")
        definitions = {node.name: node for node in ast.parse(code).body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        for symbol in (EchoGRU, fit_echo_model, make_echo_questions):
            expected = ast.parse(inspect.getsource(symbol)).body[0]
            self.assertEqual(ast.dump(definitions[symbol.__name__]), ast.dump(expected))
        self.assertNotIn("render_replay", code)
        self.assertNotIn("replay_html", code)
        self.assertNotIn("matplotlib", code)
        self.assertIn("CHAPTER_DIRECTORY", code)

    def test_static_diagrams_are_embedded_and_all_saved_links_resolve(self):
        attachments = 0
        for cell in self.notebook["cells"]:
            if cell["cell_type"] != "markdown":
                continue
            text = "".join(cell["source"])
            for target in re.findall(r"\]\(([^)]+)\)", text):
                if target.startswith("attachment:"):
                    name = target.removeprefix("attachment:")
                    embedded = base64.b64decode(cell["attachments"][name]["image/png"])
                    self.assertEqual(embedded, (HERE / "images" / name).read_bytes())
                    with Image.open(BytesIO(embedded)) as image:
                        self.assertGreater(image.width, 500)
                    attachments += 1
                elif not target.startswith(("https:", "#")):
                    self.assertTrue((HERE / target.split("#")[0]).exists(), target)
        self.assertGreaterEqual(attachments, 7)

    def test_explicit_capture_rebuilds_refresh_embedded_images_without_touching_code_or_outputs(self):
        snapshot = json.loads(json.dumps(self.notebook))
        attached = next(cell for cell in snapshot["cells"] if cell.get("attachments"))
        for image in attached["attachments"].values():
            image["image/png"] = ""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "notebook.ipynb"
            path.write_bytes((json.dumps(snapshot, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
            with patch("memory_gate_flows.NOTEBOOK", path):
                refresh_image_attachments()
            refreshed = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(refreshed, self.notebook)

    def test_saved_audio_and_animation_describe_this_exact_run(self):
        prefix = HERE / "images" / "01a-musical-gru-memory"
        with Image.open(prefix.with_suffix(".gif")) as image:
            metadata = json.loads(image.info["comment"])
            self.assertEqual(metadata["SourceFingerprint"], source_fingerprint())
            self.assertEqual(image.n_frames, len(self.frames))
            self.assertEqual(image.size, (1200, 700))
            self.assertEqual(metadata["TrainingSteps"], 240)
            self.assertEqual(metadata["Distractions"], 20)
            for mode in MODES:
                self.assertEqual(metadata["Predictions"][mode], [
                    list(trace.generated) for trace in self.comparison.traces[mode]
                ])
            self.assertEqual([entry["note"] for entry in metadata["Frames"]], [frame.note for frame in self.frames])
        audio_bytes = prefix.with_suffix(".wav").read_bytes()
        with wave.open(BytesIO(audio_bytes), "rb") as audio:
            self.assertEqual(audio.getnframes(), len(synthesize_timeline(self.frames)))
        self.assertEqual(audio_bytes, wav_bytes(synthesize_timeline(self.frames)))
        html = prefix.with_suffix(".html").read_text(encoding="utf-8")
        self.assertIn("data:audio/wav;base64,", html)
        self.assertIn("data:image/png;base64,", html)
        self.assertIn('audio.addEventListener("seeked", sync)', html)
        self.assertIn("audio.currentTime", html)
        self.assertIn("width:100%;height:auto", html)
        self.assertNotIn("https:", html)

    def test_independent_replays_do_not_share_dom_ids(self):
        image = BytesIO()
        Image.new("RGB", (4, 4), "white").save(image, format="png")
        frames = [MusicFrame("test", "", None, 0.1)]
        audio = wav_bytes(synthesize_timeline(frames))
        first = replay_html(frames, [image.getvalue()], audio)
        second = replay_html(frames, [image.getvalue()], audio)
        self.assertNotEqual(re.search(r'<section id="([^"]+)"', first)[1],
                            re.search(r'<section id="([^"]+)"', second)[1])


if __name__ == "__main__":
    unittest.main()

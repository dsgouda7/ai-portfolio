"""Verify self-contained HTML embedding and pixel-preserving frame packaging."""

import base64
from copy import deepcopy
import hashlib
from html import unescape
from html.parser import HTMLParser
from io import BytesIO
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from xml.etree import ElementTree

import numpy as np
from PIL import Image

from notebook_replays import (
    NOTEBOOK_SIZE_LIMIT, REPLAY_FILES, _packed_html, embed_notebook_images,
    refresh_embedded_replays, saved_replay,
)


HERE = Path(__file__).parent


class FrameParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        if tag == "iframe":
            self.frame = dict(attrs)


class MediaParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []
        self.audio = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag in ("img", "audio", "video", "source", "script", "link", "iframe"):
            for key in ("src", "href", "poster"):
                if values.get(key):
                    self.sources.append(values[key])
        if tag == "audio":
            self.audio.append(values["src"])


class NotebookReplayTests(unittest.TestCase):
    def fixture(self, directory):
        pixels = np.arange(8 * 9 * 4, dtype=np.uint16).reshape(8, 9, 4).astype(np.uint8)
        png = BytesIO()
        Image.fromarray(pixels).save(png, format="PNG")
        source = "data:image/png;base64," + base64.b64encode(png.getvalue()).decode()
        html = f'<img src="{source}"><button onclick="document.title=\'next\'">Next</button><script>var x=1;</script>'
        path = directory / REPLAY_FILES[0]
        path.write_bytes(html.encode())
        return pixels, path

    def test_full_html_and_scripts_are_embedded_in_an_isolated_frame(self):
        with TemporaryDirectory() as directory:
            pixels, path = self.fixture(Path(directory))
            result = saved_replay(path.name, images_directory=path.parent).data
        self.assertIn('sandbox="allow-scripts"', result)
        self.assertNotIn("allow-same-origin", result)
        parser = FrameParser()
        parser.feed(result)
        self.assertNotIn("src", parser.frame)
        document = unescape(re.search(r'srcdoc="([^"]+)"', result)[1])
        self.assertIn("<script>var x=1;</script>", document)
        self.assertIn("document.title='next'", document)
        self.assertNotIn("data:image/png", document)
        encoded = re.search(r"data:image/webp;base64,([^\"<]+)", document)[1]
        with Image.open(BytesIO(base64.b64decode(encoded))) as decoded:
            np.testing.assert_array_equal(np.asarray(decoded.convert("RGBA")), pixels)

    def test_missing_invalid_or_frame_free_replays_report_errors(self):
        with self.assertRaisesRegex(ValueError, "Unknown saved"):
            saved_replay("../private.html")
        with TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                saved_replay(REPLAY_FILES[0], images_directory=Path(directory))
        with self.assertRaisesRegex(ValueError, "no embedded PNG"):
            _packed_html(b"<p>not a replay</p>")

    def test_musical_frame_payload_and_clock_survive_lossless_packaging(self):
        with TemporaryDirectory() as directory:
            pixels, path = self.fixture(Path(directory))
            png = BytesIO()
            Image.fromarray(pixels).save(png, format="PNG")
            encoded = base64.b64encode(png.getvalue()).decode()
            payload = {"images": [encoded, encoded], "starts": [0, 0.55]}
            source = (path.read_text() + "<script>const data = " + json.dumps(payload)
                      + ';image.src="data:image/png;base64,"+data.images[1];</script>')
            packed = _packed_html(source.encode())
        start = packed.index('{"images":')
        decoded, _ = json.JSONDecoder().raw_decode(packed[start:])
        self.assertEqual(decoded["starts"], payload["starts"])
        self.assertIn('image.src="data:image/webp;base64,"', packed)
        for frame in decoded["images"]:
            with Image.open(BytesIO(base64.b64decode(frame))) as image:
                np.testing.assert_array_equal(np.asarray(image.convert("RGBA")), pixels)

    def test_refresh_updates_only_marked_display_outputs_and_keeps_sources(self):
        with TemporaryDirectory() as directory:
            _, path = self.fixture(Path(directory))
            experiment = {"cell_type": "code", "source": ["print('keep')"], "metadata": {},
                          "outputs": [{"output_type": "stream", "text": "keep"}], "execution_count": 8}
            display = {"cell_type": "code", "source": [f'saved_replay("{path.name}")'],
                       "metadata": {"saved_replay": {"file": path.name}}, "outputs": [], "execution_count": None}
            book = {"cells": [experiment, display]}
            original = deepcopy(experiment)
            refresh_embedded_replays(book, images_directory=path.parent)
            self.assertEqual(book["cells"][0], original)
            self.assertEqual(display["source"], [f'saved_replay("{path.name}")'])
            self.assertEqual(display["metadata"]["saved_replay"]["source_sha256"],
                             hashlib.sha256(path.read_bytes()).hexdigest())
            snapshot = deepcopy(book)
            refresh_embedded_replays(book, images_directory=path.parent)
            self.assertEqual(book, snapshot)
            display["outputs"][0]["data"]["text/html"] = ""
            refresh_embedded_replays(book, images_directory=path.parent)
            self.assertEqual(book, snapshot)

    def test_notebook_has_all_saved_players_with_persistent_html_outputs(self):
        book = json.loads((HERE / "01A-sequence-memory-theory.ipynb").read_text(encoding="utf-8"))
        cells = [cell for cell in book["cells"] if "saved_replay" in cell.get("metadata", {})]
        self.assertEqual({cell["metadata"]["saved_replay"]["file"] for cell in cells}, set(REPLAY_FILES))
        self.assertEqual(len(cells), len(REPLAY_FILES))
        for cell in cells:
            filename = cell["metadata"]["saved_replay"]["file"]
            self.assertEqual("".join(cell["source"]), f'saved_replay("{filename}")')
            self.assertEqual(len(cell["outputs"]), 1)
            html = cell["outputs"][0]["data"]["text/html"]
            self.assertIn('srcdoc="', html)
            self.assertIn(f'data-saved-replay="{filename}"', html)
        self.assertLess((HERE / "01A-sequence-memory-theory.ipynb").stat().st_size, NOTEBOOK_SIZE_LIMIT)

    def test_png_and_svg_links_become_persistent_attachments_without_network_paths(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            _, path = self.fixture(directory)
            pixels = np.zeros((4, 4, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(directory / "picture.png")
            svg = '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0L1 1"/></svg>'
            (directory / "vector.svg").write_text(svg, encoding="utf-8")
            cell = {"cell_type": "markdown", "source": [
                "![Picture](images/picture.png)\n[Zoomable vector](images/vector.svg)\n",
            ]}
            book = {"cells": [cell]}
            embed_notebook_images(book, images_directory=directory)
            self.assertNotIn("images/", "".join(cell["source"]))
            self.assertIn("attachment:vector.svg", "".join(cell["source"]))
            self.assertEqual(cell["attachments"]["vector.svg"]["image/svg+xml"], svg)
            self.assertEqual(
                base64.b64decode(cell["attachments"]["picture.png"]["image/png"]),
                (directory / "picture.png").read_bytes(),
            )
            snapshot = deepcopy(book)
            embed_notebook_images(book, images_directory=directory)
            self.assertEqual(book, snapshot)

    def test_notebook_visuals_and_audio_have_no_separate_asset_dependencies(self):
        book = json.loads((HERE / "01A-sequence-memory-theory.ipynb").read_text(encoding="utf-8"))
        players = {}
        for cell in book["cells"]:
            if cell["cell_type"] == "markdown":
                source = "".join(cell["source"])
                self.assertNotRegex(source, r"\]\(images/[^)]+\.(?:png|gif|svg|webp|wav|html)\)")
                for filename in re.findall(r"\]\(attachment:([^)]+)\)", source):
                    self.assertIn(filename, cell.get("attachments", {}))
                    bundle = cell["attachments"][filename]
                    if "image/svg+xml" in bundle:
                        vector = ElementTree.fromstring(bundle["image/svg+xml"])
                        for element in vector.iter():
                            for key, value in element.attrib.items():
                                if key.endswith("href"):
                                    self.assertTrue(value.startswith(("#", "data:")))
            marker = cell.get("metadata", {}).get("saved_replay")
            outputs = [
                item for item in cell.get("outputs", []) if "text/html" in item.get("data", {})
            ]
            if marker:
                self.assertEqual(len(outputs), 1)
            for output in outputs:
                document = "".join(output["data"]["text/html"])
                frame = re.search(r'srcdoc="([^"]+)"', document)
                if frame:
                    document = unescape(frame[1])
                media = MediaParser()
                media.feed(document)
                self.assertTrue(all(value.startswith(("data:", "#")) for value in media.sources))
                if marker:
                    self.assertIsNotNone(frame)
                    players[marker["file"]] = media
        self.assertEqual(set(players), set(REPLAY_FILES))
        story = next(cell for cell in book["cells"] if cell["id"] == "968f14e7")
        original = story["attachments"]["01a-backprop-saved-cards.png"]["image/png"]
        self.assertEqual(
            base64.b64decode(original),
            (HERE / "images" / "01a-backprop-saved-cards.png").read_bytes(),
        )
        for kind in ("lstm", "gru"):
            media = players[f"01a-musical-{kind}-memory.html"]
            self.assertEqual(len(media.audio), 1)
            recording = base64.b64decode(media.audio[0].split(",", 1)[1], validate=True)
            self.assertEqual(recording, (HERE / "images" / f"01a-musical-{kind}-memory.wav").read_bytes())


if __name__ == "__main__":
    unittest.main()

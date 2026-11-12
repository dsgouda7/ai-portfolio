"""Embed saved HTML players without external paths or shared script globals."""

import base64
from functools import lru_cache
import hashlib
from html import escape
from io import BytesIO
import json
from pathlib import Path
import re

from IPython.display import HTML
from PIL import Image


HERE = Path(__file__).resolve().parent
ERROR_FEEDBACK_FILE = "01a-error-feedback-tracker.html"
LSTM_FORWARD_FILE = "01a-lstm-forward-walkthrough.html"
NATIVE_PLAYER_MARKERS = {
    ERROR_FEEDBACK_FILE: "error-feedback-tracker",
    LSTM_FORWARD_FILE: "lstm-forward-walkthrough",
}
REPLAY_FILES = (
    "01a-backprop-saved-cards.html",
    "01a-lstm-gate-flow.html",
    "01a-lstm-token-layer-flow.html",
    "01a-musical-lstm-memory.html",
    "01a-gru-gate-flow.html",
    "01a-gru-token-layer-flow.html",
    "01a-musical-gru-memory.html",
    ERROR_FEEDBACK_FILE,
    LSTM_FORWARD_FILE,
)
EMBED_VERSION = 1
NOTEBOOK_SIZE_LIMIT = 65_000_000
PNG_URL = re.compile(r"data:image/png;base64,([A-Za-z0-9+/=\\\r\n]+)")
IMAGE_LINK = re.compile(r"(!?)\[([^\]\n]+)\]\((?:images/|attachment:)([^)\n]+)\)")
IMAGE_MIMES = {
    ".png": "image/png", ".gif": "image/gif", ".svg": "image/svg+xml",
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
}


def embed_notebook_images(notebook: dict, *, images_directory: Path | None = None,
                          refresh_existing: bool = False) -> None:
    directory = images_directory if images_directory is not None else HERE / "images"
    for cell in notebook["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        source = "".join(cell["source"])

        def embed(match):
            image, caption, filename = match.groups()
            mime = IMAGE_MIMES.get(Path(filename).suffix.lower())
            if mime is None:
                return match[0]
            attachments = cell.setdefault("attachments", {})
            if refresh_existing or mime not in attachments.get(filename, {}):
                path = directory / filename
                data = path.read_text(encoding="utf-8") if mime == "image/svg+xml" else (
                    base64.b64encode(path.read_bytes()).decode("ascii")
                )
                attachments[filename] = {mime: data}
            embedded = f"![{caption}](attachment:{filename})"
            if image:
                return embedded
            return f"<details><summary>{caption}</summary>\n\n{embedded}\n\n</details>"

        updated = IMAGE_LINK.sub(embed, source)
        if updated != source:
            cell["source"] = updated.splitlines(keepends=True)


@lru_cache(maxsize=len(REPLAY_FILES))
def _packed_html(raw: bytes) -> str:
    html = raw.decode("utf-8")
    converted = {}

    def image_url(encoded):
        encoded = encoded.replace("\\\r\n", "").replace("\\\n", "").replace("\r", "").replace("\n", "")
        if encoded not in converted:
            image_bytes = base64.b64decode(encoded, validate=True)
            buffer = BytesIO()
            with Image.open(BytesIO(image_bytes)) as image:
                image.save(buffer, format="WEBP", lossless=True, exact=True, method=6)
            converted[encoded] = "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        return converted[encoded]

    packed = PNG_URL.sub(lambda match: image_url(match[1]), html)
    start = packed.find('{"images":')
    if start >= 0:
        payload, length = json.JSONDecoder().raw_decode(packed[start:])
        payload["images"] = [image_url(encoded).split(",", 1)[1] for encoded in payload["images"]]
        packed = packed[:start] + json.dumps(payload) + packed[start + length:]
        packed = packed.replace("data:image/png;base64,", "data:image/webp;base64,")
    if not converted:
        raise ValueError("The saved replay contains no embedded PNG frames to package.")
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '</head><body>' + packed
        + "<style>body{margin:8px;font:16px system-ui;overflow-x:hidden}"
        "img,.animation img{max-width:100%;max-height:calc(100vh - 165px);object-fit:contain}"
        "button{min-height:36px;font:inherit;padding:4px 10px}"
        "audio{max-width:100%}.animation .anim-controls{max-width:100%}</style>"
        "</body></html>"
    )


def _player_height(filename: str) -> int:
    if filename == LSTM_FORWARD_FILE:
        return 1200
    return 1800 if filename == ERROR_FEEDBACK_FILE else 1050 if "musical" in filename else 740


def saved_replay(filename: str, *, images_directory: Path | None = None) -> HTML:
    if filename not in REPLAY_FILES:
        raise ValueError(f"Unknown saved lesson replay: {filename}")
    path = (images_directory if images_directory is not None else HERE / "images") / filename
    raw = path.read_bytes()
    if filename in NATIVE_PLAYER_MARKERS:
        packed = raw.decode("utf-8")
        marker = NATIVE_PLAYER_MARKERS[filename]
        if f'data-player="{marker}"' not in packed or "<script>" not in packed:
            raise ValueError("The saved tracker does not contain its interactive player.")
    else:
        packed = _packed_html(raw)
    height = _player_height(filename)
    title = filename.removesuffix(".html").replace("01a-", "").replace("-", " ")
    return HTML(
        f'<iframe title="{escape(title, quote=True)}" sandbox="allow-scripts" '
        f'srcdoc="{escape(packed, quote=True)}" '
        f'style="display:block;width:100%;height:{height}px;border:0;background:white" '
        f'data-saved-replay="{filename}"></iframe>'
    )


def refresh_embedded_replays(notebook: dict, *, images_directory: Path | None = None) -> None:
    directory = images_directory if images_directory is not None else HERE / "images"
    embed_notebook_images(notebook, images_directory=directory, refresh_existing=True)
    for cell in notebook["cells"]:
        replay = cell.get("metadata", {}).get("saved_replay")
        if replay is None:
            continue
        filename = replay["file"]
        native_markdown = filename in NATIVE_PLAYER_MARKERS and cell["cell_type"] == "markdown"
        if filename not in REPLAY_FILES or (not native_markdown and cell["cell_type"] != "code"):
            raise ValueError("Saved replay metadata must identify a supported display cell.")
        source = "".join(cell["source"]).strip()
        expected_source = "" if filename in NATIVE_PLAYER_MARKERS else f'saved_replay("{filename}")'
        if native_markdown and source and hashlib.sha256(source.encode()).hexdigest() != replay.get("output_sha256"):
            raise ValueError("The saved Markdown player changed; do not overwrite its source.")
        if not native_markdown and source != expected_source:
            raise ValueError("The saved replay display cell changed; do not overwrite its output.")
        digest = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        outputs = cell.get("outputs", [])
        output_html = source if native_markdown else (
            "".join(outputs[0].get("data", {}).get("text/html", "")) if len(outputs) == 1 else ""
        )
        if (replay.get("source_sha256") == digest and replay.get("embed_version") == EMBED_VERSION
                and output_html.startswith("<iframe ")
                and f"height:{_player_height(filename)}px" in output_html
                and replay.get("output_sha256") == hashlib.sha256(output_html.encode()).hexdigest()):
            continue
        html = saved_replay(filename, images_directory=directory).data
        if native_markdown:
            html = html.replace("\r", "&#13;").replace("\n", "&#10;")
            cell["source"] = [html]
            cell.pop("outputs", None)
            cell.pop("execution_count", None)
        else:
            cell["outputs"] = [{
                "output_type": "display_data",
                "data": {"text/html": html, "text/plain": f"Embedded saved player: {filename}"},
                "metadata": {},
            }]
            cell["execution_count"] = None
        replay.update({
            "source_sha256": digest,
            "output_sha256": hashlib.sha256(html.encode()).hexdigest(),
            "embed_version": EMBED_VERSION,
        })

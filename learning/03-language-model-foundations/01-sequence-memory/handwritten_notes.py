"""Shared offline paper and handwriting for measured calculation players."""

import base64
from functools import lru_cache
from html import escape
from io import BytesIO
from pathlib import Path
import re

from PIL import Image


HERE = Path(__file__).resolve().parent
PAPER = HERE / "images" / "01a-handwritten-paper.png"
FONT = HERE / "images" / "kalam" / "Kalam-Regular.ttf"


@lru_cache(maxsize=1)
def paper_style() -> str:
    buffer = BytesIO()
    with Image.open(PAPER) as image:
        image.save(buffer, format="WEBP", quality=80, method=6)
    paper = base64.b64encode(buffer.getvalue()).decode("ascii")
    font = base64.b64encode(FONT.read_bytes()).decode("ascii")
    return (
        "@font-face{font-family:LessonHand;src:url(data:font/ttf;base64,"
        + font + ") format('truetype');font-style:normal;font-weight:400;font-display:swap}"
        ".paper{background:#fff url(data:image/webp;base64," + paper + ") center/100% 100%;"
        "color:#23334a;font-family:LessonHand,cursive}"
        ".paper sub,.paper sup{font-size:.72em;line-height:0}"
        ".paper .ink{color:#234875}.paper .pencil{color:#596273}"
    )


def handwritten_math(value: str) -> str:
    """Keep the original numbers while making subscripts and operators readable."""
    text = escape(value)
    text = text.replace("prod_(t=2..21)", "∏<sub>t=2..21</sub>")
    text = text.replace("sum W_out[k]", "∑<sub>k</sub> W_out[k]")
    text = text.replace("sum (", "∑<sub>t</sub> (")
    text = text.replace("learning_rate", "step size")
    text = re.sub(r"\b([A-Za-z]+)_\(([^)]+)\)", r"\1<sub>\2</sub>", text)
    text = re.sub(r"\b([A-Za-z]+)_([A-Za-z0-9]+)", r"\1<sub>\2</sub>", text)
    text = re.sub(r"\^([0-9]+)", r"<sup>\1</sup>", text)
    return text.replace(" * ", " × ")

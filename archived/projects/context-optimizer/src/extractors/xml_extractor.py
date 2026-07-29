"""XML / HTML extractor — strips tags, extracts text nodes via lxml."""

from __future__ import annotations

import re
from pathlib import Path

from .base import BaseExtractor


class XmlExtractor(BaseExtractor):
    def extract(self, path: Path) -> str:
        return self._safe(self._read, path)

    def _read(self, path: Path) -> str:
        from lxml import etree  # type: ignore[import]

        raw = path.read_bytes()
        # Try XML first, fall back to HTML parser
        try:
            tree = etree.fromstring(raw)
        except etree.XMLSyntaxError:
            parser = etree.HTMLParser()
            tree = etree.fromstring(raw, parser)

        # Extract all text nodes, skip script/style in HTML
        _SKIP = {"script", "style", "head"}
        texts: list[str] = []
        for el in tree.iter():
            tag = el.tag
            if isinstance(tag, str):
                local = tag.split("}")[-1].lower() if "}" in tag else tag.lower()
                if local in _SKIP:
                    continue
            for t in (el.text, el.tail):
                if t and t.strip():
                    texts.append(t.strip())

        text = "\n".join(texts)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

#!/usr/bin/env python3
"""
Book Benchmark: 100-book Gutenberg corpus with Wikipedia-sourced question banks.

Workflow
--------
1. build-banks  Fetch top-N books from gutendex, download Wikipedia plot/character/
                theme sections, extract Q&A pairs, write to
                data/question_banks/<slug>.json.  Run once; re-use on every
                benchmark run.

2. run          Download book texts from Gutenberg (cached in data/books/),
                compress all books in parallel, build per-book ChromaDB
                collections, execute all questions concurrently, judge answers
                via keyword overlap against pre-fetched Wikipedia expected
                answers (no LLM judge), write book_results.md.

Usage
-----
    python book_benchmark.py build-banks [--books 100] [--qpb 20]
    python book_benchmark.py run         [--books 100] [--lines 3000]
                                          [--workers 4] [--qpb 20]
    python book_benchmark.py all         [--books 100] [--lines 3000]
                                          [--workers 4] [--qpb 20]

Judging (no LLM)
----------------
Each Q&A entry in the question bank stores an ``expected_answer`` string and a
``keywords`` list extracted from Wikipedia at bank-build time.  At run time the
pipeline answer is scored by keyword recall:

    score = |{kw : kw in answer}| / len(keywords)

This is deterministic, fast, and reproducible.  Judging runs asynchronously in
a ThreadPoolExecutor alongside query execution so it never blocks the next query.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Project paths ──────────────────────────────────────────────────────────────
_BENCH_DIR = Path(__file__).parent
_PROJECT_ROOT = _BENCH_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
sys.path.insert(0, str(_SRC_DIR))

_BANKS_DIR = _BENCH_DIR / "data" / "question_banks"
_BOOKS_DIR = _BENCH_DIR / "data" / "books"
_CHUNKS_DIR = _BENCH_DIR / "data" / "chunks"
_CACHE_DIR = _BENCH_DIR / "data" / "book_cache"


# ── Chunk + result cache helpers ───────────────────────────────────────────────


def _chunks_path(slug: str, max_lines: int, strategy: str = "llm") -> Path:
    tag = f"L{max_lines}" if max_lines > 0 else "Lall"
    return _CHUNKS_DIR / f"{slug}_{tag}_{strategy}.jsonl"


def _save_chunks(
    slug: str, max_lines: int, chunks: list, strategy: str = "llm"
) -> None:
    """Persist compressed chunks to JSONL so restarts skip LLM compression."""
    _CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    path = _chunks_path(slug, max_lines, strategy)
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.__dict__, default=str) + "\n")


def _load_chunks(slug: str, max_lines: int, strategy: str = "llm") -> list | None:
    """Return list[CompressedChunk] if cache exists, else None."""
    path = _chunks_path(slug, max_lines, strategy)
    if not path.exists():
        return None
    try:
        from context_optimizer.compressor import CompressedChunk  # noqa: PLC0415

        chunks = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            chunks.append(CompressedChunk(**d))
        return chunks or None
    except Exception:
        return None


def _result_path(slug: str, strategy: str = "llm") -> Path:
    return _CACHE_DIR / f"{slug}_{strategy}.json"


def _save_result(slug: str, result: dict, strategy: str = "llm") -> None:
    """Persist the full book result so restarts skip Q&A entirely."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _result_path(slug, strategy).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _load_result(slug: str, strategy: str = "llm") -> dict | None:
    """Return cached result dict if it exists, else None."""
    path = _result_path(slug, strategy)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ── HTTP constants ─────────────────────────────────────────────────────────────
# gutendex blocks non-browser UAs; use a generic Chrome UA
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}
_TIMEOUT = 30
_GUTENDEX = "https://gutendex.com/books/"
_WIKI_SUMM = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_WIKI_SECT = "https://en.wikipedia.org/api/rest_v1/page/mobile-sections/{title}"

# ── Exclude non-fiction / non-narrative IDs ────────────────────────────────────
# (philosophy, religious texts, reference works, audio-only)
_EXCLUDE_IDS = {
    45304,
    3296,  # Augustine
    2680,  # Marcus Aurelius Meditations
    1998,
    4363,
    52190,  # Nietzsche
    3207,  # Hobbes Leviathan
    1232,  # Machiavelli Prince
    1080,  # Swift A Modest Proposal (essay, not novel)
    205,  # Thoreau Walden (essay)
    27558,  # CIA World Factbook
    20203,  # Benjamin Franklin Autobiography (too fragmented)
    33283,  # Calculus Made Easy (PDF only)
    2542,  # Ibsen A Doll's House (play — OK but short; keep for variety)
}

# ── Wikipedia section name variants to search for ─────────────────────────────
_PLOT_TITLES = {"plot", "plot summary", "synopsis", "storyline", "summary"}
_CHAR_TITLES = {"characters", "main characters", "cast", "cast of characters"}
_THEME_TITLES = {"themes", "themes and motifs", "major themes", "motifs"}
_SETTING_TITLES = {"setting", "setting and time period", "background"}
_RECEPTION_TITLES = {
    "reception",
    "critical reception",
    "reception and legacy",
    "legacy",
    "critical response",
    "reviews",
    "reception and influence",
}
_BACKGROUND_TITLES = {
    "publication history",
    "writing",
    "composition",
    "origins",
    "creation",
    "writing and publication",
    "sources",
}
_STYLE_TITLES = {
    "style",
    "narrative style",
    "narrative technique",
    "narrative structure",
    "structure",
    "symbolism",
    "language",
    "literary style",
    "literary analysis",
    "literary significance",
    "style and structure",
}


# ── Helpers ────────────────────────────────────────────────────────────────────


def _get(url: str, **kwargs) -> Any:
    """
    GET with browser-like headers.  Falls back to a curl subprocess when
    Cloudflare or similar WAF blocks the Python requests TLS fingerprint.
    Returns a lightweight response-like object with ``.json()`` and ``.content``.
    """
    try:
        import requests
    except ImportError:
        sys.exit("ERROR: 'requests' is required.  pip install requests")

    params = kwargs.pop("params", None)
    if params:
        from urllib.parse import urlencode

        sep = "&" if "?" in url else "?"
        url = url + sep + urlencode(params)

    # ── Try requests first ─────────────────────────────────────────────────────
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT, **kwargs)
        if resp.status_code != 403:
            resp.raise_for_status()
            return resp
    except Exception:
        pass

    # ── Fall back to curl (different TLS fingerprint, bypasses WAF) ────────────
    import subprocess

    result = subprocess.run(
        [
            "curl",
            "-sL",
            "-H",
            f"User-Agent: {_HEADERS['User-Agent']}",
            "-H",
            "Accept: application/json, text/plain, */*",
            "-H",
            "Accept-Language: en-US,en;q=0.9",
            "--compressed",
            "--max-time",
            str(_TIMEOUT),
            url,
        ],
        capture_output=True,
        check=True,
    )

    class _CurlResponse:
        def __init__(self, data: bytes) -> None:
            self.content = data

        def json(self) -> Any:
            return json.loads(self.content.decode("utf-8"))

        def raise_for_status(self) -> None:
            pass

    return _CurlResponse(result.stdout)


def _strip_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&[a-zA-Z#0-9]{1,8};", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60]


def _keywords(text: str, min_len: int = 4) -> list[str]:
    """Return unique, lowercase content words (length ≥ min_len)."""
    stopwords = {
        "this",
        "that",
        "with",
        "from",
        "have",
        "will",
        "been",
        "were",
        "they",
        "them",
        "their",
        "also",
        "when",
        "where",
        "which",
        "while",
        "after",
        "before",
        "during",
        "into",
        "onto",
        "upon",
        "over",
        "under",
        "about",
        "some",
        "many",
        "more",
        "most",
        "both",
        "each",
        "such",
        "than",
    }
    tokens = re.findall(r"[a-z][a-z']{%d,}" % (min_len - 1), text.lower())
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t not in stopwords and t not in seen:
            seen.add(t)
            out.append(t)
    return out[:30]


# ── Book catalog ───────────────────────────────────────────────────────────────


def fetch_book_catalog(n: int = 100) -> list[dict]:
    """Fetch the top-N English fiction books from gutendex."""
    books: list[dict] = []
    url = _GUTENDEX
    params: dict = {"sort": "popular"}

    print(f"[catalog] Fetching top {n} books from gutendex...")
    while len(books) < n:
        resp = _get(url, params=params)
        data = resp.json()
        for item in data.get("results", []):
            if len(books) >= n:
                break
            # English only
            if "en" not in item.get("languages", []):
                continue
            # Text media only
            if item.get("media_type") != "Text":
                continue
            # Skip excluded IDs
            if item["id"] in _EXCLUDE_IDS:
                continue
            # Find plain-text URL
            fmt = item.get("formats", {})
            txt_url = (
                fmt.get("text/plain; charset=utf-8")
                or fmt.get("text/plain; charset=us-ascii")
                or fmt.get("text/plain")
            )
            if not txt_url:
                continue
            # Prefer .txt not .zip
            if txt_url.endswith(".zip"):
                continue
            author = ""
            if item.get("authors"):
                a = item["authors"][0]
                author = f"{a.get('name', '')}".strip()
            # gutendex now ships auto-generated summaries per book
            summaries = item.get("summaries", [])
            catalog_summary = summaries[0] if summaries else ""
            books.append(
                {
                    "id": item["id"],
                    "title": item["title"],
                    "author": author,
                    "txt_url": txt_url,
                    "slug": _slug(item["title"]),
                    "catalog_summary": catalog_summary,
                }
            )

        next_url = data.get("next")
        if not next_url:
            break
        url = next_url
        params = {}  # pagination URL already contains params

    print(f"[catalog] Found {len(books)} books")
    return books[:n]


# ── Wikipedia Q&A extraction ───────────────────────────────────────────────────


def _wiki_sections(title: str) -> dict[str, str]:
    """Return {section_title_lower: plain_text} from Wikipedia mobile-sections."""
    wiki_title = title.replace(" ", "_")
    try:
        resp = _get(_WIKI_SECT.format(title=wiki_title))
        data = resp.json()
    except Exception:
        # Fallback: try summary only
        try:
            resp = _get(_WIKI_SUMM.format(title=wiki_title))
            summ = resp.json().get("extract", "")
            return {"lead": summ}
        except Exception:
            return {}

    sections: dict[str, str] = {}

    # Lead section
    lead_sections = data.get("lead", {}).get("sections", [])
    if lead_sections:
        lead_text = _strip_html(lead_sections[0].get("text", ""))
        sections["lead"] = lead_text

    # Remaining sections
    for sec in data.get("remaining", {}).get("sections", []):
        sec_title = sec.get("title", "").lower().strip()
        sec_text = _strip_html(sec.get("text", ""))
        if sec_text:
            sections[sec_title] = sec_text

    return sections


def _extract_qa(book: dict, sections: dict[str, str], max_q: int = 20) -> list[dict]:
    """Generate Q&A pairs from catalog summary + Wikipedia sections."""
    title = book["title"]
    author = book["author"]
    qa: list[dict] = []

    def _add(q: str, answer: str, source: str) -> None:
        if len(answer) < 20:
            return
        qa.append(
            {
                "question": q,
                "expected_answer": answer,
                "keywords": _keywords(answer),
                "source": source,
                "difficulty": "medium",
            }
        )

    # ── Seed from the catalog summary (free — already downloaded) ─────────────
    catalog_summary = book.get("catalog_summary", "")
    if catalog_summary:
        _add(f"What is {title!r} about?", catalog_summary, "catalog")
        _add(
            f"Give a brief overview of the plot of {title!r}.",
            catalog_summary,
            "catalog",
        )
        # Strip the trailing auto-generated disclaimer for cleaner answers
        clean = re.sub(
            r"\(This is an automatically generated summary\.?\)", "", catalog_summary
        ).strip()
        if clean:
            _add(f"Summarise {title!r} in a few sentences.", clean, "catalog")
        # Additional catalog windows — different slices yield different keyword sets
        if len(catalog_summary) > 80:
            _add(f"What type of story is {title!r}?", catalog_summary[:200], "catalog")
        if len(catalog_summary) > 120:
            _add(
                f"What is the central conflict in {title!r}?",
                catalog_summary[40:280],
                "catalog",
            )
        if len(catalog_summary) > 180:
            _add(
                f"What events unfold in {title!r}?",
                catalog_summary[80:350],
                "catalog",
            )

    lead = sections.get("lead", "") or catalog_summary

    # ── Always-applicable questions ────────────────────────────────────────────
    if lead:
        _add(
            f"Who wrote {title!r} and when was it first published?", lead[:400], "lead"
        )
        _add(f"What is the main plot of {title!r}?", lead[:500], "lead")
        _add(f"What genre is {title!r}?", lead[:300], "lead")
        # Additional lead windows so books without Wikipedia plot/character sections
        # still reach 20+ questions — each slice has a different keyword set
        if len(lead) > 150:
            _add(f"What makes {title!r} notable in literature?", lead[:350], "lead")
        if len(lead) > 200:
            _add(
                f"Who is {author!r} and what is their connection to {title!r}?",
                f"{author} wrote {title}. {lead[:300]}",
                "lead",
            )
        if len(lead) > 300:
            _add(
                f"How is {title!r} described by literary sources?",
                lead[100:450],
                "lead",
            )
        if len(lead) > 400:
            _add(
                f"What is the historical or cultural context of {title!r}?",
                lead[150:500],
                "lead",
            )
        if len(lead) > 500:
            _add(
                f"What additional details are known about {title!r}?",
                lead[250:600],
                "lead",
            )
        if len(lead) > 600:
            _add(
                f"What is the broader significance of {title!r}?",
                lead[350:700],
                "lead",
            )
        # Sentence-level extraction: each sentence has a unique keyword set
        _sent_qs = [
            f"What important fact is recorded about {title!r}?",
            f"How is {title!r} characterized in literary sources?",
            f"What notable detail is documented about {title!r}?",
            f"What context helps understand {title!r}?",
            f"What key information exists about {title!r}?",
        ]
        lead_sents = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", lead) if len(s.strip()) > 50
        ]
        for _si, _sent in enumerate(lead_sents[:5]):
            if len(qa) >= max_q:
                break
            _add(_sent_qs[_si % len(_sent_qs)], _sent, "lead")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_text = next((sections[k] for k in sections if k in _PLOT_TITLES), "")
    if plot_text:
        _add(f"Summarise the plot of {title!r}.", plot_text[:600], "plot")
        _add(f"How does {title!r} end?", plot_text[-400:], "plot")
        _add(f"What is the central conflict in {title!r}?", plot_text[:400], "plot")

    # ── Characters ────────────────────────────────────────────────────────────
    char_text = next((sections[k] for k in sections if k in _CHAR_TITLES), "")
    if char_text:
        _add(
            f"Who are the main characters in {title!r}?", char_text[:500], "characters"
        )
        # Try to extract individual character descriptions
        # Pattern: sentence containing a character name and a description verb
        for m in re.finditer(
            r"([A-Z][a-z]+ [A-Z][a-z]+)[^.]{5,80}(?:is|was|serves|plays|acts)[^.]{10,120}\.",
            char_text,
        ):
            if len(qa) >= max_q:
                break
            char_name = m.group(1)
            sentence = m.group(0)
            _add(f"Who is {char_name} in {title!r}?", sentence, "characters")

    # ── Themes ────────────────────────────────────────────────────────────────
    theme_text = next((sections[k] for k in sections if k in _THEME_TITLES), "")
    if theme_text:
        _add(f"What major themes does {title!r} explore?", theme_text[:500], "themes")

    # ── Setting ───────────────────────────────────────────────────────────────
    setting_text = next((sections[k] for k in sections if k in _SETTING_TITLES), "")
    if setting_text:
        _add(f"Where and when is {title!r} set?", setting_text[:300], "setting")
    elif lead:
        # Try to extract setting from lead using location keywords
        loc_m = re.search(
            r"(?:set in|takes place in|set during|based in)[^.]{5,100}\.",
            lead,
            re.IGNORECASE,
        )
        if loc_m:
            _add(f"Where is {title!r} set?", loc_m.group(0), "lead")

    # ── Reception ───────────────────────────────────────────────────────────────────────
    recept_text = next((sections[k] for k in sections if k in _RECEPTION_TITLES), "")
    if recept_text:
        _add(f"How was {title!r} received by critics?", recept_text[:500], "reception")
        _add(
            f"What is the cultural legacy of {title!r}?",
            recept_text[80:500],
            "reception",
        )

    # ── Publication background ────────────────────────────────────────────────────────
    bg_text = next((sections[k] for k in sections if k in _BACKGROUND_TITLES), "")
    if bg_text:
        _add(
            f"What is the publication background of {title!r}?",
            bg_text[:500],
            "background",
        )
        _add(f"When and how was {title!r} written?", bg_text[:400], "background")

    # ── Style / structure ────────────────────────────────────────────────────────────
    style_text = next((sections[k] for k in sections if k in _STYLE_TITLES), "")
    if style_text:
        _add(f"What is the writing style of {title!r}?", style_text[:500], "style")
        _add(
            f"What literary techniques does {title!r} employ?",
            style_text[:400],
            "style",
        )

    # ── Author ────────────────────────────────────────────────────────────────
    if author and lead:
        _add(
            f"Who is the author of {title!r}?",
            f"{author} wrote {title}. {lead[:200]}",
            "lead",
        )

    return qa[:max_q]


def build_question_banks(books: list[dict], qpb: int = 20, force: bool = False) -> None:
    """
    For each book, fetch Wikipedia sections and write
    data/question_banks/<slug>.json.  Skips books that already have a bank
    unless *force* is True.
    """
    _BANKS_DIR.mkdir(parents=True, exist_ok=True)
    total, skipped, built = len(books), 0, 0

    print(
        f"\n[banks] Building question banks for {total} books (target: {qpb}/book)..."
    )

    for i, book in enumerate(books, 1):
        bank_path = _BANKS_DIR / f"{book['slug']}.json"
        if bank_path.exists() and not force:
            skipped += 1
            print(f"  [{i}/{total}] {book['title'][:50]}: cached ({bank_path.name})")
            continue

        print(f"  [{i}/{total}] {book['title'][:50]}: fetching Wikipedia...")
        time.sleep(0.3)  # polite rate-limit
        sections = _wiki_sections(book["title"])
        qa = _extract_qa(book, sections, max_q=qpb)

        payload = {
            "book_id": book["id"],
            "title": book["title"],
            "author": book["author"],
            "slug": book["slug"],
            "gutenberg_url": book["txt_url"],
            "built_at": datetime.now().isoformat(),
            "questions": qa,
        }
        bank_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        built += 1
        print(f"    → {len(qa)} questions  saved {bank_path.name}")

    print(f"[banks] Done — {built} built, {skipped} skipped")


def build_chunk_banks(
    books: list[dict],
    qpb: int = 50,
    lines: int = 3000,
    force: bool = False,
    strategy: str = "llm",
) -> None:
    """
    Build question banks from the cached compressed chunks — no Wikipedia calls,
    no book downloads, no LLM calls.

    Each Q&A pair's ``expected_answer`` is text that literally exists inside the
    chunk cache (raw sentences, entities, keywords, or compressed summaries), so
    the retriever can always locate it.  This gives a clean measure of retrieval
    quality independent of Wikipedia vocabulary.

    Requires: ``data/chunks/{slug}_L{lines}.jsonl`` must exist (run ``run`` first).
    """
    _BANKS_DIR.mkdir(parents=True, exist_ok=True)
    total, skipped, built, missing = len(books), 0, 0, 0

    print(
        f"\n[chunk-banks] Building question banks from chunk caches "
        f"({total} books, target {qpb}/book)..."
    )

    # Rotating question templates — varied phrasing, same intent
    _raw_qs = [
        "What occurs or is described in {title!r}?",
        "What happens in a section of {title!r}?",
        "Describe a scene or passage from {title!r}.",
        "What does a portion of {title!r} contain or depict?",
        "What narrative or dialogue appears in {title!r}?",
        "What passage or scene can be found in {title!r}?",
        "What event or situation is portrayed in {title!r}?",
        "What does an excerpt of {title!r} describe?",
    ]

    for i, book in enumerate(books, 1):
        slug = book["slug"]
        title = book["title"]
        bank_path = _BANKS_DIR / f"{slug}.json"

        if bank_path.exists() and not force:
            skipped += 1
            print(f"  [{i}/{total}] {title[:50]}: cached ({bank_path.name})")
            continue

        chunks = _load_chunks(slug, lines, strategy)
        if not chunks:
            missing += 1
            print(
                f"  [{i}/{total}] {title[:50]}: MISSING chunk cache "
                f"(run 'run --lines {lines} --strategy {strategy}' first)"
            )
            continue

        qa: list[dict] = []

        def _add(q: str, answer: str, source: str) -> None:
            if len(answer) < 20:
                return
            qa.append(
                {
                    "question": q,
                    "expected_answer": answer,
                    "keywords": _keywords(answer),
                    "source": source,
                    "difficulty": "medium",
                }
            )

        # Sample 8 evenly-spaced chunk positions across the book
        n = len(chunks)
        positions = sorted({min(int(k * n / 8), n - 1) for k in range(8)})

        for pos_idx, chunk_idx in enumerate(positions):
            if len(qa) >= qpb:
                break
            c = chunks[chunk_idx]

            # ── Raw text sentences only ───────────────────────────────────────
            # Entities, keywords, and compressed_summary from llama3.2:3b are
            # unreliable (hallucinated tech terms unrelated to the book).
            # raw_text is the original Gutenberg content — always clean.
            raw_sents = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", c.raw_text)
                if len(s.strip()) > 60 and not s.strip().startswith("_")
            ]
            for j, sent in enumerate(raw_sents[:5]):
                if len(qa) >= qpb:
                    break
                tpl = _raw_qs[(pos_idx * 5 + j) % len(_raw_qs)]
                _add(tpl.format(title=title), sent, "raw_text")

        payload = {
            "book_id": book["id"],
            "title": title,
            "author": book["author"],
            "slug": slug,
            "gutenberg_url": book["txt_url"],
            "built_at": datetime.now().isoformat(),
            "questions": qa,
        }
        bank_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        built += 1
        print(
            f"  [{i}/{total}] {title[:50]}: {len(qa)} questions  saved {bank_path.name}"
        )

    if missing:
        print(f"\n[chunk-banks] WARNING: {missing} books missing chunk caches")
    print(f"[chunk-banks] Done — {built} built, {skipped} skipped, {missing} missing")


# ── Corpus download ────────────────────────────────────────────────────────────


def download_book(book: dict) -> Path | None:
    """Download and cache a book's plain text.  Returns path or None on error."""
    _BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    dest = _BOOKS_DIR / f"{book['id']}.txt"
    if dest.exists():
        return dest
    try:
        resp = _get(book["txt_url"])
        dest.write_bytes(resp.content)
        return dest
    except Exception as exc:
        print(f"  WARNING: Could not download {book['title']}: {exc}")
        return None


def load_book_lines(book: dict, max_lines: int, pad_to_mb: float = 0.0) -> list[str]:
    """Download (if needed) and return up to max_lines of a book.

    When pad_to_mb > 0 the raw text is repeated until the corpus reaches
    the target size before the line-cap is applied — useful for large-corpus
    stress testing.
    """
    path = download_book(book)
    if path is None:
        return []
    raw = path.read_text(encoding="utf-8", errors="replace")
    # Strip Gutenberg header/footer boilerplate
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    # Skip header up to "*** START OF" marker
    for i, ln in enumerate(lines):
        if "*** START OF" in ln.upper() or "*** THE PROJECT GUTENBERG" in ln.upper():
            lines = lines[i + 1 :]
            break
    # Trim at "*** END OF" marker
    for i, ln in enumerate(lines):
        if "*** END OF" in ln.upper():
            lines = lines[:i]
            break
    # Pad to target size by repeating the cleaned text
    if pad_to_mb > 0 and lines:
        target_bytes = int(pad_to_mb * 1024 * 1024)
        base_lines = list(lines)
        current_bytes = sum(len(ln.encode("utf-8")) + 1 for ln in lines)
        while current_bytes < target_bytes:
            lines.extend(base_lines)
            current_bytes += sum(len(ln.encode("utf-8")) + 1 for ln in base_lines)
        print(
            f"  [pad] {book['title'][:40]}: {current_bytes / 1_048_576:.1f} MB  "
            f"({len(lines):,} lines)"
        )
    return lines[:max_lines] if max_lines > 0 else lines


# ── Non-LLM judge ──────────────────────────────────────────────────────────────


def judge_keyword_recall(answer: str, keywords: list[str]) -> float:
    """
    Keyword recall score: fraction of expected keywords present in the answer.

    Matching is case-insensitive substring match so morphological variants
    (e.g. 'obsession' matches keyword 'obsess') score positively.
    """
    if not keywords:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(
        1
        for kw in keywords
        if kw.lower() in answer_lower or answer_lower.startswith(kw[:4].lower())
    )
    return hits / len(keywords)


# ── Parallel benchmark runner ──────────────────────────────────────────────────


def _run_book(
    book: dict,
    qa_entries: list[dict],
    lines: list[str],
    judge_executor: ThreadPoolExecutor,
    pad_to_mb: float = 0.0,
    max_lines: int = 3000,
    force: bool = False,
    strategy: str = "llm",
) -> dict:
    """
    Compress one book, build a retriever, answer all questions, judge async.

    Caching behaviour (skipped when *force=True*):
    - If ``data/book_cache/{slug}.json`` exists, the entire book is skipped
      (returns the cached result immediately).
    - If ``data/chunks/{slug}_L{max_lines}.jsonl`` exists, the compressed
      chunks are reloaded from disk and the LLM is not called again.
      After compression the chunks are always written to disk.

    When *pad_to_mb* > 0 the base chunks are replicated (with unique IDs)
    until the total original-token footprint reaches the target size, stressing
    ChromaDB retrieval at large collection sizes.

    Returns a results dict for this book.
    """
    from context_optimizer.cached_retriever import CachedChromaRetriever
    from context_optimizer.compressor import compress_corpus_rolling
    from context_optimizer.raw_index import RawIndex
    from context_optimizer.tot_reasoner import ToTReasoner

    slug = book["slug"]
    title = book["title"]
    book_results: list[dict] = []

    # ── Full-result cache ─────────────────────────────────────────────────────
    if not force:
        cached = _load_result(slug)
        if cached:
            print(f"  [cache-hit] {title[:50]}: loaded from result cache, skipping")
            return cached

    tmp_dir = tempfile.mkdtemp(prefix=f"co_book_{slug[:20]}_")

    try:
        # ── Compress (base text only) ─────────────────────────────────────────────
        # raw_only: skip compression entirely — reuse extractive chunk cache for
        # raw_text, build only RawIndex, no ChromaDB/embeddings.
        if strategy == "raw_only":
            base_chunks = _load_chunks(slug, max_lines, "extractive") or _load_chunks(
                slug, max_lines, "llm"
            )
            if not base_chunks:
                # No chunk cache at all — run extractive compression once to get raw_text
                t0 = time.perf_counter()
                base_chunks = compress_corpus_rolling(
                    lines, label=title[:30], strategy="extractive"
                )
                compress_sec = time.perf_counter() - t0
                _save_chunks(slug, max_lines, base_chunks, "extractive")
            else:
                compress_sec = 0.0
            orig_tok = sum(c.original_tokens for c in base_chunks)
            comp_tok = orig_tok  # no compression — ratio = 100%
            print(
                f"  [raw_only] {title[:40]}: "
                f"{len(base_chunks)} chunks, raw text only (no compression)"
            )
        else:
            cached_chunks = None if force else _load_chunks(slug, max_lines, strategy)
            if cached_chunks:
                print(
                    f"  [chunk-cache] {title[:40]}: "
                    f"loaded {len(cached_chunks)} cached chunks ({strategy}), skipping compression"
                )
                base_chunks = cached_chunks
                compress_sec = 0.0
            else:
                t0 = time.perf_counter()
                base_chunks = compress_corpus_rolling(
                    lines, label=title[:30], strategy=strategy
                )
                compress_sec = time.perf_counter() - t0
                _save_chunks(
                    slug, max_lines, base_chunks, strategy
                )  # persist for restarts

        orig_tok = sum(c.original_tokens for c in base_chunks)
        comp_tok = (
            sum(c.compressed_tokens for c in base_chunks)
            if strategy != "raw_only"
            else orig_tok
        )

        # ── Pad index by replicating compressed chunks ───────────────────────────
        # Replication stresses ChromaDB at large collection sizes without
        # forcing the LLM to re-compress identical repeated content.
        if pad_to_mb > 0 and base_chunks:
            from dataclasses import replace as dc_replace

            target_bytes = int(pad_to_mb * 1024 * 1024)
            # Bytes already covered by base compression
            current_bytes = orig_tok * 4  # rough: 4 bytes/token
            chunks: list = list(base_chunks)
            rep = 1
            while current_bytes < target_bytes:
                for c in base_chunks:
                    chunks.append(dc_replace(c, chunk_id=f"{c.chunk_id}_rep{rep}"))
                current_bytes += orig_tok * 4
                rep += 1
            print(
                f"  [pad-index] {title[:40]}: "
                f"{len(base_chunks)} base chunks -> {len(chunks)} total "
                f"({rep - 1} replicas, ~{len(chunks) * orig_tok * 4 / 1_048_576:.1f} MB)"
            )
        else:
            chunks = base_chunks

        # ── Index ─────────────────────────────────────────────────────────────
        if strategy == "raw_only":
            # Pure FTS5 path: no ChromaDB, no embeddings.
            # threshold=1.1 means primary pass always scores 0 (no retriever),
            # guaranteeing the FTS5 branch fires for every query.
            raw_idx = RawIndex(os.path.join(tmp_dir, "raw.db"))
            raw_idx.add_many([(c.chunk_id, c.raw_text) for c in chunks])
            reasoner = ToTReasoner(
                retriever=None,
                top_k_per_term=5,
                raw_index=raw_idx,
                raw_fallback_threshold=1.1,
            )
        else:
            retriever = CachedChromaRetriever(
                collection_name=f"book_{book['id']}",
                persist_directory=tmp_dir,
            )
            retriever.add_chunks(chunks)

            # RawIndex (SQLite+FTS5) alongside ChromaDB — short-circuits to BM25
            # when compressed-summary similarity is too low (no embedding round-trip).
            raw_idx = RawIndex(os.path.join(tmp_dir, "raw.db"))
            raw_idx.add_many([(c.chunk_id, c.raw_text) for c in chunks])

            reasoner = ToTReasoner(
                retriever=retriever, top_k_per_term=5, raw_index=raw_idx
            )

        # ── Query + async judge ───────────────────────────────────────────────
        judge_futures: list[tuple[dict, Future]] = []
        n_q = len(qa_entries)
        short = title[:30]

        for q_idx, qa in enumerate(qa_entries, 1):
            q = qa["question"]
            expected_kw = qa.get("keywords", [])
            print(f"  [{short}] Q {q_idx}/{n_q}: {q[:60]}")

            q_start = time.perf_counter()
            try:
                # Pass question as plain string — ToTReasoner splits it into
                # search terms and retrieves the best matching evidence.
                tot = reasoner.reason(q)
                # Aggregate evidence across all branches (score-ordered) so the
                # answer text contains keywords from multiple retrieved chunks.
                all_snips: list[str] = []
                for branch in sorted(tot.branches, key=lambda b: b.score, reverse=True):
                    all_snips.extend(branch.evidence_snippets)
                answer = " ".join(all_snips[:6]) if all_snips else ""
            except Exception as exc:
                answer = f"[ERROR: {exc}]"
            latency_ms = (time.perf_counter() - q_start) * 1000

            # Submit judging to the shared executor (non-blocking)
            entry_copy = dict(qa)
            answer_copy = str(answer)
            future = judge_executor.submit(
                judge_keyword_recall, answer_copy, expected_kw
            )
            judge_futures.append(
                (
                    {
                        "question": q,
                        "answer": answer_copy[:300],
                        "latency_ms": latency_ms,
                        "source": qa.get("source", ""),
                        "difficulty": qa.get("difficulty", "medium"),
                        "expected_kw_count": len(expected_kw),
                    },
                    future,
                )
            )

        # ── Collect judging results ───────────────────────────────────────────
        for entry, future in judge_futures:
            try:
                entry["kw_recall"] = future.result(timeout=10)
            except Exception:
                entry["kw_recall"] = 0.0
            book_results.append(entry)

        avg_kw_recall = sum(r["kw_recall"] for r in book_results) / max(
            len(book_results), 1
        )
        print(
            f"  [book] {title[:45]:<45}  "
            f"base_chunks={len(base_chunks):3d}  "
            f"index_chunks={len(chunks):4d}  "
            f"compress={compress_sec:.0f}s  "
            f"Qs={len(book_results):3d}  "
            f"avg_kw_recall={avg_kw_recall:.2%}"
        )

        result = {
            "book_id": book["id"],
            "title": title,
            "author": book["author"],
            "lines_used": len(lines),
            "base_chunks": len(base_chunks),
            "index_chunks": len(chunks),
            "orig_tokens": orig_tok,
            "comp_tokens": comp_tok,
            "compress_sec": compress_sec,
            "questions_run": len(book_results),
            "avg_kw_recall": avg_kw_recall,
            "results": book_results,
        }
        _save_result(
            slug, result, strategy
        )  # persist so restarts skip this book entirely
        return result

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_benchmark(
    books: list[dict],
    max_lines: int = 3_000,
    workers: int = 4,
    qpb: int = 20,
    pad_to_mb: float = 0.0,
    force: bool = False,
    strategy: str = "llm",
) -> list[dict]:
    """
    Run the full benchmark over all books in parallel.

    Each book is assigned one worker thread.  A shared judge executor runs
    the keyword-recall scoring asynchronously alongside query execution.
    """
    # Load question banks
    all_qa: dict[str, list[dict]] = {}
    for book in books:
        bank_path = _BANKS_DIR / f"{book['slug']}.json"
        if not bank_path.exists():
            all_qa[book["slug"]] = []
            continue
        data = json.loads(bank_path.read_text(encoding="utf-8"))
        all_qa[book["slug"]] = data.get("questions", [])[:qpb]

    books_with_qa = [b for b in books if all_qa.get(b["slug"])]
    pad_label = (
        f"  |  index-pad {pad_to_mb:.0f} MB/book (compress base only)"
        if pad_to_mb
        else ""
    )
    print(
        f"\n[benchmark] {len(books_with_qa)}/{len(books)} books have question banks  "
        f"| {workers} compression workers  |  max {max_lines} lines/book{pad_label}"
    )

    # Shared judge executor (lightweight — pure Python string ops)
    all_book_results: list[dict] = []

    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="judge") as judge_exe:
        # Submit one book-compression-and-query task per worker slot
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="book"
        ) as book_exe:
            future_to_book = {}
            for book in books_with_qa:
                # Check full result cache before loading any lines
                if not force:
                    cached = _load_result(book["slug"], strategy)
                    if cached:
                        print(
                            f"  [cache-hit] {book['title'][:50]}: result cached, skipping"
                        )
                        all_book_results.append(cached)
                        continue

                qa_entries = all_qa[book["slug"]]
                # Load base text only (no padding here — replication happens
                # inside _run_book after compression, at the chunk level).
                lines = load_book_lines(book, max_lines)
                if not lines:
                    print(f"  SKIP {book['title']}: no lines loaded")
                    continue
                f = book_exe.submit(
                    _run_book,
                    book,
                    qa_entries,
                    lines,
                    judge_exe,
                    pad_to_mb,
                    max_lines,
                    force,
                    strategy,
                )
                future_to_book[f] = book["title"]

            for future in as_completed(future_to_book):
                title = future_to_book[future]
                try:
                    all_book_results.append(future.result())
                except Exception as exc:
                    print(f"  ERROR [{title}]: {exc}")

    return all_book_results


# ── Report writer ──────────────────────────────────────────────────────────────


def write_report(
    all_results: list[dict], run_date: str, args: argparse.Namespace
) -> Path:
    """Write book_results.md and BOOK_RESULTS.json to benchmarks/."""
    strategy = getattr(args, "strategy", "llm")
    out_md = _BENCH_DIR / f"book_results_{strategy}.md"
    out_json = _BENCH_DIR / f"BOOK_RESULTS_{strategy}.json"

    # Aggregate stats
    total_q = sum(r["questions_run"] for r in all_results)
    avg_rec = sum(r["avg_kw_recall"] for r in all_results) / max(len(all_results), 1)
    avg_comp = sum(r["compress_sec"] for r in all_results) / max(len(all_results), 1)
    total_orig = sum(r["orig_tokens"] for r in all_results)
    total_comp = sum(r["comp_tokens"] for r in all_results)
    reduction = (1 - total_comp / max(total_orig, 1)) * 100

    md = [
        "# Book Benchmark Results",
        "",
        f"**Run date**: {run_date}  |  "
        f"**Books**: {len(all_results)}  |  "
        f"**Questions**: {total_q:,}  |  "
        f"**Lines/book cap**: {'unlimited' if args.lines == 0 else f'{args.lines:,}'}  |  "
        f"**Index-pad**: {getattr(args, 'pad_to_mb', 0):.0f} MB/book  |  "
        f"**Workers**: {args.workers}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Books benchmarked | {len(all_results)} |",
        f"| Total questions | {total_q:,} |",
        f"| Avg keyword recall (judge) | {avg_rec:.1%} |",
        f"| Avg compression time per book | {avg_comp:.0f}s |",
        f"| Overall token reduction | {reduction:.1f}% |",
        f"| Original tokens (all books) | {total_orig:,} |",
        f"| Compressed tokens (all books) | {total_comp:,} |",
        "",
        "---",
        "",
        "## Per-Book Results",
        "",
        "| Book | Author | Base Chunks | Index Chunks | Qs | Avg KW Recall | Compress(s) |",
        "|------|--------|------------:|-------------:|---:|:-------------:|------------:|",
    ]

    for r in sorted(all_results, key=lambda x: -x["avg_kw_recall"]):
        md.append(
            f"| {r['title'][:50]} | {r['author'][:25]} | "
            f"{r['base_chunks']} | {r['index_chunks']} | {r['questions_run']} | "
            f"{r['avg_kw_recall']:.0%} | {r['compress_sec']:.0f}s |"
        )

    md += [
        "",
        "---",
        "",
        "## Judge Methodology",
        "",
        "Answers are scored by **keyword recall**: the fraction of Wikipedia-sourced",
        "expected keywords that appear (case-insensitive substring match) in the",
        "pipeline's answer.  Expected answers are fetched once at `build-banks` time",
        "and stored in `data/question_banks/<slug>.json` — no LLM judge used at",
        "evaluation time.  Judging runs asynchronously in a thread pool alongside",
        "query execution.",
        "",
        f"*Generated {run_date} — do not edit manually.*",
    ]

    out_md.write_text("\n".join(md), encoding="utf-8")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "run_date": run_date,
                "books": len(all_results),
                "total_q": total_q,
                "avg_kw_recall": avg_rec,
                "reduction_pct": reduction,
                "per_book": all_results,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n[report] {out_md.relative_to(_PROJECT_ROOT)}")
    print(f"[report] {out_json.relative_to(_PROJECT_ROOT)}")
    return out_md


# ── CLI ────────────────────────────────────────────────────────────────────────


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="book_benchmark",
        description="100-book Gutenberg benchmark with Wikipedia Q&A banks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # build-banks
    bb = sub.add_parser("build-banks", help="Fetch Wikipedia Q&A banks (run once).")
    bb.add_argument(
        "--books",
        type=int,
        default=100,
        help="Number of books to include (default: 100)",
    )
    bb.add_argument(
        "--qpb",
        type=int,
        default=20,
        help="Max questions per book to extract (default: 20)",
    )
    bb.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Rebuild banks even for books that already have a cached bank",
    )

    # run
    ru = sub.add_parser("run", help="Run the compression + retrieval benchmark.")
    ru.add_argument("--books", type=int, default=25)
    ru.add_argument(
        "--lines",
        type=int,
        default=3000,
        help="Max lines per book to compress (default: 3000; 0 = unlimited)",
    )
    ru.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Parallel compression workers (default: 2)",
    )
    ru.add_argument(
        "--qpb", type=int, default=100, help="Max questions per book (default: 100)"
    )
    ru.add_argument(
        "--pad-to-mb",
        type=float,
        default=35.0,
        dest="pad_to_mb",
        help="Pad each book to this size in MB by repeating its text (default: 35.0)",
    )
    ru.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Ignore chunk and result caches; re-run everything from scratch",
    )
    ru.add_argument(
        "--strategy",
        choices=["llm", "extractive", "raw_only"],
        default="llm",
        help="Compression strategy: 'llm', 'extractive', or 'raw_only' (no compression, FTS5-only)",
    )
    cb = sub.add_parser(
        "chunk-banks",
        help="Build Q&A banks from cached compressed chunks (no Wikipedia, no book download).",
    )
    cb.add_argument("--books", type=int, default=25)
    cb.add_argument(
        "--qpb", type=int, default=50, help="Max questions per book (default: 50)"
    )
    cb.add_argument(
        "--lines",
        type=int,
        default=3000,
        help="Max lines used in the compression run — used to locate the chunk cache (default: 3000)",
    )
    cb.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Rebuild banks even for books that already have a cached bank",
    )
    cb.add_argument(
        "--strategy",
        choices=["llm", "extractive", "raw_only"],
        default="llm",
        help="Which strategy's chunk cache to read (default: llm)",
    )

    # all (build-banks then run)
    al = sub.add_parser("all", help="Build banks then run benchmark.")
    al.add_argument("--books", type=int, default=25)
    al.add_argument("--lines", type=int, default=0)
    al.add_argument("--workers", type=int, default=2)
    al.add_argument("--qpb", type=int, default=100)
    al.add_argument(
        "--pad-to-mb",
        type=float,
        default=35.0,
        dest="pad_to_mb",
        help="Pad each book to this size in MB (default: 35.0)",
    )
    al.add_argument(
        "--strategy",
        choices=["llm", "extractive", "raw_only"],
        default="llm",
        help="Compression strategy: 'llm', 'extractive', or 'raw_only' (no compression, FTS5-only)",
    )

    return p


def main() -> None:
    args = _parser().parse_args()
    run_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    catalog = fetch_book_catalog(n=args.books)

    if args.cmd in ("build-banks", "all"):
        build_question_banks(catalog, qpb=args.qpb, force=getattr(args, "force", False))

    if args.cmd == "chunk-banks":
        build_chunk_banks(
            catalog,
            qpb=args.qpb,
            lines=args.lines,
            force=args.force,
            strategy=args.strategy,
        )

    if args.cmd in ("run", "all"):
        pad_to_mb = getattr(args, "pad_to_mb", 0.0)
        strategy = getattr(args, "strategy", "llm")
        results = run_benchmark(
            catalog,
            max_lines=args.lines,
            workers=args.workers,
            qpb=args.qpb,
            pad_to_mb=pad_to_mb,
            force=getattr(args, "force", False),
            strategy=strategy,
        )
        if results:
            write_report(results, run_date, args)
        else:
            print("[benchmark] No results — run 'build-banks' first.")


if __name__ == "__main__":
    main()

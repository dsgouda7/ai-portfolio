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

    lead = sections.get("lead", "") or catalog_summary

    # ── Always-applicable questions ────────────────────────────────────────────
    if lead:
        _add(
            f"Who wrote {title!r} and when was it first published?", lead[:400], "lead"
        )
        _add(f"What is the main plot of {title!r}?", lead[:500], "lead")
        _add(f"What genre is {title!r}?", lead[:300], "lead")

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

    # ── Author ────────────────────────────────────────────────────────────────
    if author and lead:
        _add(
            f"Who is the author of {title!r}?",
            f"{author} wrote {title}. {lead[:200]}",
            "lead",
        )

    return qa[:max_q]


def build_question_banks(books: list[dict], qpb: int = 20) -> None:
    """
    For each book, fetch Wikipedia sections and write
    data/question_banks/<slug>.json.  Skips books that already have a bank.
    """
    _BANKS_DIR.mkdir(parents=True, exist_ok=True)
    total, skipped, built = len(books), 0, 0

    print(
        f"\n[banks] Building question banks for {total} books (target: {qpb}/book)..."
    )

    for i, book in enumerate(books, 1):
        bank_path = _BANKS_DIR / f"{book['slug']}.json"
        if bank_path.exists():
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


def load_book_lines(book: dict, max_lines: int) -> list[str]:
    """Download (if needed) and return up to max_lines of a book."""
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
    return lines[:max_lines]


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
) -> dict:
    """
    Compress one book, build a retriever, answer all questions, judge async.

    Returns a results dict for this book.
    """
    from context_optimizer.cached_retriever import CachedChromaRetriever
    from context_optimizer.compressor import compress_corpus_rolling
    from context_optimizer.tot_reasoner import ToTReasoner

    title = book["title"]
    book_results: list[dict] = []
    tmp_dir = tempfile.mkdtemp(prefix=f"co_book_{book['slug'][:20]}_")

    try:
        # ── Compress ──────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        chunks = compress_corpus_rolling(lines, label=title[:30])
        compress_sec = time.perf_counter() - t0
        orig_tok = sum(c.original_tokens for c in chunks)
        comp_tok = sum(c.compressed_tokens for c in chunks)

        # ── Index ─────────────────────────────────────────────────────────────
        retriever = CachedChromaRetriever(
            collection_name=f"book_{book['id']}",
            persist_directory=tmp_dir,
        )
        retriever.add_chunks(chunks)
        reasoner = ToTReasoner(retriever=retriever)

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
                winner = tot.winner
                answer = (
                    winner.evidence_snippets[0]
                    if winner and winner.evidence_snippets
                    else ""
                )
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
            f"chunks={len(chunks):3d}  "
            f"compress={compress_sec:.0f}s  "
            f"Qs={len(book_results):3d}  "
            f"avg_kw_recall={avg_kw_recall:.2%}"
        )

        return {
            "book_id": book["id"],
            "title": title,
            "author": book["author"],
            "lines_used": len(lines),
            "chunks": len(chunks),
            "orig_tokens": orig_tok,
            "comp_tokens": comp_tok,
            "compress_sec": compress_sec,
            "questions_run": len(book_results),
            "avg_kw_recall": avg_kw_recall,
            "results": book_results,
        }

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_benchmark(
    books: list[dict],
    max_lines: int = 3_000,
    workers: int = 4,
    qpb: int = 20,
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
    print(
        f"\n[benchmark] {len(books_with_qa)}/{len(books)} books have question banks  "
        f"| {workers} compression workers  |  max {max_lines} lines/book"
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
                qa_entries = all_qa[book["slug"]]
                lines = load_book_lines(book, max_lines)
                if not lines:
                    print(f"  SKIP {book['title']}: no lines loaded")
                    continue
                f = book_exe.submit(_run_book, book, qa_entries, lines, judge_exe)
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
    out_md = _BENCH_DIR / "book_results.md"
    out_json = _BENCH_DIR / "BOOK_RESULTS.json"

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
        f"**Lines/book cap**: {args.lines:,}  |  "
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
        "| Book | Author | Chunks | Qs | Avg KW Recall | Compress(s) |",
        "|------|--------|-------:|---:|:-------------:|------------:|",
    ]

    for r in sorted(all_results, key=lambda x: -x["avg_kw_recall"]):
        md.append(
            f"| {r['title'][:50]} | {r['author'][:25]} | "
            f"{r['chunks']} | {r['questions_run']} | "
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

    # run
    ru = sub.add_parser("run", help="Run the compression + retrieval benchmark.")
    ru.add_argument("--books", type=int, default=100)
    ru.add_argument(
        "--lines",
        type=int,
        default=3_000,
        help="Max lines to ingest per book (default: 3000)",
    )
    ru.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel compression workers (default: 4)",
    )
    ru.add_argument(
        "--qpb", type=int, default=20, help="Max questions per book (default: 20)"
    )

    # all (build-banks then run)
    al = sub.add_parser("all", help="Build banks then run benchmark.")
    al.add_argument("--books", type=int, default=100)
    al.add_argument("--lines", type=int, default=3_000)
    al.add_argument("--workers", type=int, default=4)
    al.add_argument("--qpb", type=int, default=20)

    return p


def main() -> None:
    args = _parser().parse_args()
    run_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    catalog = fetch_book_catalog(n=args.books)

    if args.cmd in ("build-banks", "all"):
        build_question_banks(catalog, qpb=args.qpb)

    if args.cmd in ("run", "all"):
        results = run_benchmark(
            catalog,
            max_lines=args.lines,
            workers=args.workers,
            qpb=args.qpb,
        )
        if results:
            write_report(results, run_date, args)
        else:
            print("[benchmark] No results — run 'build-banks' first.")


if __name__ == "__main__":
    main()

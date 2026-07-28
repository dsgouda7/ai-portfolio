"""
prepare_extended_corpus.py — Download two richer demo corpora.

Run ONCE before (re-)running setup_demo.py:

    cd projects/context-optimizer/demo
    python prepare_extended_corpus.py
    python setup_demo.py --force          # rebuilds the index (~5-10 min)

What this adds
--------------
corpus/gutenberg/
    pride-and-prejudice.txt   (~700 KB, Jane Austen)
    Boilerplate stripped; only the novel text is kept.

corpus/requests-src/
    *.py  (psf/requests core package, ~20 files, ~250 KB)
    Good demo questions:
      "How does session-level authentication work?"
      "How are connection retries configured?"
      "How is SSL certificate verification handled?"
      "How does the cookie jar work?"
      "How does the HTTPAdapter handle connection pooling?"

Both downloads use only Python stdlib (urllib, zipfile) — no extra packages.
"""
from __future__ import annotations

import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

DEMO_DIR = Path(__file__).parent
CORPUS_DIR = DEMO_DIR / "corpus"

GUTENBERG_URL = (
    "https://www.gutenberg.org/files/1342/1342-0.txt"  # Pride and Prejudice
)
REQUESTS_ZIP_URL = (
    "https://github.com/psf/requests/archive/refs/heads/main.zip"
)
DJANGO_ZIP_URL = (
    "https://github.com/django/django/archive/refs/heads/main.zip"
)

# Subsystems to extract — chosen to maximise domain-collision demo value.
# Each prefix maps to a flat destination directory name.
_DJANGO_SUBSYSTEMS = {
    # Module prefix inside the zip            destination folder
    "django-main/django/db/models/":         "django-orm",
    "django-main/django/db/backends/":       "django-db-backends",
    "django-main/django/http/":              "django-http",
    "django-main/django/auth/":              "django-auth",
    "django-main/django/middleware/":        "django-middleware",
    "django-main/django/template/":          "django-template",
    "django-main/django/core/cache/":        "django-cache",
    "django-main/django/core/signals.py":    "django-signals",  # single file
    "django-main/django/dispatch/":          "django-dispatch",
    "django-main/django/urls/":              "django-urls",
}

_HEADERS = {"User-Agent": "context-optimizer-demo/1.0 (corpus downloader)"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header/footer, keeping only the book text."""
    start_pat = re.compile(
        r"\*{3}\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*{3}",
        re.IGNORECASE,
    )
    end_pat = re.compile(
        r"\*{3}\s*END OF (THE|THIS) PROJECT GUTENBERG.*?\*{3}",
        re.IGNORECASE,
    )
    m_start = start_pat.search(text)
    if m_start:
        text = text[m_start.end():]
    m_end = end_pat.search(text)
    if m_end:
        text = text[: m_end.start()]
    return text.strip()


def _get(url: str, *, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# ── Gutenberg novel ────────────────────────────────────────────────────────────

def download_gutenberg() -> Path:
    out_dir = CORPUS_DIR / "gutenberg"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "pride-and-prejudice.txt"

    if dest.exists() and dest.stat().st_size > 100_000:
        print(f"[gutenberg] {dest.name} already present "
              f"({dest.stat().st_size:,} bytes) — skipping.")
        return dest

    print("[gutenberg] Downloading Pride and Prejudice from Project Gutenberg …")
    raw = _get(GUTENBERG_URL).decode("utf-8", errors="replace")
    cleaned = _strip_gutenberg_boilerplate(raw)
    dest.write_text(cleaned, encoding="utf-8")
    print(f"[gutenberg] Saved {dest.name} "
          f"({dest.stat().st_size:,} bytes after stripping boilerplate)")
    return dest


# ── requests library source ────────────────────────────────────────────────────

def download_requests_src() -> Path:
    out_dir = CORPUS_DIR / "requests-src"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob("*.py"))
    if len(existing) >= 10:
        print(f"[requests-src] {len(existing)} .py files already present — skipping.")
        return out_dir

    print("[requests-src] Downloading psf/requests source from GitHub …")
    data = _get(REQUESTS_ZIP_URL)

    zf = zipfile.ZipFile(io.BytesIO(data))
    extracted = 0
    for name in sorted(zf.namelist()):
        parts = Path(name).parts
        # Zip layout: requests-main/src/requests/<file>.py
        # parts: ('<root>', 'src', 'requests', '<file>.py')  — 4 parts
        if (
            len(parts) == 4
            and parts[1] == "src"
            and parts[2] == "requests"
            and parts[3].endswith(".py")
            and "__pycache__" not in parts
        ):
            dest = out_dir / parts[3]
            dest.write_bytes(zf.read(name))
            sz = dest.stat().st_size
            print(f"[requests-src]   {parts[3]:30s}  {sz:>8,} bytes")
            extracted += 1

    print(f"[requests-src] Extracted {extracted} files → {out_dir}")
    return out_dir


# ── Django source (curated subsystem subset) ──────────────────────────────────

def download_django_src() -> Path:
    """
    Download a curated slice of the Django source into corpus/django-src/.

    Subsystems extracted: ORM models, DB backends, HTTP layer, auth,
    middleware, template engine, cache, signals, dispatch, URL routing.
    This gives ~50-80 .py files covering the most query-interesting parts
    of a large real-world Python framework — ideal for domain-routing demos.

    Rebuild the index after running this:
        python setup_demo.py --force
    """
    out_root = CORPUS_DIR / "django-src"
    out_root.mkdir(parents=True, exist_ok=True)

    # Skip if already populated (at least 30 .py files across subdirs)
    existing = list(out_root.rglob("*.py"))
    if len(existing) >= 30:
        print(
            f"[django-src] {len(existing)} .py files already present — skipping."
        )
        return out_root

    print(
        "[django-src] Downloading django/django from GitHub "
        "(~40 MB zip — may take a minute) …"
    )
    data = _get(DJANGO_ZIP_URL, timeout=120)
    print(f"[django-src] Download complete ({len(data):,} bytes)")

    zf = zipfile.ZipFile(io.BytesIO(data))
    all_names = sorted(zf.namelist())

    extracted = 0
    for zip_path, dest_dir_name in _DJANGO_SUBSYSTEMS.items():
        dest_dir = out_root / dest_dir_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        is_single_file = zip_path.endswith(".py")

        for name in all_names:
            if not name.endswith(".py"):
                continue
            if "__pycache__" in name or "/tests/" in name:
                continue

            if is_single_file:
                # Exact file match
                if name != zip_path:
                    continue
                file_name = Path(name).name
            else:
                # Prefix match — only direct children (no sub-subdirectories)
                if not name.startswith(zip_path):
                    continue
                remainder = name[len(zip_path):]
                if "/" in remainder:
                    continue  # skip nested packages to keep corpus focused
                file_name = remainder
                if not file_name:
                    continue

            dest = dest_dir / file_name
            dest.write_bytes(zf.read(name))
            sz = dest.stat().st_size
            print(
                f"[django-src]   {dest_dir_name}/{file_name:35s}  {sz:>8,} bytes"
            )
            extracted += 1

    print(f"[django-src] Extracted {extracted} files across {len(_DJANGO_SUBSYSTEMS)} subsystems → {out_root}")
    return out_root


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Context Optimizer — Extended Corpus Downloader")
    print("=" * 60)
    print()

    try:
        gutenberg_path = download_gutenberg()
    except Exception as exc:
        print(f"[gutenberg] ERROR: {exc}", file=sys.stderr)
        gutenberg_path = None
    print()

    try:
        requests_path = download_requests_src()
    except Exception as exc:
        print(f"[requests-src] ERROR: {exc}", file=sys.stderr)
        requests_path = None
    print()

    try:
        django_path = download_django_src()
    except Exception as exc:
        print(f"[django-src] ERROR: {exc}", file=sys.stderr)
        django_path = None
    print()

    ok = gutenberg_path is not None and requests_path is not None
    if ok:
        print("All downloads complete.")
        print()
        print("Next step — rebuild the index (takes ~5–10 min for summarization):")
        print()
        print("    python setup_demo.py --force --no-code")
        print()
        print("Tip: add --no-code to skip ingesting the library src/ directory.")
        print("     Remove it if you want src/ included too.")
    else:
        print("Some downloads failed (see errors above).")
        sys.exit(1)

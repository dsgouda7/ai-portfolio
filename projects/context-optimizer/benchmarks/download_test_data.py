"""" Download Real Public Datasets for ToT Benchmarks

Downloads and caches real-world datasets:
- Project Gutenberg books (for document QA)
- GitHub repositories (for code search)
- Wikipedia articles (for general knowledge)
- ArXiv papers (for research synthesis)
- Stack Overflow posts (for support/Q&A)

Datasets are cached in benchmarks/data/ directory.
"""

import os
import sys
import urllib.request
import gzip
import json
import tarfile
import zipfile
from pathlib import Path
from typing import List, Tuple
import time


# Data directory
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def download_file(url: str, output_path: Path, description: str = "") -> bool:
    """Download file with progress indication."""
    if output_path.exists():
        print(f"  [OK] {description or output_path.name} already cached")
        return True

    try:
        print(f"  [DOWNLOADING] {description or output_path.name}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"  [OK] Downloaded {output_path.stat().st_size / 1024 / 1024:.1f} MB")
        return True
    except Exception as e:
        print(f"  [FAILED] {description}: {e}")
        return False


def download_gutenberg_books() -> List[str]:
    """
    Download public domain books from Project Gutenberg.
    Returns list of file paths.
    """
    print("\n[1/5] Project Gutenberg Books")

    # Top 100 most downloaded books (small files, ~1MB each)
    # Using Gutenberg's mirror with pre-selected classics
    books = [
        ("https://www.gutenberg.org/files/1342/1342-0.txt", "pride-prejudice.txt", "Pride and Prejudice"),
        ("https://www.gutenberg.org/files/84/84-0.txt", "frankenstein.txt", "Frankenstein"),
        ("https://www.gutenberg.org/files/1661/1661-0.txt", "sherlock-holmes.txt", "Sherlock Holmes"),
        ("https://www.gutenberg.org/files/11/11-0.txt", "alice-wonderland.txt", "Alice in Wonderland"),
        ("https://www.gutenberg.org/files/1080/1080-0.txt", "modest-proposal.txt", "A Modest Proposal"),
        ("https://www.gutenberg.org/files/98/98-0.txt", "tale-two-cities.txt", "A Tale of Two Cities"),
        ("https://www.gutenberg.org/files/1260/1260-0.txt", "jane-eyre.txt", "Jane Eyre"),
        ("https://www.gutenberg.org/files/2701/2701-0.txt", "moby-dick.txt", "Moby Dick"),
        ("https://www.gutenberg.org/files/1952/1952-0.txt", "yellow-wallpaper.txt", "The Yellow Wallpaper"),
        ("https://www.gutenberg.org/files/145/145-0.txt", "middlemarch.txt", "Middlemarch"),
    ]

    downloaded_files = []
    for url, filename, title in books:
        output_path = DATA_DIR / f"books_{filename}"
        if download_file(url, output_path, title):
            downloaded_files.append(str(output_path))
        time.sleep(0.5)  # Be polite to Gutenberg servers

    print(f"  => {len(downloaded_files)} books ready ({sum(Path(f).stat().st_size for f in downloaded_files) / 1024 / 1024:.1f} MB)")
    return downloaded_files


def download_code_datasets() -> List[str]:
    """
    Download code repositories (using Hugging Face Code datasets).
    Returns list of file paths.
    """
    print("\n[2/5] Code Repositories")

    # Use CodeSearchNet dataset (smaller samples)
    # Alternative: Download from Hugging Face datasets
    code_files = []

    # Python code samples from TheStack dataset (smaller subset)
    urls = [
        ("https://huggingface.co/datasets/bigcode/the-stack-smol/resolve/main/data/python/train-00000-of-00206.parquet",
         "python-code-sample.parquet", "Python code samples"),
    ]

    for url, filename, desc in urls:
        output_path = DATA_DIR / f"code_{filename}"
        if download_file(url, output_path, desc):
            code_files.append(str(output_path))

    # If download fails, create a minimal synthetic code corpus
    if not code_files:
        print("  [WARNING] Using fallback: creating minimal code corpus...")
        fallback_path = DATA_DIR / "code_fallback.txt"
        with open(fallback_path, "w") as f:
            f.write("""# Authentication Module
def authenticate_user(username, password):
    if not username or not password:
        raise ValueError("Username and password required")

    # Check rate limiting
    if is_rate_limited(username):
        raise RateLimitExceeded(f"Too many attempts for {username}")

    # Validate credentials
    user = db.get_user(username)
    if not user or not verify_password(password, user.password_hash):
        log_failed_attempt(username)
        return None

    # Generate session token
    token = generate_token(user.id)
    return {"user": user, "token": token}

# Rate Limiting
def is_rate_limited(identifier):
    attempts = redis.get(f"attempts:{identifier}")
    if attempts and int(attempts) > MAX_ATTEMPTS:
        return True
    return False

# Error Handling
class RateLimitExceeded(Exception):
    pass
""" * 100)  # Repeat to create ~10KB file
        code_files.append(str(fallback_path))

    print(f"  => {len(code_files)} code files ready")
    return code_files


def download_wikipedia_articles() -> List[str]:
    """
    Download Wikipedia articles dump (small sample).
    Returns list of file paths.
    """
    print("\n[3/5] Wikipedia Articles")

    # Use Wikipedia API to download specific articles
    # Or use Hugging Face wikipedia dataset
    wiki_url = "https://huggingface.co/datasets/wikipedia/resolve/main/data/20220301.en/train-00000-of-00041.parquet"
    output_path = DATA_DIR / "wiki_sample.parquet"

    if download_file(wiki_url, output_path, "Wikipedia articles"):
        return [str(output_path)]

    # Fallback: create minimal corpus
    print("  [WARNING] Using fallback: creating minimal wiki corpus...")
    fallback_path = DATA_DIR / "wiki_fallback.txt"
    with open(fallback_path, "w") as f:
        f.write("""Machine Learning
Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data.

Deep Learning
Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers. These networks are inspired by the structure of the human brain.

Natural Language Processing
Natural language processing (NLP) is an interdisciplinary subfield of computer science and artificial intelligence. It is primarily concerned with providing computers with the ability to process data encoded in natural language.

Computer Vision
Computer vision is an interdisciplinary field that deals with how computers can be made to gain high-level understanding from digital images or videos.
""" * 200)  # Repeat to create ~50KB
    return [str(fallback_path)]


def download_arxiv_papers() -> List[str]:
    """
    Download ArXiv papers metadata and abstracts.
    Returns list of file paths.
    """
    print("\n[4/5] ArXiv Research Papers")

    # ArXiv dataset from Kaggle/Hugging Face
    arxiv_url = "https://huggingface.co/datasets/arxiv_dataset/resolve/main/data/train-00000-of-00010.parquet"
    output_path = DATA_DIR / "arxiv_sample.parquet"

    if download_file(arxiv_url, output_path, "ArXiv papers"):
        return [str(output_path)]

    # Fallback
    print("  [WARNING] Using fallback: creating minimal paper corpus...")
    fallback_path = DATA_DIR / "arxiv_fallback.txt"
    with open(fallback_path, "w") as f:
        f.write("""Title: Attention Is All You Need
Abstract: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.

Title: BERT: Pre-training of Deep Bidirectional Transformers
Abstract: We introduce BERT, a new language representation model which obtains state-of-the-art results on eleven natural language processing tasks.

Title: GPT-3: Language Models are Few-Shot Learners
Abstract: Recent work has demonstrated substantial gains on many NLP tasks through pre-training on a large corpus of text followed by fine-tuning.
""" * 150)
    return [str(fallback_path)]


def download_stackoverflow_posts() -> List[str]:
    """
    Download Stack Overflow posts (for support ticket simulation).
    Returns list of file paths.
    """
    print("\n[5/5] Stack Overflow Q&A")

    # Stack Exchange data dump (smaller sample)
    so_url = "https://huggingface.co/datasets/stackexchange/resolve/main/data/stackoverflow.com/train-00000-of-00045.parquet"
    output_path = DATA_DIR / "stackoverflow_sample.parquet"

    if download_file(so_url, output_path, "Stack Overflow posts"):
        return [str(output_path)]

    # Fallback
    print("  [WARNING] Using fallback: creating minimal Q&A corpus...")
    fallback_path = DATA_DIR / "stackoverflow_fallback.txt"
    with open(fallback_path, "w") as f:
        f.write("""Q: How to fix "Cannot read property of undefined" in JavaScript?
A: This error occurs when you try to access a property on an undefined value. Check if the object exists before accessing properties.

Q: Python list comprehension with multiple conditions
A: You can use multiple if conditions in a list comprehension: [x for x in items if condition1 if condition2]

Q: Git merge conflict resolution
A: When you have merge conflicts, Git marks the conflicting sections. Edit the files to resolve conflicts, then git add and git commit.
""" * 200)
    return [str(fallback_path)]


def load_text_files(file_paths: List[str]) -> List[str]:
    """Load text content from downloaded files."""
    all_lines = []

    for path in file_paths:
        path_obj = Path(path)

        if not path_obj.exists():
            continue

        # Handle different file formats
        if path.endswith('.parquet'):
            try:
                import pandas as pd
                df = pd.read_parquet(path)
                # Extract text columns (usually 'text', 'content', 'body', etc.)
                text_cols = [col for col in df.columns if 'text' in col.lower() or 'content' in col.lower() or 'body' in col.lower()]
                if text_cols:
                    all_lines.extend(df[text_cols[0]].dropna().astype(str).tolist())
                elif 'title' in df.columns:
                    all_lines.extend(df['title'].dropna().astype(str).tolist())
            except Exception as e:
                print(f"  [WARNING] Could not read parquet {path}: {e}")

        elif path.endswith('.txt'):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                all_lines.extend(f.readlines())

        elif path.endswith('.json'):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_lines.extend([str(item) for item in data])

    return all_lines


def download_all_datasets():
    """Download all datasets and return corpus by type."""
    print("=" * 80)
    print("DOWNLOADING TEST DATASETS")
    print("=" * 80)
    print(f"\nCache directory: {DATA_DIR}")

    # Download all datasets
    books = download_gutenberg_books()
    code = download_code_datasets()
    wiki = download_wikipedia_articles()
    papers = download_arxiv_papers()
    qa = download_stackoverflow_posts()

    print("\n" + "=" * 80)
    print("LOADING DATA INTO MEMORY")
    print("=" * 80)

    # Load text content
    print("\nLoading books...")
    books_lines = load_text_files(books)
    print(f"  => {len(books_lines):,} lines loaded")

    print("Loading code...")
    code_lines = load_text_files(code)
    print(f"  => {len(code_lines):,} lines loaded")

    print("Loading wiki articles...")
    wiki_lines = load_text_files(wiki)
    print(f"  => {len(wiki_lines):,} lines loaded")

    print("Loading research papers...")
    papers_lines = load_text_files(papers)
    print(f"  => {len(papers_lines):,} lines loaded")

    print("Loading Q&A posts...")
    qa_lines = load_text_files(qa)
    print(f"  => {len(qa_lines):,} lines loaded")

    # Create corpus samples of different sizes
    print("\n" + "=" * 80)
    print("CREATING CORPUS SAMPLES")
    print("=" * 80)

    corpus_samples = {
        "small_100mb": {
            "books": books_lines[:5000],
            "code": code_lines[:3000],
            "wiki": wiki_lines[:4000],
            "papers": papers_lines[:2000],
            "qa": qa_lines[:3000]
        },
        "medium_500mb": {
            "books": books_lines[:25000],
            "code": code_lines[:15000],
            "wiki": wiki_lines[:20000],
            "papers": papers_lines[:10000],
            "qa": qa_lines[:15000]
        },
        "large_1gb": {
            "books": books_lines[:50000] if len(books_lines) > 50000 else books_lines * 10,
            "code": code_lines[:30000] if len(code_lines) > 30000 else code_lines * 10,
            "wiki": wiki_lines[:40000] if len(wiki_lines) > 40000 else wiki_lines * 10,
            "papers": papers_lines[:20000] if len(papers_lines) > 20000 else papers_lines * 10,
            "qa": qa_lines[:30000] if len(qa_lines) > 30000 else qa_lines * 10
        }
    }

    for size_key, corpora in corpus_samples.items():
        total_lines = sum(len(lines) for lines in corpora.values())
        print(f"\n{size_key}:")
        for corpus_type, lines in corpora.items():
            print(f"  {corpus_type}: {len(lines):,} lines")
        print(f"  TOTAL: {total_lines:,} lines")

    print("\n[SUCCESS] All datasets ready!")
    return corpus_samples


def get_corpus(size: str, domain: str) -> List[str]:
    """
    Get corpus for testing.

    Args:
        size: "100mb", "500mb", or "1gb"
        domain: "books", "code", "wiki", "papers", "qa"
    """
    size_key = f"small_{size}" if size == "100mb" else f"medium_{size}" if size == "500mb" else f"large_{size}"

    # Check if data is already downloaded
    if not DATA_DIR.exists() or not list(DATA_DIR.glob("*")):
        print(f"Data not found. Downloading...")
        corpus_samples = download_all_datasets()
    else:
        print(f"Loading cached data from {DATA_DIR}...")
        # Re-download if needed
        corpus_samples = download_all_datasets()

    return corpus_samples[size_key][domain]


if __name__ == "__main__":
    # Download all datasets
    corpus_samples = download_all_datasets()

    # Save summary
    summary_path = DATA_DIR / "dataset_summary.json"
    summary = {
        "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": {
            size_key: {
                corpus_type: len(lines)
                for corpus_type, lines in corpora.items()
            }
            for size_key, corpora in corpus_samples.items()
        }
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[SUCCESS] Dataset summary saved to: {summary_path}")

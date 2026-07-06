"""
Step-by-step walkthrough of TF-IDF sentence scoring in extractive compression.
Run:  python explain_tfidf.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PASSAGE = """
Albert Einstein published the special theory of relativity in 1905.
Relativity changed our understanding of space and time completely.
Einstein also developed the general theory of relativity in 1915.
General relativity describes gravity as the curvature of spacetime.
The equation E equals m c squared shows mass and energy equivalence.
Einstein received the Nobel Prize in Physics in 1921 for photoelectric effect.
He was born in Germany and later became an American citizen.
The photoelectric effect explained how light behaves as packets of energy.
These packets are now called photons by modern physicists.
Einstein fled Nazi Germany in 1933 and settled at Princeton New Jersey.
""".strip()

print("INPUT TEXT")
print("=" * 65)
for i, s in enumerate(PASSAGE.split(". ")):
    print(f"  S{i}: {s.strip()}")

# ── Step 1: Split into sentences ─────────────────────────────────────────────
sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", PASSAGE) if s.strip()]

print(f"\n\nSTEP 1: Split into {len(sentences)} sentences")
print("=" * 65)
for i, s in enumerate(sentences):
    print(f"  [{i}] {s}")

# ── Step 2: Count word frequencies (TF) ──────────────────────────────────────
_stop = frozenset(
    "a an the and or but in on at to for of with by from is are was were be been "
    "have has had do does did that this it he she they we you i his her its their "
    "our not so as if up out no about what which will would could should may might "
    "then than when where who whom whose how much many more most some any all".split()
)

freq: dict[str, int] = {}
for s in sentences:
    for w in re.findall(r"[a-z]+", s.lower()):
        if w not in _stop and len(w) >= 3:
            freq[w] = freq.get(w, 0) + 1

print(f"\n\nSTEP 2: Word frequencies (TF) — top 15 content words")
print("=" * 65)
print("  Note: stopwords removed ('the', 'and', 'in', etc.)")
print()
for word, count in sorted(freq.items(), key=lambda x: -x[1])[:15]:
    bar = "|" * count
    print(f"  {word:<20} {count}  {bar}")

# ── Step 3: Score each sentence ───────────────────────────────────────────────
max_f = max(freq.values())

def _score(s: str) -> float:
    words = [w for w in re.findall(r"[a-z]+", s.lower())
             if w not in _stop and len(w) >= 3]
    return sum(freq.get(w, 0) / max_f for w in words) / len(words) if words else 0.0

print(f"\n\nSTEP 3: Score each sentence  (score = mean TF of content words)")
print("=" * 65)
print("  High score = sentence uses words that appear frequently ACROSS the block")
print("  Low score  = sentence uses rare, unique words → WILL BE DROPPED")
print()
scored = [(i, s, _score(s)) for i, s in enumerate(sentences)]
for idx, sent, score in scored:
    bar = "|" * round(score * 30)
    print(f"  S{idx} [{score:.3f}] {bar}")
    print(f"        {sent[:70]}")

# ── Step 4: Keep top 35% ──────────────────────────────────────────────────────
n_keep = max(2, round(len(sentences) * 0.35))
top = sorted(scored, key=lambda x: x[2], reverse=True)[:n_keep]
top.sort(key=lambda x: x[0])
kept_indices = {idx for idx, _, _ in top}

print(f"\n\nSTEP 4: Keep top {n_keep}/{len(sentences)} sentences (ratio=0.35)")
print("=" * 65)
for idx, sent, score in scored:
    status = "KEPT   " if idx in kept_indices else "DROPPED"
    print(f"  [{score:.3f}] {status}  S{idx}: {sent[:65]}")

summary = " ".join(s for _, s, _ in top)
print(f"\n\nOUTPUT SUMMARY ({len(summary.split())} words, {len(summary)//4} tokens)")
print("=" * 65)
print(f"  {summary}")

# ── The core problem ──────────────────────────────────────────────────────────
print(f"\n\nTHE PROBLEM — WHY THIS BREAKS FOR LARGE BLOCKS")
print("=" * 65)
print("""
On a small passage (above), TF-IDF works reasonably well:
  sentences that repeat key topic words (Einstein, relativity) score high.

On a 500 KB Wikipedia block containing MANY DIFFERENT topics:
  A sentence about "1919 solar eclipse" appears exactly ONCE in 500 KB.
  Its words ("eclipse", "Eddington", "1919") have freq=1 out of 50,000+ words.
  Score ≈ 1/max_f ≈ near zero → DROPPED.

  A sentence about "the" repeated context appears many times → KEPT.

Result: the extractive "summary" is full of generic repeated sentences
and drops all the rare but important facts (dates, names, measurements).

When that summary is embedded, the vector represents "generic repeated text"
not "the specific content of this block" → poor retrieval.

This is why the LLM triple format is needed: it reads the block as a whole
and writes out the rare important facts that TF-IDF would always discard.
""")

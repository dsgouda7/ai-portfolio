# Book Benchmark Results

**Run date**: 2026-07-01 17:58  |  **Books**: 25  |  **Questions**: 495  |  **Lines/book cap**: 3,000  |  **Index-pad**: 0 MB/book  |  **Workers**: 4

## Summary

| Metric | Value |
|--------|-------|
| Books benchmarked | 25 |
| Total questions | 495 |
| Avg keyword recall (judge) | 21.6% |
| Avg compression time per book | 0s |
| Overall token reduction | 0.0% |
| Original tokens (all books) | 1,213,886 |
| Compressed tokens (all books) | 1,213,886 |

---

## Per-Book Results

| Book | Author | Base Chunks | Index Chunks | Qs | Avg KW Recall | Compress(s) |
|------|--------|------------:|-------------:|---:|:-------------:|------------:|
| Middlemarch | Eliot, George | 95 | 95 | 20 | 43% | 0s |
| Alice's Adventures in Wonderland | Carroll, Lewis | 74 | 74 | 20 | 42% | 0s |
| The Secret of Chimneys | Christie, Agatha | 84 | 84 | 20 | 33% | 0s |
| A Room with a View | Forster, E. M. (Edward Mo | 90 | 90 | 20 | 33% | 0s |
| The King in Yellow | Chambers, Robert W. (Robe | 95 | 95 | 20 | 32% | 0s |
| The Great Gatsby | Fitzgerald, F. Scott (Fra | 89 | 89 | 20 | 31% | 0s |
| Dracula | Stoker, Bram | 101 | 101 | 20 | 30% | 0s |
| Twenty years after | Dumas, Alexandre | 87 | 87 | 20 | 30% | 0s |
| Jane Eyre: An Autobiography | Brontë, Charlotte | 95 | 95 | 20 | 30% | 0s |
| The Adventures of Sherlock Holmes | Doyle, Arthur Conan | 92 | 92 | 20 | 27% | 0s |
| Pride and Prejudice | Austen, Jane | 96 | 96 | 17 | 26% | 0s |
| The Count of Monte Cristo | Dumas, Alexandre | 88 | 88 | 20 | 26% | 0s |
| The Brothers Karamazov | Dostoyevsky, Fyodor | 98 | 98 | 20 | 25% | 0s |
| The Blue Castle: a novel | Montgomery, L. M. (Lucy M | 94 | 94 | 20 | 24% | 0s |
| Adventures of Huckleberry Finn | Twain, Mark | 90 | 90 | 20 | 23% | 0s |
| Crime and Punishment | Dostoyevsky, Fyodor | 101 | 101 | 20 | 19% | 0s |
| My Life — Volume 1 | Wagner, Richard | 102 | 102 | 20 | 18% | 0s |
| The Enchanted April | Von Arnim, Elizabeth | 96 | 96 | 20 | 17% | 0s |
| The Expedition of Humphry Clinker | Smollett, T. (Tobias) | 104 | 104 | 20 | 13% | 0s |
| Frankenstein; or, the modern prometheus | Shelley, Mary Wollstonecr | 101 | 101 | 20 | 7% | 0s |
| Moby Dick; Or, The Whale | Melville, Herman | 97 | 97 | 18 | 5% | 0s |
| Romeo and Juliet | Shakespeare, William | 52 | 52 | 20 | 4% | 0s |
| The Complete Works of William Shakespeare | Shakespeare, William | 64 | 64 | 20 | 2% | 0s |
| The strange case of Dr. Jekyll and Mr. Hyde | Stevenson, Robert Louis | 72 | 72 | 20 | 0% | 0s |
| Little Women; Or, Meg, Jo, Beth, and Amy | Alcott, Louisa May | 91 | 91 | 20 | 0% | 0s |

---

## Judge Methodology

Answers are scored by **keyword recall**: the fraction of Wikipedia-sourced
expected keywords that appear (case-insensitive substring match) in the
pipeline's answer.  Expected answers are fetched once at `build-banks` time
and stored in `data/question_banks/<slug>.json` — no LLM judge used at
evaluation time.  Judging runs asynchronously in a thread pool alongside
query execution.

*Generated 2026-07-01 17:58 — do not edit manually.*
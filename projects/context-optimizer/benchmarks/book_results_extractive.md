# Book Benchmark Results

**Run date**: 2026-07-01 17:56  |  **Books**: 25  |  **Questions**: 495  |  **Lines/book cap**: 3,000  |  **Index-pad**: 0 MB/book  |  **Workers**: 2

## Summary

| Metric | Value |
|--------|-------|
| Books benchmarked | 25 |
| Total questions | 495 |
| Avg keyword recall (judge) | 35.1% |
| Avg compression time per book | 1s |
| Overall token reduction | 68.6% |
| Original tokens (all books) | 1,213,886 |
| Compressed tokens (all books) | 381,730 |

---

## Per-Book Results

| Book | Author | Base Chunks | Index Chunks | Qs | Avg KW Recall | Compress(s) |
|------|--------|------------:|-------------:|---:|:-------------:|------------:|
| Adventures of Huckleberry Finn | Twain, Mark | 90 | 90 | 20 | 47% | 1s |
| The Complete Works of William Shakespeare | Shakespeare, William | 64 | 64 | 20 | 46% | 1s |
| Middlemarch | Eliot, George | 95 | 95 | 20 | 46% | 0s |
| The King in Yellow | Chambers, Robert W. (Robe | 95 | 95 | 20 | 44% | 1s |
| The Count of Monte Cristo | Dumas, Alexandre | 88 | 88 | 20 | 44% | 0s |
| Jane Eyre: An Autobiography | Brontë, Charlotte | 95 | 95 | 20 | 39% | 1s |
| Little Women; Or, Meg, Jo, Beth, and Amy | Alcott, Louisa May | 91 | 91 | 20 | 38% | 1s |
| A Room with a View | Forster, E. M. (Edward Mo | 90 | 90 | 20 | 38% | 0s |
| The Enchanted April | Von Arnim, Elizabeth | 96 | 96 | 20 | 38% | 1s |
| Twenty years after | Dumas, Alexandre | 87 | 87 | 20 | 37% | 1s |
| The Adventures of Sherlock Holmes | Doyle, Arthur Conan | 92 | 92 | 20 | 37% | 1s |
| Alice's Adventures in Wonderland | Carroll, Lewis | 74 | 74 | 20 | 37% | 0s |
| The Brothers Karamazov | Dostoyevsky, Fyodor | 98 | 98 | 20 | 36% | 1s |
| The Blue Castle: a novel | Montgomery, L. M. (Lucy M | 94 | 94 | 20 | 36% | 0s |
| The Secret of Chimneys | Christie, Agatha | 84 | 84 | 20 | 35% | 1s |
| Pride and Prejudice | Austen, Jane | 96 | 96 | 17 | 35% | 0s |
| Crime and Punishment | Dostoyevsky, Fyodor | 101 | 101 | 20 | 35% | 0s |
| The strange case of Dr. Jekyll and Mr. Hyde | Stevenson, Robert Louis | 72 | 72 | 20 | 34% | 1s |
| My Life — Volume 1 | Wagner, Richard | 102 | 102 | 20 | 31% | 1s |
| Dracula | Stoker, Bram | 101 | 101 | 20 | 30% | 1s |
| The Expedition of Humphry Clinker | Smollett, T. (Tobias) | 104 | 104 | 20 | 28% | 1s |
| Moby Dick; Or, The Whale | Melville, Herman | 97 | 97 | 18 | 26% | 0s |
| Frankenstein; or, the modern prometheus | Shelley, Mary Wollstonecr | 101 | 101 | 20 | 26% | 0s |
| The Great Gatsby | Fitzgerald, F. Scott (Fra | 89 | 89 | 20 | 21% | 1s |
| Romeo and Juliet | Shakespeare, William | 52 | 52 | 20 | 14% | 0s |

---

## Judge Methodology

Answers are scored by **keyword recall**: the fraction of Wikipedia-sourced
expected keywords that appear (case-insensitive substring match) in the
pipeline's answer.  Expected answers are fetched once at `build-banks` time
and stored in `data/question_banks/<slug>.json` — no LLM judge used at
evaluation time.  Judging runs asynchronously in a thread pool alongside
query execution.

*Generated 2026-07-01 17:56 — do not edit manually.*
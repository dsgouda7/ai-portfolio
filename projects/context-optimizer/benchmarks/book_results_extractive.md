# Book Benchmark Results

**Run date**: 2026-07-06 10:20  |  **Books**: 25  |  **Questions**: 479  |  **Lines/book cap**: 3,000  |  **Index-pad**: 0 MB/book  |  **Workers**: 1

## Summary

| Metric | Value |
|--------|-------|
| Books benchmarked | 25 |
| Total questions | 479 |
| Avg keyword recall (all tiers) | 31.8% |
| Avg compression time per book | 0s |
| Overall token reduction | 68.6% |
| Original tokens (all books) | 1,173,064 |
| Compressed tokens (all books) | 367,899 |

## Recall by Difficulty Tier

How well does the pipeline answer questions at each complexity level.
Easy = surface recall (author/genre/summary).  Expert = cross-context scholarly.

| Tier | Questions | Avg KW Recall | Interpretation |
|------|----------:|:-------------:|----------------|
| easy | 20 | 23.7% | Author / genre / one-line summary |
| medium | 431 | 33.7% | Main plot / characters / central conflict |
| hard | 4 | 16.1% | Themes / setting / specific events |
| very_hard | 12 | 14.0% | Style / critical reception / symbolism |
| expert | 12 | 18.0% | Publication history / cross-section context |

---

## Per-Book Results

| Book | Author | Base Chunks | Index Chunks | Qs | Avg KW Recall | Compress(s) |
|------|--------|------------:|-------------:|---:|:-------------:|------------:|
| The Complete Works of William Shakespeare | Shakespeare, William | 64 | 64 | 20 | 46% | 1s |
| Middlemarch | Eliot, George | 95 | 95 | 20 | 46% | 0s |
| The King in Yellow | Chambers, Robert W. (Robe | 95 | 95 | 20 | 44% | 1s |
| The Count of Monte Cristo | Dumas, Alexandre | 88 | 88 | 20 | 44% | 0s |
| Jane Eyre: An Autobiography | Brontë, Charlotte | 95 | 95 | 20 | 39% | 1s |
| Little Women; Or, Meg, Jo, Beth, and Amy | Alcott, Louisa May | 91 | 91 | 20 | 38% | 1s |
| A Room with a View | Forster, E. M. (Edward Mo | 90 | 90 | 20 | 38% | 0s |
| Twenty years after | Dumas, Alexandre | 87 | 87 | 20 | 37% | 1s |
| The Adventures of Sherlock Holmes | Doyle, Arthur Conan | 92 | 92 | 20 | 37% | 1s |
| Alice's Adventures in Wonderland | Carroll, Lewis | 74 | 74 | 20 | 37% | 0s |
| The Blue Castle: a novel | Montgomery, L. M. (Lucy M | 94 | 94 | 20 | 36% | 0s |
| The Secret of Chimneys | Christie, Agatha | 84 | 84 | 20 | 35% | 1s |
| Pride and Prejudice | Austen, Jane | 96 | 96 | 17 | 35% | 0s |
| Crime and Punishment | Dostoyevsky, Fyodor | 101 | 101 | 20 | 35% | 0s |
| The strange case of Dr. Jekyll and Mr. Hyde | Stevenson, Robert Louis | 72 | 72 | 20 | 34% | 1s |
| My Life — Volume 1 | Wagner, Richard | 102 | 102 | 20 | 31% | 1s |
| Dracula | Stoker, Bram | 101 | 101 | 20 | 30% | 1s |
| The Mysteries of Udolpho | Radcliffe, Ann Ward | 97 | 97 | 16 | 27% | 0s |
| Moby Dick; Or, The Whale | Melville, Herman | 97 | 97 | 18 | 26% | 0s |
| Frankenstein; or, the modern prometheus | Shelley, Mary Wollstonecr | 101 | 101 | 20 | 26% | 0s |
| The Extraordinary Adventures of Arsène Lupin, Gent | Leblanc, Maurice | 90 | 90 | 16 | 24% | 0s |
| The Green Mummy | Hume, Fergus | 95 | 95 | 16 | 18% | 0s |
| Romeo and Juliet | Shakespeare, William | 52 | 52 | 20 | 14% | 0s |
| Romeo and Juliet | Shakespeare, William | 52 | 52 | 20 | 14% | 0s |
| Carmen | Mérimée, Prosper | 67 | 67 | 16 | 6% | 0s |

---

## Judge Methodology

Answers are scored by **keyword recall**: the fraction of Wikipedia-sourced
expected keywords that appear (case-insensitive substring match) in the
pipeline's answer.  Expected answers are fetched once at `build-banks` time
and stored in `data/question_banks/<slug>.json` — no LLM judge used at
evaluation time.  Judging runs asynchronously in a thread pool alongside
query execution.

Difficulty tiers are assigned by question source:
- **easy**: catalog summary / author / genre (surface recall)
- **medium**: plot summary / characters / central conflict (narrative)
- **hard**: themes / setting / specific events (analytical)
- **very_hard**: reception / style / symbolism (critical analysis)
- **expert**: publication history / cross-section context (scholarly)

*Generated 2026-07-06 10:20 — do not edit manually.*
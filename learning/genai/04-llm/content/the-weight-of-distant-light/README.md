# Multi-Genre Fiction Corpus for LLM Fine-Tuning

**Seven original novels across diverse genres, written as plain-text training corpus.**

## Corpus Summary

| Novel | Genre | Chapters | Words | Size | Files |
|---|---|---|---|---|---|
| The Weight of Distant Light | Sci-Fi (Generation Ship) | 32 | ~54,800 | ~331 KB | `chapter-001.txt` – `chapter-032.txt` |
| The Tidebound Accord | Fantasy (Epic) | 25 | ~92,000 | ~590 KB | `fantasy-chapter-001.txt` – `fantasy-chapter-025.txt` |
| The Cartographer's Cipher | Mystery/Thriller (Noir) | 13 | ~45,000 | ~280 KB | `mystery-chapter-001.txt` – `mystery-chapter-013.txt` |
| The Silk Merchant's Daughter | Historical Fiction (Tang Dynasty) | 15 | ~49,500 | ~330 KB | `historical-chapter-001.txt` – `historical-chapter-015.txt` |
| Neural Drift | Cyberpunk | 16 | ~57,000 | ~350 KB | `cyberpunk-chapter-001.txt` – `cyberpunk-chapter-016.txt` |
| The Hollow Beneath | Horror/Gothic | 20 | ~52,000 | ~320 KB | `horror-chapter-001.txt` – `horror-chapter-020.txt` |
| The Weight of Tides | Literary Fiction | 20 | ~72,000 | ~370 KB | `literary-chapter-001.txt` – `literary-chapter-020.txt` |
| **TOTAL** | **7 genres** | **141** | **~422,300** | **~2.57 MB** | **141 chapter files + 7 READMEs** |

## Format

- **Structure:** One chapter per `.txt` file, beginning with `Chapter N: Title` followed by plain paragraphs separated by blank lines
- **Encoding:** UTF-8 plain text
- **Content:** 100% original fiction (no copyrighted/existing IP), suitable for machine learning training
- **Style:** Each novel maintains consistent characters, world-building, and narrative continuity across its full arc

## Individual Novel Synopses

See genre-specific README files for detailed synopses:
- [The Weight of Distant Light](README_scifi.md) - Original README preserved as `README_scifi.md`
- [fantasy_README.md](fantasy_README.md) - The Tidebound Accord
- [mystery_README.md](mystery_README.md) - The Cartographer's Cipher
- [historical_README.md](historical_README.md) - The Silk Merchant's Daughter
- [cyberpunk_README.md](cyberpunk_README.md) - Neural Drift
- [horror_README.md](horror_README.md) - The Hollow Beneath
- [literary_README.md](literary_README.md) - The Weight of Tides

## Training Use Cases

This multi-genre corpus supports:
- **Style diversity:** Train models to handle fantasy world-building, noir detective prose, historical period detail, cyberpunk jargon, gothic atmosphere, literary character depth, and hard sci-fi exposition
- **Vocabulary breadth:** Spans archaic Tang Dynasty terminology, modern crime investigation, future tech slang, fantasy magic systems, horror psychological tension, and scientific concepts
- **Narrative structures:** Quest arcs, murder mysteries, family sagas, conspiracy thrillers, generation-spanning epics, psychological horror escalation
- **Concept comparison:** Non-instructional training on the full corpus vs. genre-specific subsets to measure domain adaptation vs. catastrophic forgetting

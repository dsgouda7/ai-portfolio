# Multi-Genre Fiction Corpus for LLM Fine-Tuning

**Seven original novels across diverse genres, written as plain-text training corpus.**

## Corpus Summary

| Novel | Genre | Chapters | Words | Size | Directory |
|---|---|---|---|---|---|
| The Weight of Distant Light | Sci-Fi (Generation Ship) | 40 | ~101,000 | ~495 KB | [`the-weight-of-distant-light/`](the-weight-of-distant-light/) |
| The Tidebound Accord | Fantasy (Epic) | 33 | ~100,000 | ~486 KB | [`the-tidebound-accord/`](the-tidebound-accord/) |
| The Cartographer's Cipher | Mystery/Thriller (Noir) | 21 | ~80,000 | ~389 KB | [`the-cartographers-cipher/`](the-cartographers-cipher/) |
| The Silk Merchant's Daughter | Historical Fiction (Tang Dynasty) | 23 | ~76,000 | ~372 KB | [`the-silk-merchants-daughter/`](the-silk-merchants-daughter/) |
| Neural Drift | Cyberpunk | 24 | ~77,000 | ~375 KB | [`neural-drift/`](neural-drift/) |
| The Hollow Beneath | Horror/Gothic | 28 | ~90,000 | ~439 KB | [`the-hollow-beneath/`](the-hollow-beneath/) |
| The Weight of Tides | Literary Fiction | 28 | ~95,000 | ~465 KB | [`the-weight-of-tides/`](the-weight-of-tides/) |
| **TOTAL** | **7 genres** | **197** | **~619,000** | **~3.0 MB** | **7 novel directories** |

## Directory Structure

```
content/
├── the-weight-of-distant-light/
│   ├── README.md
│   ├── chapter-001.txt
│   ├── chapter-002.txt
│   └── ... (32 chapters total)
├── the-tidebound-accord/
│   ├── README.md
│   ├── chapter-001.txt
│   └── ... (25 chapters total)
├── the-cartographers-cipher/
│   └── ... (13 chapters)
├── the-silk-merchants-daughter/
│   └── ... (15 chapters)
├── neural-drift/
│   └── ... (16 chapters)
├── the-hollow-beneath/
│   └── ... (20 chapters)
└── the-weight-of-tides/
    └── ... (20 chapters)
```

Each novel directory contains:
- `README.md` — detailed synopsis and metadata
- `chapter-001.txt` through `chapter-NNN.txt` — one chapter per file

## Format

- **Structure:** One chapter per `.txt` file, beginning with `Chapter N: Title` followed by plain paragraphs separated by blank lines
- **Encoding:** UTF-8 plain text
- **Content:** 100% original fiction (no copyrighted/existing IP), suitable for machine learning training
- **Style:** Each novel maintains consistent characters, world-building, and narrative continuity across its full arc

## Individual Novel Synopses

Each novel directory contains a detailed README:
- [the-weight-of-distant-light/README.md](the-weight-of-distant-light/README.md) - Generation ship first contact sci-fi
- [the-tidebound-accord/README.md](the-tidebound-accord/README.md) - Epic fantasy quest
- [the-cartographers-cipher/README.md](the-cartographers-cipher/README.md) - Noir detective mystery
- [the-silk-merchants-daughter/README.md](the-silk-merchants-daughter/README.md) - Tang Dynasty historical fiction
- [neural-drift/README.md](neural-drift/README.md) - Cyberpunk memory broker conspiracy
- [the-hollow-beneath/README.md](the-hollow-beneath/README.md) - Gothic/cosmic horror
- [the-weight-of-tides/README.md](the-weight-of-tides/README.md) - Literary marine biology first contact

## Training Use Cases

This multi-genre corpus supports:
- **Style diversity:** Train models to handle fantasy world-building, noir detective prose, historical period detail, cyberpunk jargon, gothic atmosphere, literary character depth, and hard sci-fi exposition
- **Vocabulary breadth:** Spans archaic Tang Dynasty terminology, modern crime investigation, future tech slang, fantasy magic systems, horror psychological tension, and scientific concepts
- **Narrative structures:** Quest arcs, murder mysteries, family sagas, conspiracy thrillers, generation-spanning epics, psychological horror escalation
- **Concept comparison:** Non-instructional training on the full corpus vs. genre-specific subsets to measure domain adaptation vs. catastrophic forgetting

## Loading the Corpus

The notebook includes helper functions that support:
- Loading all novels combined
- Loading specific novels only by directory name
- Limiting chapters per novel for fast CPU demos
- Automatic discovery of available novels

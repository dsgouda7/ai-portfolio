# Riverside House Multi-Genre Fiction Corpus

**Eight original novels across diverse genres, shared across the GenAI learning arc.**

This directory is the canonical Riverside manuscript source for transformer examples,
fine-tuning experiments, retrieval and evaluation exercises, and later operational
chapters. Chapter directories may derive small fixtures from it, but they should not
maintain private copies of the manuscripts.

## Corpus Summary

| Novel | Genre | Chapters | Words | Size | Directory |
|---|---|---|---|---|---|
| The Weight of Distant Light | Sci-Fi (Generation Ship) | 40 | ~82,700 | ~496 KiB | [`the-weight-of-distant-light/`](the-weight-of-distant-light/) |
| The Tidebound Accord | Fantasy (Epic) | 33 | ~79,700 | ~487 KiB | [`the-tidebound-accord/`](the-tidebound-accord/) |
| The Cartographer's Cipher | Mystery/Thriller (Noir) | 21 | ~62,900 | ~390 KiB | [`the-cartographers-cipher/`](the-cartographers-cipher/) |
| The Silk Merchant's Daughter | Historical Fiction (Tang Dynasty) | 23 | ~62,200 | ~373 KiB | [`the-silk-merchants-daughter/`](the-silk-merchants-daughter/) |
| Neural Drift | Cyberpunk | 24 | ~60,900 | ~376 KiB | [`neural-drift/`](neural-drift/) |
| The Hollow Beneath | Horror/Gothic | 28 | ~70,000 | ~440 KiB | [`the-hollow-beneath/`](the-hollow-beneath/) |
| The Weight of Tides | Literary Fiction | 28 | ~74,900 | ~467 KiB | [`the-weight-of-tides/`](the-weight-of-tides/) |
| The Everglades Cipher | Noir-Historical Detective | 28 chapters + 10 appendices | ~112,200 | ~674 KiB | [`the-everglades-cipher/`](the-everglades-cipher/) |
| **TOTAL** | **8 genres** | **235 chapters + 10 appendices** | **~605,400** | **~3.6 MiB** | **8 novel directories** |

## Directory Structure

```
content/
 the-weight-of-distant-light/
    README.md
    chapter-001.txt
    chapter-002.txt
    ... (40 chapters total)
 the-tidebound-accord/
    README.md
    chapter-001.txt
    ... (33 chapters total)
 the-cartographers-cipher/
    ... (21 chapters)
 the-silk-merchants-daughter/
    ... (23 chapters)
 neural-drift/
    ... (24 chapters)
 the-hollow-beneath/
    ... (28 chapters)
 the-weight-of-tides/
   ... (28 chapters)
 the-everglades-cipher/
   chapter-001.txt
   ... (28 chapters)
   appendix-a-historical-context.txt
   ... (10 appendices)
```

Each novel directory contains:
- `README.md` - detailed synopsis, provenance, and metadata
- `chapter-001.txt` through `chapter-NNN.txt` - one chapter per file

The Everglades directory also contains ten `appendix-*.txt` reference documents. The GPU practice notebook inventories but excludes them from chapter-based train, validation, and test splits.

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
- [the-everglades-cipher/README.md](the-everglades-cipher/README.md) - Miami noir and historical cipher mystery

## GenAI Use Cases

This multi-genre corpus supports:
- **Transformer foundations:** Ground attention, generation, and source-to-target transformation in one Riverside manuscript world
- **Style diversity:** Train models to handle fantasy world-building, noir detective prose, historical period detail, cyberpunk jargon, gothic atmosphere, literary character depth, and hard sci-fi exposition
- **Vocabulary breadth:** Spans archaic Tang Dynasty terminology, modern crime investigation, future tech slang, fantasy magic systems, horror psychological tension, and scientific concepts
- **Narrative structures:** Quest arcs, murder mysteries, family sagas, conspiracy thrillers, generation-spanning epics, psychological horror escalation
- **Concept comparison:** Non-instructional training on the full corpus vs. genre-specific subsets to measure domain adaptation vs. catastrophic forgetting
- **Retrieval and evaluation:** Supply stable fictional entities, passages, and editorial questions without external copyrighted text

## Loading the Corpus

Code under `learning/genai/` should resolve this directory from the repository root as
`learning/genai/content`. The fine-tuning notebooks include helper functions that support:
- Loading all novels combined
- Loading specific novels only by directory name
- Limiting chapters per novel for fast CPU demos
- Automatic discovery of available novels

The [GPU practice notebook](../03-llm-finetuning/04-llm-finetuning-practice.ipynb) uses an explicit eight-novel allowlist and reserves validation and test chapters before creating token blocks.

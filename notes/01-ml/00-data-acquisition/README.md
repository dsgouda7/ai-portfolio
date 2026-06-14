# Data Acquisition and Collection — Chapter Overview

> **The scenario**: You're building a model to predict SpaceX first-stage landing success. Your training data lives in three places: a versioned REST API (v5, incompatible with v3), Wikipedia HTML tables covering launches from 2010–2014 that predate the API, and a CSV your colleague emailed you with undocumented column names. Before any model runs, you need a reliable, reproducible pipeline that fetches, validates, and stores this data — one that doesn't break when the API version increments, the HTML table gets redesigned, or a new engineer joins the team.

## What You'll Learn

- How to paginate REST APIs correctly — and why even "small" APIs require pagination logic
- The exponential backoff formula for rate limiting and how to respect `Retry-After` headers
- When web scraping is the only option, and how to build selectors that survive site redesigns
- How to detect silently changed HTML source pages before they corrupt your dataset
- The three categories of missing data (MCAR, MAR, MNAR) and why they require different treatments
- How type coercion with `errors='coerce'` makes wrangling failures visible rather than silent
- How to design a reproducible pipeline that produces the same output on every run
- The `data/raw/` vs `data/processed/` separation that makes pipelines recoverable

## Key Concepts

| Concept | Section | Core Challenge |
|---------|---------|----------------|
| Offset, cursor, and link-header pagination | § 1 | APIs truncate results silently |
| Exponential backoff with jitter | § 1 | Rate limiting without thundering herd |
| Schema drift validation with Pydantic | § 1 | SpaceX v3 → v5 field name changes |
| HTML parsing hierarchy: requests → BS4 → read_html | § 2 | Fragile vs robust selectors |
| Content hashing for change detection | § 2 | Silent page redesigns corrupt data |
| MCAR / MAR / MNAR taxonomy | § 3 | Different missing-data strategies |
| Type coercion and boolean normalization | § 3 | "TRUE"/1/False/NaN all meaning the same thing |
| Idempotent pipeline design | § 4 | Running twice produces the same result |
| Raw vs processed data separation | § 4 | `data/raw/` is append-only, never modified |

## The Full Chapter

[data-acquisition.md](data-acquisition.md)

---

*Companion notebooks in `learning/ibm-data-science/spacex-analysis/`:*

- `01-spacex-data-collection.ipynb` — SpaceX API pagination, response validation, JSONL output
- `02-webscraping.ipynb` — BeautifulSoup, `pd.read_html`, multi-header table post-processing
- `03-data_wrangling.ipynb` — type coercion, null handling, deduplication, validation asserts

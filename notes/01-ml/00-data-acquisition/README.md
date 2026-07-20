# Ch.00 — Data Acquisition and Collection

| Resource | Description |
|----------|-------------|
| [data-acquisition.md](data-acquisition.md) | Full chapter reference |
| [notebook-exercise.ipynb](notebook-exercise.ipynb) | Exercises with TODO stubs |
| [notebook-solution.ipynb](notebook-solution.ipynb) | Fully implemented solution |

*Companion notebooks:*
- [`learning/ibm-data-science/spacex-analysis/01-spacex-data-collection.ipynb`](../../../learning/ibm-data-science/spacex-analysis/01-spacex-data-collection.ipynb)
- [`learning/ibm-data-science/spacex-analysis/02-webscraping.ipynb`](../../../learning/ibm-data-science/spacex-analysis/02-webscraping.ipynb)
- [`learning/ibm-data-science/spacex-analysis/03-data_wrangling.ipynb`](../../../learning/ibm-data-science/spacex-analysis/03-data_wrangling.ipynb)

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

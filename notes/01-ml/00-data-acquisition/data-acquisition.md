# Data Acquisition and Collection

> **The story.** On 30 June 2023, a service called **Pushshift** went dark. For eight years, Pushshift had been the backbone of social media research — an archive of five billion Reddit posts and comments, maintained by developer **Jason Baumgartner** as a public good and accessible via free API. Researchers at Stanford, MIT, Carnegie Mellon, and hundreds of other institutions had built their data pipelines on it. The triggering event was prosaic: Reddit raised their API pricing from free to $0.24 per thousand calls, announced in April 2023, effective in July. Pushshift, which made millions of calls daily, couldn't absorb the cost and shut down within days. Overnight, hundreds of published papers in computational linguistics, sociology, political science, and machine learning became non-reproducible. Their methodology sections read, simply: *"data collected via the Pushshift API."* No local copy stored. Five years of citation-worthy datasets went unreachable in two weeks.
>
> Two categories of research teams existed in the aftermath. The first had treated Pushshift the way most engineers treat external services: as infrastructure they could assume would persist. Their pipelines opened with `requests.get(PUSHSHIFT_URL, ...)` with nothing stored locally. When the endpoint disappeared, step one of their pipeline disappeared with it. The second category had done something that felt redundant at the time: after every API call, they wrote the raw JSON response to an append-only file in `data/raw/`. Not parsed DataFrames — exact bytes from the API, gzip-compressed, timestamped. Their processing code read from `data/raw/` and ran without network access. When Pushshift went dark, they repointed step one at their local archive. When Reddit subsequently changed their own API's field names and pagination schema in August 2023, the schema validation layer in their fetch step caught the drift on the first call — not three weeks later, silently corrupting a training set.
>
> This pattern — **fetch into raw storage, validate schema at ingest, separate raw from processed** — predates Pushshift and will outlast whatever service replaces it. The SpaceX open data API migrated from v3 to v5 in 2021, renaming `launch_success` to `success`, restructuring the nested `rocket` object, and changing the pagination mechanism. Pre-2021 scrapers of Wikipedia's historical launch tables broke in 2022 when Wikimedia redesigned their infobox templates. Data sources mutate, get paywalled, go dark, or return subtly wrong data after silent redesigns. The only engineering response that works is to treat raw data as precious, immutable, and locally owned.
>
> **Where you are in the curriculum.** This is chapter zero of the ML track — not because data acquisition is trivial, but because nothing downstream is trustworthy without it. Every chapter in this library that trains a model, evaluates metrics, or engineers features assumes you already have a clean, versioned, validated dataset. Where does that come from? This chapter. Whether you're building a SpaceX landing predictor, a house valuation model, or a content moderation classifier, the first 40% of project time is typically data collection and cleaning. The models get the attention; the pipelines carry the weight.
>
> **Notation.** $n$ — retry attempt number (0-indexed); $c$ — base delay constant (seconds); $\varepsilon$ — uniform random jitter $\in [0,\, c)$; $t_{max}$ — maximum backoff cap (seconds); $t_{wait}$ — computed wait before next retry; MCAR — Missing Completely At Random; MAR — Missing At Random; MNAR — Missing Not At Random; JSONL — newline-delimited JSON (one record per line, append-safe, streamable by pandas).

---

## Common Misconceptions

Three beliefs that appear in production pipelines with disturbing regularity. Each has a logical-sounding justification and a real-world failure mode.

**Misconception 1: "I can always re-fetch it."**

**Why it's seductive:** The data is on the internet. The API is free. You can re-run the collection script whenever you need fresh data. Storing raw responses wastes disk space and complicates the pipeline. Why maintain a local copy of what a live source already holds?

**The truth:** "Always" is the word that kills reproducibility. APIs get versioned, paywalled, rate-limited into impracticality, or shut down entirely. For an ML pipeline specifically, the requirement isn't "can I get data?" — it's "can I get the *same* data that produced this model?" Re-fetching live data regenerates today's state, not the state your model was trained on. When your model's performance degrades in production, you need to know whether the training data shifted or the world did. You can only answer that question if you have your original raw data stored, versioned, and timestamped. The Pushshift incident is a case where "always" turned out to mean "for eight years, until it didn't."

*"If you can't re-run your data collection step and get the same bytes, your experiment isn't reproducible — it's just lucky."*

---

**Misconception 2: "The API returns everything."**

**Why it's seductive:** The endpoint is `/launches`. You call it, you get launches. The documentation doesn't mention pagination. The response has 10 items and you happen to know there are roughly 10 recent launches. It all looks right.

**The truth:** Most REST APIs paginate by default, often with a silent per-page cap. If your client doesn't handle pagination, you receive only the first page — and the result looks plausible. The SpaceX v5 API defaults to 10 results per call without explicit parameters and caps at 100 per call; calling `/launches` with no pagination logic returns the 10 most recent launches, not the full corpus of 200+. A model trained on "the last 10 launches" has dramatically different characteristics than one trained on the full 5-year history — different booster generations, different landing infrastructure, different success rates. The failure mode isn't an error; it's a training set that looks complete and isn't.

*"No pagination logic means your 'complete dataset' is probably page one."*

---

**Misconception 3: "Cleaning data once is enough."**

**Why it's seductive:** You spend two days cleaning the dataset: filling nulls, coercing types, deduplicating records. You save the cleaned Parquet file. Future runs load the cleaned file. The cleaning is done. Why re-clean what's already clean?

**The truth:** Cleaning is not a one-time event when the source mutates. The SpaceX API added launch records retroactively for historical corrections. The Wikipedia tables you scraped had column headers changed in a 2022 editor update. A data partner's export started encoding a boolean field as `"yes"/"no"` instead of `True/False` after a schema migration nobody communicated. Your one-time cleaned file now produces a silently broken merged dataset because you're joining a 2021 clean against a 2024 ingest with incompatible boolean representations. The correct model: cleaning code runs every time against raw inputs. Cleaning is a deterministic function of raw data, not a manual operation you perform once and forget. The processed artifact is re-derivable; the raw artifact is not.

*"A cleaned CSV is a snapshot, not a pipeline. Snapshots go stale."*

---

## 0 · The Challenge

You have a prediction task: given a SpaceX launch, will the first-stage booster successfully land? Binary classification. You need a training set.

Here is what actually exists:

- **A REST API** at `api.spacexdata.com/v5/launches` returning JSON records for launches from 2006 onward. The API is on version 5. The previous version (v3) used different field names (`launch_success` vs `success`), a different nested structure for rocket metadata, and a different pagination scheme. You don't know which version any predecessor scripts used.
- **Wikipedia HTML tables** covering launches from 2010 to 2014 that predate the SpaceX API era. These are the only structured source of early Falcon 9 test outcomes. The tables have been edited by volunteers over a decade and contain multi-header rows, merged cells, and footnote rows embedded in the data rows.
- **A CSV** with 47 launches from a 2019 internal tracking sheet. Column names don't match the API. It's unclear whether these records are a subset of API records or supplementary entries not yet reflected in the API.
- **A pile of unknown unknowns:** rate limits you'll hit in testing, SSL certificates that rotate, field names that changed since the schema was last documented, a Wikipedia infobox template that was redesigned in 2022.

Before any model runs, the pipeline must:

1. Fetch all three sources reliably and completely, handling pagination and rate limits
2. Validate each source against a known schema before persisting anything
3. Merge sources using a consistent primary key with explicit conflict resolution
4. Document provenance — which records came from which source, when they were fetched
5. Run identically on your machine, your colleague's machine, and a CI server

This is the unglamorous work that determines whether everything downstream is trustworthy.

---

## 1 · REST API Data Collection

### The HTTP fundamentals you actually need

REST APIs communicate over HTTP. For data engineers, three things matter: the request method, the response status code, and the payload format.

**Request methods.** For collection you almost always want `GET` — retrieve a resource without modifying state. Occasionally a search API requires `POST` with a JSON body describing filter criteria (Elasticsearch, some analytics APIs). Use `GET` unless documentation explicitly requires `POST` for read operations.

**Status codes.** These are contracts the server makes about what happened:

| Code | Meaning | Your response |
|------|---------|---------------|
| 200 | OK — response body contains data | Parse and persist |
| 400 | Bad request — your parameters are malformed | Fix the request; do not retry |
| 401 | Unauthorized — authentication required or expired | Refresh credentials; do not retry blindly |
| 403 | Forbidden — authenticated but not permitted | Check API plan or IP allowlist |
| 404 | Not found — this resource doesn't exist | Skip or log; may be legitimately absent |
| 429 | Too Many Requests — rate limited | Back off; respect `Retry-After` header |
| 500 | Internal server error | Retry with backoff; log the response body |
| 503 | Service unavailable | Retry with backoff; circuit-break if persistent |

The 429 and 503 codes are retryable with backoff. The 4xx codes (except 429) are your fault; retrying won't help.

**Authentication patterns.** The SpaceX public API requires no authentication. Most production APIs use one of: an **API key in a header** (`Authorization: Bearer <token>` or a custom `X-Api-Key` header), **OAuth 2.0** (a token exchange flow for user-delegated access — unnecessary for server-to-server pipelines), or **IP allowlisting** (no token, but your outbound IP must be registered). Never hardcode credentials in source code. Use environment variables (`os.environ["API_KEY"]`) or a secrets manager.

### Pagination: the silent truncation problem

Every API enforces a maximum items-per-page limit. When your client doesn't handle pagination, you receive only the first page. The response looks complete. It isn't.

Three pagination patterns appear in practice:

**Offset pagination** — the most common. Pass `?offset=0&limit=100` for page one, `?offset=100&limit=100` for page two. Simple to implement, but inconsistent if records are added or deleted mid-pagination (you may skip or double-count). Correct for append-only sources like historical launch records.

**Cursor pagination** — the robust approach for mutable sources. The API returns a `next_cursor` field in each response; your next request passes `?after=<next_cursor>`. Cursors encode server-side position, immune to concurrent insertions and deletions. Used by Twitter, Stripe, Shopify, and SpaceX's `/launches/query` endpoint.

**Link-header pagination** — the REST-idiomatic approach. The API returns `Link: <next_url>; rel="next"` in response headers. Follow the link without constructing URLs manually. GitHub uses this.

The rule: **always check for `has_more`, `next_cursor`, or a `Link` header, regardless of whether you expect pagination.** The SpaceX corpus has ~200 launches today; it will have more tomorrow. Write the loop now.

### Rate limiting and exponential backoff

The SpaceX public API imposes a limit of 100 requests per minute. Exceed it and you receive a `429 Too Many Requests` with a `Retry-After: <seconds>` header. When no such header is present, the correct backoff formula is:

$$t_{wait} = \min\!\left(c \cdot 2^n + \varepsilon,\; t_{max}\right)$$

where $n$ is the attempt number (0-indexed), $c$ is a base delay constant (typically 1 second), $\varepsilon \sim \text{Uniform}(0,\, c)$ is random jitter, and $t_{max}$ caps the wait (typically 60 seconds). The jitter term prevents the **thundering herd problem**: if 50 workers all back off for the same duration, they all hammer the server simultaneously when the wait expires. Jitter distributes their retries across a window.

`requests` handles this automatically when configured with `urllib3.util.retry.Retry`:

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def build_session(
 retries: int = 5,
 backoff_factor: float = 1.0,
 status_forcelist: tuple = (429, 500, 502, 503, 504),
) -> requests.Session:
 """
 Build a requests session with automatic retry and exponential backoff.
 backoff_factor=1.0 → waits 0s, 2s, 4s, 8s, 16s between successive retries.
 """
 session = requests.Session()
 retry = Retry(
 total=retries,
 backoff_factor=backoff_factor,
 status_forcelist=status_forcelist,
 respect_retry_after_header=True, # obeys 429 Retry-After header
 )
 adapter = HTTPAdapter(max_retries=retry)
 session.mount("https://", adapter)
 session.mount("http://", adapter)
 return session
```

`respect_retry_after_header=True` is the critical flag: it reads the `Retry-After` value from the 429 response instead of computing the backoff independently, which is both more polite and more reliable.

### Schema validation: catching drift at ingest time

When SpaceX migrated from v3 to v5, `launch_success` was renamed to `success`. Any v3-era pipeline consuming v5 data silently reads `None` for the landing outcome — the target variable. The model trains on all-null targets. No exception is raised. The failure surfaces weeks later as a suspiciously useless model.

The fix is schema validation at ingest time using `pydantic`:

```python
from pydantic import BaseModel, field_validator
from typing import Optional
from datetime import datetime


class LaunchRecord(BaseModel):
 id: str
 name: str
 date_utc: datetime
 success: Optional[bool] # null for upcoming launches — legitimate absence
 upcoming: bool
 cores: list[dict]

 @field_validator("date_utc", mode="before")
 @classmethod
 def parse_date(cls, v):
 if isinstance(v, str):
 return datetime.fromisoformat(v.rstrip("Z"))
 return v
```

If `success` is absent from a response record — meaning the API returned an unexpected schema — `pydantic` raises `ValidationError` immediately at the fetch stage, not silently at training time when the null propagates into your target column. Schema drift becomes a loud error, not a quiet data quality bug.

### Incremental vs full refresh

**Full refresh**: fetch everything from scratch on each run. Correct and simple; expensive for large sources.

**Incremental**: fetch only records created or modified since the last successful run. Requires a reliable `updated_at` field on the API, a stored high-water mark, and logic to handle retroactive corrections to historical records.

For SpaceX launches (~200 records, occasionally corrected retroactively): full refresh is correct. The 2-second cost of re-fetching 200 records is not worth the complexity of incremental logic.

For a source with 50 million records and a 1,000/minute rate limit: incremental is mandatory. Store `last_fetched_at = datetime.utcnow()` after each successful run. On the next run, pass `?min_date=<last_fetched_at>` to the API.

### The complete fetch pattern

```python
import jsonlines
from pathlib import Path

LAUNCHES_URL = "https://api.spacexdata.com/v5/launches"
RAW_OUTPUT = Path("data/raw/launches.jsonl")


def fetch_all_launches(output: Path = RAW_OUTPUT) -> int:
 """
 Full-refresh fetch of SpaceX launches to JSONL.
 Schema-validated at ingest. Returns number of records written.
 """
 session = build_session()
 output.parent.mkdir(parents=True, exist_ok=True)

 written, offset, limit = 0, 0, 100
 with jsonlines.open(output, mode="w") as writer:
 while True:
 resp = session.get(
 LAUNCHES_URL,
 params={"offset": offset, "limit": limit},
 timeout=15,
 )
 resp.raise_for_status()
 records = resp.json()
 if not records:
 break
 for raw in records:
 launch = LaunchRecord(**raw) # raises on schema drift
 writer.write(launch.model_dump(mode="json"))
 written += len(records)
 offset += limit

 return written
```

JSONL (newline-delimited JSON) is preferred over a single large JSON array for three reasons: it is append-safe (add records without rewriting the file), streamable (`pd.read_json("file.jsonl", lines=True)` processes it without loading everything into memory), and inspectable (`head data/raw/launches.jsonl` shows real records immediately).

---

## 2 · Web Scraping

### When scraping is the only option

Scraping extracts data from HTML pages designed for human browsers. You reach for it in three situations:

1. **No API exists.** SpaceX launches from 2010–2014 were documented on Wikipedia before SpaceX published any API. Wikipedia has no launch data API; the structured data only exists as HTML tables maintained by volunteers.
2. **The API predates the data.** The SpaceX API covers pre-2017 launches but with sparser metadata than the contemporaneous Wikipedia records, which accumulated detail through years of community edits.
3. **The API is paywalled and a public page carries the same data.** This is a greyer zone — see the legal constraints section.

### The parsing hierarchy

Web scraping is a four-stage pipeline with a clear tool at each stage:

```
URL
 → requests.get() raw HTML bytes
 → BeautifulSoup(html) navigable parse tree
 → .find() / .select() target element
 → pd.read_html(str(element))[0] DataFrame
 (for <table> elements)
```

**`requests.get(url)`** fetches raw HTML. Works for server-rendered pages. Does not execute JavaScript. Wikipedia uses server-side rendering; `requests` is sufficient. For JavaScript-rendered pages (React SPAs, dynamic tables), you need `playwright` or `selenium` to drive a real browser.

**`BeautifulSoup`** parses HTML into a navigable tree. The `"html.parser"` backend is built into Python's standard library; `"lxml"` is faster if available. Provides `.find()`, `.find_all()`, `.select()` (CSS selectors), and `.get_text()` for navigation and extraction.

**CSS selectors vs positional indexing.** Two ways to locate a target table:

```python
# Robust: locate by attribute — stable across minor site edits
table = soup.find("table", {"class": "wikitable plainrowheaders"})

# Fragile: locate by position — breaks on any table addition or removal
table = soup.find_all("table")[3]
```

The positional approach breaks on the first page restructure. The attribute-based approach requires the class name to remain stable — more durable but still not guaranteed. For Wikipedia specifically: `wikitable` is an established MediaWiki convention used across thousands of articles. Stable enough for a production scraper, as long as you add change detection.

**`pd.read_html()`** parses `<table>` HTML into a list of DataFrames. It handles standard HTML tables correctly and is faster to write than manual cell-by-cell extraction. The shortcut for the SpaceX historical table:

```python
# Shortcut: fetch and parse in one call (hits network on every call)
df = pd.read_html(url)[1]

# Production approach: fetch once, parse from local cache
html_bytes = Path("data/raw/html/wiki_launches.html").read_bytes()
df = pd.read_html(html_bytes.decode("utf-8"))[1]
```

The local cache approach means your development loop hits the network exactly once. Every subsequent iteration reads from disk.

### Robust selectors and fragility analysis

Every CSS selector has a fragility budget — the number of structural changes required to break it:

| Selector | Fragility | Why |
|----------|-----------|-----|
| `soup.find("table", {"id": "launch-history"})` | Low | IDs are changed only deliberately |
| `soup.find("table", {"class": "wikitable"})` | Medium | Class names change rarely for established templates |
| `soup.select("div.content > table:nth-of-type(2)")` | Medium | Positional within scope; breaks on sibling table additions |
| `soup.find_all("tr")[4].find_all("td")[2]` | High | Breaks on any row or column change |

For the SpaceX Wikipedia tables, `wikitable` class is the right handle. Add hash-based change detection below to catch redesigns before they produce wrong data silently.

### Change detection: hash, don't assume

If the source HTML changes, your scraper should fail loudly, not silently ingest a malformed structure. A content hash achieves this:

```python
import hashlib
import json
from pathlib import Path

HASH_STORE = Path("data/raw/.content_hashes.json")


def fetch_with_change_detection(url: str, name: str) -> bytes:
 """
 Fetch URL; raise ValueError if the content hash differs from the last
 known state. On first run, stores the hash and proceeds normally.
 """
 import requests
 resp = requests.get(url, timeout=15)
 resp.raise_for_status()

 current_hash = hashlib.sha256(resp.content).hexdigest()
 hashes = json.loads(HASH_STORE.read_text()) if HASH_STORE.exists() else {}
 last_hash = hashes.get(name)

 if last_hash and last_hash != current_hash:
 raise ValueError(
 f"Content hash mismatch for '{name}': "
 f"expected {last_hash[:8]}..., got {current_hash[:8]}...\n"
 "Source page structure may have changed. Inspect before proceeding."
 )

 hashes[name] = current_hash
 HASH_STORE.parent.mkdir(parents=True, exist_ok=True)
 HASH_STORE.write_text(json.dumps(hashes, indent=2))
 return resp.content
```

On first run: stores the hash and returns content. On subsequent runs: if the content changed, raises before any parsing. You inspect the change, update your selector if needed, then deliberately reset the stored hash. The structural change is now a known failure, not silent corruption that shows up two weeks later in model evaluation.

### Ethical and legal constraints

Scraping is not unconditionally legal or ethical. Three constraints apply:

**`robots.txt`** — a voluntary specification at `<domain>/robots.txt` listing paths crawlers should not access. Check it before scraping: use `urllib.robotparser.RobotFileParser`. Wikipedia explicitly permits scraping article content. Most commercial sites disallow automated access to data pages. Respecting `robots.txt` is an industry convention, not always a legal requirement — but ignoring it for commercial data extraction has produced litigation (HiQ v. LinkedIn, 2022).

**Terms of service** — the legal contract governing data use. Some ToS agreements explicitly prohibit automated access or data extraction. Violation can constitute breach of contract or CFAA violation depending on jurisdiction. Read the ToS before scraping commercially.

**Rate limiting yourself** — even if a site doesn't actively rate-limit your requests, you should throttle them. Add `time.sleep(1)` between page requests. Cache pages locally during development so you're not re-hitting the server on every code change. You're not exempt from rate etiquette because you wrote the script.

### Multi-header rows and the post-processing reality

`pd.read_html()` returns a DataFrame, but the SpaceX Wikipedia table requires post-processing before it's usable:

```python
df = pd.read_html(str(table))[0]

# Multi-level column headers → flatten to single string
if isinstance(df.columns, pd.MultiIndex):
 df.columns = [" ".join(filter(None, map(str, col))).strip() for col in df.columns]

# Footnote rows: rows where all values are identical (usually a merged cell note)
df = df[df.apply(lambda r: r.nunique() > 1, axis=1)]

# Drop rows that are entirely NaN (from merged cells rendered as empty)
df = df.dropna(how="all")

# Strip citation markers: "[1]", "[note 1]", "[a]"
import re
df = df.replace(r"\[.*?\]", "", regex=True)
df = df.replace(r"^\s*$", pd.NA, regex=True)
```

The output is still raw — types are strings, boolean-looking fields contain "Success"/"Failure"/"N/A" text, dates are inconsistently formatted. That cleanup belongs in the wrangling stage (§3). Scraping is responsible for: fetch, parse, persist the raw parsed structure. Interpretation is downstream.

---

## 3 · Data Wrangling Fundamentals

### The four categories of raw data problems

Raw data from heterogeneous sources — REST APIs, HTML scrapers, CSV exports from someone's spreadsheet — fails in four predictable ways. Identifying the category determines the fix.

**Missing values** — fields that are `None`, `NaN`, `""`, or absent entirely. The cause matters: data that never existed is different from data that was collected but lost, which is different from data whose absence is itself informative.

**Wrong types** — a numeric field containing `"N/A"`, a boolean field containing the string `"TRUE"`, a date field containing both ISO 8601 strings and Unix timestamps. The pipeline reads them as `object` dtype and continues silently until a model tries to operate on them.

**Inconsistent representation** — `"Successful"`, `"Success"`, `"success"`, `"1"`, `True` all meaning the same outcome in different source extracts. Near-matching records: the same launch appears in both the API data and the Wikipedia scrape with slightly different payload masses because one source was retroactively corrected after the other was scraped.

**Duplicate records** — exact or near-duplicate rows from joining two sources that share records. An outer join on `flight_number` between the API and the CSV may produce 3 rows for flight 42 if the CSV had corrections not yet reflected in the API.

### Missing value taxonomy: MCAR, MAR, MNAR

Not all missing values call for the same response. The statistical framework distinguishes three generating mechanisms, each with a different treatment:

**MCAR — Missing Completely At Random.** The probability of a value being absent is independent of any variable in the dataset. A 2% API response dropout due to network timeouts during your initial fetch; sensor noise causing random loss of payload telemetry.

*Treatment:* If the missing fraction is small (< 5% of rows), dropping those rows is defensible. For larger fractions, impute with the column median (numeric) or mode (categorical). Imputing MCAR data introduces no systematic bias because the missing cases are a random sample of all cases.

**MAR — Missing At Random.** The probability of missing depends on *other observed columns* but not on the value of the missing field itself. SpaceX payload mass is more likely to be missing for early Falcon 1 launches (2006–2009) because data recording standards were less rigorous in that period. The missingness correlates with launch year, not with the payload mass value.

*Treatment:* Conditional imputation — compute the imputed value from correlated columns. Group by `rocket_type` and `year`, then impute with group median. Simpler fallback: add a binary indicator column `payload_mass_missing` and impute globally with median. The indicator preserves the signal that early launches had incomplete metadata.

**MNAR — Missing Not At Random.** The probability of missing depends on the *value that would have been observed*. Landing outcome (`success`) is absent for launches where no landing was attempted — not because data was lost, but because "no recovery was attempted" is itself a meaningful state.

*Treatment:* Create separate columns rather than imputing. If you impute `success = False` for no-landing-attempted launches, you introduce false negatives that teach the model the wrong lesson. If you drop these rows, you remove evidence of early test-flight configurations.

```python
import pandas as pd


def handle_landing_outcome(df: pd.DataFrame) -> pd.DataFrame:
 """
 Separate landing outcome into attempt flag and success flag.
 Preserves MNAR semantics: missing outcome ≠ failed outcome.
 Three states: attempted+succeeded, attempted+failed, not attempted.
 """
 df = df.copy()
 df["landing_attempted"] = df["landing_outcome"].notna()
 df["landing_success"] = (
 df["landing_outcome"]
 .map(lambda v: True if str(v).lower() in ("success", "true", "1")
 else False if pd.notna(v) else pd.NA)
 .astype(pd.BooleanDtype())
 )
 return df
```

### Type coercion: normalizing the chaos

Real-world data collections accumulate type inconsistencies at every join seam. The SpaceX materials contain all of these:

| Source value | Intended type | Correct coercion |
|-------------|---------------|------------------|
| `"TRUE"`, `"FALSE"`, `"1"`, `"0"`, `True`, `False`, `None`, `"N/A"` | bool | Explicit map dict; unrecognized values → `pd.NA` |
| `"2021-01-01"`, `"Jan 1, 2021"`, `1609459200` (Unix) | datetime | `pd.to_datetime(unit="s")` for Unix; `pd.to_datetime()` for strings |
| `"9525.0 kg"`, `"9525"`, `9525.0`, `"N/A"` | float (kg) | Strip units with regex, then `pd.to_numeric(errors="coerce")` |
| `"LEO"`, `"GTO"`, `"ISS"`, `"Starlink"`, `""` | categorical | Map to standard enum values; empty string → `pd.NA` |

A normalization function that handles all boolean variants cleanly:

```python
def coerce_boolean(series: pd.Series) -> pd.Series:
 """Normalize any boolean-ish representation to pd.BooleanDtype."""
 _TRUE = {"true", "yes", "1", "success", "successful"}
 _FALSE = {"false", "no", "0", "failure", "failed", "n/a", "none", "nan", ""}
 return (
 series
 .astype(str)
 .str.lower()
 .str.strip()
 .map(lambda v: True if v in _TRUE else False if v in _FALSE else pd.NA)
 .astype(pd.BooleanDtype())
 )
```

The `errors="coerce"` argument in `pd.to_numeric()` and `pd.to_datetime()` is the single most important parameter in wrangling work: it replaces unparseable values with `NaN` rather than raising an exception. This makes coercion failures observable — you can count the resulting nulls after coercion and decide whether the count is acceptable — rather than having to choose between silent corruption and hard failures that halt your pipeline.

*"errors='coerce' is how type coercion should fail: visibly, auditably, without halting the pipeline."*

### Deduplication: exact vs near-duplicate

**Exact deduplication** is one line:

```python
df = df.drop_duplicates(subset=["flight_number"])
```

**Near-deduplication** — the same logical record appearing in two sources with slightly different field values — requires a merge strategy. For the SpaceX case, the API record and the Wikipedia-scraped record for flight 42 may agree on `flight_number` but differ on `payload_mass_kg` (e.g., 9,525 vs 9,600) because one source was retroactively corrected after the other was scraped.

Define a **priority order** for each field based on source reliability, then merge with explicit conflict resolution:

```python
def merge_with_priority(
 api_df: pd.DataFrame, wiki_df: pd.DataFrame
) -> pd.DataFrame:
 """
 Merge API and Wikipedia records by flight_number.
 API wins for outcome fields (more frequently updated, authoritative).
 Wikipedia wins for descriptive text fields (more historically complete).
 """
 merged = api_df.merge(
 wiki_df, on="flight_number", suffixes=("_api", "_wiki"), how="outer"
 )
 merged["landing_success"] = merged["landing_success_api"].combine_first(
 merged["landing_success_wiki"]
 )
 merged["launch_site"] = merged["launch_site_wiki"].combine_first(
 merged["launch_site_api"]
 )
 return merged
```

`combine_first()` takes the left value when it's non-null, falling back to the right. Document the priority choice explicitly — in six months, you won't remember which source was authoritative for which field.

### Validation before downstream use

The final step of wrangling is not "save the file." It's "assert the output meets the contract the downstream ML code expects." If assertions fail, the pipeline halts with a clear error. If they pass, downstream code receives a validated artifact.

```python
def validate_launches(df: pd.DataFrame) -> None:
 """
 Assert data quality invariants before writing to data/processed/.
 Raises AssertionError with a descriptive message on any violation.
 """
 assert len(df) >= 100, f"Suspiciously few records: {len(df)} (expected ≥ 100)"

 required_nonnull = ["flight_number", "launch_date", "rocket_type"]
 for col in required_nonnull:
 n_null = df[col].isna().sum()
 assert n_null == 0, f"Required column '{col}' has {n_null} null values"

 assert df["payload_mass_kg"].dropna().gt(0).all(), \
 "payload_mass_kg contains non-positive values"

 if "latitude" in df.columns:
 assert df["latitude"].dropna().between(-90, 90).all(), \
 "latitude out of range [-90, 90]"

 dupes = df["flight_number"].duplicated().sum()
 assert dupes == 0, f"{dupes} duplicate flight_number values after deduplication"
```

Write validation functions before you write wrangling functions. The assertions describe the contract you're trying to satisfy; the wrangling code is your attempt to satisfy it. Running validation as the last step of every pipeline run makes it impossible to accidentally produce a corrupt output that persists silently.

*"Validation failures are free. Silent validation failures cost you two weeks debugging a model that was never given real data."*

---

## 4 · Building a Reproducible Pipeline

### The three enemies of reproducibility

**Mutable sources** — API endpoints that return different data on different days because records are retroactively corrected, added, or removed. You can't control the source. You can control whether you stored a timestamped local copy before the mutation occurred.

**Local state** — files that the pipeline reads but that aren't in version control and don't exist on any other machine. The new team member who clones your repo gets different results because `manual_corrections.csv` lives only on your laptop. Every file the pipeline reads must either be fetched programmatically by the pipeline itself or committed to the repository.

**Implicit ordering** — step 3 assumes step 2 has already run. If step 3 is run against stale intermediate data (perhaps from a previous experiment), it proceeds silently with the wrong inputs. The fix: each step reads inputs from versioned paths and writes outputs to new versioned paths. No step reads from the same path it writes to.

### Idempotency: running twice should produce the same result

A pipeline is idempotent if executing it multiple times produces identical output. This is harder than it sounds. Common failure modes:

- Appending to an output file on each run rather than overwriting (row counts grow with each execution)
- Using `datetime.utcnow()` as a deduplication key (changes on every run by definition)
- Writing intermediate files with non-deterministic filenames

The fix: write outputs to deterministic paths with explicit overwrite semantics. If raw data is unchanged, skip the network fetch entirely:

```python
from pathlib import Path
import hashlib


def fetch_if_changed(url: str, dest: Path) -> bool:
 """
 Fetch URL to dest only if content hash differs from last fetch.
 Returns True if a new fetch was performed, False if skipped.
 """
 import requests
 resp = requests.get(url, timeout=15)
 resp.raise_for_status()

 new_hash = hashlib.sha256(resp.content).hexdigest()
 hash_file = dest.with_suffix(".sha256")

 if dest.exists() and hash_file.exists():
 if hash_file.read_text().strip() == new_hash:
 return False # content unchanged; skip write

 dest.parent.mkdir(parents=True, exist_ok=True)
 dest.write_bytes(resp.content)
 hash_file.write_text(new_hash)
 return True
```

### Raw vs processed: the only non-negotiable layout rule

```
data/
├── raw/ # Written once by fetch scripts. Never overwritten.
│ ├── launches.jsonl
│ ├── html/
│ │ └── wiki_launches_2010_2014.html
│ └── csv/
│ └── manual_corrections.csv
└── processed/ # Written by wrangling scripts. Regenerable from raw.
 ├── launches_clean.parquet
 └── launches_merged.parquet
```

The rule: `data/raw/` is **append-only and never overwritten in place**. If the API returns new data, a new timestamped file appears in `data/raw/`. Processed files are derivatives that can always be regenerated by running the wrangling pipeline over the raw files. If `data/processed/` is deleted (corrupt, wrong schema version, needs regeneration with updated logic), it is recoverable in minutes. If `data/raw/` is deleted, you've lost the ground truth.

Commit `data/raw/` to version control when files are small. For large binary files, use DVC (Data Version Control) to track them externally while keeping the directory structure and hashes in git.

### Delta Lake and versioned storage for production scale

For the SpaceX dataset (~200 launches, < 1 MB total), the file-based approach above is correct — the engineering overhead of a distributed storage system isn't justified. For a production ML pipeline ingesting millions of events per day, you need a storage layer with built-in versioning, schema enforcement, and efficient partial reads.

Delta Lake provides three properties that matter for data ingestion at scale:

**Write-once append, no in-place updates** — new data is always written as new Parquet files. Existing files are never modified. This makes Delta tables naturally append-only and S3-compatible without requiring file locking.

**Time travel** — every write creates a new table version. `spark.read.format("delta").option("versionAsOf", 42).load(path)` reads the table exactly as it existed at version 42. You can reproduce any past training run by specifying the data version it trained on.

**Schema enforcement** — attempting to write a column absent from the table schema raises an error at write time, not at query time three days later when a downstream job silently starts returning nulls. Combined with `mergeSchema=True` for intentional additions, schema evolution becomes auditable and explicit rather than emergent.

The `projects/data-engineering/databricks_rag/` pipeline in this repository demonstrates the Delta Lake pattern for production-scale ingestion. The file-based `data/raw/` pattern is the portable analog of the same principles: immutable raw storage, derived processed artifacts, and versioned traceability.

### Logging and observability

When a data pipeline fails at 3 AM, you have one debugging tool: the logs. Log enough to reconstruct what happened without re-running the pipeline.

```python
import logging

logging.basicConfig(
 level=logging.INFO,
 format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)


def run_pipeline() -> None:
 log.info("Pipeline start")

 rows_fetched = fetch_all_launches()
 log.info(f"Fetched {rows_fetched} raw launch records from API")

 df_raw = pd.read_json("data/raw/launches.jsonl", lines=True)
 log.info(f"Raw shape: {df_raw.shape} | dtypes: {dict(df_raw.dtypes)}")
 log.info(f"Null counts by column: {df_raw.isna().sum().to_dict()}")

 df_clean = wrangle(df_raw)
 log.info(
 f"After wrangling: {len(df_clean)} rows, "
 f"{df_clean.isna().sum().sum()} total nulls remaining"
 )

 try:
 validate_launches(df_clean)
 log.info("Validation passed")
 except AssertionError as e:
 log.error(f"Validation FAILED: {e}")
 raise

 df_clean.to_parquet("data/processed/launches_clean.parquet", index=False)
 log.info("Pipeline complete — output written to data/processed/launches_clean.parquet")
```

Log row counts at each stage. If the pipeline produces 47 rows when you expected 200, stage-level counts tell you whether 153 rows were lost in fetch, schema validation rejection, type coercion, deduplication, or the final validation assertion. Without stage counts, you know the output is wrong but not where the attrition happened.

---

## 5 · Quick Reference

| Source type | Recommended tool | Key gotcha | Pattern |
|-------------|-----------------|------------|---------|
| JSON REST API (paginated) | `requests` + `Retry` adapter | Pagination silently truncates to page 1 | Loop while `resp.json()` is non-empty; increment offset |
| JSON REST API (rate-limited) | `Retry(respect_retry_after_header=True)` | 429 `Retry-After` header must be obeyed | `status_forcelist=(429,)` in Retry |
| JSON REST API (schema drift) | `pydantic.BaseModel` | Field renames return `None`, not errors | Validate every record at ingest; catch `ValidationError` |
| HTML table, simple page | `pd.read_html(url)` | Multi-header rows, footnote rows mixed in | Post-process: flatten MultiIndex columns, drop uniform rows |
| HTML page, complex structure | `requests` + `BeautifulSoup` | Positional selectors break on redesign | Locate by class/id; add SHA-256 change detection |
| Any scraped source | Content hash check | Structural changes produce silent wrong data | Store `sha256(resp.content)`; raise on mismatch |
| CSV with unknown provenance | `pd.read_csv` + dtype audit | Silent `object` dtype masks numeric fields | `df.dtypes`, then `pd.to_numeric(errors="coerce")` |
| Boolean field (inconsistent) | Explicit map dict | `"N/A"` must map to `pd.NA`, not `False` | `coerce_boolean()` from § 3 |
| Missing values: MCAR | Median/mode imputation | Valid only if < ~5% rows affected | `df[col].fillna(df[col].median())` |
| Missing values: MAR | Group-conditional imputation | Group by the correlated column | `df.groupby("year")[col].transform("median")` |
| Missing values: MNAR | Binary indicator column | Missingness IS a feature; don't impute | `df["col_missing"] = df["col"].isna().astype(int)` |
| Deduplication (exact) | `.drop_duplicates(subset=[pk])` | Default drops on all columns, not just primary key | Always pass `subset=` |
| Deduplication (near) | Outer merge + `combine_first()` | Define priority order per field, in code | Document which source wins for which field |
| Pipeline reproducibility | `data/raw/` append-only | Overwriting raw files destroys provenance | `data/raw/` committed or DVC-tracked; `data/processed/` gitignored |
| Large-scale ingestion | Delta Lake | Schema enforcement must be at write time, not query time | `mergeSchema=False` for strict enforcement |

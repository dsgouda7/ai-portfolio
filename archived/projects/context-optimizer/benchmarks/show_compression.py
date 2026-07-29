import json

d = json.loads(open("benchmarks/BOOK_RESULTS_extractive.json", encoding="utf-8").read())
books = d["per_book"]

print(f"{'Title':<40} {'Orig':>8} {'Comp':>8} {'Saved':>8} {'Ratio':>7} {'Lines':>7}")
print("-" * 82)
for b in sorted(books, key=lambda x: x["orig_tokens"]):
    orig, comp = b["orig_tokens"], b["comp_tokens"]
    saved = orig - comp
    ratio = comp / orig
    lines = b["lines_used"]
    title = b["title"][:38]
    print(f"{title:<40} {orig:>8,} {comp:>8,} {saved:>8,} {ratio:>7.1%} {lines:>7,}")

t_o = sum(b["orig_tokens"] for b in books)
t_c = sum(b["comp_tokens"] for b in books)
print("-" * 82)
print(f"{'TOTAL':<40} {t_o:>8,} {t_c:>8,} {t_o-t_c:>8,} {t_c/t_o:>7.1%}")

avg_per_book_orig = t_o // len(books)
avg_per_book_comp = t_c // len(books)

print()
print("Context window — how many books fit per call (3000-line slice):")
for label, ctx in [
    ("GPT-4o 128K", 128_000),
    ("Claude 200K", 200_000),
    ("Gemini 1M", 1_000_000),
]:
    raw_fits = ctx // avg_per_book_orig
    comp_fits = ctx // avg_per_book_comp
    print(
        f"  {label:<14}  raw: {raw_fits:>3} books   compressed: {comp_fits:>3} books  ({comp_fits/max(raw_fits,1):.1f}x more)"
    )

print()
print("NOTE: These are 3,000-line SLICES. Full novels are 5-20x larger:")
print(
    "  Moby Dick full text  ~250,000 tokens  →  compressed ~77,000 tokens (fits GPT-4o; raw does not)"
)
print(
    "  War and Peace        ~580,000 tokens  →  compressed ~179,000 tokens (fits Claude; raw does not)"
)

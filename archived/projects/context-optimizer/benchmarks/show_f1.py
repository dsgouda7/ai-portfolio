"""Print F1 + token compression stats, optionally side-by-side comparison.

Usage:
    python show_f1.py                        # extractive only
    python show_f1.py raw_only               # raw_only only
    python show_f1.py extractive raw_only    # side-by-side
"""

import json
import re
import sys
from pathlib import Path

_BENCH = Path(__file__).parent
strategies = sys.argv[1:] if len(sys.argv) > 1 else ["extractive"]
# Always default to extractive if no args given but also show comparison helper
if not strategies:
    strategies = ["extractive"]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]{3,}", text.lower())


def load_stats(strategy: str) -> dict:
    path = _BENCH / f"BOOK_RESULTS_{strategy}.json"
    if not path.exists():
        raise FileNotFoundError(f"No results for strategy '{strategy}': {path}")
    d = json.loads(path.read_text(encoding="utf-8"))
    stats = {}
    total_orig = total_comp = 0
    for b in d["per_book"]:
        orig, comp = b["orig_tokens"], b["comp_tokens"]
        total_orig += orig
        total_comp += comp
        precisions, recalls, f1s = [], [], []
        for q in b["results"]:
            ans_tokens = set(tokenize(q["answer"]))
            n_kw = q["expected_kw_count"]
            matched = q["kw_recall"] * n_kw
            recall = q["kw_recall"]
            precision = matched / max(len(ans_tokens), 1)
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        stats[b["title"]] = {
            "ratio": comp / orig,
            "P": sum(precisions) / len(precisions),
            "R": sum(recalls) / len(recalls),
            "F1": sum(f1s) / len(f1s),
            "ans_tok": sum(len(tokenize(q["answer"])) for q in b["results"])
            / max(len(b["results"]), 1),
        }
    stats["__total__"] = {
        "ratio": total_comp / total_orig,
        "orig": total_orig,
        "comp": total_comp,
    }
    return stats


if len(strategies) == 1:
    strat = strategies[0]
    data = load_stats(strat)
    books = {k: v for k, v in data.items() if k != "__total__"}
    print(f"\nStrategy: {strat}")
    print(f"{'Title':<38} {'Ratio':>6} {'P':>6} {'R':>6} {'F1':>6} {'ans_tok':>8}")
    print("-" * 76)
    all_p = all_r = all_f1 = 0
    for title, s in sorted(books.items(), key=lambda x: -x[1]["R"]):
        all_p += s["P"]
        all_r += s["R"]
        all_f1 += s["F1"]
        print(
            f"{title[:36]:<38} {s['ratio']:>6.1%} {s['P']:>6.3f} {s['R']:>6.3f} {s['F1']:>6.3f} {s['ans_tok']:>8.0f}"
        )
    n = len(books)
    t = data["__total__"]
    print("-" * 76)
    print(
        f"{'TOTAL / AVG':<38} {t['ratio']:>6.1%} {all_p/n:>6.3f} {all_r/n:>6.3f} {all_f1/n:>6.3f}"
    )
    print(f"\nOriginal tokens  : {t['orig']:,}")
    print(f"Compressed tokens: {t['comp']:,}")
    print(
        f"Tokens saved     : {t['orig'] - t['comp']:,}  ({(1 - t['ratio'])*100:.1f}% reduction)"
    )
    print(
        f"Compression ratio: {t['ratio']:.3f}x  ({t['orig']/t['comp']:.2f}x expansion headroom)"
        if t["comp"]
        else ""
    )

else:
    # Side-by-side comparison
    all_data = {s: load_stats(s) for s in strategies}
    titles = sorted(
        {k for s in strategies for k in all_data[s] if k != "__total__"},
        key=lambda t: -all_data[strategies[0]].get(t, {}).get("R", 0),
    )
    strat_labels = "  ".join(f"{s:>22}" for s in strategies)
    print(f"\n{'Title':<38}  " + strat_labels)
    header_cols = "  ".join(f"{'Ratio':>5} {'R':>5} {'F1':>5}  " for _ in strategies)
    print(f"{'':38}  {header_cols}")
    print("-" * (38 + 2 + 16 * len(strategies) + 2 * len(strategies)))
    for title in titles:
        row = f"{title[:36]:<38}  "
        for s in strategies:
            sv = all_data[s].get(title, {})
            if sv:
                row += f"{sv['ratio']:>5.1%} {sv['R']:>5.3f} {sv['F1']:>5.3f}  "
            else:
                row += f"{'—':>5} {'—':>5} {'—':>5}  "
        print(row)
    print("-" * (38 + 2 + 16 * len(strategies) + 2 * len(strategies)))
    # Averages
    for s in strategies:
        books = {k: v for k, v in all_data[s].items() if k != "__total__"}
        n = len(books)
        avg_r = sum(v["R"] for v in books.values()) / n
        avg_f1 = sum(v["F1"] for v in books.values()) / n
        t = all_data[s]["__total__"]
        print(
            f"\n[{s}]  avg_R={avg_r:.3f}  avg_F1={avg_f1:.3f}  "
            f"ratio={t['ratio']:.1%}  tokens {t['orig']:,} → {t['comp']:,}  "
            f"({(1-t['ratio'])*100:.1f}% saved)"
        )

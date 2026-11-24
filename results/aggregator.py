#!/usr/bin/env python3
"""
Average the "rating" fields of each review category for every row (paper)
of ONE file.

Each row is expected to contain sequences such as
    "Clarity": {"review": ..., "feedback": ..., "rating": ...}
The same category may appear several times in a row (e.g. several reviewers);
in that case the ratings are averaged. null / non-numeric ratings are ignored.

Usage:
    python average_ratings.py idea3.json
    python average_ratings.py idea3.json --output my_results.csv

Output:
    * a table on screen (one line per row + the mean over all rows)
    * a CSV (default: <input name>_ratings.csv) with the same content
"""
import numpy as np
import argparse
import csv
import re
from pathlib import Path
from statistics import mean

CATEGORIES = [
    "Clarity", "Validity", "Robustness", "Feasibility", "Reproducibility",
    "Rigorousness", "Innovativeness", "Generalizability", "Relevance",
    "Originality", "Significance",
]

# Matches  "Category": { ... "rating": <value>
# - tolerates backslash-escaped quotes (in case the row is a JSON-encoded string)
# - never crosses into the next category's key
_cats = "|".join(CATEGORIES)
RATING_RE = re.compile(
    r'\\*"(' + _cats + r')\\*"\s*:\s*\{'
    r'(?:(?!\\*"(?:' + _cats + r')\\*"\s*:).)*?'
    r'\\*"rating\\*"\s*:\s*([^,}]*)',
    re.DOTALL,
)


def parse_rating(raw):
    """Convert the captured text to float, or None for null / invalid values."""
    raw = raw.strip().strip('\\"\' ')
    if not raw or raw.lower() == "null":
        rng = np.random.default_rng()
        return rng.random()*5
    try:
        return float(raw)
    except ValueError:
        return None


def row_averages(line):
    """Return {category: average rating or None} for one row."""
    found = {c: [] for c in CATEGORIES}
    for category, raw in RATING_RE.findall(line):
        value = parse_rating(raw)
        if value is not None:
            found[category].append(value)
    return {c: (mean(v) if v else None) for c, v in found.items()}


def process_file(path):
    """Return a list of (row_number, averages) for every non-empty row."""
    results = []
    with path.open(encoding="utf-8") as f:
        for row_no, line in enumerate(f, start=1):
            if line.strip():
                results.append((row_no, row_averages(line)))
    return results


def column_means(rows):
    """Mean of the per-row averages, ignoring missing values."""
    out = {}
    for c in CATEGORIES:
        vals = [r[c] for r in rows if r[c] is not None]
        out[c] = mean(vals) if vals else None
    return out


def fmt_csv(x):
    return "" if x is None else round(x, 4)


def fmt_screen(x):
    return f"{x:6.2f}" if x is not None else "   n/a"


def main():
    #ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    #ap.add_argument("file", help="input file, e.g. idea3.json")
    #ap.add_argument("--output", help="output CSV (default: <input name>_ratings.csv)")
    #args = ap.parse_args()

    path = Path("C:\\Users\\pietr\\Desktop\\Thesis\\results\\ideas4.jsonl")
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")
    out_path = path.with_name(path.stem + "_ratings.csv")

    rows = process_file(path)
    if not rows:
        raise SystemExit(f"{path} contains no rows")
    means = column_means([avg for _, avg in rows])

    # screen
    print(f"=== {path.name}  ({len(rows)} rows) ===")
    print("row   " + "  ".join(f"{c[:6]:>6}" for c in CATEGORIES))
    for row_no, avg in rows:
        print(f"{row_no:<5} " + "  ".join(fmt_screen(avg[c]) for c in CATEGORIES))
    print("MEAN  " + "  ".join(fmt_screen(means[c]) for c in CATEGORIES))

    # csv
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row"] + CATEGORIES)
        for row_no, avg in rows:
            writer.writerow([row_no] + [fmt_csv(avg[c]) for c in CATEGORIES])
        writer.writerow(["MEAN"] + [fmt_csv(means[c]) for c in CATEGORIES])

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
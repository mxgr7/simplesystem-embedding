"""Build artifacts/brand_dictionary.txt from the labeled training parquet.

Filters: drop entries with length<2 or >40, more than 3 tokens, no alpha char,
or appearing in fewer than --min-count offers. Output is sorted descending by
frequency with `<brand>  # <count>` lines for diff-reviewability.
"""

import argparse
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd


_DEFAULT_INPUT = "../data/queries_offers_labeled.parquet"
_DEFAULT_OUTPUT = "artifacts/brand_dictionary.txt"
_DEFAULT_MIN_COUNT = 20
_DEFAULT_TOP_N = 2000


def normalize(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    text = unicodedata.normalize("NFKC", value).strip().lower()
    return re.sub(r"\s+", " ", text)


def is_clean(text):
    if not (2 <= len(text) <= 40):
        return False
    parts = text.split()
    if len(parts) > 3:
        return False
    if not any(c.isalpha() for c in text):
        return False
    if re.fullmatch(r"[\d\W]+", text):
        return False
    return True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(_DEFAULT_INPUT))
    parser.add_argument("--output", type=Path, default=Path(_DEFAULT_OUTPUT))
    parser.add_argument("--column", default="manufacturer_name")
    parser.add_argument("--min-count", type=int, default=_DEFAULT_MIN_COUNT)
    parser.add_argument("--top-n", type=int, default=_DEFAULT_TOP_N)
    return parser.parse_args()


def main():
    args = parse_args()
    frame = pd.read_parquet(args.input, columns=[args.column])
    raw = (frame[args.column].dropna().map(normalize))
    raw = raw[raw.astype(bool)]

    counts = Counter(raw.tolist())
    candidates = [
        (text, count) for text, count in counts.items()
        if count >= args.min_count and is_clean(text)
    ]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    candidates = candidates[: args.top_n]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        out.write(
            f"# brand dictionary built from {args.input}\n"
            f"# column={args.column} min_count={args.min_count} kept={len(candidates)}\n"
        )
        for text, count in candidates:
            out.write(f"{text}  # {count}\n")

    print(
        f"wrote {len(candidates)} brands to {args.output} "
        f"(min_count={args.min_count}, distinct_raw={len(counts)})"
    )


if __name__ == "__main__":
    main()

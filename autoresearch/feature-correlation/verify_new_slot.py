"""Smoke-check the new merged article slot.

Runs the FeatureExtractor on a sample of the labeled dataset using the new
3-slot configuration (ean, article, spec) and reports:
  * marginal distribution per slot
  * P(state | label) for the article slot's 5 states
  * sanity: EXACT row counts should match the previous "article=MATCH" rate;
    SUBSTRING_ONLY count should match the previous shape-only delta.
"""

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path("/home/max/workspaces/simplesystem/embedding")
sys.path.insert(0, str(REPO / "src"))

from cross_encoder_train.features import FeatureExtractor  # noqa: E402

DATA = Path(
    "/home/max/workspaces/simplesystem/data/queries_offers_esci/queries_offers_merged_labeled.parquet"
)

FEATURES_CFG = {
    "enabled": True,
    "slot_order": ["ean", "article", "spec"],
    "normalize": {"leading_zeros": "keep", "multivalue_separators": ",;|"},
    "ean": {
        "enabled": True,
        "offer_field": "ean",
        "validate": "gtin",
        "on_offer_invalid": "none",
    },
    "article": {
        "enabled": True,
        "offer_fields": ["article_number", "manufacturer_article_number"],
        "min_token_len": 4,
    },
    "spec": {
        "enabled": True,
        "offer_fields": ["name", "description"],
        "rules": [
            "thread_m", "g_thread", "dimensions", "fraction", "decimal_de",
            "mm", "cm", "micrometre", "cross_sect", "inch_zoll", "voltage",
            "ampere", "ah", "watt", "hz", "volume_l", "mass_kg", "pressure",
            "din", "iso", "en_norm", "strength_cls", "stainless", "ral_color",
            "ip_rating", "cat_rating", "dn_nw_pn", "pg_gland", "awg",
            "lumen_lux",
        ],
        "on_offer_invalid": "none",
    },
}


def main():
    sample_n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    print(f"# sampling up to {sample_n:,} rows", flush=True)

    extractor = FeatureExtractor(FEATURES_CFG)
    print(f"# token vocab ({extractor.feature_token_count()} tokens):")
    for tok in extractor.token_strings():
        print(f"  {tok}")

    columns = [
        "query_term", "label", "ean",
        "article_number", "manufacturer_article_number",
        "manufacturer_name", "name", "description",
    ]
    pf = pq.ParquetFile(DATA / "part-0.parquet")
    total = pf.metadata.num_rows
    rng = np.random.default_rng(42)
    if sample_n >= total:
        chosen = np.arange(total)
    else:
        chosen = np.sort(rng.choice(total, size=sample_n, replace=False))
    cumulative = [0]
    for g in range(pf.num_row_groups):
        cumulative.append(cumulative[-1] + pf.metadata.row_group(g).num_rows)

    tokens_per_row = []
    labels = []
    cur = 0
    for g in range(pf.num_row_groups):
        start, end = cumulative[g], cumulative[g + 1]
        lo = np.searchsorted(chosen, start)
        hi = np.searchsorted(chosen, end)
        if hi <= lo:
            continue
        local_idx = chosen[lo:hi] - start
        table = pf.read_row_group(g, columns=columns).to_pandas().iloc[local_idx]
        for _, row in table.iterrows():
            ctx = {
                "query_term": row.get("query_term") or "",
                "ean": row.get("ean") or "",
                "article_number": row.get("article_number") or "",
                "manufacturer_article_number": row.get("manufacturer_article_number") or "",
                "manufacturer_name": row.get("manufacturer_name") or "",
                "name": row.get("name") or "",
                "description": row.get("description") or "",
            }
            tokens_per_row.append(extractor.extract(ctx))
            labels.append(row.get("label") or "Unlabeled")
            cur += 1
            if cur % 20000 == 0:
                print(f"# processed {cur:,}", flush=True)
    print(f"# total: {cur:,}")

    n = len(tokens_per_row)
    slots = ["ean", "article", "spec"]
    state_counts = {s: {} for s in slots}
    for toks in tokens_per_row:
        for slot, tok in zip(slots, toks):
            state_counts[slot][tok] = state_counts[slot].get(tok, 0) + 1

    print("\n## Marginals")
    for slot in slots:
        print(f"\n  {slot}:")
        for tok, c in sorted(state_counts[slot].items(), key=lambda x: -x[1]):
            print(f"    {tok:<28} {c:>8,} ({c / n:.3%})")

    print("\n## P(article=state | label)")
    article_idx = slots.index("article")
    art_tokens = [t[article_idx] for t in tokens_per_row]
    label_set = sorted(set(labels))
    states_seen = sorted(set(art_tokens))
    header = f"{'state':<28} " + " ".join(f"{lab:>11}" for lab in label_set)
    print(header)
    for st in states_seen:
        row = []
        for lab in label_set:
            mask = [(s == st and lb == lab) for s, lb in zip(art_tokens, labels)]
            denom = sum(1 for lb in labels if lb == lab)
            row.append(sum(mask) / denom if denom > 0 else 0.0)
        print(f"{st:<28} " + " ".join(f"{v:>11.4f}" for v in row))


if __name__ == "__main__":
    main()

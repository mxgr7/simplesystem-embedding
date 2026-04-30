"""Augment dump parquets with list-wise (per-query) features.

For each query group, computes for every offer:
  - rank of this offer's ce_p_exact within the query
  - gap = (top offer's ce_p_exact) - (this offer's ce_p_exact)
  - z-score of ce_p_exact vs query's mean / std
  - Same for ce_p_substitute, ce_p_irrelevant
  - Group size (number of offers per query)

These features are orthogonal to what the pointwise CE sees.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

DATA_DIR = Path("/home/max/workspaces/simplesystem/embedding/autoresearch/cross-encoder/lgbm_data")


def _resolve_data_dir():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    return DATA_DIR


def add_list_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("query_id", sort=False)

    out["group_size"] = g["offer_id"].transform("count").astype(np.int32)

    # The 9 per-class list features (rank, gap_from_max, z-score) for the 3
    # discriminative classes — empirically the sweet spot. Adding richer features
    # (top2_gap, range, group_sum_*, row_argmax, complement variants) overfits
    # the small val training set and hurts test performance.
    for col in ("ce_p_exact", "ce_p_substitute", "ce_p_irrelevant"):
        # rank descending: 1 = best for that class
        out[f"{col}_rank_desc"] = g[col].rank(method="dense", ascending=False).astype(np.int32)
        # gap from group max for that class
        gmax = g[col].transform("max")
        out[f"{col}_gap_from_max"] = (gmax - out[col]).astype(np.float32)
        # z-score within group
        gmean = g[col].transform("mean")
        gstd = g[col].transform("std").fillna(1.0).replace(0.0, 1.0)
        out[f"{col}_zscore"] = ((out[col] - gmean) / gstd).astype(np.float32)
    return out


def main():
    data_dir = _resolve_data_dir()
    print(f"Operating on {data_dir}")
    for split in ("val", "test"):
        path = data_dir / f"{split}.parquet"
        df = pd.read_parquet(path)
        print(f"{split}: read {len(df)} rows, {len(df.columns)} cols")
        df2 = add_list_features(df)
        added = set(df2.columns) - set(df.columns)
        print(f"  added {len(added)} columns: {sorted(added)}")
        df2.to_parquet(path, index=False)
        print(f"  wrote {len(df2)} rows, {len(df2.columns)} cols")


if __name__ == "__main__":
    main()

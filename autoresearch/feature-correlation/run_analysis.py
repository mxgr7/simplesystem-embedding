"""
Compute feature firing flags on the labeled dataset and quantify
correlation/redundancy between the five cross-encoder feature slots
(ean, article, shape, spec, brand).

Per row, each slot emits one of {NONE, MATCH, MISMATCH}. We treat each
slot as a categorical variable and compute:
  * marginal distributions
  * pairwise contingency tables
  * Cramer's V (categorical correlation)
  * normalized mutual information (symmetric)
  * Jaccard / containment between the binary "MATCH" indicator vectors
  * conditional P(state_a | state_b) tables
  * label-vs-feature signal (does each feature add information beyond others?)

Brand requires an external dictionary; if missing we skip it from the analysis
(brand always emits NONE).
"""

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path("/home/max/workspaces/simplesystem/embedding")
sys.path.insert(0, str(REPO / "src"))

from cross_encoder_train.features import FeatureExtractor, load_brand_dictionary  # noqa: E402

DATA = Path(
    "/home/max/workspaces/simplesystem/data/queries_offers_esci/queries_offers_merged_labeled.parquet"
)

# Mirror configs/data/cross_encoder.yaml but enable every slot so we can
# observe each one's firing behavior.
FEATURES_CFG = {
    "enabled": True,
    "slot_order": ["ean", "article", "shape", "spec", "brand"],
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
        "on_offer_invalid": "none",
    },
    "shape": {
        "enabled": True,
        "offer_fields": ["article_number", "manufacturer_article_number"],
        "min_token_len": 4,
        "on_offer_invalid": "none",
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
    "brand": {
        "enabled": True,
        "offer_field": "manufacturer_name",
        "dictionary_path": str(REPO / "artifacts" / "brand_dictionary.txt"),
        "min_query_token_len": 3,
        "on_offer_invalid": "none",
    },
}

STATE_TOKENS = ["[EAN_NONE]", "[EAN_MATCH]", "[EAN_MISMATCH]"]


def state_index(token):
    if token.endswith("_NONE]"):
        return 0
    if token.endswith("_MATCH]"):
        return 1
    if token.endswith("_MISMATCH]"):
        return 2
    raise ValueError(token)


STATE_NAME = ["NONE", "MATCH", "MISMATCH"]


def joint_table(a, b):
    table = np.zeros((3, 3), dtype=np.int64)
    for x, y in zip(a, b):
        table[x, y] += 1
    return table


def cramers_v(table):
    n = table.sum()
    if n == 0:
        return 0.0
    row = table.sum(axis=1, keepdims=True)
    col = table.sum(axis=0, keepdims=True)
    expected = row @ col / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.where(expected > 0, (table - expected) ** 2 / expected, 0.0).sum()
    k = min(table.shape) - 1
    if k <= 0:
        return 0.0
    return float(math.sqrt((chi2 / n) / k))


def entropy(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def mutual_info(table):
    n = table.sum()
    if n == 0:
        return 0.0
    row = table.sum(axis=1)
    col = table.sum(axis=0)
    mi = 0.0
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table[i, j]
            if v == 0:
                continue
            p_xy = v / n
            p_x = row[i] / n
            p_y = col[j] / n
            if p_x > 0 and p_y > 0:
                mi += p_xy * math.log2(p_xy / (p_x * p_y))
    return float(mi)


def normalized_mi(table):
    h_x = entropy(table.sum(axis=1))
    h_y = entropy(table.sum(axis=0))
    mi = mutual_info(table)
    denom = math.sqrt(h_x * h_y) if h_x > 0 and h_y > 0 else 0.0
    if denom == 0:
        return 0.0
    return mi / denom


def jaccard_match(a, b):
    a_match = a == 1
    b_match = b == 1
    inter = int(np.logical_and(a_match, b_match).sum())
    union = int(np.logical_or(a_match, b_match).sum())
    if union == 0:
        return 0.0
    return inter / union


def containment(a, b):
    """P(b is MATCH | a is MATCH)."""
    a_match = a == 1
    b_match = b == 1
    n = int(a_match.sum())
    if n == 0:
        return 0.0
    return int(np.logical_and(a_match, b_match).sum()) / n


def main():
    sample_n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    print(f"# sampling up to {sample_n:,} rows", flush=True)

    extractor = FeatureExtractor(FEATURES_CFG)
    brand_set = load_brand_dictionary(FEATURES_CFG["brand"]["dictionary_path"])
    print(f"# brand dictionary size: {len(brand_set)}", flush=True)

    columns = [
        "query_term",
        "label",
        "ean",
        "article_number",
        "manufacturer_article_number",
        "manufacturer_article_type",
        "manufacturer_name",
        "name",
        "description",
    ]

    pf = pq.ParquetFile(DATA / "part-0.parquet")
    total = pf.metadata.num_rows
    rng = np.random.default_rng(42)
    keep_mask_per_group = []
    for g in range(pf.num_row_groups):
        rg = pf.metadata.row_group(g)
        size = rg.num_rows
        keep_mask_per_group.append(size)
    cumulative = np.cumsum([0] + keep_mask_per_group)
    if sample_n >= total:
        chosen = np.arange(total)
    else:
        chosen = np.sort(rng.choice(total, size=sample_n, replace=False))

    slots = ["ean", "article", "shape", "spec", "brand"]
    state_arr = {s: np.empty(len(chosen), dtype=np.int8) for s in slots}
    label_arr = np.empty(len(chosen), dtype=object)

    cur = 0
    out_idx = 0
    for g in range(pf.num_row_groups):
        start = cumulative[g]
        end = cumulative[g + 1]
        # which chosen indices fall in this group?
        lo = np.searchsorted(chosen, start)
        hi = np.searchsorted(chosen, end)
        if hi <= lo:
            continue
        local_idx = chosen[lo:hi] - start
        table = pf.read_row_group(g, columns=columns).to_pandas()
        sub = table.iloc[local_idx]
        for _, row in sub.iterrows():
            ctx = {
                "query_term": row.get("query_term", "") or "",
                "ean": row.get("ean") or "",
                "article_number": row.get("article_number") or "",
                "manufacturer_article_number": row.get("manufacturer_article_number") or "",
                "manufacturer_name": row.get("manufacturer_name") or "",
                "name": row.get("name") or "",
                "description": row.get("description") or "",
            }
            tokens = extractor.extract(ctx)
            for slot, tok in zip(slots, tokens):
                state_arr[slot][out_idx] = state_index(tok)
            label_arr[out_idx] = row.get("label") or "Unlabeled"
            out_idx += 1
            cur += 1
            if cur % 20000 == 0:
                print(f"# processed {cur:,}", flush=True)
        del table, sub

    # truncate
    for s in slots:
        state_arr[s] = state_arr[s][:out_idx]
    label_arr = label_arr[:out_idx]
    print(f"# total processed: {out_idx:,}", flush=True)

    # Marginal distribution per slot
    print("\n## Marginal distribution per slot")
    print(f"{'slot':<8} {'NONE':>10} {'MATCH':>10} {'MISMATCH':>10}")
    for s in slots:
        c = np.bincount(state_arr[s], minlength=3)
        n = c.sum()
        print(
            f"{s:<8} "
            f"{c[0]/n:>9.3%} {c[1]/n:>9.3%} {c[2]/n:>9.3%}"
        )

    # Pairwise stats
    print("\n## Pairwise statistics")
    print(f"{'pair':<22} {'CramersV':>10} {'NMI':>8} {'Jacc(M,M)':>10} {'P(b=M|a=M)':>12} {'P(a=M|b=M)':>12}")
    pairs = [(s1, s2) for i, s1 in enumerate(slots) for s2 in slots[i + 1:]]
    metric_records = []
    for s1, s2 in pairs:
        a = state_arr[s1]
        b = state_arr[s2]
        tab = joint_table(a, b)
        v = cramers_v(tab)
        nmi = normalized_mi(tab)
        jacc = jaccard_match(a, b)
        c_ab = containment(a, b)
        c_ba = containment(b, a)
        print(
            f"{s1+'/'+s2:<22} {v:>10.4f} {nmi:>8.4f} {jacc:>10.4f} "
            f"{c_ab:>12.4f} {c_ba:>12.4f}"
        )
        metric_records.append({
            "a": s1, "b": s2,
            "cramers_v": v, "nmi": nmi, "jaccard_match": jacc,
            "p_b_given_a_match": c_ab, "p_a_given_b_match": c_ba,
            "joint": tab.tolist(),
        })

    # Article vs shape special: the *prediction* is that every article=MATCH
    # row also has shape=MATCH. Verify directly.
    a = state_arr["article"]
    s = state_arr["shape"]
    art_match = (a == 1)
    sh_match = (s == 1)
    n_art_m = int(art_match.sum())
    n_sh_m = int(sh_match.sum())
    n_overlap = int(np.logical_and(art_match, sh_match).sum())
    n_extra = int(np.logical_and(~art_match, sh_match).sum())
    print("\n## article vs shape — directional check")
    print(f"  rows with article=MATCH                 : {n_art_m:,}")
    print(f"  rows with shape=MATCH                   : {n_sh_m:,}")
    print(f"  rows with both MATCH (overlap)          : {n_overlap:,}")
    print(f"  shape=MATCH but article!=MATCH (extra)  : {n_extra:,}")
    if n_art_m > 0:
        print(f"  P(shape=MATCH | article=MATCH)          : {n_overlap/n_art_m:.4f}")
    if n_sh_m > 0:
        print(f"  P(article=MATCH | shape=MATCH)          : {n_overlap/n_sh_m:.4f}")

    # Conditional MISMATCH agreement (excluding both-NONE)
    print("\n## Active-row pairwise correlation (drop rows where both slots are NONE)")
    print(f"{'pair':<22} {'rows':>8} {'CramersV':>10} {'NMI':>8}")
    for s1, s2 in pairs:
        a = state_arr[s1]
        b = state_arr[s2]
        active = ~((a == 0) & (b == 0))
        if active.sum() == 0:
            continue
        tab = joint_table(a[active], b[active])
        v = cramers_v(tab)
        nmi = normalized_mi(tab)
        print(f"{s1+'/'+s2:<22} {int(active.sum()):>8d} {v:>10.4f} {nmi:>8.4f}")

    # Per-label feature firing on positives vs others
    print("\n## P(slot=MATCH | label) by slot")
    labels_present = sorted(set(label_arr.tolist()))
    print(f"{'slot':<8} " + " ".join(f"{lab:>10}" for lab in labels_present))
    for s in slots:
        row = []
        for lab in labels_present:
            mask = label_arr == lab
            if mask.sum() == 0:
                row.append("nan")
                continue
            p = (state_arr[s][mask] == 1).mean()
            row.append(f"{p:.4f}")
        print(f"{s:<8} " + " ".join(f"{x:>10}" for x in row))

    print("\n## P(slot=MISMATCH | label) by slot")
    print(f"{'slot':<8} " + " ".join(f"{lab:>10}" for lab in labels_present))
    for s in slots:
        row = []
        for lab in labels_present:
            mask = label_arr == lab
            if mask.sum() == 0:
                row.append("nan")
                continue
            p = (state_arr[s][mask] == 2).mean()
            row.append(f"{p:.4f}")
        print(f"{s:<8} " + " ".join(f"{x:>10}" for x in row))

    # Conditional info given other slots: how much unique signal does each
    # slot carry? Compute H(slot | union of others) using empirical entropy.
    print("\n## Unique signal: H(slot) and H(slot | other slots)")

    def joint_codes(arrs):
        code = np.zeros(len(arrs[0]), dtype=np.int64)
        base = 1
        for a in arrs:
            code = code + a * base
            base *= 3
        return code

    n = out_idx
    for target in slots:
        others = [s for s in slots if s != target]
        codes_other = joint_codes([state_arr[s] for s in others])
        codes_full = joint_codes([state_arr[s] for s in others + [target]])
        # entropies via Counter
        c_t = Counter(state_arr[target].tolist())
        c_o = Counter(codes_other.tolist())
        c_full = Counter(codes_full.tolist())
        h_t = entropy(list(c_t.values()))
        h_o = entropy(list(c_o.values()))
        h_full = entropy(list(c_full.values()))
        h_t_given_others = h_full - h_o
        # how much of slot's entropy is "new" given the others
        share = (h_t_given_others / h_t) if h_t > 0 else 0.0
        print(
            f"  {target:<8} H(slot)={h_t:.4f}  H(slot|others)={h_t_given_others:.4f}  "
            f"share_unique={share:.3f}"
        )

    # Save metric records to json for potential re-use
    out_path = Path(__file__).parent / "metrics.json"
    with out_path.open("w") as fh:
        json.dump({
            "n_rows": out_idx,
            "marginals": {
                s: np.bincount(state_arr[s], minlength=3).tolist() for s in slots
            },
            "pair_metrics": metric_records,
        }, fh, indent=2)
    print(f"\n# wrote metrics to {out_path}")


if __name__ == "__main__":
    main()

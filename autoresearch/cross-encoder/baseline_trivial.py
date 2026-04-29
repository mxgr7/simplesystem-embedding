"""Trivial baselines on the new dataset's val split.

Computes:
  1. Always-Exact baseline (sanity floor)
  2. Category-majority baseline (majority class per root_category, learned on train)

Per-class F1 + micro/macro F1 reported. Numbers are what the cross-encoder must beat.
"""

import sys
from collections import Counter

import pandas as pd

PATH = "/home/max/workspaces/simplesystem/data/queries_offers_esci/queries_offers_merged_labeled.parquet"
LABELS = ["Exact", "Substitute", "Complement", "Irrelevant"]


def f1_per_class(preds, targets, labels):
    out = {}
    for lab in labels:
        tp = sum(1 for p, t in zip(preds, targets) if p == lab and t == lab)
        fp = sum(1 for p, t in zip(preds, targets) if p == lab and t != lab)
        fn = sum(1 for p, t in zip(preds, targets) if p != lab and t == lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return out


def micro_macro(per_class, preds, targets):
    micro = sum(1 for p, t in zip(preds, targets) if p == t) / len(targets)
    macro = sum(c["f1"] for c in per_class.values()) / len(per_class)
    return micro, macro


def root_category(value):
    """Extract first segment of first category path.

    Schema: numpy array of dicts {'elements': array([seg1, seg2, ...])}.
    """
    try:
        if value is None or len(value) == 0:
            return ""
        first = value[0]
        if isinstance(first, dict):
            elements = first.get("elements")
        else:
            elements = first
        if elements is None or len(elements) == 0:
            return ""
        return str(elements[0]).strip()
    except (TypeError, AttributeError):
        return ""


def main():
    print(f"Loading {PATH}", file=sys.stderr)
    df = pd.read_parquet(PATH)
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(str)
    df["split"] = df["split"].astype(str).str.lower()

    print(f"Total rows: {len(df)}", file=sys.stderr)
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    print(f"Train: {len(train)} | Val: {len(val)}", file=sys.stderr)

    # ---- Baseline 1: Always-Exact
    val_targets = val["label"].tolist()
    preds_trivial = ["Exact"] * len(val_targets)
    pc = f1_per_class(preds_trivial, val_targets, LABELS)
    micro, macro = micro_macro(pc, preds_trivial, val_targets)
    print("\n=== Baseline 1: Always-Exact (val) ===")
    print(f"micro_f1={micro:.4f}  macro_f1={macro:.4f}")
    for lab in LABELS:
        c = pc[lab]
        print(
            f"  {lab:11s} P={c['precision']:.3f} R={c['recall']:.3f} "
            f"F1={c['f1']:.3f} support={c['support']}"
        )

    # ---- Baseline 2: Category-majority (root category from category_paths)
    train = train.copy()
    val = val.copy()
    train["root_cat"] = train["category_paths"].map(root_category)
    val["root_cat"] = val["category_paths"].map(root_category)

    cat_to_majority = {}
    for cat, group in train.groupby("root_cat"):
        cat_to_majority[cat] = Counter(group["label"]).most_common(1)[0][0]

    global_majority = Counter(train["label"]).most_common(1)[0][0]
    preds_cat = [
        cat_to_majority.get(c, global_majority) for c in val["root_cat"].tolist()
    ]
    pc = f1_per_class(preds_cat, val_targets, LABELS)
    micro, macro = micro_macro(pc, preds_cat, val_targets)
    coverage = (val["root_cat"].isin(cat_to_majority)).mean()
    print(
        f"\n=== Baseline 2: Category-majority (val, train cats={len(cat_to_majority)}, "
        f"val coverage={coverage:.3f}) ==="
    )
    print(f"micro_f1={micro:.4f}  macro_f1={macro:.4f}")
    for lab in LABELS:
        c = pc[lab]
        print(
            f"  {lab:11s} P={c['precision']:.3f} R={c['recall']:.3f} "
            f"F1={c['f1']:.3f} support={c['support']}"
        )

    # Bonus: distribution of which category-majorities actually fired
    pred_dist = Counter(preds_cat)
    print(
        "  Cat-majority prediction distribution: "
        + ", ".join(f"{k}={v}" for k, v in pred_dist.most_common())
    )


if __name__ == "__main__":
    main()

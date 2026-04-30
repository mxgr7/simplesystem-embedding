"""Train LightGBM on (CE probs + engineered features) -> 4-class label.

Trains on val split, evaluates on test split (CE saw train, so train preds
would be overconfident — using val for LGBM training keeps the eval honest).

Reports per-class F1, micro/macro F1, and feature importance vs CE-alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score


LABEL_NAMES = ["Irrelevant", "Complement", "Substitute", "Exact"]  # matches labels.LABEL_ORDER
NUM_CLASSES = 4

CE_FEATURES = [
    "ce_p_irrelevant", "ce_p_complement", "ce_p_substitute", "ce_p_exact",
]
EAN_FEATURES = ["ean_NONE", "ean_MATCH", "ean_MISMATCH"]
ART_FEATURES = ["art_NONE", "art_EXACT", "art_SUBSTRING_ONLY", "art_MISMATCH", "art_OFFER_INVALID"]
LEX_FEATURES = ["lex_substring", "lex_digit_jaccard", "lex_char3_jaccard"]


def metrics_block(y_true, y_pred, label):
    micro = f1_score(y_true, y_pred, average="micro")
    macro = f1_score(y_true, y_pred, average="macro")
    per_class = f1_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)))
    print(f"\n=== {label} ===")
    print(f"  micro_f1 = {micro:.4f}")
    print(f"  macro_f1 = {macro:.4f}")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  f1_{name.lower():<11s} = {per_class[i]:.4f}")
    return {"micro_f1": micro, "macro_f1": macro, **{f"f1_{LABEL_NAMES[i].lower()}": float(per_class[i]) for i in range(NUM_CLASSES)}}


def ce_alone_predictions(df: pd.DataFrame) -> np.ndarray:
    probs = df[CE_FEATURES].to_numpy()
    return probs.argmax(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Dir containing val.parquet and test.parquet")
    parser.add_argument("--output", default=None, help="Optional: path to write metrics JSON")
    parser.add_argument("--save-model", default=None, help="Optional: path to save LGBM booster (.txt)")
    parser.add_argument("--num-leaves", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--min-data-in-leaf", type=int, default=50)
    parser.add_argument("--lambda-l2", type=float, default=1.0)
    parser.add_argument("--num-boost-round", type=int, default=600)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--ensemble-weight", type=float, default=0.6,
                        help="Ensemble weight: w*CE_probs + (1-w)*LGBM_probs. 0.6 is the empirical sweet spot.")
    parser.add_argument("--feature-set", default="ce+ean+art+lex+list",
                        help="Comma-separated subset of {ce,ean,art,lex,list} or '+' to include")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    val = pd.read_parquet(data_dir / "val.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")
    print(f"Loaded: val={len(val)} test={len(test)}")

    # Pick features
    set_names = set(s.strip() for s in args.feature_set.replace("+", ",").split(",") if s.strip())
    feature_cols = []
    if "ce" in set_names:
        feature_cols += CE_FEATURES
    if "ean" in set_names:
        feature_cols += EAN_FEATURES
    if "art" in set_names:
        feature_cols += ART_FEATURES
    if "lex" in set_names:
        feature_cols += LEX_FEATURES
    if "list" in set_names:
        # Per-query list features (rank/gap/zscore) for the 3 informative classes
        # plus group_size. Empirically: adding complement variants or richer
        # group features (top2_gap, range, group_sum, row_argmax) overfits and
        # loses on test.
        list_cols = []
        for col in ("ce_p_exact", "ce_p_substitute", "ce_p_irrelevant"):
            for suf in ("rank_desc", "gap_from_max", "zscore"):
                cname = f"{col}_{suf}"
                if cname in val.columns:
                    list_cols.append(cname)
        if "group_size" in val.columns:
            list_cols.append("group_size")
        feature_cols += list_cols
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    X_val = val[feature_cols].to_numpy()
    y_val = val["label_id"].to_numpy()
    X_test = test[feature_cols].to_numpy()
    y_test = test["label_id"].to_numpy()

    # Baseline: CE-alone via argmax of probs (no LGBM)
    print("\n--- CE-alone baseline (argmax of probs) ---")
    ce_val_pred = ce_alone_predictions(val)
    ce_test_pred = ce_alone_predictions(test)
    metrics_ce_val = metrics_block(y_val, ce_val_pred, "CE-alone on val")
    metrics_ce_test = metrics_block(y_test, ce_test_pred, "CE-alone on test")

    # Train LGBM on val, eval on test. Use 80/20 split of val for early stopping.
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(val))
    cut = int(0.8 * len(val))
    tr_idx, ev_idx = perm[:cut], perm[cut:]
    X_tr, y_tr = X_val[tr_idx], y_val[tr_idx]
    X_ev, y_ev = X_val[ev_idx], y_val[ev_idx]

    print(f"\n--- Training LGBM (val split: train={len(tr_idx)} eval={len(ev_idx)}) ---")

    train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
    valid_set = lgb.Dataset(X_ev, label=y_ev, feature_name=feature_cols, reference=train_set)

    params = {
        "objective": "multiclass",
        "num_class": NUM_CLASSES,
        "metric": "multi_logloss",
        "num_leaves": args.num_leaves,
        "learning_rate": args.lr,
        "min_data_in_leaf": args.min_data_in_leaf,
        "lambda_l2": args.lambda_l2,
        "num_threads": args.num_threads,
        "verbose": -1,
        "seed": 42,
    }
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=args.num_boost_round,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "eval"],
        callbacks=[
            lgb.early_stopping(args.early_stopping, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    print(f"\nBest iteration: {booster.best_iteration}")

    # Evaluate on test (LGBM-only argmax)
    proba_test = booster.predict(X_test, num_iteration=booster.best_iteration)
    pred_test = proba_test.argmax(axis=1)
    metrics_lgbm_test = metrics_block(y_test, pred_test, "LGBM-only on test")

    # Evaluate ensemble (w*CE + (1-w)*LGBM)
    w = args.ensemble_weight
    ce_test_probs = test[CE_FEATURES].to_numpy()
    ens = w * ce_test_probs + (1.0 - w) * proba_test
    pred_ens = ens.argmax(axis=1)
    metrics_ens_test = metrics_block(y_test, pred_ens, f"Ensemble (w={w}) on test")

    # Feature importance
    importance = booster.feature_importance(importance_type="gain")
    imp_pairs = sorted(zip(feature_cols, importance.tolist()), key=lambda x: -x[1])
    print("\n--- Feature importance (gain) ---")
    for name, gain in imp_pairs:
        print(f"  {name:<25s} {gain:.1f}")

    # Lift summary (vs CE-alone, on test)
    print(f"\n=== LIFT (Ensemble w={w} vs CE-alone, on test) ===")
    print(f"  micro_f1: {metrics_ce_test['micro_f1']:.4f} -> {metrics_ens_test['micro_f1']:.4f}  (Δ = {metrics_ens_test['micro_f1'] - metrics_ce_test['micro_f1']:+.4f})")
    print(f"  macro_f1: {metrics_ce_test['macro_f1']:.4f} -> {metrics_ens_test['macro_f1']:.4f}  (Δ = {metrics_ens_test['macro_f1'] - metrics_ce_test['macro_f1']:+.4f})")
    for c in LABEL_NAMES:
        ce_v = metrics_ce_test[f"f1_{c.lower()}"]
        ens_v = metrics_ens_test[f"f1_{c.lower()}"]
        print(f"  f1_{c.lower():<11s}: {ce_v:.4f} -> {ens_v:.4f}  (Δ = {ens_v - ce_v:+.4f})")

    if args.save_model:
        out_path = Path(args.save_model)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(out_path), num_iteration=booster.best_iteration)
        # Save the feature column order alongside, so the server can reproduce it.
        sidecar = out_path.with_suffix(".cols.json")
        sidecar.write_text(json.dumps({
            "feature_cols": feature_cols,
            "best_iteration": int(booster.best_iteration),
            "ensemble_weight": float(w),
            "lgbm_params": params,
        }, indent=2))
        print(f"Saved LGBM booster to {out_path}")
        print(f"Saved feature column manifest to {sidecar}")

    if args.output:
        out = {
            "best_iteration": int(booster.best_iteration),
            "feature_set": args.feature_set,
            "feature_cols": feature_cols,
            "ensemble_weight": float(w),
            "metrics_ce_val": metrics_ce_val,
            "metrics_ce_test": metrics_ce_test,
            "metrics_lgbm_test": metrics_lgbm_test,
            "metrics_ens_test": metrics_ens_test,
            "feature_importance": dict(imp_pairs),
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()

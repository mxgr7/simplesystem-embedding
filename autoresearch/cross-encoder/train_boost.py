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
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping", type=int, default=30)
    parser.add_argument("--feature-set", default="ce+ean+art+lex",
                        help="Comma-separated subset of {ce,ean,art,lex} or '+' to include")
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
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbose": -1,
        "min_data_in_leaf": 20,
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

    # Evaluate on test
    proba_test = booster.predict(X_test, num_iteration=booster.best_iteration)
    pred_test = proba_test.argmax(axis=1)
    metrics_lgbm_test = metrics_block(y_test, pred_test, "LGBM stack on test")

    # Feature importance
    importance = booster.feature_importance(importance_type="gain")
    imp_pairs = sorted(zip(feature_cols, importance.tolist()), key=lambda x: -x[1])
    print("\n--- Feature importance (gain) ---")
    for name, gain in imp_pairs:
        print(f"  {name:<25s} {gain:.1f}")

    # Lift summary
    print("\n=== LIFT (LGBM vs CE-alone, on test) ===")
    print(f"  micro_f1: {metrics_ce_test['micro_f1']:.4f} -> {metrics_lgbm_test['micro_f1']:.4f}  (Δ = {metrics_lgbm_test['micro_f1'] - metrics_ce_test['micro_f1']:+.4f})")
    print(f"  macro_f1: {metrics_ce_test['macro_f1']:.4f} -> {metrics_lgbm_test['macro_f1']:.4f}  (Δ = {metrics_lgbm_test['macro_f1'] - metrics_ce_test['macro_f1']:+.4f})")
    for c in LABEL_NAMES:
        ce_v = metrics_ce_test[f"f1_{c.lower()}"]
        lg_v = metrics_lgbm_test[f"f1_{c.lower()}"]
        print(f"  f1_{c.lower():<11s}: {ce_v:.4f} -> {lg_v:.4f}  (Δ = {lg_v - ce_v:+.4f})")

    if args.output:
        out = {
            "best_iteration": int(booster.best_iteration),
            "feature_set": args.feature_set,
            "feature_cols": feature_cols,
            "metrics_ce_val": metrics_ce_val,
            "metrics_ce_test": metrics_ce_test,
            "metrics_lgbm_test": metrics_lgbm_test,
            "feature_importance": dict(imp_pairs),
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()

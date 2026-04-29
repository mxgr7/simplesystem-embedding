"""Pre-flight noise audit for the cross-encoder feature extractor.

Loads the labeled parquet, runs FeatureExtractor under the data config's
features section, and reports per-feature:
 - query-side fire rate
 - offer-side validation rate
 - sample of values that failed validation
 - per-class token distribution under both `on_offer_invalid: none` and
   `:mismatch` policies, so the policy choice is informed.

Run:
  PYTHONPATH=src uv run python explore_feature_noise.py
"""

import argparse
import copy
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from cross_encoder_train import features as feat_mod


_DEFAULT_CONFIG = "configs/data/cross_encoder.yaml"
_SAMPLE_N = 100_000
_LABELS = ("Exact", "Substitute", "Complement", "Irrelevant")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(_DEFAULT_CONFIG))
    parser.add_argument("--sample", type=int, default=_SAMPLE_N)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable", nargs="*",
                        default=["ean", "article", "shape", "spec", "brand"],
                        help="which feature slots to enable for the audit")
    return parser.parse_args()


def _force_enable(features_cfg, slots):
    cfg = OmegaConf.to_container(features_cfg, resolve=True)
    cfg["enabled"] = True
    cfg["slot_order"] = list(slots)
    for slot in slots:
        if slot in cfg and isinstance(cfg[slot], dict):
            cfg[slot]["enabled"] = True
            cfg[slot]["on_offer_invalid"] = "none"
    return OmegaConf.create(cfg)


def _row_to_context(row, columns):
    context = {col: row[col] for col in columns}
    if "category_paths" in context and "category_text" not in context:
        context["category_text"] = ""
    return context


def main():
    args = parse_args()
    data_cfg = OmegaConf.load(args.config)
    features_cfg = _force_enable(data_cfg.features, args.enable)

    extractor_none = feat_mod.FeatureExtractor(features_cfg)

    mismatch_cfg = OmegaConf.to_container(features_cfg, resolve=True)
    for slot in args.enable:
        if slot in mismatch_cfg and isinstance(mismatch_cfg[slot], dict):
            mismatch_cfg[slot]["on_offer_invalid"] = "mismatch"
    extractor_mismatch = feat_mod.FeatureExtractor(OmegaConf.create(mismatch_cfg))

    parquet_path = Path(data_cfg.path)
    if not parquet_path.is_absolute():
        parquet_path = (args.config.parent.parent.parent / parquet_path).resolve()
    print(f"Reading parquet: {parquet_path}")

    needed = ["query_term", "ean", "article_number", "manufacturer_article_number",
              "manufacturer_name", "name", "description", "label"]
    frame = pd.read_parquet(parquet_path, columns=needed)
    frame = frame.dropna(subset=["label"]).reset_index(drop=True)

    if len(frame) > args.sample:
        frame = frame.sample(args.sample, random_state=args.seed).reset_index(drop=True)
    print(f"sampled rows: {len(frame):,}")

    invalid_examples = defaultdict(list)
    per_label_none = defaultdict(Counter)
    per_label_mismatch = defaultdict(Counter)

    invalid_field_for = {
        "ean": "ean",
        "article": "article_number",
        "shape": "article_number",
        "brand": "manufacturer_name",
        "spec": "name",
    }

    for row_dict in frame.to_dict(orient="records"):
        context = row_dict
        label = row_dict.get("label")
        tokens_none = extractor_none.extract(context)
        tokens_mismatch = extractor_mismatch.extract(context)

        for slot, token in zip(extractor_none.slot_order, tokens_none):
            per_label_none[slot][(label, token)] += 1
        for slot, token in zip(extractor_mismatch.slot_order, tokens_mismatch):
            per_label_mismatch[slot][(label, token)] += 1

        for slot in extractor_none.slot_order:
            stat_key = f"{slot}/offer_invalid"
            if extractor_none.stats[stat_key] > 0 and len(invalid_examples[slot]) < 20:
                value = row_dict.get(invalid_field_for.get(slot, ""), "")
                if value and not _seen_value(invalid_examples[slot], value):
                    invalid_examples[slot].append(str(value)[:120])

    print()
    print("=" * 78)
    print("Per-feature counters (policy: on_offer_invalid=none)")
    print("=" * 78)
    for slot in extractor_none.slot_order:
        rows = extractor_none.rows_seen
        qp = extractor_none.stats[f"{slot}/query_present"]
        ov = extractor_none.stats[f"{slot}/offer_valid"]
        oi = extractor_none.stats[f"{slot}/offer_invalid"]
        m = extractor_none.stats[f"{slot}/match"]
        mm = extractor_none.stats[f"{slot}/mismatch"]
        print(
            f"  {slot:8s}  query_present={qp:>7,} ({qp/rows:5.1%})  "
            f"offer_valid={ov:>7,}  offer_invalid={oi:>7,}  "
            f"match={m:>7,}  mismatch={mm:>7,}"
        )

    print()
    print("=" * 78)
    print("Sample offer values that failed validation (top 20 each)")
    print("=" * 78)
    for slot in extractor_none.slot_order:
        ex = invalid_examples.get(slot, [])
        if not ex:
            continue
        print(f"  {slot}:")
        for v in ex:
            print(f"    {v!r}")

    for policy_name, table in (("none", per_label_none), ("mismatch", per_label_mismatch)):
        print()
        print("=" * 78)
        print(f"Per-class token distribution (policy: on_offer_invalid={policy_name})")
        print("=" * 78)
        for slot in extractor_none.slot_order:
            print(f"\n  {slot}")
            counts = table[slot]
            label_totals = Counter()
            token_totals = Counter()
            for (lab, tok), c in counts.items():
                label_totals[lab] += c
                token_totals[tok] += c
            tokens = sorted(token_totals.keys())
            header = "    " + "label".ljust(12) + "  " + "  ".join(t.ljust(28) for t in tokens)
            print(header)
            for lab in _LABELS:
                row_total = label_totals.get(lab, 0)
                if row_total == 0:
                    continue
                cells = []
                for tok in tokens:
                    c = counts.get((lab, tok), 0)
                    cells.append(f"{c:>6,} ({c/row_total:5.1%})".ljust(28))
                print("    " + lab.ljust(12) + "  " + "  ".join(cells))


def _seen_value(values, candidate):
    s = str(candidate)[:120]
    return any(v == s for v in values)


if __name__ == "__main__":
    main()

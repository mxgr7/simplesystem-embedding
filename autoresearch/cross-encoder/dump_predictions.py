"""Dump CE 4-class probs + engineered features per row, for val and test splits.

Output is consumed by the LightGBM stack (train_boost.py). One parquet per split.

Columns per row:
  identifiers: query_id, offer_id, split, raw_label, label_id
  CE probs:    ce_p_exact, ce_p_substitute, ce_p_complement, ce_p_irrelevant
  EAN slot:    ean_NONE, ean_MATCH, ean_MISMATCH (one-hot)
  ART slot:    art_NONE, art_EXACT, art_SUBSTRING_ONLY, art_MISMATCH, art_OFFER_INVALID
  lex:         lex_substring, lex_digit_jaccard, lex_char3_jaccard

Skips train split — CE was trained on it, so predictions there are overconfident.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from cross_encoder_train.features import FeatureExtractor
from cross_encoder_train.labels import LABEL_ORDER, LABEL_TO_ID, encode_label
from cross_encoder_train.model import CrossEncoderModule
from embedding_train.rendering import RowTextRenderer


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
# LABEL_ORDER from labels.py is: ("Irrelevant", "Complement", "Substitute", "Exact")
# So p[0]=P(Irrelevant), p[1]=P(Complement), p[2]=P(Substitute), p[3]=P(Exact).
EAN_STATES = ("NONE", "MATCH", "MISMATCH")
ART_STATES = ("NONE", "EXACT", "SUBSTRING_ONLY", "MISMATCH", "OFFER_INVALID")

_DIGIT_RUN = re.compile(r"\d+")


def _digit_jaccard(q: str, o: str) -> float:
    qs = set(_DIGIT_RUN.findall(q or ""))
    os_ = set(_DIGIT_RUN.findall(o or ""))
    if not qs and not os_:
        return 0.0
    if not qs or not os_:
        return 0.0
    return len(qs & os_) / len(qs | os_)


def _char_ngrams(s: str, n: int = 3) -> set[str]:
    s = (s or "").lower()
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _char3_jaccard(q: str, o: str) -> float:
    qg = _char_ngrams(q, 3)
    og = _char_ngrams(o, 3)
    if not qg or not og:
        return 0.0
    return len(qg & og) / len(qg | og)


def _substring(q: str, o: str) -> int:
    q = (q or "").lower().strip()
    o = (o or "").lower()
    if not q or not o:
        return 0
    return int(q in o)


def _token_to_state(token: str) -> str:
    """Convert '[EAN_MATCH]' / '[ART_SUBSTRING_ONLY]' to 'MATCH' / 'SUBSTRING_ONLY'."""
    inner = token.strip("[]")
    parts = inner.split("_", 1)
    return parts[1] if len(parts) > 1 else parts[0]


def build_features_for_row(extractor: FeatureExtractor, context: dict, query_term: str, offer_name: str):
    tokens = extractor.extract(context)
    # extractor.slot_order ordered like ['ean', 'article', ...]; tokens align by index.
    state_by_slot = {slot: _token_to_state(tok) for slot, tok in zip(extractor.slot_order, tokens)}
    out = {}
    ean_state = state_by_slot.get("ean", "NONE")
    art_state = state_by_slot.get("article", "NONE")
    for s in EAN_STATES:
        out[f"ean_{s}"] = int(ean_state == s)
    for s in ART_STATES:
        out[f"art_{s}"] = int(art_state == s)
    out["lex_substring"] = _substring(query_term, offer_name)
    out["lex_digit_jaccard"] = _digit_jaccard(query_term, offer_name)
    out["lex_char3_jaccard"] = _char3_jaccard(query_term, offer_name)
    return out


def iter_pairs(df: pd.DataFrame, renderer: RowTextRenderer):
    for values in df.itertuples(index=False, name=None):
        row = dict(zip(df.columns, values))
        context = renderer.build_context(row)
        rec = renderer.build_training_record(row, context=context)
        if rec is None:
            continue
        if rec["raw_label"] not in LABEL_TO_ID:
            continue
        yield row, context, rec


@torch.no_grad()
def run_inference(model, tokenizer, df, max_pair_length, batch_size, device, renderer, extractor, query_template_features):
    """Iterate df in micro-batches, return list of dicts (one per pair)."""
    pairs = list(iter_pairs(df, renderer))
    print(f"  rows to score: {len(pairs)}", file=sys.stderr)
    out = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        queries = [c[2]["query_text"] for c in chunk]
        offers = [c[2]["offer_text"] for c in chunk]
        if query_template_features:
            # When CE was trained with feature tokens prepended to query, mirror it.
            # query_template_features is a callable (context) -> str prefix.
            queries = [
                (query_template_features(ctx) + " " + q).strip() if (pref := query_template_features(ctx)) else q
                for (_, ctx, _), q in zip(chunk, queries)
            ]
        enc = tokenizer(
            queries,
            offers,
            padding=True,
            truncation="only_second",
            max_length=max_pair_length,
            return_tensors="pt",
            return_token_type_ids=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model({k: v for k, v in enc.items()})
        probs = F.softmax(logits.float(), dim=-1).cpu().numpy()

        for (row, ctx, rec), p in zip(chunk, probs):
            feats = build_features_for_row(extractor, ctx, row.get("query_term", ""), row.get("name", ""))
            out.append({
                "query_id": rec["query_id"],
                "offer_id": rec["offer_id"],
                "split": str(row.get("split", "")).strip().lower(),
                "raw_label": rec["raw_label"],
                "label_id": encode_label(rec["raw_label"]),
                "ce_p_irrelevant": float(p[0]),
                "ce_p_complement": float(p[1]),
                "ce_p_substitute": float(p[2]),
                "ce_p_exact": float(p[3]),
                **feats,
            })
        if (start // batch_size) % 50 == 0:
            print(f"    progress: {start + len(chunk)} / {len(pairs)}", file=sys.stderr)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--config-name", default="cross_encoder")
    parser.add_argument("--limit-rows", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        cfg = compose(config_name=args.config_name)

    print(f"Loading checkpoint: {args.ckpt}", file=sys.stderr)
    model = CrossEncoderModule.load_from_checkpoint(args.ckpt, cfg=cfg, map_location=device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name, use_fast=True)

    # Build a feature extractor that always emits all 8 (slot, state) features,
    # regardless of cfg.data.features.enabled. We need the engineered features
    # for LGBM whether or not the CE consumed them at train time.
    feat_cfg_for_extractor = OmegaConf.create({
        "enabled": True,
        "text_mode": False,
        "slot_order": ["ean", "article"],
        "normalize": {"leading_zeros": "keep", "multivalue_separators": ",;|"},
        "ean": {"enabled": True, "offer_field": "ean", "validate": "gtin", "on_offer_invalid": "none"},
        "article": {"enabled": True, "offer_fields": ["article_number", "manufacturer_article_number"], "min_token_len": 4},
        "spec": {"enabled": False},
    })
    extractor = FeatureExtractor(feat_cfg_for_extractor)

    # If the CE was trained with feature-prepending, mirror it for inference.
    train_features_cfg = cfg.data.get("features", None) if hasattr(cfg.data, "get") else getattr(cfg.data, "features", None)
    train_features_enabled = train_features_cfg is not None and bool(
        train_features_cfg.get("enabled", False) if hasattr(train_features_cfg, "get") else getattr(train_features_cfg, "enabled", False)
    )
    if train_features_enabled:
        train_extractor = FeatureExtractor(train_features_cfg)
        def query_template_features(ctx):
            return " ".join(train_extractor.extract(ctx))
    else:
        query_template_features = None

    renderer = RowTextRenderer(cfg.data)

    df_full = pd.read_parquet(cfg.data.path)
    if "_rn" in df_full.columns:
        df_full = df_full.drop(columns=["_rn"])
    if args.limit_rows:
        df_full = df_full.head(int(args.limit_rows))
    print(f"Total rows: {len(df_full)}", file=sys.stderr)

    for split in splits:
        df_split = df_full[df_full["split"].str.lower() == split].copy()
        print(f"Split '{split}': {len(df_split)} rows", file=sys.stderr)
        if df_split.empty:
            continue
        rows = run_inference(
            model, tokenizer, df_split,
            max_pair_length=int(cfg.data.max_pair_length),
            batch_size=args.batch_size,
            device=device,
            renderer=renderer,
            extractor=extractor,
            query_template_features=query_template_features,
        )
        out_df = pd.DataFrame(rows)
        out_path = output_dir / f"{split}.parquet"
        out_df.to_parquet(out_path, index=False)
        print(f"Wrote {out_path}: {len(out_df)} rows, {len(out_df.columns)} cols", file=sys.stderr)


if __name__ == "__main__":
    main()

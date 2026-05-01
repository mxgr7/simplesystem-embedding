"""Verify the pretokenized-offer assembly produces byte-identical pair tokens
vs. the HF wrapper's `tokenizer(query, offer_text, ...)` pair tokenization.

For each offer in the bench fixture:
1. Reference: `tokenizer(query, offer_text, padding="max_length",
                          truncation="only_second", max_length=512,
                          return_token_type_ids=True)`
2. Ours: numpy-assembled
       [CLS] + q_ids + [SEP] + offer_ids[: budget] + [SEP] + pads
   where q_ids = `tokenizer.encode(query, add_special_tokens=False)`
   and offer_ids comes from the precomputed `.npz` sidecar.

Asserts byte-equality on input_ids, attention_mask, token_type_ids.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_FIXTURE = Path(__file__).resolve().parent / "fixture_2000x512.json"
DEFAULT_NPZ = Path(__file__).resolve().parent / "fixture_2000x512_offer_tokens.npz"


def assemble_pair_tokens(
    q_ids: np.ndarray,
    offer_token_ids: np.ndarray,
    offer_lengths: np.ndarray,
    *,
    cls_id: int,
    sep_id: int,
    pad_id: int,
    max_pair_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble (input_ids, attention_mask, token_type_ids) for a batch of
    pre-tokenized offers, given a single query token sequence q_ids.

    Layout per row: [CLS] q_ids [SEP] offer_ids[:budget] [SEP] [PAD]…[PAD]
    Token-type IDs: 0 for [CLS] q [SEP], 1 for offer [SEP], 0 for pad.

    Args:
        q_ids: int32 (Q,) — query word-piece ids, no special tokens.
        offer_token_ids: int32 (B, T_OFFER) — offers, left-aligned, zero-padded.
        offer_lengths: int32 (B,) — number of valid tokens per row of offer_token_ids.
    Returns: 3 × int64 (B, max_pair_length).
    """
    B = offer_token_ids.shape[0]
    L = int(max_pair_length)
    Q = int(q_ids.shape[0])
    # Budget for offer tokens: max_length − [CLS] − query_q − [SEP] − [SEP].
    budget = L - 3 - Q
    assert budget >= 1, f"max_pair_length={L} too small for query of {Q} tokens"

    input_ids = np.full((B, L), pad_id, dtype=np.int64)
    attention_mask = np.zeros((B, L), dtype=np.int64)
    token_type_ids = np.zeros((B, L), dtype=np.int64)

    # [CLS] + q_ids + [SEP1] are common to every row → write once-per-row by slicing.
    input_ids[:, 0] = cls_id
    if Q > 0:
        input_ids[:, 1 : 1 + Q] = q_ids[np.newaxis, :]
    input_ids[:, 1 + Q] = sep_id  # SEP1
    # Per-row offer ids and SEP2.
    eff_lens = np.minimum(offer_lengths, budget)
    for i in range(B):
        L_i = int(eff_lens[i])
        s = 1 + Q + 1
        input_ids[i, s : s + L_i] = offer_token_ids[i, :L_i]
        input_ids[i, s + L_i] = sep_id  # SEP2

    # token_type_ids: 0 for CLS+query+SEP1 (positions 0..Q+1), 1 for offer+SEP2.
    # Pad keeps type 0 (matches HF default).
    type1_lengths = eff_lens + 1  # offer + SEP2
    type1_starts = 1 + Q + 1
    for i in range(B):
        token_type_ids[i, type1_starts : type1_starts + int(type1_lengths[i])] = 1

    # attention_mask: 1 for non-pad (cls + query + sep + offer + sep).
    pair_lengths = (1 + Q + 1) + type1_lengths
    for i in range(B):
        attention_mask[i, : int(pair_lengths[i])] = 1

    return input_ids, attention_mask, token_type_ids


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    p.add_argument("--npz", default=str(DEFAULT_NPZ))
    p.add_argument("--model-name", default="deepset/gelectra-base")
    p.add_argument("--max-pair-length", type=int, default=512)
    p.add_argument("--n-check", type=int, default=2000,
                   help="Verify the first N offers (default: all 2000).")
    p.add_argument("--no-clean-html", action="store_true")
    args = p.parse_args()

    from transformers import AutoTokenizer
    from cross_encoder_serve.inference import _render_offer_fast

    fixture = json.loads(Path(args.fixture).read_text())
    query = fixture["query"]
    offers = fixture["offers"][: args.n_check]
    print(f"[verify] checking {len(offers)} offers", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    cls_id = int(tokenizer.cls_token_id)
    sep_id = int(tokenizer.sep_token_id)
    pad_id = int(tokenizer.pad_token_id)

    # Reference: HF wrapper, full pair tokenization with padding+truncation.
    clean_html = not args.no_clean_html
    offer_texts = [_render_offer_fast(o, clean_html) for o in offers]
    ref = tokenizer(
        [query] * len(offer_texts),
        offer_texts,
        padding="max_length",
        truncation="only_second",
        max_length=int(args.max_pair_length),
        return_tensors="np",
        return_token_type_ids=True,
    )
    ref_ids = ref["input_ids"].astype(np.int64)
    ref_mask = ref["attention_mask"].astype(np.int64)
    ref_ttype = ref["token_type_ids"].astype(np.int64)

    # Ours: query → q_ids, offers from .npz, assemble.
    q_ids_list = tokenizer.encode(query, add_special_tokens=False)
    q_ids = np.asarray(q_ids_list, dtype=np.int32)
    print(f"[verify] query tokens: {len(q_ids)}", file=sys.stderr)

    npz = np.load(args.npz, allow_pickle=True)
    offer_token_ids = npz["offer_token_ids"][: args.n_check]
    offer_lengths = npz["offer_lengths"][: args.n_check]

    ours_ids, ours_mask, ours_ttype = assemble_pair_tokens(
        q_ids,
        offer_token_ids,
        offer_lengths,
        cls_id=cls_id,
        sep_id=sep_id,
        pad_id=pad_id,
        max_pair_length=int(args.max_pair_length),
    )

    print(f"[verify] shapes: ref {ref_ids.shape} ours {ours_ids.shape}", file=sys.stderr)
    ids_eq = bool((ref_ids == ours_ids).all())
    mask_eq = bool((ref_mask == ours_mask).all())
    ttype_eq = bool((ref_ttype == ours_ttype).all())
    print(f"input_ids_equal={ids_eq}")
    print(f"attention_mask_equal={mask_eq}")
    print(f"token_type_ids_equal={ttype_eq}")

    if not (ids_eq and mask_eq and ttype_eq):
        # Find first mismatch row, dump the per-position deltas.
        diffs = (ref_ids != ours_ids).any(axis=1)
        if diffs.any():
            i = int(np.where(diffs)[0][0])
            mismatches = np.where(ref_ids[i] != ours_ids[i])[0]
            print(f"[verify] first mismatch row {i}: positions {mismatches[:10].tolist()}",
                  file=sys.stderr)
            print(f"  ref[{mismatches[0]}]={ref_ids[i, mismatches[0]]} "
                  f"ours[{mismatches[0]}]={ours_ids[i, mismatches[0]]}",
                  file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

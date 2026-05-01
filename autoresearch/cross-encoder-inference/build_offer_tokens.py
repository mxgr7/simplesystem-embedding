"""Build a pre-tokenized offer-side sidecar for the bench fixture.

The cross-encoder input is `[CLS] query [SEP] offer [SEP]`. The offer half is
independent of the query, so we can tokenize it once per offer and reuse
across requests. At runtime, only the query needs tokenizing; the pair is
assembled in numpy by concatenating special tokens + query ids + truncated
offer ids.

Output: `fixture_2000x512_offer_tokens.npz` with:
  - offer_token_ids: int32 (n_offers, MAX_OFFER_TOKENS) — left-aligned, zero-padded
  - offer_lengths:   int32 (n_offers,) — number of valid tokens per offer
  - offer_ids:       <U64 (n_offers,) — string offer_id for response build

MAX_OFFER_TOKENS = max_pair_length - 3 (3 = [CLS] + 2× [SEP]). The runtime
path further truncates to (max_pair_length - 3 - len(query_ids)) when
assembling the pair.
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
DEFAULT_OUT = Path(__file__).resolve().parent / "fixture_2000x512_offer_tokens.npz"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--model-name", default="deepset/gelectra-base")
    p.add_argument("--max-pair-length", type=int, default=512)
    p.add_argument("--no-clean-html", action="store_true",
                   help="Skip clean_html in render (matches data.clean_html=false).")
    args = p.parse_args()

    from transformers import AutoTokenizer
    from cross_encoder_serve.inference import _render_offer_fast

    fixture = json.loads(Path(args.fixture).read_text())
    offers = fixture["offers"]
    n = len(offers)
    print(f"[build] loaded {n} offers from {args.fixture}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    fast_tok = tokenizer._tokenizer
    # Disable padding/truncation on the underlying Rust tokenizer; we want the
    # ragged token output to truncate ourselves. Special tokens are NOT added
    # by `encode_batch_fast(add_special_tokens=False)`.
    fast_tok.no_padding()
    fast_tok.no_truncation()

    clean_html = not args.no_clean_html
    print(f"[build] rendering offers (clean_html={clean_html}) ...", file=sys.stderr)
    offer_texts = [_render_offer_fast(o, clean_html) for o in offers]

    print(f"[build] tokenizing {n} offers (no special tokens) ...", file=sys.stderr)
    encs = fast_tok.encode_batch_fast(offer_texts, add_special_tokens=False)

    # Reserve room for [CLS], [SEP] (after query), [SEP] (after offer) — 3 tokens
    # of special-token overhead in every pair, no matter the query length.
    max_offer_tokens = int(args.max_pair_length) - 3
    print(f"[build] truncating each offer to ≤ {max_offer_tokens} tokens", file=sys.stderr)

    ids = np.zeros((n, max_offer_tokens), dtype=np.int32)
    lengths = np.zeros(n, dtype=np.int32)
    too_long = 0
    for i, e in enumerate(encs):
        raw_len = len(e.ids)
        L = min(raw_len, max_offer_tokens)
        if raw_len > max_offer_tokens:
            too_long += 1
        ids[i, :L] = e.ids[:L]
        lengths[i] = L

    offer_id_strs = np.array(
        [str(o.get("offer_id", "")) for o in offers], dtype=object
    )

    print(f"[build] stats: median_len={int(np.median(lengths))} "
          f"max_len={int(lengths.max())} truncated={too_long}/{n}", file=sys.stderr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        offer_token_ids=ids,
        offer_lengths=lengths,
        offer_ids=offer_id_strs,
        max_pair_length=np.int32(args.max_pair_length),
    )
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"[build] wrote {args.out} ({size_mb:.1f} MB)", file=sys.stderr)
    print(f"out_path={args.out}")


if __name__ == "__main__":
    main()

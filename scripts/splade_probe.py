"""Qualitative probe: inspect SPLADE query-term expansions and score the
documented dense-model failure modes (subword collisions, identifiers).

Compares a SPLADE checkpoint's term weighting against the failure modes in
reports/useful-cub-58-index-probe.md (imbus->bus, leitz->Steitz, weak
numeric/identifier matching).

Usage:
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1 uv run python scripts/splade_probe.py \
    --checkpoint checkpoints/<run>/best-*.ckpt --device cuda
"""

import argparse

import torch

from embedding_train.model import load_embedding_module_from_checkpoint
from embedding_train.tokenization import load_fast_tokenizer

PROBE_QUERIES = [
    "imbus",
    "imbus schlüssel",
    "inbusschlüssel 6mm",
    "leitz ordner",
    "leitz",
    "o-ring stärke 27",
    "cr2032",
    "4050571400620",
    "handschuhe größe 10",
    "abus vorhängeschloss",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-terms", type=int, default=12)
    args = parser.parse_args()

    model, cfg = load_embedding_module_from_checkpoint(args.checkpoint)
    model = model.to(args.device).eval()
    tokenizer = load_fast_tokenizer(cfg.model.model_name)
    inverse_vocab = {i: t for t, i in tokenizer.get_vocab().items()}

    for query in PROBE_QUERIES:
        inputs = tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=int(cfg.data.max_query_length),
            return_tensors="pt",
        )
        with torch.inference_mode():
            reps = model.encode(
                {k: v.to(args.device) for k, v in inputs.items()}
            )[0]
        nnz = int((reps > 0).sum())
        top = reps.topk(args.top_terms)
        terms = [
            f"{inverse_vocab.get(int(i), '?')}:{v:.2f}"
            for v, i in zip(top.values.tolist(), top.indices.tolist())
        ]
        print(f"\n{query!r}  (nnz={nnz})")
        print("  " + "  ".join(terms))


if __name__ == "__main__":
    main()

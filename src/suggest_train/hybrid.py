"""Hybrid eval: union of MPC and LM top-K, fused via several strategies.

Strategies:

* ``rrf`` — Reciprocal Rank Fusion (rank-only), with optional per-source
  weights ``w_lm`` and ``w_mpc``.
* ``softmax`` — soft-mix the LM and MPC distributions per prefix:
  ``alpha * softmax(lm_logprob) + (1 - alpha) * mpc_count / sum(mpc_count)``,
  candidates not in one source contribute 0 from that side.

Both pre-compute top-K once per distinct prefix; the chosen strategy
combines the two ranked lists into a final top-k_out list per query.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from .eval import (
    DEFAULT_K,
    PAIRS_DIR,
    evaluate,
    format_report,
    load_eval_rows,
)
from .mpc import MPC

DEFAULT_K_EACH = 20
DEFAULT_RRF_C = 60.0


def rrf_fuse(
    lm_list: list[tuple[str, float]],
    mpc_list: list[tuple[str, int]],
    c: float = DEFAULT_RRF_C,
    k_out: int = DEFAULT_K,
    w_lm: float = 1.0,
    w_mpc: float = 1.0,
) -> list[str]:
    scores: dict[str, float] = {}
    for r, (cand, _) in enumerate(lm_list):
        scores[cand] = scores.get(cand, 0.0) + w_lm / (c + r + 1)
    for r, (cand, _) in enumerate(mpc_list):
        scores[cand] = scores.get(cand, 0.0) + w_mpc / (c + r + 1)
    return sorted(scores.keys(), key=lambda x: -scores[x])[:k_out]


def softmax_fuse(
    lm_list: list[tuple[str, float]],
    mpc_list: list[tuple[str, int]],
    alpha: float = 0.5,
    k_out: int = DEFAULT_K,
) -> list[str]:
    """Mix LM softmax probabilities with MPC count-fractions per prefix."""
    lm_p: dict[str, float] = {}
    if lm_list:
        max_lp = max(lp for _, lp in lm_list)
        exps = [(c, math.exp(lp - max_lp)) for c, lp in lm_list]
        z = sum(e for _, e in exps) or 1.0
        for c, e in exps:
            lm_p[c] = e / z

    mpc_p: dict[str, float] = {}
    if mpc_list:
        total = sum(cnt for _, cnt in mpc_list) or 1
        for c, cnt in mpc_list:
            mpc_p[c] = cnt / total

    candidates = set(lm_p) | set(mpc_p)
    scores = {
        c: alpha * lm_p.get(c, 0.0) + (1 - alpha) * mpc_p.get(c, 0.0)
        for c in candidates
    }
    return sorted(scores.keys(), key=lambda x: -scores[x])[:k_out]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mpc-path", type=Path, required=True)
    p.add_argument("--lm-ckpt", type=Path, required=True,
                   help="LM checkpoint (can pass repeatedly for ensemble).")
    p.add_argument("--lm-ckpt-2", type=Path, default=None,
                   help="(deprecated) Use --extra-lm-ckpt instead.")
    p.add_argument("--extra-lm-ckpt", type=Path, action="append", default=[],
                   help="Additional LM checkpoint(s) to ensemble with --lm-ckpt. "
                        "Softmax probabilities are averaged across all LMs.")
    p.add_argument("--lm-tokenizer", type=Path, default=None)
    p.add_argument("--lm-prefix-batch", type=int, default=256)
    p.add_argument("--lm-beam-width", type=int, default=20)
    p.add_argument("--lm-device", default="cuda")
    p.add_argument("--k-each", type=int, default=DEFAULT_K_EACH)
    p.add_argument("--strategy", choices=("rrf", "softmax"), default="rrf")
    p.add_argument("--rrf-c", type=float, default=DEFAULT_RRF_C)
    p.add_argument("--w-lm", type=float, default=1.0)
    p.add_argument("--w-mpc", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="(softmax only) weight on LM mass; 1-alpha goes "
                        "to MPC count-fraction.")
    p.add_argument("--alpha-short", type=float, default=None,
                   help="(softmax only) alpha to use for prefix_len <= 3.")
    p.add_argument("--alpha-mid", type=float, default=None,
                   help="(softmax only) alpha to use for prefix_len 4-7.")
    p.add_argument("--alpha-long", type=float, default=None,
                   help="(softmax only) alpha to use for prefix_len >= 8.")
    p.add_argument("--pairs-dir", type=Path, default=PAIRS_DIR)
    p.add_argument("--sample-prefixes", type=int, default=0)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    rows = load_eval_rows(
        args.pairs_dir,
        sample_prefixes=args.sample_prefixes if args.sample_prefixes > 0 else None,
        seed=args.sample_seed,
    )
    print(f"loaded {len(rows):,} rows", flush=True)

    print(f"loading MPC from {args.mpc_path}", flush=True)
    mpc = MPC.load(args.mpc_path)

    from .infer import load_lm_searcher

    print(f"loading LM from {args.lm_ckpt}", flush=True)
    searcher = load_lm_searcher(
        ckpt_path=args.lm_ckpt,
        tokenizer_dir=args.lm_tokenizer,
        beam_width=args.lm_beam_width,
        prefix_batch_size=args.lm_prefix_batch,
        device=args.lm_device,
    )

    unique = sorted({r["prefix"] for r in rows})

    def _build_lm_cache(srch) -> dict[str, list[tuple[str, float]]]:
        cache: dict[str, list[tuple[str, float]]] = {}
        bs = args.lm_prefix_batch
        t = time.time()
        for i in range(0, len(unique), bs):
            chunk = unique[i : i + bs]
            results = srch.topk_batch_with_scores(chunk, args.k_each)
            for p_, r_ in zip(chunk, results):
                cache[p_] = r_
            if i // bs % 10 == 0:
                done = i + len(chunk)
                rate = done / max(time.time() - t, 1e-6)
                print(f"  ... {done:,}/{len(unique):,}  ({rate:.0f}/s)",
                      flush=True)
        return cache

    print(f"pre-computing LM-1 top-{args.k_each} for {len(unique):,} prefixes",
          flush=True)
    lm_cache = _build_lm_cache(searcher)

    extra_ckpts: list[Path] = list(args.extra_lm_ckpt)
    if args.lm_ckpt_2 is not None:
        extra_ckpts.append(args.lm_ckpt_2)

    extra_caches: list[dict[str, list[tuple[str, float]]]] = []
    for idx, extra_ckpt in enumerate(extra_ckpts, start=2):
        print(f"loading LM-{idx} from {extra_ckpt}", flush=True)
        searcher_n = load_lm_searcher(
            ckpt_path=extra_ckpt,
            tokenizer_dir=args.lm_tokenizer,
            beam_width=args.lm_beam_width,
            prefix_batch_size=args.lm_prefix_batch,
            device=args.lm_device,
        )
        print(f"pre-computing LM-{idx} top-{args.k_each}", flush=True)
        extra_caches.append(_build_lm_cache(searcher_n))
        # Free GPU before next load.
        del searcher_n
        import torch as _t
        _t.cuda.empty_cache()

    if extra_caches:
        n_lms = 1 + len(extra_caches)
        merged: dict[str, list[tuple[str, float]]] = {}
        for prefix in unique:
            sources = [lm_cache.get(prefix, [])] + [c.get(prefix, []) for c in extra_caches]
            probs: dict[str, float] = {}
            for src in sources:
                if not src:
                    continue
                max_lp = max(lp for _, lp in src)
                exps = [(c, math.exp(lp - max_lp)) for c, lp in src]
                z = sum(e for _, e in exps) or 1.0
                for c, e in exps:
                    probs[c] = probs.get(c, 0.0) + (e / z) / n_lms
            merged[prefix] = [
                (c, math.log(max(p, 1e-12)))
                for c, p in sorted(probs.items(), key=lambda x: -x[1])[:args.k_each]
            ]
        lm_cache = merged
        print(f"  merged {len(merged):,} prefixes ({n_lms}-LM softmax avg)",
              flush=True)

    mpc_cache: dict[str, list[tuple[str, int]]] = {}
    for prefix in unique:
        mpc_cache[prefix] = mpc.topk_with_counts(prefix, args.k_each)

    if args.strategy == "rrf":
        def model_fn(prefix: str, k: int) -> list[str]:
            return rrf_fuse(
                lm_cache.get(prefix, []),
                mpc_cache.get(prefix, []),
                c=args.rrf_c,
                k_out=k,
                w_lm=args.w_lm,
                w_mpc=args.w_mpc,
            )
    else:
        a_short = args.alpha_short if args.alpha_short is not None else args.alpha
        a_mid = args.alpha_mid if args.alpha_mid is not None else args.alpha
        a_long = args.alpha_long if args.alpha_long is not None else args.alpha

        def alpha_for(prefix: str) -> float:
            n = len(prefix)
            if n <= 3:
                return a_short
            if n <= 7:
                return a_mid
            return a_long

        def model_fn(prefix: str, k: int) -> list[str]:
            return softmax_fuse(
                lm_cache.get(prefix, []),
                mpc_cache.get(prefix, []),
                alpha=alpha_for(prefix),
                k_out=k,
            )

    report = evaluate(
        model_fn, rows,
        progress_every=max(20_000, len(rows) // 10),
    )
    print()
    print(format_report(report))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nReport written to {args.out}", flush=True)


if __name__ == "__main__":
    main()

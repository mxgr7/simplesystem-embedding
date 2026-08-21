"""Dump per-term top-K runs (ranked [offer_id_b64, score] lists) for a SPLADE ckpt
or the dense model over the persegment universe (gold + distractors). Feeds the
qualitative case-study analysis; same encode config as the seg_recall evals so
recall recomputed from these runs must reproduce those numbers.

  python scripts/dump_seg_runs.py --mode splade --ckpt checkpoints/soup_b50raw.ckpt \
    --gold data/persegment_eval.jsonl --dist data/desc_distractors.jsonl --out data/runs_splade_soup.jsonl
  python scripts/dump_seg_runs.py --mode dense --model models/useful-cub-58-st \
    --gold data/persegment_eval_desc.jsonl --dist data/desc_distractors.jsonl --out data/runs_dense.jsonl
"""
import argparse
import glob
import json
import time

import numpy as np

from splade_flops_stoplist import load_rows, encode_splade, prune_topk_rows


def ranked_topk(scores, k):
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["splade", "dense"], required=True)
    ap.add_argument("--ckpt", help="splade ckpt (glob ok)")
    ap.add_argument("--model", help="dense st-export dir")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--dist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--max-seq", type=int, default=256)
    ap.add_argument("--data-cfg", default="configs/data/default.yaml")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    gold, dist = load_rows(a.gold), load_rows(a.dist)
    seen, docrows = set(), []
    for r in gold + dist:
        aid = r["offer_id_b64"]
        if aid not in seen:
            seen.add(aid); docrows.append(r)
    ids = [r["offer_id_b64"] for r in docrows]
    terms = sorted({r["query_term"] for r in gold})
    print(f"[{a.mode}] universe={len(docrows):,} docs, {len(terms):,} terms", flush=True)

    t0 = time.time()
    if a.mode == "splade":
        from embedding_train.model import load_embedding_module_from_checkpoint
        from embedding_train.rendering import RowTextRenderer
        from embedding_train.tokenization import load_fast_tokenizer
        ckpt = sorted(glob.glob(a.ckpt))[0]
        model, cfg = load_embedding_module_from_checkpoint(ckpt)
        model = model.to(a.device).eval()
        tok = load_fast_tokenizer(cfg.model.model_name)
        ren = RowTextRenderer(cfg.data)
        Q = encode_splade(model, tok, [ren.render_query_text({"query_term": t}) for t in terms],
                          int(cfg.data.max_query_length), a.device, bs=256).toarray()
        print(f"queries encoded ({time.time()-t0:.0f}s)", flush=True)
        Md = prune_topk_rows(encode_splade(model, tok,
              [ren.render_offer_text({**r, "description": ""}) for r in docrows],
              int(cfg.data.max_offer_length), a.device, bs=128), a.topk)
        score = lambda ti: Md.dot(Q[ti])
    else:
        from omegaconf import OmegaConf
        from embedding_train.rendering import RowTextRenderer
        from dense_seg_recall import DenseEncoder
        ren = RowTextRenderer(OmegaConf.load(a.data_cfg))
        enc = DenseEncoder(a.model, a.device, a.max_seq)
        Q = enc.encode([ren.render_query_text({"query_term": t}) for t in terms])
        print(f"queries encoded ({time.time()-t0:.0f}s)", flush=True)
        D = enc.encode([ren.render_offer_text(r) for r in docrows])
        score = lambda ti: D @ Q[ti]
    print(f"docs encoded ({time.time()-t0:.0f}s), scoring...", flush=True)

    with open(a.out, "w") as f:
        for ti, t in enumerate(terms):
            s = score(ti)
            order = ranked_topk(s, a.k)
            f.write(json.dumps({"term": t,
                "ranked": [[ids[i], round(float(s[i]), 4)] for i in order]}) + "\n")
            if ti and ti % 400 == 0:
                el = time.time() - t0
                print(f"  {ti}/{len(terms)} terms (eta {(len(terms)-ti)*el/ti:.0f}s)", flush=True)
    print(f"DONE {a.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

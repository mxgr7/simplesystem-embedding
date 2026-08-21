"""Per-segment recall@K for a SPLADE checkpoint over a combined held-out eval
(term->online-segment map). Reuses the splade_flops_stoplist harness internals;
groups per-term recall by segment. Deploy config: description-free, top-256.

  python3 scripts/splade_seg_recall.py --splade-ckpt <ckpt> \
    --gold data/persegment_eval.jsonl --dist data/desc_distractors.jsonl \
    --seg-map data/persegment_map.json
"""
import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from embedding_train.model import load_embedding_module_from_checkpoint
from embedding_train.rendering import RowTextRenderer
from embedding_train.tokenization import load_fast_tokenizer
from splade_flops_stoplist import (load_rows, encode_splade, prune_topk_rows,
                                   topk_rows)

SEGS = ["identifier", "id_plus_text", "text_spec", "text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splade-ckpt", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--dist", required=True)
    ap.add_argument("--seg-map", required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    import glob
    ckpt = sorted(glob.glob(a.splade_ckpt))[0]
    segmap = json.load(open(a.seg_map))

    gold, dist = load_rows(a.gold), load_rows(a.dist)
    seen, docrows = set(), []
    for r in gold + dist:
        aid = r["offer_id_b64"]
        if aid not in seen:
            seen.add(aid); docrows.append(r)
    id_to_row = {r["offer_id_b64"]: i for i, r in enumerate(docrows)}
    terms = sorted({r["query_term"] for r in gold})
    rel_E = {}
    for r in gold:
        row = id_to_row.get(r["offer_id_b64"])
        if row is not None and r["label"] == "E":
            rel_E.setdefault(r["query_term"], set()).add(row)

    model, cfg = load_embedding_module_from_checkpoint(ckpt)
    model = model.to(a.device).eval()
    tok = load_fast_tokenizer(cfg.model.model_name)
    ren = RowTextRenderer(cfg.data)
    qtexts = [ren.render_query_text({"query_term": t}) for t in terms]
    Mq = encode_splade(model, tok, qtexts, int(cfg.data.max_query_length), a.device, bs=256)
    dtexts = [ren.render_offer_text({**r, "description": ""}) for r in docrows]
    Md = prune_topk_rows(encode_splade(model, tok, dtexts,
                         int(cfg.data.max_offer_length), a.device, bs=128), a.topk)
    Q = Mq.toarray()

    by_seg = defaultdict(list)
    for ti, term in enumerate(terms):
        if term not in rel_E or not rel_E[term]:
            continue
        top = topk_rows(Md.dot(Q[ti]), a.k)
        by_seg[segmap.get(term, "?")].append(len(top & rel_E[term]) / len(rel_E[term]))
    print(f"=== per-segment recall@{a.k} (E) {a.tag} ===")
    allv = []
    for s in SEGS:
        v = by_seg.get(s, [])
        allv += v
        print(f"  {s:14s} n={len(v):>4}  R@{a.k}={np.mean(v):.4f}" if v else f"  {s:14s} n=0")
    print(f"  {'OVERALL':14s} n={len(allv):>4}  R@{a.k}={np.mean(allv):.4f}")


if __name__ == "__main__":
    main()

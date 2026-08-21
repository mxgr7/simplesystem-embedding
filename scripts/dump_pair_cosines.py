"""Compute dense (useful-cub-58) query-doc cosines for an explicit pair list —
sem_cosine fill for pairs whose prod vectors are missing in CH. Single-vector
cosine with the standard query:/passage:+description template (serve-equivalent
for docs with one embedding).

  python scripts/dump_pair_cosines.py --pairs data/union_sem_missing.tsv \
    --model models/useful-cub-58-st --gold data/persegment_eval_desc.jsonl \
    --dist data/desc_distractors.jsonl --out data/union_sem_fill.tsv
"""
import argparse
import time

import numpy as np
from omegaconf import OmegaConf

from embedding_train.rendering import RowTextRenderer
from splade_flops_stoplist import load_rows
from dense_seg_recall import DenseEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--dist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-cfg", default="configs/data/default.yaml")
    ap.add_argument("--max-seq", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    # rsplit: raw log terms can contain literal tabs; article_ids never do
    pairs = [l.rstrip("\n").rsplit("\t", 1) for l in open(a.pairs) if l.strip()]
    need_terms = sorted({t for t, _ in pairs})
    need_docs = {d for _, d in pairs}
    rows = {}
    for r in load_rows(a.gold) + load_rows(a.dist):
        if r["offer_id_b64"] in need_docs and r["offer_id_b64"] not in rows:
            rows[r["offer_id_b64"]] = r
    print(f"{len(pairs):,} pairs | {len(need_terms):,} terms | "
          f"{len(rows):,}/{len(need_docs):,} docs found", flush=True)

    ren = RowTextRenderer(OmegaConf.load(a.data_cfg))
    enc = DenseEncoder(a.model, a.device, a.max_seq)
    t0 = time.time()
    doc_ids = list(rows)
    D = enc.encode([ren.render_offer_text(rows[d]) for d in doc_ids])
    Q = enc.encode([ren.render_query_text({"query_term": t}) for t in need_terms])
    print(f"encoded ({time.time()-t0:.0f}s)", flush=True)
    drow = {d: i for i, d in enumerate(doc_ids)}
    qrow = {t: i for i, t in enumerate(need_terms)}
    import json
    with open(a.out, "w") as f:
        for t, d in pairs:
            if d in drow:
                f.write(json.dumps({"term": t, "article_id": d,
                    "cosine": round(float(D[drow[d]] @ Q[qrow[t]]), 6)}) + "\n")
    print(f"DONE {a.out}", flush=True)


if __name__ == "__main__":
    main()

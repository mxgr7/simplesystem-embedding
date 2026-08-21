"""SPLADE (soup_b50raw) query-doc scores for an explicit pair list (JSONL:
{"term","article_id"}). Deploy config: description-free render, top-256 pruned doc
vectors, raw dot product. Doc rows from one or more row files (offer_id_b64 keyed).

  python scripts/dump_pair_splade_scores.py --ckpt checkpoints/soup_b50raw.ckpt \
    --pairs data/gold_pairs.jsonl --rows data/gold_pair_rows.jsonl \
    --out data/gold_splade_scores.jsonl
"""
import argparse
import glob
import json
import time

from embedding_train.model import load_embedding_module_from_checkpoint
from embedding_train.rendering import RowTextRenderer
from embedding_train.tokenization import load_fast_tokenizer
from splade_flops_stoplist import load_rows, encode_splade, prune_topk_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    pairs = [json.loads(l) for l in open(a.pairs) if l.strip()]
    need_docs = {p["article_id"] for p in pairs}
    rows = {}
    for path in a.rows:
        for r in load_rows(path):
            if r["offer_id_b64"] in need_docs and r["offer_id_b64"] not in rows:
                rows[r["offer_id_b64"]] = r
    terms = sorted({p["term"] for p in pairs})
    print(f"{len(pairs):,} pairs | {len(terms):,} terms | "
          f"{len(rows):,}/{len(need_docs):,} docs found", flush=True)

    model, cfg = load_embedding_module_from_checkpoint(sorted(glob.glob(a.ckpt))[0])
    model = model.to(a.device).eval()
    tok = load_fast_tokenizer(cfg.model.model_name)
    ren = RowTextRenderer(cfg.data)
    t0 = time.time()
    doc_ids = list(rows)
    Md = prune_topk_rows(encode_splade(model, tok,
          [ren.render_offer_text({**rows[d], "description": ""}) for d in doc_ids],
          int(cfg.data.max_offer_length), a.device, bs=128), a.topk)
    Q = encode_splade(model, tok, [ren.render_query_text({"query_term": t}) for t in terms],
                      int(cfg.data.max_query_length), a.device, bs=256).toarray()
    print(f"encoded ({time.time()-t0:.0f}s)", flush=True)
    drow = {d: i for i, d in enumerate(doc_ids)}
    qrow = {t: i for i, t in enumerate(terms)}
    with open(a.out, "w") as f:
        for p in pairs:
            d = p["article_id"]
            if d in drow:
                s = float(Md[drow[d]].dot(Q[qrow[p["term"]]]).ravel()[0])
                f.write(json.dumps({"term": p["term"], "article_id": d,
                                    "splade_score": round(s, 4)}) + "\n")
    print(f"DONE {a.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

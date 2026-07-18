"""Encode the recall-study universe on the box: dense (useful-cub-58) doc vectors
for gold-test + distractor articles and query vectors for the terms, plus SPLADE
(capable-auk-759) doc vectors for the distractors. SPLADE gold-article + query
vectors already exist (splade_poc_*.jsonl) and are reused offline.

Outputs:
  dense_docs.npz   : ids (str[]), vecs (float16 [N,128], L2-normalized by encode)
  dense_queries.npz: terms (str[]), vecs (float16 [T,128])
  splade_distractors.jsonl : {article_id, splade: {token_id: weight}}
"""
import argparse
import json
import time

import numpy as np
import torch

from embedding_train.model import load_embedding_module_from_checkpoint
from embedding_train.rendering import RowTextRenderer
from embedding_train.tokenization import load_fast_tokenizer


@torch.inference_mode()
def encode(model, tokenizer, texts, max_len, device, bs=256):
    out = []
    for i in range(0, len(texts), bs):
        inp = tokenizer(texts[i:i + bs], padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")
        out.append(model.encode({k: v.to(device) for k, v in inp.items()}).float().cpu())
    return torch.cat(out, 0)


def load_articles(path, dedup):
    rows, seen = [], set()
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        aid = r["offer_id_b64"]
        if dedup and aid in seen:
            continue
        seen.add(aid)
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-ckpt", required=True)
    ap.add_argument("--splade-ckpt", required=True)
    ap.add_argument("--eval-jsonl", default="data/esci_gold_eval.jsonl")
    ap.add_argument("--distractors", default="data/recall_distractors.jsonl")
    ap.add_argument("--out-prefix", default="data/recall_")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    gold = load_articles(args.eval_jsonl, dedup=True)
    dist = load_articles(args.distractors, dedup=True)
    terms = sorted({json.loads(l)["query_term"] for l in open(args.eval_jsonl) if l.strip()})
    print(f"gold={len(gold)} distractors={len(dist)} terms={len(terms)}", flush=True)

    # --- dense (useful-cub-58): doc vectors (gold+distractor) + query vectors ---
    t0 = time.time()
    dmodel, dcfg = load_embedding_module_from_checkpoint(args.dense_ckpt)
    dmodel = dmodel.to(args.device).eval()
    dtok = load_fast_tokenizer(dcfg.model.model_name)
    dren = RowTextRenderer(dcfg.data)
    all_docs = gold + dist
    doc_texts = [dren.render_offer_text(r) for r in all_docs]
    q_texts = [dren.render_query_text({"query_term": t}) for t in terms]
    dvecs = encode(dmodel, dtok, doc_texts, int(dcfg.data.max_offer_length), args.device)
    qvecs = encode(dmodel, dtok, q_texts, int(dcfg.data.max_query_length), args.device)
    ids = np.array([r["offer_id_b64"] for r in all_docs])
    np.savez(f"{args.out_prefix}dense_docs.npz", ids=ids, vecs=dvecs.half().numpy())
    np.savez(f"{args.out_prefix}dense_queries.npz",
             terms=np.array(terms), vecs=qvecs.half().numpy())
    print(f"dense done: {len(all_docs)} docs + {len(terms)} queries, dim={dvecs.shape[1]}, "
          f"{time.time()-t0:.0f}s", flush=True)
    del dmodel
    torch.cuda.empty_cache()

    # --- SPLADE (capable-auk-759): distractor doc vectors ---
    t0 = time.time()
    smodel, scfg = load_embedding_module_from_checkpoint(args.splade_ckpt)
    smodel = smodel.to(args.device).eval()
    stok = load_fast_tokenizer(scfg.model.model_name)
    sren = RowTextRenderer(scfg.data)
    d_texts = [sren.render_offer_text(r) for r in dist]
    svecs = encode(smodel, stok, d_texts, int(scfg.data.max_offer_length), args.device, bs=128)
    with open(f"{args.out_prefix}splade_distractors.jsonl", "w") as f:
        for r, v in zip(dist, svecs):
            idx = torch.nonzero(v > 0, as_tuple=False).flatten().tolist()
            f.write(json.dumps({"article_id": r["offer_id_b64"],
                                "splade": {str(i): float(v[i]) for i in idx}}) + "\n")
    print(f"splade distractors done: {len(dist)}, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""Evaluate an embedding checkpoint (dense or SPLADE) on the ESCI gold TEST
split by re-ranking each term's judged candidate pool.

Loads the checkpoint's own cfg, so every article is rendered with the template
the model was trained on (e5 `query:`/`passage:` prefixes for the dense model,
prefix-free German for SPLADE). Scores are dot products of encode() outputs —
identical to the model's own validation_similarity. Metrics mirror
pipeline/esci_serp_eval.py exactly (condensed judged-only lists, Amazon-ESCI
gains E=1/S=0.1/C=0.01/I=0, NDCG@10 vs term-pool ideal, first-E MRR,
E-capped recall@10) so numbers are comparable across the whole ESCI thread.

Usage (on the GPU box):
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1 uv run python scripts/esci_gold_eval.py \
    --checkpoint ~/checkpoints/embedding/useful-cub-58/best-*.ckpt \
    --eval-jsonl data/esci_gold_eval.jsonl --name dense --output data/esci_gold_dense.json
"""

import argparse
import json
import math
import time

import torch

from embedding_train.model import load_embedding_module_from_checkpoint
from embedding_train.rendering import RowTextRenderer
from embedding_train.tokenization import load_fast_tokenizer

GAIN = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
K = 10


# --- metrics (vendored verbatim from pipeline/esci_serp_eval.py) ------------
def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def eval_thread(ranked_ids, pool):
    """ranked_ids: article ids sorted by model score desc; pool: {aid: label}."""
    condensed = [pool[aid] for aid in ranked_ids if aid in pool]
    n_rel_e = sum(1 for l in pool.values() if l == "E")
    out = {"n_pool": len(pool), "n_rel_e": n_rel_e,
           "ndcg10": None, "mrr_e": None, "recall10_e": None}
    if not condensed or n_rel_e == 0:
        return out
    ideal = sorted((GAIN[l] for l in pool.values()), reverse=True)[:K]
    out["ndcg10"] = dcg([GAIN[l] for l in condensed[:K]]) / dcg(ideal)
    first_e = next((i for i, l in enumerate(condensed) if l == "E"), None)
    out["mrr_e"] = 0.0 if first_e is None else 1.0 / (first_e + 1)
    out["recall10_e"] = sum(1 for l in condensed[:K] if l == "E") / min(n_rel_e, K)
    return out


# --- encoding ---------------------------------------------------------------
@torch.inference_mode()
def encode_texts(model, tokenizer, texts, max_length, device, batch_size=128):
    reps = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        inputs = tokenizer(chunk, padding=True, truncation=True,
                           max_length=max_length, return_tensors="pt")
        rep = model.encode({k: v.to(device) for k, v in inputs.items()})
        reps.append(rep.float().cpu())
    return torch.cat(reps, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--name", required=True, help="label for this run (dense/splade)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    t0 = time.time()
    rows = [json.loads(line) for line in open(args.eval_jsonl) if line.strip()]
    model, cfg = load_embedding_module_from_checkpoint(args.checkpoint)
    model = model.to(args.device).eval()
    tokenizer = load_fast_tokenizer(cfg.model.model_name)
    renderer = RowTextRenderer(cfg.data)
    print(f"[{args.name}] {len(rows):,} pairs | model={cfg.model.model_name} "
          f"| arch={cfg.data.get('architecture', 'dense') or 'dense'}")

    # Unique query terms and unique articles -> encode each once.
    query_terms = sorted({r["query_term"] for r in rows})
    q_idx = {t: i for i, t in enumerate(query_terms)}
    q_texts = [renderer.render_query_text({"query_term": t}) for t in query_terms]

    art_rows, a_idx, seen = [], {}, {}
    for r in rows:
        aid = r["offer_id_b64"]
        if aid not in seen:
            seen[aid] = len(art_rows)
            art_rows.append(r)
    a_idx = seen
    a_texts = [renderer.render_offer_text(r) for r in art_rows]

    q_emb = encode_texts(model, tokenizer, q_texts,
                         int(cfg.data.max_query_length), args.device, args.batch_size)
    a_emb = encode_texts(model, tokenizer, a_texts,
                         int(cfg.data.max_offer_length), args.device, args.batch_size)
    print(f"[{args.name}] encoded {len(q_texts):,} queries + {len(a_texts):,} "
          f"articles in {time.time()-t0:.0f}s | dim={q_emb.shape[1]}")

    # Group pool + score per term.
    pools = {}
    for r in rows:
        pools.setdefault(r["query_term"], []).append(
            (r["offer_id_b64"], r["label"]))

    records = []
    for term, items in pools.items():
        pool = {aid: label for aid, label in items}
        qv = q_emb[q_idx[term]]
        aids = list(pool.keys())
        av = a_emb[[a_idx[aid] for aid in aids]]
        scores = (av @ qv)
        order = torch.argsort(scores, descending=True).tolist()
        ranked_ids = [aids[i] for i in order]
        m = eval_thread(ranked_ids, pool)
        m["term"] = term
        records.append(m)

    elig = [r for r in records if r["ndcg10"] is not None]
    def macro(metric):
        return sum(r[metric] for r in elig) / len(elig) if elig else None
    summary = {
        "name": args.name,
        "checkpoint": args.checkpoint,
        "n_terms": len(records),
        "n_eligible": len(elig),
        "ndcg10": macro("ndcg10"),
        "mrr_e": macro("mrr_e"),
        "recall10_e": macro("recall10_e"),
    }
    print(f"[{args.name}] eligible terms={len(elig)}/{len(records)} | "
          f"NDCG@10={summary['ndcg10']:.4f} MRR={summary['mrr_e']:.4f} "
          f"recall@10={summary['recall10_e']:.4f}")

    with open(args.output, "w") as f:
        json.dump({"summary": summary, "records": records}, f)
    print(f"[{args.name}] wrote {args.output} in {time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()

"""Quick check: does top-k doc pruning hurt SPLADE recall? (run before the big encode)

Encodes the gold recall universe (gold-test pool + described distractors) with
capable-auk-759 description-free, then scores full-catalog recall with the DOC
vectors pruned to top-k in {full, 512, 256, 128}. Query vectors are left full
(query-side is pruning-sensitive; we only prune docs). Tells us the safe k.
"""
import argparse
import json
import time

import numpy as np
import torch

from embedding_train.model import load_embedding_module_from_checkpoint
from embedding_train.rendering import RowTextRenderer
from embedding_train.tokenization import load_fast_tokenizer

DEV = "cuda"
KS = [10, 50, 100]


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


@torch.inference_mode()
def encode(model, tok, texts, ml, bs=512):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=ml,
                  return_tensors="pt")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            r = model.encode({k: v.to(DEV) for k, v in enc.items()})
        out.append(r.float())
    return torch.cat(out, 0)


def prune_topk(reps, k):
    if k is None:
        return reps
    vals, idx = torch.topk(reps, min(k, reps.shape[1]), dim=1)
    pruned = torch.zeros_like(reps)
    pruned.scatter_(1, idx, torch.clamp(vals, min=0))
    return pruned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splade-ckpt", required=True)
    ap.add_argument("--gold", default="data/esci_gold_eval_desc.jsonl")
    ap.add_argument("--dist", default="data/desc_distractors.jsonl")
    args = ap.parse_args()

    model, cfg = load_embedding_module_from_checkpoint(args.splade_ckpt)
    model = model.to(DEV).eval()
    tok = load_fast_tokenizer(cfg.model.model_name)
    ren = RowTextRenderer(cfg.data)
    ml = int(cfg.data.max_offer_length)

    gold = load(args.gold); dist = load(args.dist)
    seen, docs = set(), []
    for r in gold + dist:
        a = r["offer_id_b64"]
        if a in seen:
            continue
        seen.add(a); docs.append(r)
    idx_of = {r["offer_id_b64"]: i for i, r in enumerate(docs)}
    terms = sorted({r["query_term"] for r in gold})
    relE, relES = {}, {}
    for r in gold:
        i = idx_of[r["offer_id_b64"]]
        if r["label"] == "E":
            relE.setdefault(r["query_term"], set()).add(i)
        if r["label"] in ("E", "S"):
            relES.setdefault(r["query_term"], set()).add(i)

    t0 = time.time()
    dtexts = [ren.render_offer_text({**r, "description": ""}) for r in docs]
    qtexts = [ren.render_query_text({"query_term": t}) for t in terms]
    D = encode(model, tok, dtexts, ml)          # [N, vocab]
    Q = encode(model, tok, qtexts, int(cfg.data.max_query_length))
    print(f"encoded {len(docs):,} docs + {len(terms)} queries in {time.time()-t0:.0f}s", flush=True)

    for k in (None, 512, 256, 128):
        Dp = prune_topk(D, k)
        aggE = {kk: [] for kk in KS}; aggES = {kk: [] for kk in KS}
        for qi, term in enumerate(terms):
            scores = (Dp @ Q[qi]).cpu().numpy()
            order = np.argpartition(-scores, max(KS))[:max(KS)]
            order = order[np.argsort(-scores[order])]
            for kk in KS:
                top = set(order[:kk].tolist())
                if term in relE:
                    aggE[kk].append(len(top & relE[term]) / len(relE[term]))
                if term in relES:
                    aggES[kk].append(len(top & relES[term]) / len(relES[term]))
        nnz = float((Dp > 0).sum(1).float().mean())
        label = "full" if k is None else f"top{k}"
        print(f"{label:>6} nnz~{nnz:>4.0f} | "
              f"E@10 {np.mean(aggE[10]):.4f} @50 {np.mean(aggE[50]):.4f} @100 {np.mean(aggE[100]):.4f} | "
              f"ES@100 {np.mean(aggES[100]):.4f}", flush=True)


if __name__ == "__main__":
    main()

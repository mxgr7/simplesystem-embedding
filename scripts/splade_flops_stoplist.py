"""Measure our SPLADE's real query x doc FLOPS + the stopword-mask recall/FLOPS
tradeoff, on the gold universe (gold-test pool + described distractors), rendered
DESCRIPTION-FREE (the deploy config).

FLOPS metric (SPLADE / Paria et al.): expected float-multiplies per query-doc pair
  weighted:  sum_j  mean_q(a_j) * mean_d(a_j)      <- comparable to SPLADE-v3's ~1.2
  binary:    sum_j  P_q(a_j>0)   * P_d(a_j>0)
Stopword mask = zero the K vocab dims most frequently active across DOCS (data-driven;
the always-on stopwords/punctuation/subword pieces). Applied to BOTH sides.

Reports, per config {full, top256, top256+mask@K}: doc nnz, query nnz, FLOPS(w/bin),
recall@{10,100} (E and E+S), macro over terms.

  PYTORCH_ALLOC_CONF=expandable_segments:True LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1 \
    uv run --extra train python scripts/splade_flops_stoplist.py \
      --splade-ckpt checkpoints/capable-auk-759/best-*.ckpt
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import time

import numpy as np
import scipy.sparse as sp
import torch

from embedding_train.model import load_embedding_module_from_checkpoint
from embedding_train.rendering import RowTextRenderer
from embedding_train.tokenization import load_fast_tokenizer

KS = [10, 100]
MASK_LEVELS = [0, 20, 50, 100, 200]
N_WORKERS = max(1, min(32, (os.cpu_count() or 8) - 2))

# fork-shared read-only state for the worker pools (copy-on-write; workers
# never mutate). Set before Pool creation, read inside workers.
_G = {}


def _render_doc(row):
    return _G["ren"].render_offer_text({**row, "description": ""})


def _recall_chunk(term_indices):
    Md, Q = _G["Md"], _G["Q"]
    terms, rel_E, rel_ES = _G["terms"], _G["rel_E"], _G["rel_ES"]
    out = []
    for ti in term_indices:
        term = terms[ti]
        scores = Md.dot(Q[ti])
        tops = {k: topk_rows(scores, k) for k in KS}
        e = ({k: len(tops[k] & rel_E[term]) / len(rel_E[term]) for k in KS}
             if rel_E.get(term) else None)
        es = ({k: len(tops[k] & rel_ES[term]) / len(rel_ES[term]) for k in KS}
              if rel_ES.get(term) else None)
        out.append((ti, e, es))
    return out


def load_rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


@torch.inference_mode()
def encode_splade(model, tok, texts, max_len, device, bs=128):
    mats = []
    for i in range(0, len(texts), bs):
        inp = tok(texts[i:i + bs], padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt")
        rep = model.encode({k: v.to(device) for k, v in inp.items()}).float().cpu().numpy()
        rep[rep < 0] = 0.0
        mats.append(sp.csr_matrix(rep))
    return sp.vstack(mats).tocsr()


def prune_topk_rows(M, k):
    """Return CSR with only the top-k entries (by value) kept per row."""
    M = M.tocsr()
    data, indices, indptr = M.data, M.indices, M.indptr
    nd, ni, npt = [], [], [0]
    for r in range(M.shape[0]):
        s, e = indptr[r], indptr[r + 1]
        d, idx = data[s:e], indices[s:e]
        if len(d) > k:
            keep = np.argpartition(-d, k)[:k]
            d, idx = d[keep], idx[keep]
        nd.append(d); ni.append(idx); npt.append(npt[-1] + len(d))
    return sp.csr_matrix((np.concatenate(nd), np.concatenate(ni), np.array(npt)),
                         shape=M.shape)


def zero_cols(M, cols):
    if len(cols) == 0:
        return M
    M = M.tocsc()
    for c in cols:
        M.data[M.indptr[c]:M.indptr[c + 1]] = 0.0
    M.eliminate_zeros()
    return M.tocsr()


def flops(M_q, M_d):
    qmean = np.asarray(M_q.mean(axis=0)).ravel()
    dmean = np.asarray(M_d.mean(axis=0)).ravel()
    qbin = np.asarray((M_q > 0).mean(axis=0)).ravel()
    dbin = np.asarray((M_d > 0).mean(axis=0)).ravel()
    return float((qmean * dmean).sum()), float((qbin * dbin).sum())


def topk_rows(scores, k):
    if k >= len(scores):
        return set(np.argsort(-scores)[:k].tolist())
    idx = np.argpartition(-scores, k)[:k]
    return set(idx[np.argsort(-scores[idx])].tolist())


def macro_recall(M_d, M_q, terms, rel_E, rel_ES):
    """Parallel over terms via fork pool; per-term results reassembled in term
    order, so the means are bit-identical to the old sequential loop."""
    _G.update(Md=M_d.tocsr(), Q=M_q.toarray(), terms=terms,
              rel_E=rel_E, rel_ES=rel_ES)
    chunks = [list(range(i, min(i + 32, len(terms))))
              for i in range(0, len(terms), 32)]
    per_term = [None] * len(terms)
    with mp.get_context("fork").Pool(N_WORKERS) as pool:
        for res in pool.imap_unordered(_recall_chunk, chunks):
            for ti, e, es in res:
                per_term[ti] = (e, es)
    accE = {k: [p[0][k] for p in per_term if p and p[0]] for k in KS}
    accES = {k: [p[1][k] for p in per_term if p and p[1]] for k in KS}
    return ({k: float(np.mean(accE[k])) for k in KS},
            {k: float(np.mean(accES[k])) for k in KS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splade-ckpt", required=True)
    ap.add_argument("--gold", default="data/esci_gold_eval_desc.jsonl")
    ap.add_argument("--dist", default="data/desc_distractors.jsonl")
    ap.add_argument("--out", default="splade_flops_stoplist.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    ckpt = sorted(glob.glob(args.splade_ckpt))[0]

    gold = load_rows(args.gold)
    dist = load_rows(args.dist)
    seen, docrows = set(), []
    for r in gold + dist:
        aid = r["offer_id_b64"]
        if aid not in seen:
            seen.add(aid); docrows.append(r)
    id_to_row = {r["offer_id_b64"]: i for i, r in enumerate(docrows)}
    N = len(docrows)
    terms = sorted({r["query_term"] for r in gold})
    rel_E, rel_ES = {}, {}
    for r in gold:
        row = id_to_row.get(r["offer_id_b64"])
        if row is None:
            continue
        if r["label"] == "E":
            rel_E.setdefault(r["query_term"], set()).add(row)
        if r["label"] in ("E", "S"):
            rel_ES.setdefault(r["query_term"], set()).add(row)
    print(f"universe N={N:,} terms={len(terms)} E={len(rel_E)} ES={len(rel_ES)}", flush=True)

    model, cfg = load_embedding_module_from_checkpoint(ckpt)
    model = model.to(args.device).eval()
    tok = load_fast_tokenizer(cfg.model.model_name)
    ren = RowTextRenderer(cfg.data)

    t0 = time.time()
    qtexts = [ren.render_query_text({"query_term": t}) for t in terms]
    Mq = encode_splade(model, tok, qtexts, int(cfg.data.max_query_length), args.device, bs=256)
    _G["ren"] = ren
    with mp.get_context("fork").Pool(N_WORKERS) as pool:
        dtexts = pool.map(_render_doc, docrows, chunksize=512)  # deploy: desc-free
    print(f"rendered {len(dtexts):,} docs in {time.time()-t0:.0f}s "
          f"({N_WORKERS} workers)", flush=True)
    Md = encode_splade(model, tok, dtexts, int(cfg.data.max_offer_length), args.device, bs=128)
    print(f"encoded q={Mq.shape} d={Md.shape} in {time.time()-t0:.0f}s", flush=True)

    # rank vocab dims by doc activation frequency -> stopword mask candidates
    dbin = np.asarray((Md > 0).mean(axis=0)).ravel()
    order = np.argsort(-dbin)

    result = {"universe": N, "n_terms": len(terms), "vocab": int(Md.shape[1]),
              "query_nnz_full": float(Mq.nnz / Mq.shape[0]),
              "doc_nnz_full": float(Md.nnz / Md.shape[0]), "configs": []}

    def report(tag, Md_c, Mq_c):
        fw, fb = flops(Mq_c, Md_c)
        rE, rES = macro_recall(Md_c, Mq_c, terms, rel_E, rel_ES)
        row = {"config": tag, "doc_nnz": float(Md_c.nnz / Md_c.shape[0]),
               "query_nnz": float(Mq_c.nnz / Mq_c.shape[0]),
               "flops_weighted": fw, "flops_binary": fb,
               "recallE": {str(k): rE[k] for k in KS},
               "recallES": {str(k): rES[k] for k in KS}}
        result["configs"].append(row)
        print(f"{tag:16s} docnnz={row['doc_nnz']:6.1f} qnnz={row['query_nnz']:5.1f} "
              f"FLOPS_w={fw:6.2f} FLOPS_b={fb:6.2f} "
              f"R@100 E={rE[100]:.4f} ES={rES[100]:.4f} R@10 E={rE[10]:.4f}", flush=True)

    report("full", Md, Mq)
    Md256 = prune_topk_rows(Md, 256)
    report("top256", Md256, Mq)
    for k in MASK_LEVELS:
        if k == 0:
            continue
        cols = order[:k]
        report(f"top256+mask{k}", zero_cols(Md256, cols), zero_cols(Mq.copy(), cols))
    # also: mask on full (model-native) to see best-case FLOPS drop
    for k in (50, 100):
        report(f"full+mask{k}", zero_cols(Md, order[:k]), zero_cols(Mq.copy(), order[:k]))

    result["top_doc_terms"] = [[int(j), float(dbin[j])] for j in order[:60]]
    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

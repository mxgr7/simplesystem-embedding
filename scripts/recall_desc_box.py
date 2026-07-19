"""Description-lift A/B on the box: encode one fixed universe (gold-test pool +
described distractors) TWICE per model — with vs without the `description` field —
and compute per-term full-catalog recall. Only a tiny per-term JSON comes back;
the giant sparse SPLADE reps never leave the box.

Four arms: {splade,dense} × {with,without} description. For each arm, per term,
recall@{10,50,100,400} against the gold E and E+S rel sets over the shared
universe. Also reports mean offer nnz (the density cost) for the two SPLADE arms.

Inputs (rsynced from the workspace pipeline/out):
  --gold  esci_gold_eval_desc.jsonl   (per-(term,article) rows, label + description)
  --dist  desc_distractors.jsonl      (unique described non-test articles)

Output: recall_desc_per_term.json  (aggregation + language slices done offline).

Run on the H100 box (ask first — shared):
  PYTORCH_ALLOC_CONF=expandable_segments:True uv run python scripts/recall_desc_box.py \
      --dense-ckpt data/useful-cub-58.ckpt --splade-ckpt data/capable-auk-759.ckpt
"""
import argparse
import json
import time

import numpy as np
import scipy.sparse as sp
import torch

from embedding_train.model import load_embedding_module_from_checkpoint
from embedding_train.rendering import RowTextRenderer
from embedding_train.tokenization import load_fast_tokenizer

KS = [10, 50, 100, 400]


def load_rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def render_docs(renderer, rows, drop_desc):
    texts = []
    for r in rows:
        if drop_desc:
            r = {**r, "description": ""}
        texts.append(renderer.render_offer_text(r))
    return texts


@torch.inference_mode()
def encode_dense(model, tok, texts, max_len, device, bs=256):
    out = []
    for i in range(0, len(texts), bs):
        inp = tok(texts[i:i + bs], padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt")
        out.append(model.encode({k: v.to(device) for k, v in inp.items()}).float().cpu())
    return torch.cat(out, 0).numpy().astype(np.float32)


@torch.inference_mode()
def encode_splade(model, tok, texts, max_len, device, bs=128):
    """-> scipy CSR [N, vocab], mean nnz/doc."""
    mats = []
    for i in range(0, len(texts), bs):
        inp = tok(texts[i:i + bs], padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt")
        rep = model.encode({k: v.to(device) for k, v in inp.items()}).float().cpu().numpy()
        rep[rep < 0] = 0.0
        mats.append(sp.csr_matrix(rep))
    M = sp.vstack(mats).tocsr()
    return M, M.nnz / M.shape[0]


def topk_rows(scores, k):
    if k >= len(scores):
        return set(np.argsort(-scores)[:k].tolist())
    idx = np.argpartition(-scores, k)[:k]
    return set(idx[np.argsort(-scores[idx])].tolist())


def recall_per_term(score_fn, terms, qvecs, rel_E, rel_ES):
    """score_fn(term_idx)->scores[N]; returns {term: {'E':{K:r},'ES':{K:r}}}."""
    out = {}
    for ti, term in enumerate(terms):
        scores = score_fn(ti)
        tops = {k: topk_rows(scores, k) for k in KS}
        rec = {}
        for tag, rel in (("E", rel_E), ("ES", rel_ES)):
            rs = rel.get(term)
            if not rs:
                continue
            rec[tag] = {str(k): len(tops[k] & rs) / len(rs) for k in KS}
        out[term] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-ckpt", required=True)
    ap.add_argument("--splade-ckpt", required=True)
    ap.add_argument("--gold", default="data/esci_gold_eval_desc.jsonl")
    ap.add_argument("--dist", default="data/desc_distractors.jsonl")
    ap.add_argument("--out", default="recall_desc_per_term.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    gold = load_rows(args.gold)
    dist = load_rows(args.dist)

    # universe doc list: unique gold articles (first occurrence) + distractors
    seen, docrows = set(), []
    for r in gold + dist:
        aid = r["offer_id_b64"]
        if aid in seen:
            continue
        seen.add(aid)
        docrows.append(r)
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
    print(f"universe N={N:,}, terms={len(terms)}, "
          f"terms with E={len(rel_E)}, ES={len(rel_ES)}", flush=True)

    result = {"universe": N, "n_terms": len(terms), "ks": KS,
              "per_term": {}, "nnz": {}}

    # ---- SPLADE ----
    t0 = time.time()
    smodel, scfg = load_embedding_module_from_checkpoint(args.splade_ckpt)
    smodel = smodel.to(args.device).eval()
    stok = load_fast_tokenizer(scfg.model.model_name)
    sren = RowTextRenderer(scfg.data)
    qtexts = [sren.render_query_text({"query_term": t}) for t in terms]
    sq = encode_splade(smodel, stok, qtexts, int(scfg.data.max_query_length),
                       args.device, bs=256)[0]
    for mode in ("with", "without"):
        dt = render_docs(sren, docrows, drop_desc=(mode == "without"))
        M, nnz = encode_splade(smodel, stok, dt, int(scfg.data.max_offer_length),
                               args.device, bs=128)
        result["nnz"][f"splade_{mode}"] = float(nnz)
        pt = recall_per_term(lambda ti: M.dot(sq[ti].toarray().ravel()),
                             terms, sq, rel_E, rel_ES)
        for term, rec in pt.items():
            result["per_term"].setdefault(term, {})[f"splade_{mode}"] = rec
        print(f"splade {mode}: nnz/doc {nnz:.0f}, {time.time()-t0:.0f}s", flush=True)
    del smodel, M
    torch.cuda.empty_cache()

    # ---- dense ----
    t0 = time.time()
    dmodel, dcfg = load_embedding_module_from_checkpoint(args.dense_ckpt)
    dmodel = dmodel.to(args.device).eval()
    dtok = load_fast_tokenizer(dcfg.model.model_name)
    dren = RowTextRenderer(dcfg.data)
    dqtexts = [dren.render_query_text({"query_term": t}) for t in terms]
    dq = encode_dense(dmodel, dtok, dqtexts, int(dcfg.data.max_query_length), args.device)
    for mode in ("with", "without"):
        dt = render_docs(dren, docrows, drop_desc=(mode == "without"))
        D = encode_dense(dmodel, dtok, dt, int(dcfg.data.max_offer_length), args.device)
        pt = recall_per_term(lambda ti: D @ dq[ti], terms, dq, rel_E, rel_ES)
        for term, rec in pt.items():
            result["per_term"].setdefault(term, {})[f"dense_{mode}"] = rec
        print(f"dense {mode}: {time.time()-t0:.0f}s", flush=True)

    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

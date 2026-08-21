"""Test the folded-vocab OUTPUT MASK lever on a folded SPLADE checkpoint.

Hypothesis: a model trained on folded (lowercase, diacritic-stripped) INPUT still
activates cased/diacritic vocab dimensions on the OUTPUT side (the MLM head lives in
the full cased gbert vocab). Many are redundant case-twins of a lowercase token
(System + system, Abs + abs). Zeroing every cased/diacritic output dimension — exactly
parallel to the existing special_token_vocab_mask and the stopword-mask lever — should
cut nnz (FLOPS) at ~0 retrieval cost.

This re-ranks each gold-TEST-v2.1 term's judged pool with and without that mask and
reports the trusted esci_gold_eval.py metrics (condensed judged-only, gains
E=1/S=.1/C=.01/I=0, NDCG@10 / first-E MRR / E-capped recall@10) plus mean nnz and the
score-mass the mask removes.

Because the mask m is 0/1, (a·m)·(q·m) = a·(q·m): masking the query alone gives the
masked dot, so we reuse one dense doc-embedding matrix for both variants.

Run on the box (folded input is applied here, like splade_viz_encode.py --fold):
  uv run python scripts/splade_cased_mask_eval.py \
    --checkpoint checkpoints/fold_raw/best-*.ckpt --name fold_raw \
    --eval-jsonl data/esci_gold_eval_v21_sink.jsonl
"""
import argparse
import glob
import json
import math
import re
import time
import unicodedata

import torch

from embedding_train.model import load_embedding_module_from_checkpoint
from embedding_train.rendering import RowTextRenderer
from embedding_train.tokenization import load_fast_tokenizer
from fold_text import fold

GAIN = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
K = 10


def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def eval_thread(ranked_ids, pool):
    condensed = [pool[aid] for aid in ranked_ids if aid in pool]
    n_rel_e = sum(1 for l in pool.values() if l == "E")
    out = {"ndcg10": None, "mrr_e": None, "recall10_e": None}
    if not condensed or n_rel_e == 0:
        return out
    ideal = sorted((GAIN[l] for l in pool.values()), reverse=True)[:K]
    out["ndcg10"] = dcg([GAIN[l] for l in condensed[:K]]) / dcg(ideal)
    first_e = next((i for i, l in enumerate(condensed) if l == "E"), None)
    out["mrr_e"] = 0.0 if first_e is None else 1.0 / (first_e + 1)
    out["recall10_e"] = sum(1 for l in condensed[:K] if l == "E") / min(n_rel_e, K)
    return out


def is_cased(tok):
    """Token string carries uppercase or a diacritic (i.e. folding would change it)."""
    s = tok.replace("##", "")
    if re.search(r"[A-ZÄÖÜ]", s):
        return True
    return any(unicodedata.combining(c) for c in unicodedata.normalize("NFD", s))


@torch.inference_mode()
def encode(model, tok, texts, max_len, device, bs=128):
    # Keep the full [n, vocab] matrix on CPU (36k x 31k x4 = 4.5GB); only one
    # batch's dense logits ever live on the GPU, so peak VRAM stays small.
    reps = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        inp = tok(chunk, padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt")
        rep = model.encode({k: v.to(device) for k, v in inp.items()}).float().cpu()
        reps.append(rep)
    return torch.cat(reps, 0)


def macro(records, key):
    elig = [r[key] for r in records if r[key] is not None]
    return sum(elig) / len(elig) if elig else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--no-fold", action="store_true",
                    help="skip folding the input (for cased checkpoints)")
    args = ap.parse_args()

    ckpt = sorted(glob.glob(args.checkpoint))
    ckpt = ([c for c in ckpt if "/best-" in c] or ckpt)[0]
    t0 = time.time()
    rows = [json.loads(l) for l in open(args.eval_jsonl) if l.strip()]
    model, cfg = load_embedding_module_from_checkpoint(ckpt)
    model = model.to(args.device).eval()
    tok = load_fast_tokenizer(cfg.model.model_name)
    ren = RowTextRenderer(cfg.data)
    do_fold = not args.no_fold

    query_terms = sorted({r["query_term"] for r in rows})
    q_idx = {t: i for i, t in enumerate(query_terms)}
    q_texts = [ren.render_query_text({"query_term": t}) for t in query_terms]

    art_rows, seen = [], {}
    for r in rows:
        aid = r["offer_id_b64"]
        if aid not in seen:
            seen[aid] = len(art_rows)
            art_rows.append(r)
    for r in art_rows:
        r["description"] = ""
    a_texts = [ren.render_offer_text(r) for r in art_rows]
    if do_fold:
        q_texts = [fold(t) for t in q_texts]
        a_texts = [fold(t) for t in a_texts]

    q_emb = encode(model, tok, q_texts, int(cfg.data.max_query_length),
                   args.device, args.batch_size)
    a_emb = encode(model, tok, a_texts, int(cfg.data.max_offer_length),
                   args.device, args.batch_size)
    vocab_dim = q_emb.shape[1]

    # cased/diacritic output-dimension mask (1 = keep, 0 = zero out); on CPU to
    # match the CPU-resident q_emb / a_emb.
    inv = {i: t for t, i in tok.get_vocab().items()}
    cased_ids = [i for i in range(vocab_dim) if i in inv and is_cased(inv[i])]
    keep = torch.ones(vocab_dim)
    keep[torch.tensor(cased_ids)] = 0.0
    n_cased = len(cased_ids)

    pools = {}
    for r in rows:
        pools.setdefault(r["query_term"], []).append((r["offer_id_b64"], r["label"]))

    def run(mask):
        recs = []
        qm = q_emb * mask if mask is not None else q_emb
        for term, items in pools.items():
            pool = {aid: lab for aid, lab in items}
            aids = list(pool.keys())
            av = a_emb[[seen[a] for a in aids]]
            scores = av @ qm[q_idx[term]]
            order = torch.argsort(scores, descending=True).tolist()
            recs.append(eval_thread([aids[i] for i in order], pool))
        return recs

    base = run(None)
    masked = run(keep)

    # nnz / mass stats on the doc side
    active = a_emb > 0
    nnz_full = active.sum(1).float().mean().item()
    nnz_masked = (active & (keep > 0)).sum(1).float().mean().item()
    mass_full = a_emb.sum().item()
    mass_kept = (a_emb * keep).sum().item()
    q_nnz_full = (q_emb > 0).sum(1).float().mean().item()
    q_nnz_masked = ((q_emb > 0) & (keep > 0)).sum(1).float().mean().item()

    def line(tag, recs):
        return (f"  {tag:9s} NDCG@10={macro(recs,'ndcg10'):.4f}  "
                f"MRR_e={macro(recs,'mrr_e'):.4f}  "
                f"recall@10_e={macro(recs,'recall10_e'):.4f}")

    print(f"\n=== {args.name} | folded_input={do_fold} | ckpt={ckpt.split('/')[-1]} ===")
    print(f"vocab {vocab_dim} | cased dims zeroed: {n_cased} ({100*n_cased/vocab_dim:.0f}%)")
    print(f"doc  nnz  {nnz_full:.1f} -> {nnz_masked:.1f} "
          f"({100*(nnz_full-nnz_masked)/nnz_full:.0f}% fewer)")
    print(f"query nnz {q_nnz_full:.1f} -> {q_nnz_masked:.1f}")
    print(f"doc activation mass kept after mask: {100*mass_kept/mass_full:.1f}%")
    print(line("baseline", base))
    print(line("masked", masked))
    d = macro(masked, "ndcg10") - macro(base, "ndcg10")
    print(f"  ΔNDCG@10 (masked - baseline) = {d:+.4f}")
    print(f"[{args.name}] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

"""SPLADE-viz precompute (box side).

Encode every unique gold-TEST-v2.1 query term and every unique article for a
single checkpoint into its sparse ``{token_id: weight}`` SPLADE vector, then dump
compact parquet + a vocab map for the local visualization server
(pipeline/splade_viz_server.py) to browse.

Rendering is deliberately identical to scripts/esci_gold_eval.py: the checkpoint's
own ``cfg.data`` renderer + tokenizer + ``max_query_length``/``max_offer_length``.
So the dot products the viz server computes from these sparse vectors match the
trusted gold-eval scores exactly (encode() -> log1p(relu(logits)) max-pool).

Outputs (into --out-dir, one set per checkpoint --name):
  <name>__query.parquet  columns: term (str), ids (list<int32>), weights (list<float32>)
  <name>__doc.parquet    columns: article_id (str), ids (list<int32>), weights (list<float32>)
  <name>__vocab.json     {token_id: token_str} for the whole model vocab
  manifest.json          merged registry {name: {model_name, arch, ndcg_tag, counts, max_len}}

Usage on the GPU box (one call per checkpoint):
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1 uv run --extra train \
    python scripts/splade_viz_encode.py \
      --checkpoint checkpoints/capable-auk-759/best-*.ckpt --name capable-auk-759 \
      --eval-jsonl data/esci_gold_eval_v21_sink.jsonl --out-dir out/splade_viz
"""

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd
import torch

from embedding_train.model import load_embedding_module_from_checkpoint
from embedding_train.rendering import RowTextRenderer
from embedding_train.tokenization import load_fast_tokenizer

# fold_text.py (canonical text fold) is shipped next to this script; import the
# single source of truth rather than reimplementing NFKD/casefold.
try:
    from fold_text import fold as fold_text
except ImportError:
    fold_text = None


@torch.inference_mode()
def encode_sparse(model, tokenizer, texts, max_length, device, batch_size=128,
                  tag="", log_every=20, is_query=False):
    """texts -> list of (ids np.int32, weights np.float32) keeping only w > 0.

    is_query routes to the untied query encoder when the checkpoint has one. For
    tied models it changes nothing. Without it, an untied model's queries would
    be encoded by the DOC encoder — silently wrong vectors that still look
    plausible, and these dumps feed pipeline/splade_df_metrics.py, i.e. the
    coverage/postings numbers the whole comparison rests on.
    """
    out = []
    t0 = time.time()
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for bi, i in enumerate(range(0, len(texts), batch_size)):
        chunk = texts[i:i + batch_size]
        inputs = tokenizer(chunk, padding=True, truncation=True,
                           max_length=max_length, return_tensors="pt")
        rep = model.encode({k: v.to(device) for k, v in inputs.items()},
                           is_query=is_query)
        rep = rep.float().cpu().numpy()
        for vec in rep:
            idx = np.nonzero(vec > 0.0)[0]
            out.append((idx.astype(np.int32), vec[idx].astype(np.float32)))
        if (bi + 1) % log_every == 0 or bi + 1 == n_batches:
            done = i + len(chunk)
            rate = done / max(time.time() - t0, 1e-9)
            eta = (len(texts) - done) / max(rate, 1e-9)
            print(f"  [{tag}] {done:,}/{len(texts):,} ({rate:.0f}/s, ETA {eta:.0f}s)",
                  flush=True)
    return out


def resolve_checkpoint(pattern):
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise SystemExit(f"no checkpoint matches: {pattern}")
    # Prefer a best-* checkpoint over last-* when the glob catches both.
    best = [m for m in matches if "/best-" in m or os.path.basename(m).startswith("best")]
    return (best or matches)[0]


def ndcg_tag(path):
    m = re.search(r"ndcg_at_5=([0-9]+\.[0-9]+)", path)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path or glob to a .ckpt")
    ap.add_argument("--name", required=True, help="short checkpoint id (dir label)")
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--out-dir", default="out/splade_viz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--keep-description", action="store_true",
                    help="render article descriptions (default: blank them, matching "
                         "the description-free SPLADE serving/eval convention)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap input rows for a quick smoke test (0 = all)")
    ap.add_argument("--fold", action="store_true",
                    help="apply the canonical STRIP text fold (fold_text) to "
                         "rendered query/offer text before tokenizing — for models "
                         "trained on strip-folded inputs")
    ap.add_argument("--fold-type", choices=["none", "strip", "de"], default=None,
                    help="record the model's fold regime in the manifest (server "
                         "folds live-encode input to match). 'de' = ä→ae; assumes "
                         "the eval jsonl is already folded (don't also pass --fold)")
    args = ap.parse_args()
    if args.fold and fold_text is None:
        raise SystemExit("--fold requested but fold_text.py not importable "
                         "(ship it next to this script)")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = resolve_checkpoint(args.checkpoint)
    t0 = time.time()

    rows = [json.loads(l) for l in open(args.eval_jsonl) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    model, cfg = load_embedding_module_from_checkpoint(ckpt)
    model = model.to(args.device).eval()
    tokenizer = load_fast_tokenizer(cfg.model.model_name)
    renderer = RowTextRenderer(cfg.data)
    arch = cfg.data.get("architecture", None) or cfg.model.get("architecture", "dense")
    print(f"[{args.name}] {len(rows):,} pairs | model={cfg.model.model_name} "
          f"| arch={arch} | ckpt={ckpt}", flush=True)
    if arch != "splade":
        print(f"[{args.name}] WARNING: architecture is '{arch}', not 'splade' — "
              f"vectors will be dense (not token-sparse).", flush=True)

    # Key queries by query_id (the stable join term) but render query_term (the
    # text fed to the model). For most sets query_id == query_term == term; for
    # pre-folded sets (e.g. German fold) query_id is the original term and
    # query_term is the folded text, so the viz server can still join on the term.
    q_render = {}
    for r in rows:
        key = r.get("query_id") or r["query_term"]
        q_render.setdefault(key, r["query_term"])
    query_terms = sorted(q_render)
    q_texts = [renderer.render_query_text({"query_term": q_render[t]}) for t in query_terms]

    art_rows, seen = [], {}
    for r in rows:
        aid = r["offer_id_b64"]
        if aid not in seen:
            seen[aid] = len(art_rows)
            art_rows.append(r)
    if not args.keep_description:
        # Description is dead weight in the SPLADE path — serving/eval blanks it.
        for r in art_rows:
            r["description"] = ""
    a_texts = [renderer.render_offer_text(r) for r in art_rows]

    if args.fold:
        # fold(render(row)) == render(fold-each-field(row)) for tokenization:
        # the fold is per-character and distributes over the template's joins.
        q_texts = [fold_text(t) for t in q_texts]
        a_texts = [fold_text(t) for t in a_texts]

    print(f"[{args.name}] encoding {len(q_texts):,} queries "
          f"(max_len={int(cfg.data.max_query_length)}) ...", flush=True)
    q_vecs = encode_sparse(model, tokenizer, q_texts, int(cfg.data.max_query_length),
                           args.device, args.batch_size, tag=f"{args.name} q", is_query=True)
    print(f"[{args.name}] encoding {len(a_texts):,} articles "
          f"(max_len={int(cfg.data.max_offer_length)}) ...", flush=True)
    a_vecs = encode_sparse(model, tokenizer, a_texts, int(cfg.data.max_offer_length),
                           args.device, args.batch_size, tag=f"{args.name} d")

    q_df = pd.DataFrame({
        "term": query_terms,
        "ids": [list(map(int, ids)) for ids, _ in q_vecs],
        "weights": [list(map(float, w)) for _, w in q_vecs],
    })
    d_df = pd.DataFrame({
        "article_id": [r["offer_id_b64"] for r in art_rows],
        "ids": [list(map(int, ids)) for ids, _ in a_vecs],
        "weights": [list(map(float, w)) for _, w in a_vecs],
    })
    q_path = os.path.join(args.out_dir, f"{args.name}__query.parquet")
    d_path = os.path.join(args.out_dir, f"{args.name}__doc.parquet")
    q_df.to_parquet(q_path, index=False)
    d_df.to_parquet(d_path, index=False)

    vocab = tokenizer.get_vocab()  # {token_str: id}
    inv = {int(i): t for t, i in vocab.items()}
    with open(os.path.join(args.out_dir, f"{args.name}__vocab.json"), "w") as f:
        json.dump(inv, f)

    q_nnz = float(np.mean([len(w) for _, w in q_vecs])) if q_vecs else 0.0
    d_nnz = float(np.mean([len(w) for _, w in a_vecs])) if a_vecs else 0.0
    # A fold_vocab_mask checkpoint zeroes its cased/diacritic output dims inside
    # encode(), so those dims are simply absent from the vectors above — unlike the
    # post-hoc "pruned" variants, which the server masks itself. Record it so the UI
    # can say WHY this model's nnz is a fraction of an unmasked sibling's.
    vmask = getattr(model, "special_token_vocab_mask", None)
    masked_dims = int((vmask == 0).sum().item()) if vmask is not None else 0
    man_path = os.path.join(args.out_dir, "manifest.json")
    manifest = {}
    if os.path.exists(man_path):
        manifest = json.load(open(man_path))
    manifest[args.name] = {
        "model_name": cfg.model.model_name,
        "arch": arch,
        "checkpoint": ckpt,
        "ndcg_at_5": ndcg_tag(ckpt),
        "n_queries": len(query_terms),
        "n_docs": len(art_rows),
        "vocab_size": len(inv),
        "max_query_length": int(cfg.data.max_query_length),
        "max_offer_length": int(cfg.data.max_offer_length),
        "mean_query_nnz": round(q_nnz, 1),
        "mean_doc_nnz": round(d_nnz, 1),
        "description_rendered": bool(args.keep_description),
        "folded": bool(args.fold) or (args.fold_type not in (None, "none")),
        "fold_type": args.fold_type or ("strip" if args.fold else "none"),
        "fold_vocab_mask": bool(cfg.model.get("fold_vocab_mask", False)),
        "masked_dims": masked_dims,
    }
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[{args.name}] wrote {q_path} + {d_path} + vocab | "
          f"q_nnz~{q_nnz:.0f} d_nnz~{d_nnz:.0f} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

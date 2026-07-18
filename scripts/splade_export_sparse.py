"""Export SPLADE sparse vectors ({token_id: weight}) for the gold-eval terms
and articles, so they can be indexed into an Elasticsearch `sparse_vector` field
and the ES scoring verified against the offline dot-product ranking.

Each article/term is rendered with the checkpoint's own template and encoded via
encode() (log1p(relu(MLM logits)), max-pool, special tokens masked, no L2 norm).
The non-zero dims are dumped keyed by stringified vocab-token id — the same key
convention used on both the doc and query side, so ES matches them exactly.

Usage (GPU box):
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1 uv run python scripts/splade_export_sparse.py \
    --checkpoint checkpoints/capable-auk-759/best-*.ckpt \
    --eval-jsonl data/esci_gold_eval.jsonl \
    --out-terms data/splade_poc_terms.jsonl --out-articles data/splade_poc_articles.jsonl
"""

import argparse
import json
import time

import torch

from embedding_train.model import load_embedding_module_from_checkpoint
from embedding_train.rendering import RowTextRenderer
from embedding_train.tokenization import load_fast_tokenizer


@torch.inference_mode()
def encode(model, tokenizer, texts, max_length, device, batch_size=128):
    reps = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        inputs = tokenizer(chunk, padding=True, truncation=True,
                           max_length=max_length, return_tensors="pt")
        rep = model.encode({k: v.to(device) for k, v in inputs.items()})
        reps.append(rep.float().cpu())
    return torch.cat(reps, dim=0)


def to_sparse(vec):
    """[vocab] tensor -> {str(token_id): float weight} over non-zero dims."""
    idx = torch.nonzero(vec > 0, as_tuple=False).flatten().tolist()
    return {str(i): float(vec[i]) for i in idx}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--out-terms", required=True)
    ap.add_argument("--out-articles", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    t0 = time.time()
    rows = [json.loads(l) for l in open(args.eval_jsonl) if l.strip()]
    model, cfg = load_embedding_module_from_checkpoint(args.checkpoint)
    model = model.to(args.device).eval()
    tokenizer = load_fast_tokenizer(cfg.model.model_name)
    renderer = RowTextRenderer(cfg.data)

    terms = sorted({r["query_term"] for r in rows})
    q_texts = [renderer.render_query_text({"query_term": t}) for t in terms]

    art_rows, seen = [], {}
    for r in rows:
        if r["offer_id_b64"] not in seen:
            seen[r["offer_id_b64"]] = len(art_rows)
            art_rows.append(r)
    a_texts = [renderer.render_offer_text(r) for r in art_rows]

    q_emb = encode(model, tokenizer, q_texts, int(cfg.data.max_query_length), args.device)
    a_emb = encode(model, tokenizer, a_texts, int(cfg.data.max_offer_length), args.device)
    print(f"encoded {len(terms)} terms + {len(art_rows)} articles in {time.time()-t0:.0f}s")

    q_nnz = a_nnz = 0
    with open(args.out_terms, "w") as f:
        for t, v in zip(terms, q_emb):
            sv = to_sparse(v)
            q_nnz += len(sv)
            f.write(json.dumps({"term": t, "splade": sv}) + "\n")
    with open(args.out_articles, "w") as f:
        for r, v in zip(art_rows, a_emb):
            sv = to_sparse(v)
            a_nnz += len(sv)
            f.write(json.dumps({"article_id": r["offer_id_b64"], "splade": sv}) + "\n")

    print(f"mean query nnz={q_nnz/len(terms):.1f}, mean article nnz={a_nnz/len(art_rows):.1f}")
    print(f"wrote {args.out_terms} + {args.out_articles}")


if __name__ == "__main__":
    main()

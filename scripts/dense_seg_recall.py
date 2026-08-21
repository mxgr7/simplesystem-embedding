"""Per-segment recall@K for the DENSE model (useful-cub-58 sentence-transformers
export) on the same held-out eval, using its OWN template (configs/data/default.yaml:
`query:`/`passage:` prefixes + description). Cosine (normalized) scoring. So it's a
fair head-to-head vs the SPLADE per-segment numbers.

  python3 scripts/dense_seg_recall.py --model models/useful-cub-58-st \
    --gold data/persegment_eval_desc.jsonl --dist data/desc_distractors.jsonl \
    --seg-map data/persegment_map.json
"""
import argparse
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoModel, AutoTokenizer

from embedding_train.rendering import RowTextRenderer
from splade_flops_stoplist import load_rows, topk_rows

SEGS = ["identifier", "id_plus_text", "text_spec", "text"]


class DenseEncoder:
    """useful-cub-58 st-export loaded manually: XLM-R -> mean-pool -> 128-d linear
    (no bias) -> L2-normalize."""

    def __init__(self, model_dir, device, max_seq):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.enc = AutoModel.from_pretrained(model_dir).to(device).eval()
        sd = torch.load(f"{model_dir}/2_Dense/pytorch_model.bin", map_location=device)
        self.W = next(v for k, v in sd.items() if k.endswith("weight")).to(device).float()  # [128,768]
        self.device, self.max_seq = device, max_seq

    @torch.inference_mode()
    def encode(self, texts, bs=256):
        out = []
        for i in range(0, len(texts), bs):
            b = self.tok(texts[i:i + bs], padding=True, truncation=True,
                         max_length=self.max_seq, return_tensors="pt").to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = self.enc(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            pooled = (h.float() * m).sum(1) / m.sum(1).clamp(min=1e-9)
            proj = F.normalize(pooled @ self.W.t(), dim=1)
            out.append(proj.cpu().numpy())
        return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--dist", required=True)
    ap.add_argument("--seg-map", required=True)
    ap.add_argument("--data-cfg", default="configs/data/default.yaml")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--max-seq", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    ren = RowTextRenderer(OmegaConf.load(a.data_cfg))   # dense query:/passage: template
    segmap = json.load(open(a.seg_map))
    gold, dist = load_rows(a.gold), load_rows(a.dist)
    seen, docrows = set(), []
    for r in gold + dist:
        aid = r["offer_id_b64"]
        if aid not in seen:
            seen.add(aid); docrows.append(r)
    id_to_row = {r["offer_id_b64"]: i for i, r in enumerate(docrows)}
    terms = sorted({r["query_term"] for r in gold})
    rel_E = {}
    for r in gold:
        row = id_to_row.get(r["offer_id_b64"])
        if row is not None and r["label"] == "E":
            rel_E.setdefault(r["query_term"], set()).add(row)

    model = DenseEncoder(a.model, a.device, a.max_seq)
    D = model.encode([ren.render_offer_text(r) for r in docrows])
    Q = model.encode([ren.render_query_text({"query_term": t}) for t in terms])

    by_seg = defaultdict(list)
    for ti, term in enumerate(terms):
        if term not in rel_E or not rel_E[term]:
            continue
        top = topk_rows(D @ Q[ti], a.k)
        by_seg[segmap.get(term, "?")].append(len(top & rel_E[term]) / len(rel_E[term]))
    print(f"=== DENSE useful-cub-58 per-segment recall@{a.k} (E) ===")
    allv = []
    for s in SEGS:
        v = by_seg.get(s, []); allv += v
        print(f"  {s:14s} n={len(v):>4}  R@{a.k}={np.mean(v):.4f}" if v else f"  {s:14s} n=0")
    print(f"  {'OVERALL':14s} n={len(allv):>4}  R@{a.k}={np.mean(allv):.4f}")


if __name__ == "__main__":
    main()

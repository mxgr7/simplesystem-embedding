"""SPLADE inference ceiling — round 2: torch.compile + fixed-length padding.

Builds on bench1 (bf16 + pad=longest + gpu-topk256 = ~6k/s). Here we pad to a
fixed length (static shapes) so torch.compile + CUDA graphs can kick in, and
measure the forward ceiling. Texts are ~75 tokens (p99 107), so pad to 128.
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


def load_texts(path, n, ren):
    out = []
    for line in open(path):
        if len(out) >= n:
            break
        r = json.loads(line); r["description"] = ""
        t = ren.render_offer_text(r)
        if t:
            out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splade-ckpt", required=True)
    ap.add_argument("--data", default="data/desc_distractors.jsonl")
    ap.add_argument("--pad", type=int, default=128)
    args = ap.parse_args()

    model, cfg = load_embedding_module_from_checkpoint(args.splade_ckpt)
    model = model.to(DEV).eval()
    tok = load_fast_tokenizer(cfg.model.model_name)
    ren = RowTextRenderer(cfg.data)
    texts = load_texts(args.data, 60000, ren)
    print(f"{len(texts):,} texts, pad to fixed {args.pad}", flush=True)

    def make_batches(bs, n_batches):
        b = []
        for i in range(0, bs * n_batches, bs):
            e = tok(texts[i:i + bs], padding="max_length", truncation=True,
                    max_length=args.pad, return_tensors="pt")
            b.append((e["input_ids"].to(DEV), e["attention_mask"].to(DEV)))
        return b

    def eager_fwd(ids, attn):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model._encode_representations(ids, attn)

    compiled = torch.compile(eager_fwd, mode="reduce-overhead")

    @torch.inference_mode()
    def timeit(fn, batches, warmup=5):
        for b in batches[:warmup]:
            fn(b[0], b[1])
        torch.cuda.synchronize()
        t0 = time.time()
        for b in batches:
            r = fn(b[0], b[1])
        torch.cuda.synchronize()
        dt = time.time() - t0
        n = sum(b[0].shape[0] for b in batches)
        return n / dt

    print(f"\n{'config':<40} {'fwd docs/s':>12}", flush=True)
    for bs in (256, 512, 1024):
        batches = make_batches(bs, 25)
        eag = timeit(eager_fwd, batches)
        print(f"{'eager bf16 bs'+str(bs):<40} {eag:>12,.0f}", flush=True)
    for bs in (256, 512, 1024):
        batches = make_batches(bs, 25)
        comp = timeit(compiled, batches)  # first call triggers compile (warmup covers it)
        print(f"{'compiled bf16 bs'+str(bs):<40} {comp:>12,.0f}", flush=True)


if __name__ == "__main__":
    main()

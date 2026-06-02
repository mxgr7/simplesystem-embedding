"""Reconstruct the precision-bench dataset (corpus sample + exact GT) on milvus.

The original small dataset's corpus_vectors.npy/ground_truth.npy were cleaned up;
we sample ~200k vectors from the full corpus and recompute exact top-100 GT for
the existing 1000 query vectors. Relative recall (fp32 vs bfloat16 vs int8…) is
valid on any consistent corpus+GT pair.
"""
import json
import numpy as np

OUT = "reports/hnsw_eval"
FULL = "/data/datasets/hnsw_eval_full/corpus_vectors.npy"
SAMPLE = 200_000
SEED = 42
GT_TOPK = 100

full = np.load(FULL, mmap_mode="r")
n = full.shape[0]
rng = np.random.default_rng(SEED)
idx = np.sort(rng.choice(n, size=min(SAMPLE, n), replace=False))
corpus = np.ascontiguousarray(np.asarray(full[idx], dtype=np.float32))
np.save(f"{OUT}/corpus_vectors.npy", corpus)

q = np.load(f"{OUT}/query_vectors.npy").astype(np.float32)


def l2norm(a):
    return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)


qn, cn = l2norm(q), l2norm(corpus)
sims = qn @ cn.T  # (1000, 200k) ~800MB
part = np.argpartition(-sims, GT_TOPK, axis=1)[:, :GT_TOPK]
rows = np.arange(q.shape[0])[:, None]
order = np.argsort(-sims[rows, part], axis=1)
gt = part[rows, order].astype(np.int32)
np.save(f"{OUT}/ground_truth.npy", gt)

manifest = {
    "queries": int(q.shape[0]),
    "vectors_total": int(corpus.shape[0]),
    "dim": int(corpus.shape[1]),
    "gt_topk": GT_TOPK,
    "source": "sampled from /data/datasets/hnsw_eval_full/corpus_vectors.npy",
    "seed": SEED,
}
json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=2)
print("corpus", corpus.shape, corpus.dtype, "| gt", gt.shape, "| manifest", manifest)

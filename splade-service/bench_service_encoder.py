"""Gate + benchmark for the optimised service document encoder.

Two things must hold before the reindex may use this encoder:

  1. In *exact* mode (no weights cast, no compiled head, pad guard on) the new
     `SpladeEncoder.encode_packed` must be **byte-identical** to the previous
     implementation, which is kept verbatim below as the oracle.
  2. In *fast* mode the deviation must match what the gold/seg recall harness
     already signed off on (top-256 jaccard ~0.979, top-1 agreement ~0.978) --
     anything further from the reference means something other than the
     validated numeric change crept in.

Usage (on the GPU box):
    python bench_service_encoder.py --data ../../pipeline/throughput_lab/data/bench_300k.jsonl \
        --limit 60000 --request-size 2048
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np
import torch

SERVICE = os.path.dirname(os.path.abspath(__file__))
LAB = os.environ.get("LAB", "/workspace/pipeline/throughput_lab/box")
BAKEOFF = os.environ.get("BAKEOFF", "/workspace/pipeline/agg_bakeoff")
for path in (SERVICE, LAB, BAKEOFF):
    if path not in sys.path:
        sys.path.insert(0, path)

from constants import MAX_OFFER_LENGTH, TOP_K  # noqa: E402
from codec import pack_sparse_arrays, pack_sparse_rows  # noqa: E402
import backend  # noqa: E402

_HEAD = struct.Struct("<H")
_BATCH_HEAD = struct.Struct("<4sI")


def reference_encode_packed(encoder, texts):
    """The pre-optimisation implementation, verbatim, as the equivalence oracle.

    Unsorted, one tokenizer call per chunk with `padding=True`, the full
    [B,L,V] relu/log1p/mask/amax head, and per-row numpy packing.
    """
    rows = []
    for start in range(0, len(texts), encoder.batch_size):
        chunk = texts[start:start + encoder.batch_size]
        tokens = encoder.tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=MAX_OFFER_LENGTH,
            return_tensors="pt",
        )
        tokens = {
            name: value.to(encoder.device, non_blocking=True)
            for name, value in tokens.items()
        }
        with torch.inference_mode(), torch.autocast(
            device_type=encoder.device.type,
            dtype=encoder.document_dtype,
            enabled=encoder.document_dtype is not None,
        ):
            logits = encoder.model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            ).logits
            logits.relu_().log1p_()
            logits *= tokens["attention_mask"].unsqueeze(-1)
            vectors = logits.amax(dim=1)
            vectors.masked_fill_(~encoder.mask, 0)
            values, ids = torch.topk(vectors, TOP_K, dim=1, sorted=True)
        ids = ids.cpu().numpy()
        values = values.to(torch.float16).cpu().numpy()
        for row_ids, row_values in zip(ids, values):
            positive = np.isfinite(row_values) & (row_values > 0)
            rows.append(pack_sparse_arrays(row_ids[positive], row_values[positive]))
        del logits, vectors, values, ids
    return pack_sparse_rows(rows)


def split_batch(blob):
    """packed batch -> list of raw row bytes."""
    magic, count = _BATCH_HEAD.unpack_from(blob)
    assert magic == b"SPB1", magic
    offset = _BATCH_HEAD.size
    rows = []
    for _ in range(count):
        (entries,) = _HEAD.unpack_from(blob, offset)
        size = _HEAD.size + entries * 4
        rows.append(blob[offset:offset + size])
        offset += size
    assert offset == len(blob), "trailing bytes in packed batch"
    return rows


def row_to_arrays(row):
    (count,) = _HEAD.unpack_from(row)
    ids = np.frombuffer(row, dtype="<u2", count=count, offset=_HEAD.size)
    values = np.frombuffer(row, dtype="<f2", count=count, offset=_HEAD.size + count * 2)
    return ids, values


def compare(reference_rows, candidate_rows):
    assert len(reference_rows) == len(candidate_rows)
    identical = 0
    jaccard_sum = 0.0
    top1 = 0
    max_delta = 0.0
    for left, right in zip(reference_rows, candidate_rows):
        if left == right:
            identical += 1
        a_ids, a_values = row_to_arrays(left)
        b_ids, b_values = row_to_arrays(right)
        a_set, b_set = set(a_ids.tolist()), set(b_ids.tolist())
        union = len(a_set | b_set)
        jaccard_sum += (len(a_set & b_set) / union) if union else 1.0
        if len(a_ids) and len(b_ids):
            # rows are id-ascending, so recover the argmax by weight
            if int(a_ids[int(np.argmax(a_values))]) == int(b_ids[int(np.argmax(b_values))]):
                top1 += 1
        elif not len(a_ids) and not len(b_ids):
            top1 += 1
        shared = a_set & b_set
        if shared:
            a_map = dict(zip(a_ids.tolist(), a_values.tolist()))
            b_map = dict(zip(b_ids.tolist(), b_values.tolist()))
            max_delta = max(
                max_delta, max(abs(a_map[t] - b_map[t]) for t in shared)
            )
    n = len(reference_rows)
    return {
        "docs": n,
        "byte_identical": identical,
        "byte_identical_share": identical / n,
        "mean_topk_jaccard": jaccard_sum / n,
        "top1_agreement": top1 / n,
        "max_abs_weight_delta": max_delta,
    }


def build_texts(path, limit):
    from enc_bench import _v1_nul
    from rendering import render_from_nul

    texts = []
    with open(path) as handle:
        for line in handle:
            if len(texts) >= limit:
                break
            if not line.strip():
                continue
            nul = _v1_nul(json.loads(line), None, None)
            if nul is None:
                continue
            text = render_from_nul(nul)
            if text:
                texts.append(text)
    return texts


def run(encoder, texts, request_size, repeats):
    """Encode in service-sized requests; return (rows, best docs/s)."""
    best = 0.0
    rows = None
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        collected = []
        for offset in range(0, len(texts), request_size):
            blob = encoder.encode_packed(texts[offset:offset + request_size])
            collected.extend(split_batch(blob))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        best = max(best, len(texts) / elapsed)
        rows = collected
    return rows, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--limit", type=int, default=60000)
    parser.add_argument("--request-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=4096)
    parser.add_argument("--checkpoint", default=os.environ.get(
        "SPLADE_CHECKPOINT", "/root/prod_soup.ckpt"))
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME", "/root/.hf"))
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--json-out")
    cli = parser.parse_args()

    texts = build_texts(cli.data, cli.limit)
    print(f"texts: {len(texts)}", flush=True)

    report = {"docs": len(texts), "request_size": cli.request_size,
              "batch_size": cli.batch_size}

    # ---- exact mode: must be byte-identical to the previous implementation
    exact = backend.SpladeEncoder(
        cli.checkpoint, "auto", cli.cache_dir, cli.batch_size,
        document_dtype="bf16", weights_cast=False, compile_head=False,
        overlap=2, pad_guard=1,
    )
    exact.encode_packed(texts[:cli.warmup])          # warm clocks + allocator
    report["exact_encoding_version"] = exact.document_encoding_version

    if not cli.skip_reference:
        torch.cuda.synchronize()
        start = time.perf_counter()
        reference_rows = split_batch(reference_encode_packed(exact, texts))
        torch.cuda.synchronize()
        report["reference_docs_per_s"] = len(texts) / (time.perf_counter() - start)
        print(f"reference: {report['reference_docs_per_s']:.1f} docs/s", flush=True)

    exact_rows, exact_rate = run(exact, texts, cli.request_size, cli.repeats)
    report["exact_docs_per_s"] = exact_rate
    print(f"exact:     {exact_rate:.1f} docs/s", flush=True)
    if not cli.skip_reference:
        report["exact_vs_reference"] = compare(reference_rows, exact_rows)
        print(f"exact vs reference: {report['exact_vs_reference']}", flush=True)

    del exact
    torch.cuda.empty_cache()

    # ---- fast mode: the recall-gated numeric stack
    fast = backend.SpladeEncoder(
        cli.checkpoint, "auto", cli.cache_dir, cli.batch_size,
        document_dtype="bf16", weights_cast=True, compile_head=True,
        overlap=2, pad_guard=0,
    )
    fast.encode_packed(texts[:cli.warmup])
    report["fast_encoding_version"] = fast.document_encoding_version
    fast_rows, fast_rate = run(fast, texts, cli.request_size, cli.repeats)
    report["fast_docs_per_s"] = fast_rate
    print(f"fast:      {fast_rate:.1f} docs/s", flush=True)
    if not cli.skip_reference:
        report["fast_vs_reference"] = compare(reference_rows, fast_rows)
        print(f"fast vs reference: {report['fast_vs_reference']}", flush=True)

    if cli.json_out:
        with open(cli.json_out, "w") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

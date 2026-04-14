"""Train a shared IVF-PQ codebook on a sample of input rows.

The output is an empty trained FAISS index (coarse quantizer + PQ codebook,
zero vectors added). It is meant to be shared across multiple shard builds
via `index_build --trained-index <path>` so every shard adds to an index
with the same quantization state and results can be merged later.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq
import torch
from dotenv import load_dotenv

from embedding_train.faiss_index import (
    build_index_config,
    create_faiss_index,
    index_requires_training,
    minimum_training_vectors,
    resolve_faiss_index_type,
)
from embedding_train.index_build import (
    ParquetSource,
    load_index_model,
    prepare_offer_rows,
)
from embedding_train.infer import (
    build_tokenizer,
    encode_texts,
    parse_copy_columns,
    resolve_device,
)
from embedding_train.rendering import RowTextRenderer


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train an IVF-PQ codebook on a sample of input rows and save as an empty trained index."
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--input", required=True, help="Input Parquet file or directory.")
    parser.add_argument("--output", required=True, help="Path to write the trained index file.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--index-type", default="ivf_pq", choices=["ivf_pq", "ivf_flat"])
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--pq-m", type=int, default=16)
    parser.add_argument("--pq-bits", type=int, default=8)
    parser.add_argument("--nprobe", type=int, default=16)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=64)
    parser.add_argument(
        "--train-sample-size",
        type=int,
        default=0,
        help="Number of rows to encode for training. 0 = 40 × nlist (FAISS recommendation).",
    )
    parser.add_argument("--read-batch-size", type=int, default=8192)
    parser.add_argument("--encode-batch-size", type=int, default=1024)
    parser.add_argument("--max-offer-length", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    load_dotenv()
    torch.set_float32_matmul_precision("high")
    args = build_arg_parser().parse_args(argv)
    train_index(args)


def train_index(args):
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    index_type = resolve_faiss_index_type(args.index_type)
    if not index_requires_training(index_type):
        raise ValueError(f"{index_type} does not need training — nothing to do")

    index_config = build_index_config(args)
    sample_target = int(args.train_sample_size) or max(
        minimum_training_vectors(index_type, index_config),
        40 * int(args.nlist),
    )

    model, cfg = load_index_model(args)
    model = model.to(device)
    model.eval()
    tokenizer = build_tokenizer(cfg.model.model_name)
    renderer = RowTextRenderer(cfg.data)
    parquet_file = ParquetSource(args.input)
    copy_columns = parse_copy_columns("", parquet_file.schema.names)
    max_offer_length = int(args.max_offer_length) or int(cfg.data.max_offer_length)

    print(
        f"Training target: {sample_target:,} samples "
        f"(nlist={args.nlist}, index_type={index_type})",
        flush=True,
    )

    row_number = 0
    next_faiss_id = 0
    sample_chunks = []
    collected = 0
    t0 = time.perf_counter()

    for batch in parquet_file.iter_batches(batch_size=int(args.read_batch_size)):
        rows = batch.to_pylist()
        if not rows:
            break
        prepared_rows, offer_texts, row_number, next_faiss_id, _ = prepare_offer_rows(
            rows,
            renderer,
            copy_columns,
            row_number,
            next_faiss_id,
        )
        if not prepared_rows:
            continue
        embeddings = encode_texts(
            model,
            tokenizer,
            offer_texts,
            max_offer_length,
            int(args.encode_batch_size),
            device,
        )
        chunk = embeddings.numpy().astype("float32", copy=False)
        remaining = sample_target - collected
        if chunk.shape[0] > remaining:
            chunk = chunk[:remaining]
        sample_chunks.append(chunk)
        collected += chunk.shape[0]
        print(
            f"  collected={collected:,}/{sample_target:,} "
            f"elapsed={time.perf_counter() - t0:.1f}s",
            flush=True,
        )
        if collected >= sample_target:
            break

    if collected == 0:
        raise ValueError("No training samples collected from input")

    train_matrix = np.concatenate(sample_chunks, axis=0)
    min_required = minimum_training_vectors(index_type, index_config)
    if train_matrix.shape[0] < min_required:
        raise ValueError(
            f"Not enough training samples: got {train_matrix.shape[0]}, need {min_required}"
        )

    embedding_dim = int(train_matrix.shape[1])
    index = create_faiss_index(embedding_dim, index_config)

    t_train = time.perf_counter()
    index.train(train_matrix)
    train_seconds = time.perf_counter() - t_train

    faiss.write_index(index, str(output_path))
    print(
        f"Trained index saved: {output_path} "
        f"(train_rows={train_matrix.shape[0]:,}, train_time={train_seconds:.1f}s, "
        f"embedding_dim={embedding_dim})",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())

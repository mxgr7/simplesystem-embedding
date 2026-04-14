"""Orchestrator for multi-GPU embedding export.

Workflow:
1. Prepare per-worker input splits.  Either re-split the input into N
   contiguous row-range slices (default), or assign existing source
   shards directly to workers round-robin (``--use-source-shards``).
2. Launch N ``embedding_train.infer`` subprocesses in parallel, one per
   GPU, each reading its slice and writing embeddings to
   ``<output>/embeddings/shard_<i>.parquet``.
3. Each worker uses a disjoint ``--row-number-offset`` so row_numbers
   are globally unique across shards.

The resulting ``<output>/embeddings/`` directory is a parquet dataset
that downstream vector DB ingestion (Milvus bulk insert, Elastic bulk)
can consume directly.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run a sharded embedding export across multiple GPUs."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--input",
        required=True,
        help="Source parquet file or directory of shards.",
    )
    parser.add_argument("--output", required=True, help="Root output directory.")
    parser.add_argument(
        "--num-shards",
        type=int,
        required=True,
        help="Number of parallel workers (typically equal to the GPU count).",
    )
    parser.add_argument(
        "--gpu-ids",
        default="",
        help="Comma-separated GPU ids. Default: 0..num_shards-1.",
    )
    parser.add_argument("--mode", default="offer", choices=["offer", "query"])
    parser.add_argument(
        "--copy-columns",
        default="",
        help="Comma-separated columns from the source to copy into each output row.",
    )
    parser.add_argument(
        "--embedding-precision",
        default="float16",
        help="Embedding export precision (float32/float16/int8/sign/binary).",
    )
    parser.add_argument("--read-batch-size", type=int, default=200_000)
    parser.add_argument("--encode-batch-size", type=int, default=1024)
    parser.add_argument("--max-offer-length", type=int, default=0)
    parser.add_argument("--max-query-length", type=int, default=0)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-shard-split",
        action="store_true",
        help="Reuse existing shard parquets under <output>/shards/input_*.parquet.",
    )
    parser.add_argument(
        "--use-source-shards",
        action="store_true",
        help=(
            "Assign existing source parquet shards directly to workers "
            "round-robin instead of re-splitting. Avoids reading and "
            "rewriting the entire dataset when the input is already "
            "partitioned into multiple files."
        ),
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include rendered text columns in each output row.",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    run_parallel(args)


def resolve_gpu_ids(args):
    if args.gpu_ids:
        ids = [int(tok) for tok in args.gpu_ids.split(",") if tok.strip()]
    else:
        ids = list(range(int(args.num_shards)))
    if len(ids) != int(args.num_shards):
        raise ValueError(
            f"--gpu-ids must list exactly --num-shards={args.num_shards} ids"
        )
    return ids


def split_input_parquet(source, shard_paths):
    source_path = Path(source)
    if source_path.is_dir():
        shard_files = sorted(source_path.glob("*.parquet"))
    else:
        shard_files = [source_path]
    if not shard_files:
        raise ValueError(f"No parquet files found at {source}")

    row_counts = [pq.ParquetFile(f).metadata.num_rows for f in shard_files]
    total_rows = sum(row_counts)
    num_shards = len(shard_paths)
    per_shard = total_rows // num_shards
    boundaries = [per_shard * i for i in range(num_shards)] + [total_rows]

    writers = [None] * num_shards
    write_counts = [0] * num_shards

    global_row = 0
    for shard_file, n_rows in zip(shard_files, row_counts):
        shard_table = pq.read_table(shard_file)
        base = global_row
        global_row += n_rows

        for target_idx in range(num_shards):
            start = max(boundaries[target_idx], base)
            end = min(boundaries[target_idx + 1], base + n_rows)
            if start >= end:
                continue
            local_start = start - base
            local_end = end - base
            subset = shard_table.slice(local_start, local_end - local_start)
            if writers[target_idx] is None:
                writers[target_idx] = pq.ParquetWriter(
                    shard_paths[target_idx], subset.schema, compression="zstd"
                )
            writers[target_idx].write_table(subset)
            write_counts[target_idx] += subset.num_rows
        del shard_table

    for writer in writers:
        if writer is not None:
            writer.close()

    return write_counts


def assign_source_shards(input_path, num_workers, work_dir):
    """Assign existing source parquet shards round-robin to workers.

    Creates per-worker directories with symlinks to the assigned source
    files.  Returns (input_dirs, row_counts) where input_dirs[i] is the
    directory to pass as --input for worker i.
    """
    source = Path(input_path)
    if source.is_dir():
        shard_files = sorted(source.glob("*.parquet"))
    else:
        shard_files = [source]
    if not shard_files:
        raise ValueError(f"No parquet files found at {input_path}")

    buckets = [[] for _ in range(num_workers)]
    for i, f in enumerate(shard_files):
        buckets[i % num_workers].append(f)

    input_dirs = []
    counts = []
    inputs_root = work_dir / "inputs"
    for idx, paths in enumerate(buckets):
        worker_dir = inputs_root / f"worker_{idx}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        for existing in worker_dir.iterdir():
            existing.unlink()
        n = 0
        for p in paths:
            (worker_dir / p.name).symlink_to(p.resolve())
            n += pq.ParquetFile(str(p)).metadata.num_rows
        input_dirs.append(worker_dir)
        counts.append(n)

    return input_dirs, counts


def run_parallel(args):
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    shards_root = output_root / "shards"
    embeddings_root = output_root / "embeddings"
    shards_root.mkdir(parents=True, exist_ok=True)
    embeddings_root.mkdir(parents=True, exist_ok=True)

    gpu_ids = resolve_gpu_ids(args)
    num_shards = int(args.num_shards)
    shard_output_paths = [
        embeddings_root / f"shard_{i:04d}.parquet" for i in range(num_shards)
    ]

    if args.use_source_shards:
        print(f"Assigning source shards to {num_shards} workers", flush=True)
        shard_input_paths, counts = assign_source_shards(
            args.input, num_shards, output_root
        )
        for i, n in enumerate(counts):
            print(f"  worker {i}: {n:,} rows", flush=True)
    elif not args.skip_shard_split:
        shard_input_paths = [
            shards_root / f"input_{i}.parquet" for i in range(num_shards)
        ]
        print(f"Splitting input into {num_shards} shards", flush=True)
        t = time.perf_counter()
        counts = split_input_parquet(args.input, shard_input_paths)
        for i, n in enumerate(counts):
            print(f"  shard {i}: {n:,} rows → {shard_input_paths[i]}", flush=True)
        print(f"Split done in {time.perf_counter() - t:.1f}s", flush=True)
    else:
        shard_input_paths = [
            shards_root / f"input_{i}.parquet" for i in range(num_shards)
        ]
        counts = [
            pq.ParquetFile(p).metadata.num_rows for p in shard_input_paths
        ]

    row_offsets = [sum(counts[:i]) for i in range(num_shards)]

    print("Launching shard infer workers", flush=True)
    procs = []
    for i in range(num_shards):
        cmd = [
            sys.executable,
            "-m",
            "embedding_train.infer",
            "--checkpoint",
            args.checkpoint,
            "--input",
            str(shard_input_paths[i]),
            "--output",
            str(shard_output_paths[i]),
            "--mode",
            args.mode,
            "--device",
            f"cuda:{gpu_ids[i]}",
            "--embedding-precision",
            args.embedding_precision,
            "--read-batch-size",
            str(args.read_batch_size),
            "--encode-batch-size",
            str(args.encode_batch_size),
            "--row-number-offset",
            str(row_offsets[i]),
            "--compression",
            args.compression,
            "--overwrite",
        ]
        if args.copy_columns:
            cmd += ["--copy-columns", args.copy_columns]
        if args.max_offer_length:
            cmd += ["--max-offer-length", str(args.max_offer_length)]
        if args.max_query_length:
            cmd += ["--max-query-length", str(args.max_query_length)]
        if args.include_text:
            cmd.append("--include-text")
        print(
            f"  shard {i}: {counts[i]:,} rows, offset={row_offsets[i]:,}, gpu={gpu_ids[i]}",
            flush=True,
        )
        procs.append(subprocess.Popen(cmd, cwd=REPO_ROOT))

    failed = []
    for i, proc in enumerate(procs):
        rc = proc.wait()
        if rc != 0:
            failed.append((i, rc))
    if failed:
        raise RuntimeError(f"Shard infer workers failed: {failed}")

    print(f"Done. Embeddings written to: {embeddings_root}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

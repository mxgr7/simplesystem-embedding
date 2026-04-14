"""Orchestrator for multi-GPU index builds.

Workflow:
1. Split the input parquet into N shard files (contiguous row ranges).
2. Train a shared IVF-PQ index on a sample (one GPU).
3. Launch N parallel `index_build` subprocesses, each:
   - bound to a different `--device cuda:i`
   - reading its own shard parquet
   - loading the shared trained index via `--trained-index`
   - writing a shard artifact with its own `--faiss-id-offset`
4. Merge the N shard artifacts into a single final artifact.

The shard input files are written under `<output>/shards/input_<i>.parquet`,
shard artifacts under `<output>/shards/artifact_<i>/`, and the merged
artifact under `<output>/artifact/`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run a sharded IVF-PQ index build across multiple GPUs."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Source parquet file or directory.")
    parser.add_argument("--output", required=True, help="Root output directory.")
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument(
        "--gpu-ids",
        default="",
        help="Comma-separated GPU ids. Default: 0..num_shards-1.",
    )
    parser.add_argument("--index-type", default="ivf_pq")
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--pq-m", type=int, default=16)
    parser.add_argument("--pq-bits", type=int, default=8)
    parser.add_argument("--nprobe", type=int, default=16)
    parser.add_argument("--train-sample-size", type=int, default=0)
    parser.add_argument("--read-batch-size", type=int, default=200000)
    parser.add_argument("--encode-batch-size", type=int, default=1024)
    parser.add_argument("--max-offer-length", type=int, default=0)
    parser.add_argument("--metadata-chunk-rows", type=int, default=1_000_000)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-shard-split",
        action="store_true",
        help="Reuse existing shard parquets under <output>/shards/input_*.parquet.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse existing trained index at <output>/trained.index.",
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

    row_counts = []
    for f in shard_files:
        row_counts.append(pq.ParquetFile(f).metadata.num_rows)
    total_rows = sum(row_counts)
    num_shards = len(shard_paths)
    per_shard = total_rows // num_shards
    boundaries = [per_shard * i for i in range(num_shards)] + [total_rows]

    writers = [None] * num_shards
    schemas = [None] * num_shards
    write_counts = [0] * num_shards

    global_row = 0
    for shard_file, n_rows in zip(shard_files, row_counts):
        shard_table = pq.read_table(shard_file)
        base = global_row
        global_row += n_rows

        # Assign this source-shard's rows to target shards based on boundaries.
        for target_idx in range(num_shards):
            start = max(boundaries[target_idx], base)
            end = min(boundaries[target_idx + 1], base + n_rows)
            if start >= end:
                continue
            local_start = start - base
            local_end = end - base
            subset = shard_table.slice(local_start, local_end - local_start)
            if writers[target_idx] is None:
                schemas[target_idx] = subset.schema
                writers[target_idx] = pq.ParquetWriter(
                    shard_paths[target_idx], schemas[target_idx], compression="zstd"
                )
            writers[target_idx].write_table(subset)
            write_counts[target_idx] += subset.num_rows
        del shard_table

    for writer in writers:
        if writer is not None:
            writer.close()

    return write_counts


def run_parallel(args):
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    shards_root = output_root / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)
    artifact_root = output_root / "artifact"

    gpu_ids = resolve_gpu_ids(args)
    num_shards = int(args.num_shards)
    shard_input_paths = [shards_root / f"input_{i}.parquet" for i in range(num_shards)]
    shard_artifact_paths = [shards_root / f"artifact_{i}" for i in range(num_shards)]

    if not args.skip_shard_split:
        print(f"Splitting input into {num_shards} shards", flush=True)
        t = time.perf_counter()
        counts = split_input_parquet(args.input, shard_input_paths)
        for i, n in enumerate(counts):
            print(f"  shard {i}: {n:,} rows → {shard_input_paths[i]}", flush=True)
        print(f"Split done in {time.perf_counter() - t:.1f}s", flush=True)
    else:
        counts = [
            pq.ParquetFile(p).metadata.num_rows for p in shard_input_paths
        ]

    trained_index_path = output_root / "trained.index"
    if not args.skip_train:
        print("Training shared IVF-PQ codebook on shard 0", flush=True)
        train_cmd = [
            sys.executable,
            "-m",
            "embedding_train.index_build_train",
            "--checkpoint",
            args.checkpoint,
            "--input",
            str(shard_input_paths[0]),
            "--output",
            str(trained_index_path),
            "--device",
            f"cuda:{gpu_ids[0]}",
            "--index-type",
            args.index_type,
            "--nlist",
            str(args.nlist),
            "--pq-m",
            str(args.pq_m),
            "--pq-bits",
            str(args.pq_bits),
            "--nprobe",
            str(args.nprobe),
            "--read-batch-size",
            str(args.read_batch_size),
            "--encode-batch-size",
            str(args.encode_batch_size),
            "--overwrite",
        ]
        if args.train_sample_size:
            train_cmd += ["--train-sample-size", str(args.train_sample_size)]
        if args.max_offer_length:
            train_cmd += ["--max-offer-length", str(args.max_offer_length)]
        t = time.perf_counter()
        subprocess.run(train_cmd, check=True, cwd=REPO_ROOT)
        print(f"Training done in {time.perf_counter() - t:.1f}s", flush=True)

    id_offsets = [sum(counts[:i]) for i in range(num_shards)]
    print("Launching shard builds", flush=True)
    procs = []
    for i in range(num_shards):
        cmd = [
            sys.executable,
            "-m",
            "embedding_train.index_build",
            "--checkpoint",
            args.checkpoint,
            "--input",
            str(shard_input_paths[i]),
            "--output",
            str(shard_artifact_paths[i]),
            "--trained-index",
            str(trained_index_path),
            "--faiss-id-offset",
            str(id_offsets[i]),
            "--index-type",
            args.index_type,
            "--nlist",
            str(args.nlist),
            "--pq-m",
            str(args.pq_m),
            "--pq-bits",
            str(args.pq_bits),
            "--nprobe",
            str(args.nprobe),
            "--read-batch-size",
            str(args.read_batch_size),
            "--encode-batch-size",
            str(args.encode_batch_size),
            "--metadata-chunk-rows",
            str(args.metadata_chunk_rows),
            "--device",
            f"cuda:{gpu_ids[i]}",
            "--compression",
            args.compression,
            "--overwrite",
        ]
        if args.max_offer_length:
            cmd += ["--max-offer-length", str(args.max_offer_length)]
        print(
            f"  shard {i}: {counts[i]:,} rows, offset={id_offsets[i]:,}, gpu={gpu_ids[i]}",
            flush=True,
        )
        procs.append(subprocess.Popen(cmd, cwd=REPO_ROOT))

    failed = []
    for i, proc in enumerate(procs):
        rc = proc.wait()
        if rc != 0:
            failed.append((i, rc))
    if failed:
        raise RuntimeError(f"Shard builds failed: {failed}")

    print("Merging shards", flush=True)
    merge_cmd = [
        sys.executable,
        "-m",
        "embedding_train.index_merge",
        "--shards",
        *[str(p) for p in shard_artifact_paths],
        "--output",
        str(artifact_root),
        "--compression",
        args.compression,
    ]
    if args.overwrite:
        merge_cmd.append("--overwrite")
    subprocess.run(merge_cmd, check=True, cwd=REPO_ROOT)

    print(f"Done. Final artifact: {artifact_root}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

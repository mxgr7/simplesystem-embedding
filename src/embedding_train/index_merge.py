"""Merge shard IVF-PQ indices produced by parallel index_build runs.

Each shard must have been built against the same pre-trained index (same
coarse quantizer + PQ codebook). This script:

1. Loads each shard's FAISS index and `merge_from`s them into the first.
2. Concatenates each shard's metadata.parquet into a single output file.
3. Writes a merged manifest referencing the combined artifact.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import faiss
import pyarrow.parquet as pq

from embedding_train.index_artifact import (
    prepare_index_directory,
    read_manifest,
    write_manifest,
)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Merge shard FAISS indices + metadata parquets into a single index artifact."
    )
    parser.add_argument(
        "--shards",
        nargs="+",
        required=True,
        help="Shard artifact directories (each containing index.faiss + metadata.parquet + manifest.json).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for the merged index artifact.",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        help="Parquet compression codec for merged metadata.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    merge_shards(args)


def merge_shards(args):
    shard_paths = [Path(s) for s in args.shards]
    if len(shard_paths) < 1:
        raise ValueError("--shards requires at least one directory")

    for path in shard_paths:
        if not (path / "index.faiss").exists():
            raise FileNotFoundError(f"Missing index.faiss in shard: {path}")
        if not (path / "metadata.parquet").exists():
            raise FileNotFoundError(f"Missing metadata.parquet in shard: {path}")

    artifact_paths = prepare_index_directory(args.output, args.overwrite)

    print(f"Loading shard 0: {shard_paths[0]}", flush=True)
    merged_index = faiss.read_index(str(shard_paths[0] / "index.faiss"))

    for i, path in enumerate(shard_paths[1:], start=1):
        print(f"Merging shard {i}: {path}", flush=True)
        shard_index = faiss.read_index(str(path / "index.faiss"))
        if hasattr(merged_index, "merge_from"):
            merged_index.merge_from(shard_index, 0)
        else:
            raise TypeError(
                f"Index type {type(merged_index).__name__} does not support merge_from"
            )

    print(f"Writing merged index ({merged_index.ntotal:,} vectors)", flush=True)
    faiss.write_index(merged_index, str(artifact_paths["index_file"]))

    print("Concatenating metadata parquets", flush=True)
    writer = None
    try:
        for path in shard_paths:
            shard_metadata_path = path / "metadata.parquet"
            shard_parquet = pq.ParquetFile(str(shard_metadata_path))
            for row_group_idx in range(shard_parquet.num_row_groups):
                table = shard_parquet.read_row_group(row_group_idx)
                if writer is None:
                    writer = pq.ParquetWriter(
                        str(artifact_paths["metadata_file"]),
                        table.schema,
                        compression=args.compression,
                    )
                writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    shard_manifests = [
        read_manifest(path / "manifest.json") for path in shard_paths
    ]
    base = dict(shard_manifests[0])
    base.update(
        {
            "index_file": artifact_paths["index_file"].name,
            "metadata_file": artifact_paths["metadata_file"].name,
            "indexed_rows": int(merged_index.ntotal),
            "processed_rows": sum(
                int(m.get("processed_rows", 0)) for m in shard_manifests
            ),
            "skipped_rows": sum(
                int(m.get("skipped_rows", 0)) for m in shard_manifests
            ),
            "shards": [str(p) for p in shard_paths],
        }
    )
    write_manifest(artifact_paths["manifest_file"], base)

    print(
        f"Merged artifact written: {artifact_paths['index_dir']} "
        f"(ntotal={merged_index.ntotal:,})",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())

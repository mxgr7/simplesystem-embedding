import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from embedding_train.faiss_index import (
    build_index_config,
    create_faiss_index,
    index_requires_training,
    minimum_training_vectors,
    resolve_faiss_index_type,
)
from embedding_train.config import load_base_config
from embedding_train.index_artifact import prepare_index_directory, write_manifest
from embedding_train.infer import (
    IncrementalParquetWriter,
    build_tokenizer,
    encode_texts,
    parse_copy_columns,
    resolve_device,
)
from embedding_train.model import EmbeddingModule
from embedding_train.model import load_embedding_module_from_checkpoint
from embedding_train.rendering import RowTextRenderer


class ParquetSource:
    """Streaming parquet source backed by a single file or a directory of shards.

    Mimics the subset of pyarrow.parquet.ParquetFile that index_build relies on:
    iter_batches(batch_size), metadata.num_rows, schema.names, schema_arrow.
    Shards are read in sorted order and streamed one at a time so memory use
    stays bounded when indexing large partitioned datasets.
    """

    def __init__(self, path):
        self.path = path
        self.shard_paths = self._resolve_shard_paths(path)
        if not self.shard_paths:
            raise ValueError(f"No parquet files found at: {path}")

        first = pq.ParquetFile(str(self.shard_paths[0]))
        self.schema_arrow = first.schema_arrow
        self.schema = first.schema
        num_rows = 0
        for shard_path in self.shard_paths:
            num_rows += pq.ParquetFile(str(shard_path)).metadata.num_rows
        self.metadata = SimpleNamespace(num_rows=num_rows)

    @staticmethod
    def _resolve_shard_paths(path):
        candidate = Path(path)
        if candidate.is_dir():
            return sorted(candidate.glob("*.parquet"))
        return [candidate]

    def iter_batches(self, batch_size):
        for shard_path in self.shard_paths:
            shard = pq.ParquetFile(str(shard_path))
            for batch in shard.iter_batches(batch_size=batch_size):
                yield batch


def build_progress(total_rows):
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=24),
        TaskProgressColumn(),
        TextColumn("[dim]({task.completed:,.0f}/{task.total:,.0f} rows)"),
        TextColumn(
            "[dim]indexed={task.fields[indexed_rows]:,.0f} skipped={task.fields[skipped_rows]:,.0f}"
        ),
        TimeElapsedColumn(),
        console=Console(file=sys.stderr),
        auto_refresh=False,
        transient=False,
    )


def update_progress(
    progress, task_id, processed_rows, total_rows, indexed_rows, skipped_rows
):
    progress.update(
        task_id,
        completed=min(processed_rows, total_rows),
        indexed_rows=indexed_rows,
        skipped_rows=skipped_rows,
    )
    progress.refresh()


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Build a FAISS offer index from a Parquet dataset using the training offer template."
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional Lightning checkpoint path. If omitted, uses the base pretrained model.",
    )
    parser.add_argument(
        "--model-name",
        default="",
        help="Optional pretrained model name override when --checkpoint is omitted.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input Parquet path. Either a single .parquet file or a directory "
            "of .parquet shards (read in sorted name order)."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for the FAISS index artifact.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, cuda:0, or mps.",
    )
    parser.add_argument(
        "--read-batch-size",
        type=int,
        default=1024,
        help="Rows to stream from Parquet at a time.",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=128,
        help="Texts to tokenize and encode per forward pass.",
    )
    parser.add_argument(
        "--max-offer-length",
        type=int,
        default=0,
        help=(
            "Override cfg.data.max_offer_length for tokenization (0 keeps config value). "
            "Lowering this below the 99th percentile of actual token lengths trades off a "
            "small fraction of truncated rows for proportional forward-pass savings."
        ),
    )
    parser.add_argument(
        "--metadata-chunk-rows",
        type=int,
        default=1_000_000,
        help=(
            "Flush buffered metadata rows to disk once the buffer reaches this many rows. "
            "Large values reduce per-chunk zstd overhead; small values reduce peak memory. "
            "The default (~1M rows ≈ 16 MB of int64 columns) is the sweet spot for corpora "
            "that don't fit in RAM."
        ),
    )
    parser.add_argument(
        "--trained-index",
        default="",
        help=(
            "Path to a pre-trained FAISS index file (e.g. produced by index_build_train). "
            "When set, skips local training and loads the shared quantizer/PQ codebook. "
            "Required for multi-shard builds where all shards must share the same codebook."
        ),
    )
    parser.add_argument(
        "--faiss-id-offset",
        type=int,
        default=0,
        help=(
            "Offset added to every faiss_id and row_number written by this build. "
            "Used for multi-shard builds so each shard's ids are disjoint from the others."
        ),
    )
    parser.add_argument(
        "--index-type",
        default="flat",
        choices=["flat", "ivf_flat", "ivf_pq", "hnsw"],
        help="FAISS index type to build.",
    )
    parser.add_argument(
        "--nlist",
        type=int,
        default=1024,
        help="Number of IVF coarse clusters for ivf_flat and ivf_pq.",
    )
    parser.add_argument(
        "--train-sample-size",
        type=int,
        default=50000,
        help="Embeddings to buffer for IVF/PQ training before indexing all rows.",
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=16,
        help="Default IVF probe count saved into the index manifest.",
    )
    parser.add_argument(
        "--pq-m",
        type=int,
        default=16,
        help="Number of PQ subquantizers for ivf_pq.",
    )
    parser.add_argument(
        "--pq-bits",
        type=int,
        default=8,
        help="Bits per PQ codebook entry for ivf_pq.",
    )
    parser.add_argument(
        "--hnsw-m",
        type=int,
        default=32,
        help="HNSW graph connectivity parameter.",
    )
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=200,
        help="HNSW efConstruction value used while building the graph.",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=64,
        help="Default HNSW efSearch value saved into the index manifest.",
    )
    parser.add_argument(
        "--copy-columns",
        default="",
        help="Comma-separated input columns to retain in index metadata.",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        help="Parquet compression codec for index metadata.",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Optional maximum number of input rows to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    return parser


def build_metadata_table(rows, copy_columns, input_schema):
    arrays = {
        "faiss_id": pa.array([row["faiss_id"] for row in rows], type=pa.int64()),
        "row_number": pa.array([row["row_number"] for row in rows], type=pa.int64()),
    }

    for column in copy_columns:
        arrays[column] = pa.array(
            [row.get(column) for row in rows],
            type=input_schema.field(column).type,
        )

    return pa.table(arrays)


def prepare_offer_rows(rows, renderer, copy_columns, row_number, next_faiss_id):
    prepared_rows = []
    offer_texts = []
    skipped_rows = 0

    for row in rows:
        context = renderer.build_context(row)
        prepared_row = {
            "faiss_id": next_faiss_id,
            "row_number": row_number,
        }

        row_number += 1

        for column in copy_columns:
            prepared_row[column] = row.get(column)

        offer_text = renderer.render_offer_text(row, context=context)
        if not offer_text:
            skipped_rows += 1
            continue

        prepared_rows.append(prepared_row)
        offer_texts.append(offer_text)
        next_faiss_id += 1

    return prepared_rows, offer_texts, row_number, next_faiss_id, skipped_rows


def append_training_sample(training_chunks, embedding_matrix, train_sample_size):
    collected_rows = sum(chunk.shape[0] for chunk in training_chunks)
    remaining_rows = max(0, int(train_sample_size) - collected_rows)
    if remaining_rows <= 0:
        return

    training_chunks.append(embedding_matrix[:remaining_rows].copy())


def concatenate_embedding_chunks(chunks):
    if not chunks:
        return None

    if len(chunks) == 1:
        return chunks[0]

    return np.concatenate(chunks, axis=0)


def add_batches_to_index(index, pending_batches):
    added_rows = 0

    for prepared_rows, embedding_matrix in pending_batches:
        faiss_ids = np.asarray(
            [row["faiss_id"] for row in prepared_rows],
            dtype="int64",
        )
        index.add_with_ids(embedding_matrix, faiss_ids)
        added_rows += len(prepared_rows)

    pending_batches.clear()
    return added_rows


def run_flat_index_build(
    args,
    model,
    cfg,
    device,
    parquet_file,
    copy_columns,
):
    debug_logging = os.environ.get("DEBUG_INDEX_BUILD") == "1"
    artifact_paths = prepare_index_directory(args.output, args.overwrite)
    tokenizer = build_tokenizer(cfg.model.model_name)
    renderer = RowTextRenderer(cfg.data)
    max_offer_length = int(args.max_offer_length) or int(cfg.data.max_offer_length)
    if debug_logging:
        print("flat_build:start", flush=True)

    processed_rows = 0
    skipped_rows = 0
    row_number = 0
    next_faiss_id = 0
    prepared_row_chunks = []
    embedding_chunks = []

    for batch in parquet_file.iter_batches(batch_size=int(args.read_batch_size)):
        rows = batch.to_pylist()
        if args.limit_rows is not None:
            remaining_rows = int(args.limit_rows) - processed_rows
            if remaining_rows <= 0:
                break
            rows = rows[:remaining_rows]

        processed_rows += len(rows)
        if not rows:
            break

        (
            prepared_rows,
            offer_texts,
            row_number,
            next_faiss_id,
            batch_skipped_rows,
        ) = prepare_offer_rows(
            rows,
            renderer,
            copy_columns,
            row_number,
            next_faiss_id,
        )
        skipped_rows += batch_skipped_rows
        if not prepared_rows:
            continue

        if debug_logging:
            print(f"flat_build:prepared={len(prepared_rows)}", flush=True)

        embeddings = encode_texts(
            model,
            tokenizer,
            offer_texts,
            max_offer_length,
            int(args.encode_batch_size),
            device,
        )
        if debug_logging:
            print(f"flat_build:encoded={tuple(embeddings.shape)}", flush=True)
        prepared_row_chunks.append(prepared_rows)
        embedding_chunks.append(embeddings.numpy().astype("float32", copy=False))

    if not prepared_row_chunks or not embedding_chunks:
        raise ValueError("No non-empty offers were available to index")

    prepared_rows = [row for chunk in prepared_row_chunks for row in chunk]
    embedding_matrix = concatenate_embedding_chunks(embedding_chunks)
    embedding_dim = int(embedding_matrix.shape[1])
    if debug_logging:
        print(f"flat_build:concatenated={embedding_matrix.shape}", flush=True)
    index = create_faiss_index(embedding_dim, {"index_type": "flat"})
    faiss_ids = np.asarray([row["faiss_id"] for row in prepared_rows], dtype="int64")
    index.add_with_ids(embedding_matrix, faiss_ids)
    if debug_logging:
        print("flat_build:indexed", flush=True)

    metadata_table = build_metadata_table(
        prepared_rows,
        copy_columns,
        parquet_file.schema_arrow,
    )
    pq.write_table(
        metadata_table,
        artifact_paths["metadata_file"],
        compression=args.compression,
    )
    if debug_logging:
        print("flat_build:metadata_written", flush=True)
    faiss.write_index(index, str(artifact_paths["index_file"]))
    if debug_logging:
        print("flat_build:index_written", flush=True)
    write_manifest(
        artifact_paths["manifest_file"],
        {
            "checkpoint": str(args.checkpoint),
            "copy_columns": copy_columns,
            "embedding_dim": embedding_dim,
            "index_file": artifact_paths["index_file"].name,
            "index_type": "flat",
            "index_config": build_index_config(args),
            "indexed_rows": len(prepared_rows),
            "input": str(args.input),
            "metadata_file": artifact_paths["metadata_file"].name,
            "metric": "inner_product",
            "processed_rows": processed_rows,
            "query_model_name": str(cfg.model.model_name),
            "skipped_rows": skipped_rows,
        },
    )
    if debug_logging:
        print("flat_build:manifest_written", flush=True)

    print(
        "Index build complete:",
        {
            "device": str(device),
            "index_type": "flat",
            "processed_rows": processed_rows,
            "indexed_rows": len(prepared_rows),
            "skipped_rows": skipped_rows,
            "embedding_dim": embedding_dim,
            "output": str(artifact_paths["index_dir"]),
        },
    )


def run_index_build(args):
    device = resolve_device(args.device)
    index_config = build_index_config(args)
    index_type = resolve_faiss_index_type(index_config["index_type"])
    requires_training = index_requires_training(index_type)

    model, cfg = load_index_model(args)
    model = model.to(device)
    model.eval()

    tokenizer = build_tokenizer(cfg.model.model_name)
    renderer = RowTextRenderer(cfg.data)
    parquet_file = ParquetSource(args.input)
    total_rows = parquet_file.metadata.num_rows
    if args.limit_rows is not None:
        total_rows = min(total_rows, int(args.limit_rows))
    copy_columns = parse_copy_columns(args.copy_columns, parquet_file.schema.names)

    max_offer_length = int(args.max_offer_length) or int(cfg.data.max_offer_length)

    if index_type == "flat":
        return run_flat_index_build(
            args,
            model,
            cfg,
            device,
            parquet_file,
            copy_columns,
        )

    artifact_paths = prepare_index_directory(args.output, args.overwrite)
    metadata_writer = IncrementalParquetWriter(
        artifact_paths["metadata_file"],
        args.compression,
        overwrite=True,
    )
    metadata_buffer = []
    metadata_chunk_rows = int(args.metadata_chunk_rows)

    preloaded_index = None
    preloaded_embedding_dim = None
    if args.trained_index:
        preloaded_index = faiss.read_index(str(args.trained_index))
        if not preloaded_index.is_trained:
            raise ValueError(
                f"Pre-trained index at {args.trained_index} is not trained"
            )
        preloaded_embedding_dim = int(preloaded_index.d)
        requires_training = False

    def flush_metadata_buffer(force=False):
        if not metadata_buffer:
            return
        if not force and len(metadata_buffer) < metadata_chunk_rows:
            return
        t = time.perf_counter()
        table = build_metadata_table(
            metadata_buffer,
            copy_columns,
            parquet_file.schema_arrow,
        )
        metadata_writer.write_table(table)
        phase_times["metadata_write"] += time.perf_counter() - t
        metadata_buffer.clear()

    processed_rows = 0
    indexed_rows = 0
    skipped_rows = 0
    embedding_dim = preloaded_embedding_dim
    index = preloaded_index
    pending_batches = []
    training_chunks = []
    train_sample_size = int(index_config["train_sample_size"])
    minimum_train_vectors = minimum_training_vectors(index_type, index_config)

    if requires_training and train_sample_size < minimum_train_vectors:
        raise ValueError(
            "--train-sample-size must be at least the minimum training size for the "
            f"selected index ({minimum_train_vectors})"
        )

    row_number = int(args.faiss_id_offset)
    next_faiss_id = int(args.faiss_id_offset)

    phase_times = {
        "read_parquet": 0.0,
        "prepare_rows": 0.0,
        "encode": 0.0,
        "await_flush": 0.0,
        "submit_flush": 0.0,
        "train": 0.0,
        "metadata_write": 0.0,
    }

    flush_executor = ThreadPoolExecutor(max_workers=1)
    pending_flush_future = None

    def await_pending_flush():
        nonlocal pending_flush_future
        if pending_flush_future is None:
            return 0
        try:
            return int(pending_flush_future.result())
        finally:
            pending_flush_future = None

    with build_progress(total_rows) as progress:
        task_id = progress.add_task(
            "Indexing",
            total=max(total_rows, 1),
            indexed_rows=0,
            skipped_rows=0,
        )

        try:
            batch_iter = parquet_file.iter_batches(
                batch_size=int(args.read_batch_size)
            )
            while True:
                t = time.perf_counter()
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    phase_times["read_parquet"] += time.perf_counter() - t
                    break
                phase_times["read_parquet"] += time.perf_counter() - t

                rows = batch.to_pylist()
                if args.limit_rows is not None:
                    remaining_rows = int(args.limit_rows) - processed_rows
                    if remaining_rows <= 0:
                        break
                    rows = rows[:remaining_rows]

                processed_rows += len(rows)
                if not rows:
                    break

                t = time.perf_counter()
                (
                    prepared_rows,
                    offer_texts,
                    row_number,
                    next_faiss_id,
                    batch_skipped_rows,
                ) = prepare_offer_rows(
                    rows,
                    renderer,
                    copy_columns,
                    row_number,
                    next_faiss_id,
                )
                phase_times["prepare_rows"] += time.perf_counter() - t
                skipped_rows += batch_skipped_rows

                if prepared_rows:
                    metadata_buffer.extend(prepared_rows)
                    flush_metadata_buffer()

                    t = time.perf_counter()
                    embeddings = encode_texts(
                        model,
                        tokenizer,
                        offer_texts,
                        max_offer_length,
                        int(args.encode_batch_size),
                        device,
                        phase_times=phase_times,
                    )
                    embedding_matrix = embeddings.numpy().astype("float32", copy=False)
                    phase_times["encode"] += time.perf_counter() - t

                    if index is None:
                        embedding_dim = int(embeddings.size(1))
                        index = create_faiss_index(embedding_dim, index_config)
                    elif embedding_dim != int(embeddings.size(1)):
                        raise ValueError(
                            f"Embedding dim mismatch: model produced "
                            f"{embeddings.size(1)} but pre-trained index expects "
                            f"{embedding_dim}"
                        )

                    if requires_training and not index.is_trained:
                        pending_batches.append((prepared_rows, embedding_matrix.copy()))
                        append_training_sample(
                            training_chunks,
                            embedding_matrix,
                            train_sample_size,
                        )

                        train_matrix = concatenate_embedding_chunks(training_chunks)
                        if (
                            train_matrix is not None
                            and train_matrix.shape[0] >= train_sample_size
                        ):
                            if train_matrix.shape[0] < minimum_train_vectors:
                                raise ValueError(
                                    "Not enough embeddings to train the requested FAISS index: "
                                    f"need at least {minimum_train_vectors}, got {train_matrix.shape[0]}"
                                )

                            t = time.perf_counter()
                            index.train(train_matrix)
                            phase_times["train"] += time.perf_counter() - t
                            indexed_rows += add_batches_to_index(
                                index,
                                pending_batches,
                            )
                    else:
                        t = time.perf_counter()
                        indexed_rows += await_pending_flush()
                        phase_times["await_flush"] += time.perf_counter() - t

                        t = time.perf_counter()
                        pending_flush_future = flush_executor.submit(
                            add_batches_to_index,
                            index,
                            [(prepared_rows, embedding_matrix)],
                        )
                        phase_times["submit_flush"] += time.perf_counter() - t

                update_progress(
                    progress,
                    task_id,
                    processed_rows,
                    total_rows,
                    indexed_rows,
                    skipped_rows,
                )

            t = time.perf_counter()
            indexed_rows += await_pending_flush()
            phase_times["await_flush"] += time.perf_counter() - t
        finally:
            try:
                indexed_rows += await_pending_flush()
            finally:
                flush_executor.shutdown(wait=True)

    if index is None or embedding_dim is None or next_faiss_id == 0:
        raise ValueError("No non-empty offers were available to index")

    if requires_training and not index.is_trained:
        train_matrix = concatenate_embedding_chunks(training_chunks)
        if train_matrix is None or train_matrix.shape[0] < minimum_train_vectors:
            available_vectors = (
                0 if train_matrix is None else int(train_matrix.shape[0])
            )
            raise ValueError(
                "Not enough embeddings to train the requested FAISS index: "
                f"need at least {minimum_train_vectors}, got {available_vectors}"
            )

        t = time.perf_counter()
        index.train(train_matrix)
        phase_times["train"] += time.perf_counter() - t
        indexed_rows = add_batches_to_index(index, pending_batches)

    flush_metadata_buffer(force=True)
    metadata_writer.close()

    faiss.write_index(index, str(artifact_paths["index_file"]))
    write_manifest(
        artifact_paths["manifest_file"],
        {
            "checkpoint": str(args.checkpoint),
            "copy_columns": copy_columns,
            "embedding_dim": embedding_dim,
            "index_file": artifact_paths["index_file"].name,
            "index_type": index_type,
            "index_config": index_config,
            "indexed_rows": indexed_rows,
            "input": str(args.input),
            "metadata_file": artifact_paths["metadata_file"].name,
            "metric": "inner_product",
            "processed_rows": processed_rows,
            "query_model_name": str(cfg.model.model_name),
            "skipped_rows": skipped_rows,
        },
    )

    print(
        "Index build complete:",
        {
            "device": str(device),
            "index_type": index_type,
            "processed_rows": processed_rows,
            "indexed_rows": indexed_rows,
            "skipped_rows": skipped_rows,
            "embedding_dim": embedding_dim,
            "output": str(artifact_paths["index_dir"]),
        },
    )
    print(
        "Phase times (s):",
        {name: round(value, 3) for name, value in phase_times.items()},
    )


def main(argv=None):
    load_dotenv()
    torch.set_float32_matmul_precision("high")
    args = build_arg_parser().parse_args(argv)
    run_index_build(args)


def load_index_model(args):
    if args.checkpoint:
        return load_embedding_module_from_checkpoint(
            args.checkpoint, map_location="cpu"
        )

    cfg = load_base_config()
    if args.model_name:
        cfg = OmegaConf.merge(
            cfg,
            OmegaConf.create({"model": {"model_name": args.model_name}}),
        )

    model = EmbeddingModule(cfg)
    model.eval()
    return model, cfg


if __name__ == "__main__":
    main()

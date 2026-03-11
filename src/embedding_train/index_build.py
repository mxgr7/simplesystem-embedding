import argparse

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from dotenv import load_dotenv

from embedding_train.index_artifact import prepare_index_directory, write_manifest
from embedding_train.infer import (
    IncrementalParquetWriter,
    build_tokenizer,
    encode_texts,
    parse_copy_columns,
    resolve_device,
)
from embedding_train.model import load_embedding_module_from_checkpoint
from embedding_train.rendering import RowTextRenderer


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Build a FAISS offer index from a Parquet dataset using the training offer template."
    )
    parser.add_argument("--checkpoint", required=True, help="Lightning checkpoint path")
    parser.add_argument("--input", required=True, help="Input Parquet path")
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
        "offer_text": pa.array([row["offer_text"] for row in rows], type=pa.string()),
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

        prepared_row["offer_text"] = offer_text
        prepared_rows.append(prepared_row)
        offer_texts.append(offer_text)
        next_faiss_id += 1

    return prepared_rows, offer_texts, row_number, next_faiss_id, skipped_rows


def run_index_build(args):
    device = resolve_device(args.device)

    model, cfg = load_embedding_module_from_checkpoint(
        args.checkpoint, map_location="cpu"
    )
    model = model.to(device)
    model.eval()

    tokenizer = build_tokenizer(cfg.model.model_name)
    renderer = RowTextRenderer(cfg.data)
    parquet_file = pq.ParquetFile(args.input)
    copy_columns = parse_copy_columns(args.copy_columns, parquet_file.schema.names)
    artifact_paths = prepare_index_directory(args.output, args.overwrite)
    metadata_writer = IncrementalParquetWriter(
        artifact_paths["metadata_file"],
        args.compression,
        overwrite=True,
    )

    processed_rows = 0
    indexed_rows = 0
    skipped_rows = 0
    row_number = 0
    next_faiss_id = 0
    embedding_dim = None
    index = None

    try:
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

            embeddings = encode_texts(
                model,
                tokenizer,
                offer_texts,
                int(cfg.data.max_offer_length),
                int(args.encode_batch_size),
                device,
            )
            embedding_matrix = embeddings.numpy().astype("float32", copy=False)

            if index is None:
                embedding_dim = int(embeddings.size(1))
                index = faiss.IndexIDMap2(faiss.IndexFlatIP(embedding_dim))

            faiss_ids = np.asarray(
                [row["faiss_id"] for row in prepared_rows],
                dtype="int64",
            )
            index.add_with_ids(embedding_matrix, faiss_ids)

            metadata_table = build_metadata_table(
                prepared_rows,
                copy_columns,
                parquet_file.schema_arrow,
            )
            metadata_writer.write_table(metadata_table)
            indexed_rows += len(prepared_rows)
    finally:
        metadata_writer.close()

    if index is None or embedding_dim is None or indexed_rows == 0:
        raise ValueError("No non-empty offers were available to index")

    faiss.write_index(index, str(artifact_paths["index_file"]))
    write_manifest(
        artifact_paths["manifest_file"],
        {
            "checkpoint": str(args.checkpoint),
            "copy_columns": copy_columns,
            "embedding_dim": embedding_dim,
            "index_file": artifact_paths["index_file"].name,
            "index_type": "IndexIDMap2(IndexFlatIP)",
            "indexed_rows": indexed_rows,
            "input": str(args.input),
            "metadata_file": artifact_paths["metadata_file"].name,
            "metric": "inner_product",
            "processed_rows": processed_rows,
            "query_model_name": str(cfg.model.model_name),
            "rendered_text_column": "offer_text",
            "skipped_rows": skipped_rows,
        },
    )

    print(
        "Index build complete:",
        {
            "device": str(device),
            "processed_rows": processed_rows,
            "indexed_rows": indexed_rows,
            "skipped_rows": skipped_rows,
            "embedding_dim": embedding_dim,
            "output": str(artifact_paths["index_dir"]),
        },
    )


def main(argv=None):
    load_dotenv()
    torch.set_float32_matmul_precision("high")
    args = build_arg_parser().parse_args(argv)
    run_index_build(args)


if __name__ == "__main__":
    main()

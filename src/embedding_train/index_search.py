import argparse
import io

import faiss
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from embedding_train.config import load_base_config
from embedding_train.faiss_index import apply_search_parameters
from embedding_train.index_artifact import read_manifest, resolve_index_paths
from embedding_train.infer import (
    IncrementalParquetWriter,
    build_tokenizer,
    encode_texts,
    default_copy_columns_from_renderer,
    parse_copy_columns,
    resolve_device,
)
from embedding_train.model import EmbeddingModule
from embedding_train.model import load_embedding_module_from_checkpoint
from embedding_train.rendering import RowTextRenderer
from embedding_train.text import normalize_text


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Search a built FAISS offer index using either Parquet queries or raw query text."
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
        "--index",
        required=True,
        help="Index artifact directory produced by embedding-index-build.",
    )
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--input", help="Parquet file containing query rows.")
    query_group.add_argument(
        "--query-text",
        action="append",
        help="Raw query text to search. Repeat the flag to search multiple queries.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output Parquet path. Required when searching Parquet input.",
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
        help="Rows to stream from query Parquet at a time.",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=128,
        help="Texts to tokenize and encode per forward pass.",
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=None,
        help="Override the IVF nprobe value for search.",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=None,
        help="Override the HNSW efSearch value for search.",
    )
    parser.add_argument(
        "--copy-columns",
        default="",
        help="Comma-separated query input columns to copy to the search output.",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        help="Parquet compression codec for search results.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Maximum nearest neighbors to return for each query.",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Optional maximum number of query rows to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser


def load_index_artifact(index_path):
    artifact_paths = resolve_index_paths(index_path)
    manifest = read_manifest(artifact_paths["manifest_file"])
    index = faiss.read_index(str(artifact_paths["index_file"]))
    metadata_table = pq.read_table(artifact_paths["metadata_file"])
    metadata_rows = metadata_table.to_pylist()
    metadata_by_id = {int(row["faiss_id"]): row for row in metadata_rows}
    return artifact_paths, manifest, index, metadata_by_id


def build_query_rows(rows, renderer, copy_columns, row_number):
    prepared_rows = []
    query_texts = []
    skipped_rows = 0

    for row in rows:
        context = renderer.build_context(row)
        prepared_row = {"query_row_number": row_number}
        row_number += 1

        for column in copy_columns:
            prepared_row[column] = row.get(column)

        query_text = renderer.render_query_text(row, context=context)
        if not query_text:
            skipped_rows += 1
            continue

        prepared_row["query_text"] = query_text
        prepared_rows.append(prepared_row)
        query_texts.append(query_text)

    return prepared_rows, query_texts, row_number, skipped_rows


def build_raw_query_rows(query_texts, renderer):
    prepared_rows = []
    normalized_queries = []
    skipped_rows = 0

    for index, query_text in enumerate(query_texts or []):
        raw_query_term = normalize_text(query_text)
        if not raw_query_term:
            skipped_rows += 1
            continue

        query_row = {"query_term": raw_query_term}
        rendered_query = renderer.render_query_text(query_row)
        if not rendered_query:
            skipped_rows += 1
            continue

        prepared_rows.append(
            {
                "query_row_number": index,
                "query_text": rendered_query,
                "raw_query_term": raw_query_term,
            }
        )
        normalized_queries.append(rendered_query)

    return prepared_rows, normalized_queries, skipped_rows


def build_result_rows(query_rows, scores, indices, metadata_by_id):
    result_rows = []

    for query_row, query_scores, query_indices in zip(query_rows, scores, indices):
        for rank, (score, faiss_id) in enumerate(
            zip(query_scores, query_indices), start=1
        ):
            if int(faiss_id) < 0:
                continue

            metadata_row = metadata_by_id.get(int(faiss_id))
            if metadata_row is None:
                continue

            result_row = dict(query_row)
            result_row["rank"] = rank
            result_row["score"] = float(score)

            for key, value in metadata_row.items():
                result_row[f"match_{key}"] = value

            result_rows.append(result_row)

    return result_rows


def build_result_table(rows):
    if not rows:
        return pa.table(
            {
                "query_row_number": pa.array([], type=pa.int64()),
                "query_text": pa.array([], type=pa.string()),
                "rank": pa.array([], type=pa.int64()),
                "score": pa.array([], type=pa.float32()),
                "match_faiss_id": pa.array([], type=pa.int64()),
                "match_row_number": pa.array([], type=pa.int64()),
                "match_offer_text": pa.array([], type=pa.string()),
            }
        )

    return pa.Table.from_pylist(rows)


def format_search_report(query_rows, result_rows):
    output_buffer = io.StringIO()
    console = Console(
        file=output_buffer,
        record=True,
        force_terminal=False,
        width=120,
    )

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold cyan")
    summary.add_column()
    summary.add_row("Queries", str(len(query_rows)))
    summary.add_row("Matches", str(len(result_rows)))

    result_table = Table(title="FAISS Search Results")
    result_table.add_column("Query", style="bold")
    result_table.add_column("Rank", justify="right")
    result_table.add_column("Score", justify="right")
    result_table.add_column("Offer ID")
    result_table.add_column("Offer Text")

    for row in result_rows:
        result_table.add_row(
            row.get("query_text", ""),
            str(row.get("rank", "")),
            f"{float(row.get('score', 0.0)):.6f}",
            str(row.get("match_offer_id_b64", "")),
            str(row.get("match_offer_text", "")),
        )

    console.print(Panel(summary, title="Index Search", expand=False))
    console.print(result_table)
    return console.export_text().rstrip()


def search_embeddings(index, metadata_by_id, query_rows, query_embeddings, top_k):
    scores, indices = index.search(
        query_embeddings.numpy().astype("float32", copy=False),
        min(int(top_k), max(1, index.ntotal)),
    )
    return build_result_rows(query_rows, scores, indices, metadata_by_id)


def run_index_search(args):
    if args.input and not args.output:
        raise ValueError("--output is required when searching Parquet input")

    if int(args.top_k) < 1:
        raise ValueError("--top-k must be at least 1")

    device = resolve_device(args.device)
    artifact_paths, manifest, index, metadata_by_id = load_index_artifact(args.index)
    apply_search_parameters(
        index,
        manifest.get("index_config", {"index_type": manifest["index_type"]}),
        nprobe=args.nprobe,
        ef_search=args.ef_search,
    )

    model, cfg = load_search_model(args)
    model = model.to(device)
    model.eval()

    tokenizer = build_tokenizer(cfg.model.model_name)
    renderer = RowTextRenderer(cfg.data)
    expected_dim = int(manifest["embedding_dim"])

    if args.query_text:
        query_rows, query_texts, skipped_rows = build_raw_query_rows(
            args.query_text,
            renderer,
        )
        if not query_rows:
            raise ValueError("No non-empty query text was provided")

        query_embeddings = encode_texts(
            model,
            tokenizer,
            query_texts,
            int(cfg.data.max_query_length),
            int(args.encode_batch_size),
            device,
        )
        if int(query_embeddings.size(1)) != expected_dim:
            raise ValueError(
                "Query embedding dimension does not match the built index: "
                f"{query_embeddings.size(1)} != {expected_dim}"
            )

        result_rows = search_embeddings(
            index,
            metadata_by_id,
            query_rows,
            query_embeddings,
            args.top_k,
        )

        if args.output:
            writer = IncrementalParquetWriter(
                args.output, args.compression, args.overwrite
            )
            try:
                writer.write_table(build_result_table(result_rows))
            finally:
                writer.close()

            print(
                "Index search complete:",
                {
                    "device": str(device),
                    "queries": len(query_rows),
                    "matches": len(result_rows),
                    "skipped_rows": skipped_rows,
                    "output": str(args.output),
                    "index": str(artifact_paths["index_dir"]),
                },
            )
            return result_rows

        print(format_search_report(query_rows, result_rows))
        return result_rows

    parquet_file = pq.ParquetFile(args.input)
    copy_columns = parse_copy_columns(
        args.copy_columns,
        parquet_file.schema.names,
        default_copy_columns_from_renderer(renderer),
    )
    writer = IncrementalParquetWriter(args.output, args.compression, args.overwrite)

    processed_rows = 0
    searched_rows = 0
    skipped_rows = 0
    written_rows = 0
    row_number = 0

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

            query_rows, query_texts, row_number, batch_skipped_rows = build_query_rows(
                rows,
                renderer,
                copy_columns,
                row_number,
            )
            skipped_rows += batch_skipped_rows

            if not query_rows:
                continue

            query_embeddings = encode_texts(
                model,
                tokenizer,
                query_texts,
                int(cfg.data.max_query_length),
                int(args.encode_batch_size),
                device,
            )
            if int(query_embeddings.size(1)) != expected_dim:
                raise ValueError(
                    "Query embedding dimension does not match the built index: "
                    f"{query_embeddings.size(1)} != {expected_dim}"
                )

            searched_rows += len(query_rows)
            result_rows = search_embeddings(
                index,
                metadata_by_id,
                query_rows,
                query_embeddings,
                args.top_k,
            )
            if not result_rows:
                continue

            result_table = build_result_table(result_rows)
            writer.write_table(result_table)
            written_rows += result_table.num_rows

        if written_rows == 0:
            writer.write_table(build_result_table([]))
    finally:
        writer.close()

    print(
        "Index search complete:",
        {
            "device": str(device),
            "processed_rows": processed_rows,
            "searched_rows": searched_rows,
            "skipped_rows": skipped_rows,
            "written_rows": written_rows,
            "output": str(args.output),
            "index": str(artifact_paths["index_dir"]),
        },
    )
    return None


def main(argv=None):
    load_dotenv()
    torch.set_float32_matmul_precision("high")
    args = build_arg_parser().parse_args(argv)
    run_index_search(args)


def load_search_model(args):
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

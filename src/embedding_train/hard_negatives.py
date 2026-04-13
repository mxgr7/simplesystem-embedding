import argparse
import sys

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

from embedding_train.config import load_base_config
from embedding_train.faiss_index import apply_search_parameters
from embedding_train.index_artifact import read_manifest, resolve_index_paths
from embedding_train.infer import (
    build_tokenizer,
    encode_texts,
    resolve_device,
)
from embedding_train.model import EmbeddingModule, load_embedding_module_from_checkpoint
from embedding_train.rendering import RowTextRenderer


HARD_NEGATIVE_PROVENANCE = "hard_negative"
SEMI_HARD_NEGATIVE_PROVENANCE = "semi_hard_negative"
VALID_PROVENANCE_VALUES = (HARD_NEGATIVE_PROVENANCE, SEMI_HARD_NEGATIVE_PROVENANCE)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Mine hard or semi-hard negatives for retrieval training. "
            "Encodes queries, searches a FAISS index, excludes known positives, "
            "and writes the offers in a configurable rank band as negatives."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Lightning checkpoint path. If omitted, uses the base pretrained model.",
    )
    parser.add_argument(
        "--model-name",
        default="",
        help="Pretrained model name override when --checkpoint is omitted.",
    )
    parser.add_argument(
        "--index",
        required=True,
        help="Index artifact directory produced by embedding-index-build.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Training Parquet file containing query-offer pairs with labels.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output Parquet path for mined hard negatives.",
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
        help="Query rows to process at a time.",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=128,
        help="Texts to tokenize and encode per forward pass.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Candidates to retrieve per query before filtering positives.",
    )
    parser.add_argument(
        "--max-negatives-per-query",
        type=int,
        default=10,
        help="Maximum negatives to keep per query after positive exclusion.",
    )
    parser.add_argument(
        "--rank-start",
        type=int,
        default=0,
        help=(
            "Zero-based start of the rank band to keep after positive exclusion "
            "(inclusive). 0 keeps from the very top (hard negatives); a higher "
            "value drops the top hardest items to produce semi-hard negatives."
        ),
    )
    parser.add_argument(
        "--rank-end",
        type=int,
        default=None,
        help=(
            "Zero-based end of the rank band to keep after positive exclusion "
            "(exclusive). When omitted, defaults to rank-start + "
            "max-negatives-per-query, so --max-negatives-per-query keeps "
            "controlling the count."
        ),
    )
    parser.add_argument(
        "--provenance",
        default=HARD_NEGATIVE_PROVENANCE,
        choices=list(VALID_PROVENANCE_VALUES),
        help=(
            "Provenance label written to every output row. Use "
            f"'{SEMI_HARD_NEGATIVE_PROVENANCE}' for rank bands that skip the top."
        ),
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
        "--compression",
        default="zstd",
        help="Parquet compression codec.",
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
        help="Overwrite the output file if it already exists.",
    )
    return parser


def build_progress(total_queries):
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=24),
        TaskProgressColumn(),
        TextColumn("[dim]({task.completed:,.0f}/{task.total:,.0f} queries)"),
        TextColumn(
            "[dim]mined={task.fields[mined_negatives]:,.0f}"
        ),
        TimeElapsedColumn(),
        console=Console(file=sys.stderr),
        auto_refresh=False,
        transient=False,
    )


def load_mining_model(args):
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


def build_positive_offer_ids_by_query(
    input_path, positive_label, query_id_column, offer_id_column, limit_rows
):
    parquet_file = pq.ParquetFile(input_path)
    positive_offer_ids_by_query = {}
    processed_rows = 0

    for batch in parquet_file.iter_batches(batch_size=4096):
        rows = batch.to_pylist()
        if limit_rows is not None:
            remaining = int(limit_rows) - processed_rows
            if remaining <= 0:
                break
            rows = rows[:remaining]

        processed_rows += len(rows)

        for row in rows:
            query_id = str(row.get(query_id_column, "") or "").strip()
            offer_id = str(row.get(offer_id_column, "") or "").strip()
            label = str(row.get("label", "") or "").strip()

            if not query_id or not offer_id:
                continue

            if label == positive_label:
                positive_offer_ids_by_query.setdefault(query_id, set()).add(offer_id)

    return positive_offer_ids_by_query


def build_unique_queries(input_path, renderer, query_id_column, limit_rows):
    parquet_file = pq.ParquetFile(input_path)
    seen_query_ids = set()
    queries = []
    processed_rows = 0

    for batch in parquet_file.iter_batches(batch_size=4096):
        rows = batch.to_pylist()
        if limit_rows is not None:
            remaining = int(limit_rows) - processed_rows
            if remaining <= 0:
                break
            rows = rows[:remaining]

        processed_rows += len(rows)

        for row in rows:
            query_id = str(row.get(query_id_column, "") or "").strip()
            if not query_id or query_id in seen_query_ids:
                continue

            query_text = renderer.render_query_text(row)
            if not query_text:
                continue

            seen_query_ids.add(query_id)
            queries.append({"query_id": query_id, "query_text": query_text})

    return queries


def mine_hard_negatives_from_results(
    query_rows,
    scores,
    indices,
    metadata_by_id,
    positive_offer_ids_by_query,
    offer_id_column,
    max_negatives_per_query,
    rank_start=0,
    rank_end=None,
    provenance=HARD_NEGATIVE_PROVENANCE,
):
    if rank_start < 0:
        raise ValueError("rank_start must be non-negative")
    if rank_end is not None and rank_end <= rank_start:
        raise ValueError("rank_end must be greater than rank_start")
    if provenance not in VALID_PROVENANCE_VALUES:
        raise ValueError(
            f"provenance must be one of {VALID_PROVENANCE_VALUES}, got {provenance}"
        )

    if rank_end is None:
        rank_end = rank_start + max_negatives_per_query

    band_capacity = rank_end - rank_start
    per_query_cap = min(max_negatives_per_query, band_capacity)
    result_rows = []

    for query_row, query_scores, query_indices in zip(query_rows, scores, indices):
        query_id = query_row["query_id"]
        query_text = query_row["query_text"]
        positive_offer_ids = positive_offer_ids_by_query.get(query_id, set())
        non_positive_rank = 0
        kept_in_band = 0

        for score, faiss_id in zip(query_scores, query_indices):
            if int(faiss_id) < 0:
                continue

            metadata_row = metadata_by_id.get(int(faiss_id))
            if metadata_row is None:
                continue

            offer_id = str(metadata_row.get(offer_id_column, "") or "").strip()
            if not offer_id:
                continue

            if offer_id in positive_offer_ids:
                continue

            current_rank = non_positive_rank
            non_positive_rank += 1

            if current_rank < rank_start:
                continue
            if current_rank >= rank_end:
                break

            kept_in_band += 1
            result_rows.append({
                "query_id": query_id,
                "query_text": query_text,
                "offer_id": offer_id,
                "offer_text": str(metadata_row.get("offer_text", "")),
                "score": float(score),
                "rank": current_rank + 1,
                "provenance": provenance,
            })

            if kept_in_band >= per_query_cap:
                break

    return result_rows


def build_output_table(rows):
    if not rows:
        return pa.table({
            "query_id": pa.array([], type=pa.string()),
            "query_text": pa.array([], type=pa.string()),
            "offer_id": pa.array([], type=pa.string()),
            "offer_text": pa.array([], type=pa.string()),
            "score": pa.array([], type=pa.float32()),
            "rank": pa.array([], type=pa.int32()),
            "provenance": pa.array([], type=pa.string()),
        })

    return pa.table({
        "query_id": pa.array([r["query_id"] for r in rows], type=pa.string()),
        "query_text": pa.array([r["query_text"] for r in rows], type=pa.string()),
        "offer_id": pa.array([r["offer_id"] for r in rows], type=pa.string()),
        "offer_text": pa.array([r["offer_text"] for r in rows], type=pa.string()),
        "score": pa.array([r["score"] for r in rows], type=pa.float32()),
        "rank": pa.array([r["rank"] for r in rows], type=pa.int32()),
        "provenance": pa.array(
            [r.get("provenance", HARD_NEGATIVE_PROVENANCE) for r in rows],
            type=pa.string(),
        ),
    })


def run_hard_negative_mining(args):
    device = resolve_device(args.device)
    top_k = int(args.top_k)
    max_negatives_per_query = int(args.max_negatives_per_query)
    rank_start = int(args.rank_start)
    rank_end = int(args.rank_end) if args.rank_end is not None else None
    provenance = str(args.provenance)

    if top_k < 1:
        raise ValueError("--top-k must be at least 1")
    if max_negatives_per_query < 1:
        raise ValueError("--max-negatives-per-query must be at least 1")
    if rank_start < 0:
        raise ValueError("--rank-start must be non-negative")
    if rank_end is not None and rank_end <= rank_start:
        raise ValueError("--rank-end must be greater than --rank-start")

    effective_rank_end = (
        rank_end if rank_end is not None else rank_start + max_negatives_per_query
    )
    if top_k < effective_rank_end:
        raise ValueError(
            f"--top-k ({top_k}) must be at least the rank-band end "
            f"({effective_rank_end}); raise --top-k or lower the band."
        )
    if rank_start > 0 and provenance == HARD_NEGATIVE_PROVENANCE:
        print(
            "WARNING: --rank-start > 0 but provenance is 'hard_negative'. "
            "Pass --provenance semi_hard_negative if these rows are semi-hard.",
            file=sys.stderr,
        )

    artifact_paths, manifest, index, metadata_by_id = load_index_artifact(args.index)
    apply_search_parameters(
        index,
        manifest.get("index_config", {"index_type": manifest["index_type"]}),
        nprobe=args.nprobe,
        ef_search=args.ef_search,
    )

    model, cfg = load_mining_model(args)
    model = model.to(device)
    model.eval()

    tokenizer = build_tokenizer(cfg.model.model_name)
    renderer = RowTextRenderer(cfg.data)
    expected_dim = int(manifest["embedding_dim"])
    query_id_column = str(cfg.data.get("query_id_column", "query_id")).strip() or "query_id"
    offer_id_column = str(cfg.data.get("offer_id_column", "offer_id_b64")).strip() or "offer_id_b64"
    positive_label = str(cfg.data.positive_label)

    print("Building positive offer sets from training data...")
    positive_offer_ids_by_query = build_positive_offer_ids_by_query(
        args.input,
        positive_label,
        query_id_column,
        offer_id_column,
        args.limit_rows,
    )
    print(f"  queries with positives: {len(positive_offer_ids_by_query)}")

    print("Collecting unique queries...")
    unique_queries = build_unique_queries(
        args.input, renderer, query_id_column, args.limit_rows,
    )
    print(f"  unique queries: {len(unique_queries)}")

    if not unique_queries:
        raise ValueError("No valid queries found in the input")

    output_path = args.output
    from pathlib import Path

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_file}. Pass --overwrite to replace it."
        )

    writer = pq.ParquetWriter(
        output_path,
        build_output_table([]).schema,
        compression=args.compression,
    )

    total_mined = 0
    total_queries_searched = 0

    with build_progress(len(unique_queries)) as progress:
        task_id = progress.add_task(
            "Mining hard negatives",
            total=max(1, len(unique_queries)),
            mined_negatives=0,
        )

        try:
            for batch_start in range(0, len(unique_queries), int(args.read_batch_size)):
                query_batch = unique_queries[
                    batch_start : batch_start + int(args.read_batch_size)
                ]
                query_texts = [q["query_text"] for q in query_batch]

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
                        f"Query embedding dim {query_embeddings.size(1)} != index dim {expected_dim}"
                    )

                scores, indices = index.search(
                    query_embeddings.numpy().astype("float32", copy=False),
                    min(top_k, max(1, index.ntotal)),
                )

                result_rows = mine_hard_negatives_from_results(
                    query_batch,
                    scores,
                    indices,
                    metadata_by_id,
                    positive_offer_ids_by_query,
                    offer_id_column,
                    max_negatives_per_query,
                    rank_start=rank_start,
                    rank_end=rank_end,
                    provenance=provenance,
                )

                if result_rows:
                    writer.write_table(build_output_table(result_rows))
                    total_mined += len(result_rows)

                total_queries_searched += len(query_batch)
                progress.update(
                    task_id,
                    completed=total_queries_searched,
                    mined_negatives=total_mined,
                )
                progress.refresh()
        finally:
            writer.close()

    print(
        "Hard negative mining complete:",
        {
            "device": str(device),
            "queries_searched": total_queries_searched,
            "queries_with_positives": len(positive_offer_ids_by_query),
            "negatives_mined": total_mined,
            "max_negatives_per_query": max_negatives_per_query,
            "top_k": top_k,
            "rank_start": rank_start,
            "rank_end": effective_rank_end,
            "provenance": provenance,
            "output": str(output_path),
            "index": str(artifact_paths["index_dir"]),
        },
    )


def load_index_artifact(index_path):
    artifact_paths = resolve_index_paths(index_path)
    manifest = read_manifest(artifact_paths["manifest_file"])
    index = faiss.read_index(str(artifact_paths["index_file"]))
    metadata_table = pq.read_table(artifact_paths["metadata_file"])
    metadata_rows = metadata_table.to_pylist()
    metadata_by_id = {int(row["faiss_id"]): row for row in metadata_rows}
    return artifact_paths, manifest, index, metadata_by_id


def main(argv=None):
    load_dotenv()
    torch.set_float32_matmul_precision("high")
    args = build_arg_parser().parse_args(argv)
    run_hard_negative_mining(args)


if __name__ == "__main__":
    main()

import argparse
import io
import sys

import pyarrow.parquet as pq
import torch
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from embedding_train.faiss_index import apply_search_parameters
from embedding_train.infer import build_tokenizer, encode_texts, resolve_device
from embedding_train.index_search import load_index_artifact, search_embeddings
from embedding_train.metrics import (
    RELEVANCE_GAINS,
    compute_exact_retrieval_metrics,
    compute_ranking_metrics,
)
from embedding_train.model import load_embedding_module_from_checkpoint
from embedding_train.precision import resolve_embedding_precision, score_embedding_pairs
from embedding_train.rendering import RowTextRenderer


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate embedding checkpoints with either pairwise precision scoring or indexed retrieval."
    )
    parser.add_argument("--checkpoint", required=True, help="Lightning checkpoint path")
    parser.add_argument("--input", required=True, help="Input Parquet path")
    parser.add_argument(
        "--index",
        default="",
        help="Optional FAISS index artifact directory for retrieval evaluation.",
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
        "--embedding-precision",
        default="float32",
        help="Pairwise scoring precision: float32, float16, int8, sign, or binary.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Maximum neighbors to evaluate per query when --index is provided.",
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=None,
        help="Optional IVF nprobe override when --index is provided.",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=None,
        help="Optional HNSW efSearch override when --index is provided.",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Optional maximum number of input rows to process.",
    )
    return parser


def build_metric_deltas(metrics, baseline_metrics):
    deltas = {}

    for key, value in metrics.items():
        if key not in baseline_metrics:
            continue

        deltas[key] = value - baseline_metrics[key]

    return deltas


def build_progress(total_rows):
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=24),
        TaskProgressColumn(),
        TextColumn("[dim]({task.completed:,.0f}/{task.total:,.0f} rows)"),
        TextColumn(
            "[dim]evaluated={task.fields[evaluated_rows]:,.0f} skipped={task.fields[skipped_rows]:,.0f}"
        ),
        TimeElapsedColumn(),
        console=Console(file=sys.stderr),
        transient=False,
    )


def update_progress(
    progress, task_id, processed_rows, total_rows, evaluated_rows, skipped_rows
):
    progress.update(
        task_id,
        completed=min(processed_rows, total_rows),
        evaluated_rows=evaluated_rows,
        skipped_rows=skipped_rows,
    )


def format_metric_value(value):
    return f"{float(value):.6f}"


def format_delta_value(value):
    formatted = f"{float(value):+.6f}"
    if abs(float(value)) < 0.0000005:
        return "+0.000000"
    return formatted


def format_evaluation_report(report):
    output_buffer = io.StringIO()
    console = Console(
        file=output_buffer,
        record=True,
        force_terminal=False,
        width=100,
    )

    summary_table = Table.grid(padding=(0, 2))
    summary_table.add_column(style="bold cyan")
    summary_table.add_column()

    if report.get("evaluation_mode") == "retrieval":
        summary_table.add_row("Mode", "retrieval")
        summary_table.add_row("Index Type", str(report["index_type"]))
        summary_table.add_row("Top K", f"{int(report['top_k']):,}")
        summary_table.add_row("Processed Rows", f"{int(report['processed_rows']):,}")
        summary_table.add_row(
            "Searched Queries", f"{int(report['searched_queries']):,}"
        )
        summary_table.add_row("Skipped Rows", f"{int(report['skipped_rows']):,}")
    else:
        summary_table.add_row("Precision", str(report["embedding_precision"]))
        summary_table.add_row("Processed Rows", f"{int(report['processed_rows']):,}")
        summary_table.add_row("Evaluated Rows", f"{int(report['evaluated_rows']):,}")
        summary_table.add_row("Skipped Rows", f"{int(report['skipped_rows']):,}")

    metric_table = Table(title="Metrics")
    metric_table.add_column("Metric", style="bold")
    metric_table.add_column("Selected", justify="right")

    if report.get("evaluation_mode") == "retrieval":
        metric_order = [
            "exact_success@1",
            "exact_mrr",
            "exact_recall@5",
            "exact_recall@10",
            "ndcg@1",
            "ndcg@5",
            "ndcg@10",
            "eligible_queries",
            "evaluated_queries",
        ]
    else:
        metric_order = [
            "exact_success@1",
            "exact_mrr",
            "exact_recall@5",
            "exact_recall@10",
            "ndcg@1",
            "ndcg@5",
            "ndcg@10",
            "eligible_queries",
            "evaluated_queries",
        ]
    metrics = report["metrics"]
    baseline_metrics = report.get("baseline_metrics")
    metric_deltas = report.get("metric_deltas")

    if baseline_metrics and metric_deltas:
        metric_table.add_column("Baseline", justify="right")
        metric_table.add_column("Delta", justify="right")
        for name in metric_order:
            if name not in metrics:
                continue
            metric_table.add_row(
                name,
                format_metric_value(metrics[name]),
                format_metric_value(baseline_metrics[name]),
                format_delta_value(metric_deltas[name]),
            )
    else:
        for name in metric_order:
            if name not in metrics:
                continue
            metric_table.add_row(name, format_metric_value(metrics[name]))

    console.print(Panel(summary_table, title="Embedding Evaluation", expand=False))
    console.print(metric_table)
    return console.export_text().rstrip()


def run_evaluation(args):
    if args.index:
        return run_retrieval_evaluation(args)

    device = resolve_device(args.device)
    embedding_precision = resolve_embedding_precision(args.embedding_precision)

    model, cfg = load_embedding_module_from_checkpoint(
        args.checkpoint, map_location="cpu"
    )
    model = model.to(device)
    model.eval()

    tokenizer = build_tokenizer(cfg.model.model_name)
    renderer = RowTextRenderer(cfg.data)
    parquet_file = pq.ParquetFile(args.input)
    total_rows = parquet_file.metadata.num_rows
    if args.limit_rows is not None:
        total_rows = min(total_rows, int(args.limit_rows))

    processed_rows = 0
    evaluated_rows = 0
    skipped_rows = 0
    selected_rows = []
    baseline_rows = []

    with build_progress(total_rows) as progress:
        task_id = progress.add_task(
            "Evaluating",
            total=max(total_rows, 1),
            evaluated_rows=0,
            skipped_rows=0,
        )

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

            records = []
            for row in rows:
                record = renderer.build_training_record(row)
                if record is None:
                    skipped_rows += 1
                    continue
                records.append(record)

            if records:
                evaluated_rows += len(records)
                query_embeddings = encode_texts(
                    model,
                    tokenizer,
                    [record["query_text"] for record in records],
                    int(cfg.data.max_query_length),
                    int(args.encode_batch_size),
                    device,
                )
                offer_embeddings = encode_texts(
                    model,
                    tokenizer,
                    [record["offer_text"] for record in records],
                    int(cfg.data.max_offer_length),
                    int(args.encode_batch_size),
                    device,
                )
                selected_scores = score_embedding_pairs(
                    query_embeddings,
                    offer_embeddings,
                    embedding_precision,
                ).tolist()

                for record, score in zip(records, selected_scores):
                    selected_rows.append(
                        {
                            "query_id": record["query_id"],
                            "score": float(score),
                            "raw_label": record["raw_label"],
                        }
                    )

                if embedding_precision != "float32":
                    baseline_scores = score_embedding_pairs(
                        query_embeddings,
                        offer_embeddings,
                        "float32",
                    ).tolist()
                    for record, score in zip(records, baseline_scores):
                        baseline_rows.append(
                            {
                                "query_id": record["query_id"],
                                "score": float(score),
                                "raw_label": record["raw_label"],
                            }
                        )

            update_progress(
                progress,
                task_id,
                processed_rows,
                total_rows,
                evaluated_rows,
                skipped_rows,
            )

    metrics = compute_ranking_metrics(selected_rows)
    exact_metrics = compute_exact_retrieval_metrics(selected_rows)
    for key, value in exact_metrics.items():
        if key in {"eligible_queries", "evaluated_queries"}:
            continue
        metrics[key] = value
    report = {
        "embedding_precision": embedding_precision,
        "processed_rows": float(processed_rows),
        "evaluated_rows": float(evaluated_rows),
        "skipped_rows": float(skipped_rows),
        "metrics": metrics,
    }

    if embedding_precision != "float32":
        baseline_metrics = compute_ranking_metrics(baseline_rows)
        baseline_exact_metrics = compute_exact_retrieval_metrics(baseline_rows)
        for key, value in baseline_exact_metrics.items():
            if key in {"eligible_queries", "evaluated_queries"}:
                continue
            baseline_metrics[key] = value
        report["baseline_precision"] = "float32"
        report["baseline_metrics"] = baseline_metrics
        report["metric_deltas"] = build_metric_deltas(metrics, baseline_metrics)

    return report


def run_retrieval_evaluation(args):
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

    model, cfg = load_embedding_module_from_checkpoint(
        args.checkpoint, map_location="cpu"
    )
    model = model.to(device)
    model.eval()

    tokenizer = build_tokenizer(cfg.model.model_name)
    renderer = RowTextRenderer(cfg.data)
    retrieval_data = collect_retrieval_data(args, renderer)
    query_rows = retrieval_data["query_rows"]
    if not query_rows:
        raise ValueError("No retrieval queries were available to evaluate")

    first_metadata_row = next(iter(metadata_by_id.values()), None)
    if first_metadata_row is None or "offer_id_b64" not in first_metadata_row:
        raise ValueError(
            "Index metadata is missing offer_id_b64. Rebuild the index with the "
            "default copy columns or pass --copy-columns offer_id_b64."
        )

    expected_dim = int(manifest["embedding_dim"])
    result_rows = []
    batch_size = int(args.encode_batch_size)

    for start in range(0, len(query_rows), batch_size):
        batch_query_rows = query_rows[start : start + batch_size]
        query_embeddings = encode_texts(
            model,
            tokenizer,
            [row["query_text"] for row in batch_query_rows],
            int(cfg.data.max_query_length),
            batch_size,
            device,
        )
        if int(query_embeddings.size(1)) != expected_dim:
            raise ValueError(
                "Query embedding dimension does not match the built index: "
                f"{query_embeddings.size(1)} != {expected_dim}"
            )

        batch_result_rows = search_embeddings(
            index,
            metadata_by_id,
            batch_query_rows,
            query_embeddings,
            args.top_k,
        )
        annotate_retrieval_labels(
            batch_result_rows,
            retrieval_data["label_by_query_offer"],
        )
        result_rows.extend(batch_result_rows)

    metric_ks = tuple(k for k in (1, 5, 10) if k <= int(args.top_k))
    metrics = compute_exact_retrieval_metrics(
        result_rows,
        ks=metric_ks,
        evaluated_query_ids=[row["query_id"] for row in query_rows],
        eligible_query_ids=retrieval_data["eligible_query_ids"],
    )
    ranking_metrics = compute_ranking_metrics(result_rows, ks=metric_ks)
    for key, value in ranking_metrics.items():
        if key in {"eligible_queries", "evaluated_queries"}:
            continue
        metrics[key] = value

    return {
        "evaluation_mode": "retrieval",
        "index": str(artifact_paths["index_dir"]),
        "index_type": manifest["index_type"],
        "top_k": float(args.top_k),
        "processed_rows": float(retrieval_data["processed_rows"]),
        "searched_queries": float(len(query_rows)),
        "skipped_rows": float(retrieval_data["skipped_rows"]),
        "metrics": metrics,
    }


def collect_retrieval_data(args, renderer):
    parquet_file = pq.ParquetFile(args.input)
    total_rows = parquet_file.metadata.num_rows
    if args.limit_rows is not None:
        total_rows = min(total_rows, int(args.limit_rows))

    processed_rows = 0
    skipped_rows = 0
    query_rows_by_id = {}
    eligible_query_ids = set()
    label_by_query_offer = {}

    with build_progress(total_rows) as progress:
        task_id = progress.add_task(
            "Evaluating",
            total=max(total_rows, 1),
            evaluated_rows=0,
            skipped_rows=0,
        )

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

            for row in rows:
                record = renderer.build_training_record(row)
                if record is None:
                    skipped_rows += 1
                    continue

                query_id = record["query_id"]
                offer_id = record["offer_id"]
                if not query_id or not offer_id:
                    skipped_rows += 1
                    continue

                if query_id not in query_rows_by_id:
                    query_rows_by_id[query_id] = {
                        "query_id": query_id,
                        "query_text": record["query_text"],
                    }

                if record["raw_label"] == "Exact":
                    eligible_query_ids.add(query_id)

                label_key = (query_id, offer_id)
                existing_label = label_by_query_offer.get(label_key)
                if (
                    existing_label is None
                    or RELEVANCE_GAINS[record["raw_label"]]
                    > RELEVANCE_GAINS[existing_label]
                ):
                    label_by_query_offer[label_key] = record["raw_label"]

            update_progress(
                progress,
                task_id,
                processed_rows,
                total_rows,
                len(query_rows_by_id),
                skipped_rows,
            )

    return {
        "processed_rows": processed_rows,
        "skipped_rows": skipped_rows,
        "query_rows": list(query_rows_by_id.values()),
        "eligible_query_ids": sorted(eligible_query_ids),
        "label_by_query_offer": label_by_query_offer,
    }


def annotate_retrieval_labels(result_rows, label_by_query_offer):
    for row in result_rows:
        offer_id = row.get("match_offer_id_b64")
        row["raw_label"] = label_by_query_offer.get(
            (row["query_id"], offer_id),
            "Irrelevant",
        )


def main(argv=None):
    load_dotenv()
    torch.set_float32_matmul_precision("high")
    args = build_arg_parser().parse_args(argv)
    report = run_evaluation(args)
    print(format_evaluation_report(report))


if __name__ == "__main__":
    main()

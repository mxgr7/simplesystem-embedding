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

from embedding_train.infer import build_tokenizer, encode_texts, resolve_device
from embedding_train.metrics import compute_ranking_metrics
from embedding_train.model import load_embedding_module_from_checkpoint
from embedding_train.precision import resolve_embedding_precision, score_embedding_pairs
from embedding_train.rendering import RowTextRenderer


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate nDCG for checkpoint scoring under different embedding precisions."
    )
    parser.add_argument("--checkpoint", required=True, help="Lightning checkpoint path")
    parser.add_argument("--input", required=True, help="Input Parquet path")
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
        help="Embedding precision to evaluate: float32, float16, int8, sign, or binary.",
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
    summary_table.add_row("Precision", str(report["embedding_precision"]))
    summary_table.add_row("Processed Rows", f"{int(report['processed_rows']):,}")
    summary_table.add_row("Evaluated Rows", f"{int(report['evaluated_rows']):,}")
    summary_table.add_row("Skipped Rows", f"{int(report['skipped_rows']):,}")

    metric_table = Table(title="Metrics")
    metric_table.add_column("Metric", style="bold")
    metric_table.add_column("Selected", justify="right")

    metric_order = [
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
    report = {
        "embedding_precision": embedding_precision,
        "processed_rows": float(processed_rows),
        "evaluated_rows": float(evaluated_rows),
        "skipped_rows": float(skipped_rows),
        "metrics": metrics,
    }

    if embedding_precision != "float32":
        baseline_metrics = compute_ranking_metrics(baseline_rows)
        report["baseline_precision"] = "float32"
        report["baseline_metrics"] = baseline_metrics
        report["metric_deltas"] = build_metric_deltas(metrics, baseline_metrics)

    return report


def main(argv=None):
    load_dotenv()
    torch.set_float32_matmul_precision("high")
    args = build_arg_parser().parse_args(argv)
    report = run_evaluation(args)
    print(format_evaluation_report(report))


if __name__ == "__main__":
    main()

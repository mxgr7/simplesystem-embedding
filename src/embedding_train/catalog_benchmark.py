import argparse
import io
import math
from collections import defaultdict

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from omegaconf import OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from embedding_train.config import load_base_config
from embedding_train.infer import build_tokenizer, encode_texts, resolve_device
from embedding_train.metrics import RELEVANCE_GAINS
from embedding_train.model import EmbeddingModule
from embedding_train.model import load_embedding_module_from_checkpoint
from embedding_train.rendering import RowTextRenderer
from embedding_train.text import normalize_text


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark exact query-to-catalog retrieval with exhaustive embedding scoring."
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
        "--query-template",
        default="",
        help="Optional query template override for benchmark runs.",
    )
    parser.add_argument(
        "--offer-template",
        default="",
        help="Optional offer template override for benchmark runs.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Parquet file containing queries, offers, and labels in one table",
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
        help="Rows to stream from the input Parquet file at a time.",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=128,
        help="Texts to tokenize and encode per forward pass.",
    )
    parser.add_argument(
        "--score-batch-size",
        type=int,
        default=128,
        help="Queries to score against the catalog per batch.",
    )
    parser.add_argument(
        "--similarity",
        default="dot",
        choices=["dot", "cosine"],
        help="Exact similarity function used for exhaustive ranking.",
    )
    parser.add_argument(
        "--ks",
        default="1,5,10,100",
        help="Comma-separated cutoffs for ndcg, recall, and precision.",
    )
    parser.add_argument(
        "--relevant-labels",
        default="Exact",
        help="Comma-separated labels treated as relevant for recall, mrr, and precision.",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Optional maximum number of input rows to process.",
    )
    return parser


def parse_comma_separated_ints(raw_value):
    values = []
    for part in str(raw_value).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 1:
            raise ValueError("K values must be at least 1")
        values.append(value)

    if not values:
        raise ValueError("At least one K value is required")

    return tuple(sorted(dict.fromkeys(values)))


def parse_relevant_labels(raw_value):
    labels = []
    for part in str(raw_value).split(","):
        normalized = normalize_text(part)
        if normalized:
            labels.append(normalized)

    if not labels:
        raise ValueError("At least one relevant label is required")

    return tuple(dict.fromkeys(labels))


def resolve_row_identifier(row, primary_column, fallback_columns):
    candidate_columns = [primary_column] + list(fallback_columns)
    for column in candidate_columns:
        if not column:
            continue
        value = normalize_text(row.get(column))
        if value:
            return value

    return ""


def collect_benchmark_data(args, renderer):
    parquet_file = pq.ParquetFile(args.input)
    processed_rows = 0
    skipped_rows = 0
    query_rows_by_id = {}
    catalog_rows_by_offer_id = {}
    judgments_by_query = defaultdict(dict)

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
            query_id = resolve_row_identifier(
                row,
                renderer.query_id_column,
                ["query_id"],
            )
            offer_id = resolve_row_identifier(
                row,
                renderer.offer_id_column,
                ["offer_id_b64", "offer_id", "item_id"],
            )
            raw_label = normalize_text(row.get("label"))
            if not query_id or not offer_id or not raw_label:
                skipped_rows += 1
                continue

            if raw_label not in RELEVANCE_GAINS:
                raise ValueError(f"Unknown relevance label: {raw_label}")

            if query_id not in query_rows_by_id:
                query_text = renderer.render_query_text(row)
                if not query_text:
                    skipped_rows += 1
                    continue
                query_rows_by_id[query_id] = {
                    "query_id": query_id,
                    "query_text": query_text,
                }

            if offer_id not in catalog_rows_by_offer_id:
                offer_text = renderer.render_offer_text(row)
                if not offer_text:
                    skipped_rows += 1
                    continue
                catalog_rows_by_offer_id[offer_id] = {
                    "offer_id": offer_id,
                    "offer_text": offer_text,
                }

            existing_label = judgments_by_query[query_id].get(offer_id)
            if existing_label is None:
                judgments_by_query[query_id][offer_id] = raw_label
                continue

            if RELEVANCE_GAINS[raw_label] > RELEVANCE_GAINS[existing_label]:
                judgments_by_query[query_id][offer_id] = raw_label

    return {
        "processed_rows": processed_rows,
        "skipped_rows": skipped_rows,
        "query_rows": list(query_rows_by_id.values()),
        "catalog_rows": list(catalog_rows_by_offer_id.values()),
        "judgments_by_query": dict(judgments_by_query),
    }


def resolve_similarity(mode, query_embeddings, catalog_embeddings):
    if mode == "cosine":
        query_embeddings = F.normalize(query_embeddings, dim=1)
        catalog_embeddings = F.normalize(catalog_embeddings, dim=1)

    return query_embeddings, catalog_embeddings


def compute_dcg_from_gains(gains, k):
    dcg = 0.0
    for index, gain in enumerate(gains[: int(k)], start=1):
        dcg += float(gain) / math.log2(index + 1)
    return dcg


def score_queries_against_catalog(
    query_rows,
    query_embeddings,
    catalog_rows,
    catalog_embeddings,
    judgments_by_query,
    ks,
    relevant_labels,
    score_batch_size,
):
    max_k = max(int(k) for k in ks)
    ndcg_totals = {k: 0.0 for k in ks}
    recall_totals = {k: 0.0 for k in ks}
    precision_totals = {k: 0.0 for k in ks}
    mrr_total = 0.0
    ndcg_eligible_queries = 0
    retrieval_eligible_queries = 0
    catalog_offer_ids = [row["offer_id"] for row in catalog_rows]
    catalog_offer_id_set = set(catalog_offer_ids)

    for start in range(0, len(query_rows), int(score_batch_size)):
        batch_rows = query_rows[start : start + int(score_batch_size)]
        batch_embeddings = query_embeddings[start : start + int(score_batch_size)]
        score_matrix = torch.matmul(batch_embeddings, catalog_embeddings.T)

        for row, score_row in zip(batch_rows, score_matrix):
            query_id = row["query_id"]
            label_by_offer = judgments_by_query.get(query_id, {})
            ideal_gains = sorted(
                [
                    RELEVANCE_GAINS[label]
                    for offer_id, label in label_by_offer.items()
                    if offer_id in catalog_offer_id_set and RELEVANCE_GAINS[label] > 0
                ],
                reverse=True,
            )
            relevant_offer_ids = {
                offer_id
                for offer_id, label in label_by_offer.items()
                if label in relevant_labels and offer_id in catalog_offer_id_set
            }

            ranked_indices = torch.argsort(score_row, descending=True, stable=True)
            ranked_offer_ids = [
                catalog_offer_ids[index] for index in ranked_indices.tolist()
            ]
            top_offer_ids = ranked_offer_ids[:max_k]

            if ideal_gains:
                ndcg_eligible_queries += 1
                predicted_gains = [
                    RELEVANCE_GAINS[label_by_offer.get(offer_id, "Irrelevant")]
                    for offer_id in top_offer_ids
                ]
                for k in ks:
                    dcg = compute_dcg_from_gains(predicted_gains, k)
                    idcg = compute_dcg_from_gains(ideal_gains, k)
                    if idcg > 0:
                        ndcg_totals[k] += dcg / idcg

            if not relevant_offer_ids:
                continue

            retrieval_eligible_queries += 1
            first_relevant_rank = None
            for index, offer_id in enumerate(ranked_offer_ids, start=1):
                if offer_id in relevant_offer_ids:
                    first_relevant_rank = index
                    break

            if first_relevant_rank is not None:
                mrr_total += 1.0 / float(first_relevant_rank)

            for k in ks:
                top_k_offer_ids = top_offer_ids[: int(k)]
                relevant_hits = sum(
                    1.0
                    for offer_id in top_k_offer_ids
                    if offer_id in relevant_offer_ids
                )
                if relevant_hits > 0:
                    recall_totals[k] += 1.0
                precision_totals[k] += relevant_hits / float(int(k))

    metrics = {
        "evaluated_queries": float(len(query_rows)),
        "ndcg_eligible_queries": float(ndcg_eligible_queries),
        "retrieval_eligible_queries": float(retrieval_eligible_queries),
        "mrr": 0.0,
    }

    for k in ks:
        metrics[f"ndcg@{int(k)}"] = 0.0
        metrics[f"recall@{int(k)}"] = 0.0
        metrics[f"precision@{int(k)}"] = 0.0

    if ndcg_eligible_queries > 0:
        for k in ks:
            metrics[f"ndcg@{int(k)}"] = ndcg_totals[k] / float(ndcg_eligible_queries)

    if retrieval_eligible_queries > 0:
        metrics["mrr"] = mrr_total / float(retrieval_eligible_queries)
        for k in ks:
            metrics[f"recall@{int(k)}"] = recall_totals[k] / float(
                retrieval_eligible_queries
            )
            metrics[f"precision@{int(k)}"] = precision_totals[k] / float(
                retrieval_eligible_queries
            )

    return metrics


def format_metric_value(value):
    return f"{float(value):.6f}"


def format_benchmark_report(report):
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
    summary_table.add_row("Similarity", str(report["similarity"]))
    summary_table.add_row("Ks", ", ".join(str(k) for k in report["ks"]))
    summary_table.add_row("Relevant Labels", ", ".join(report["relevant_labels"]))
    summary_table.add_row("Processed Rows", f"{int(report['processed_rows']):,}")
    summary_table.add_row("Skipped Rows", f"{int(report['skipped_rows']):,}")
    summary_table.add_row("Query Count", f"{int(report['query_count']):,}")
    summary_table.add_row("Catalog Size", f"{int(report['catalog_size']):,}")

    metric_table = Table(title="Metrics")
    metric_table.add_column("Metric", style="bold")
    metric_table.add_column("Value", justify="right")

    metric_order = [
        "mrr",
        "evaluated_queries",
        "ndcg_eligible_queries",
        "retrieval_eligible_queries",
    ]
    for k in report["ks"]:
        metric_order.append(f"ndcg@{int(k)}")
    for k in report["ks"]:
        metric_order.append(f"recall@{int(k)}")
    for k in report["ks"]:
        metric_order.append(f"precision@{int(k)}")

    for metric_name in metric_order:
        metric_table.add_row(
            metric_name, format_metric_value(report["metrics"][metric_name])
        )

    console.print(Panel(summary_table, title="Catalog Benchmark", expand=False))
    console.print(metric_table)
    return console.export_text().rstrip()


def run_catalog_benchmark(args):
    ks = parse_comma_separated_ints(args.ks)
    relevant_labels = parse_relevant_labels(args.relevant_labels)
    device = resolve_device(args.device)

    model, cfg = load_benchmark_model(args)
    cfg = apply_template_overrides(cfg, args)
    model = model.to(device)
    model.eval()

    tokenizer = build_tokenizer(cfg.model.model_name)
    renderer = RowTextRenderer(cfg.data)

    benchmark_data = collect_benchmark_data(args, renderer)

    query_rows = benchmark_data["query_rows"]
    catalog_rows = benchmark_data["catalog_rows"]
    if not query_rows:
        raise ValueError("No benchmark queries were available")
    if not catalog_rows:
        raise ValueError("No catalog rows were available")

    query_embeddings = encode_texts(
        model,
        tokenizer,
        [row["query_text"] for row in query_rows],
        int(cfg.data.max_query_length),
        int(args.encode_batch_size),
        device,
    ).to(dtype=torch.float32)
    catalog_embeddings = encode_texts(
        model,
        tokenizer,
        [row["offer_text"] for row in catalog_rows],
        int(cfg.data.max_offer_length),
        int(args.encode_batch_size),
        device,
    ).to(dtype=torch.float32)
    query_embeddings, catalog_embeddings = resolve_similarity(
        args.similarity,
        query_embeddings,
        catalog_embeddings,
    )

    metrics = score_queries_against_catalog(
        query_rows=query_rows,
        query_embeddings=query_embeddings,
        catalog_rows=catalog_rows,
        catalog_embeddings=catalog_embeddings,
        judgments_by_query=benchmark_data["judgments_by_query"],
        ks=ks,
        relevant_labels=set(relevant_labels),
        score_batch_size=args.score_batch_size,
    )

    return {
        "similarity": args.similarity,
        "ks": ks,
        "relevant_labels": relevant_labels,
        "processed_rows": float(benchmark_data["processed_rows"]),
        "skipped_rows": float(benchmark_data["skipped_rows"]),
        "query_count": float(len(query_rows)),
        "catalog_size": float(len(catalog_rows)),
        "metrics": metrics,
    }


def load_benchmark_model(args):
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


def apply_template_overrides(cfg, args):
    data_overrides = {}
    if args.query_template:
        data_overrides["query_template"] = args.query_template
    if args.offer_template:
        data_overrides["offer_template"] = args.offer_template

    if not data_overrides:
        return cfg

    return OmegaConf.merge(cfg, OmegaConf.create({"data": data_overrides}))


def main(argv=None):
    load_dotenv()
    torch.set_float32_matmul_precision("high")
    args = build_arg_parser().parse_args(argv)
    report = run_catalog_benchmark(args)
    print(format_benchmark_report(report))


if __name__ == "__main__":
    main()

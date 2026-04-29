#!/usr/bin/env python3
"""Verify the annotator JSONL output and produce the labeled dataset.

Steps:
 1. Stream the annotations JSONL, parse {"label": ...} from each response.
 2. Print sanity stats: records, parse failures, label distribution, error
    responses, finish_reason mix.
 3. Build a parquet of (example_id, esci_label, raw_label, model,
    annotated_at) using DuckDB.
 4. Join with the unlabeled merged dataset on example_id and write the
    result to queries_offers_merged_labeled.parquet/part-0.parquet.
"""
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import duckdb

ANNOT_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/home/mgerer/annotatorv3/outputs/2026-04-28/21-09-52/annotations-full.jsonl"
)
ROOT = Path("/data/datasets/queries_offers_esci")
SRC_PARQUET = ROOT / "queries_offers_merged.parquet" / "part-0.parquet"
LABELS_DIR = ROOT / "esci_labels.parquet"
LABELS_DIR.mkdir(parents=True, exist_ok=True)
LABELS_PATH = LABELS_DIR / "part-0.parquet"
LABELED_DIR = ROOT / "queries_offers_merged_labeled.parquet"
LABELED_DIR.mkdir(parents=True, exist_ok=True)
LABELED_PATH = LABELED_DIR / "part-0.parquet"

VALID = {"Exact", "Substitute", "Complement", "Irrelevant"}


def stream_records(path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def verify_and_extract():
    print(f"[verify] reading {ANNOT_PATH}")
    label_counts = Counter()
    finish_reasons = Counter()
    parse_failures = 0
    response_errors = 0
    cached_count = 0
    total_cost = 0.0
    total_prompt_tok = 0
    total_completion_tok = 0
    total_reasoning_tok = 0
    n = 0
    invalid_examples = []
    rows_for_parquet = []  # (example_id, label, raw_content, finish_reason)
    model_name = None

    for r in stream_records(ANNOT_PATH):
        n += 1
        ex_id = r.get("row", {}).get("example_id")
        if "cached" in r and r["cached"]:
            cached_count += 1
        resp = r.get("response") or {}
        if model_name is None:
            model_name = resp.get("model")
        usage = resp.get("usage") or {}
        total_cost += float(usage.get("cost") or 0)
        total_prompt_tok += usage.get("prompt_tokens") or 0
        total_completion_tok += usage.get("completion_tokens") or 0
        total_reasoning_tok += (usage.get("completion_tokens_details") or {}).get(
            "reasoning_tokens") or 0

        if not resp:
            response_errors += 1
            rows_for_parquet.append((ex_id, None, None, None))
            continue
        choices = resp.get("choices") or []
        if not choices:
            response_errors += 1
            rows_for_parquet.append((ex_id, None, None, None))
            continue
        msg = choices[0].get("message") or {}
        finish = choices[0].get("finish_reason")
        finish_reasons[finish] += 1
        content = msg.get("content")
        label = None
        try:
            label = (json.loads(content) or {}).get("label")
        except (TypeError, json.JSONDecodeError):
            parse_failures += 1
        if label not in VALID:
            if label is not None:
                if len(invalid_examples) < 5:
                    invalid_examples.append((ex_id, label))
            label_for_storage = None
        else:
            label_for_storage = label
            label_counts[label] += 1
        rows_for_parquet.append((ex_id, label_for_storage, content, finish))

    print(f"\nrecords:                {n:,}")
    print(f"  cached (from prior runs): {cached_count:,}")
    print(f"  response errors:          {response_errors:,}")
    print(f"  parse failures:           {parse_failures:,}")
    print(f"  invalid label values:     {n - cached_count - sum(label_counts.values()) - parse_failures - response_errors}")
    print(f"\nlabel distribution:")
    total_valid = sum(label_counts.values())
    for k in ("Exact", "Substitute", "Complement", "Irrelevant"):
        c = label_counts.get(k, 0)
        pct = c / total_valid * 100 if total_valid else 0
        print(f"  {k:11} {c:>8,}  ({pct:5.1f}%)")
    print(f"\nfinish reasons: {dict(finish_reasons)}")
    if invalid_examples:
        print(f"\nfirst 5 invalid label values: {invalid_examples}")
    print(f"\ncost: ${total_cost:.2f}  (avg ${total_cost/max(n,1):.6f}/row)")
    print(f"prompt tokens:     {total_prompt_tok:,}")
    print(f"completion tokens: {total_completion_tok:,}")
    print(f"reasoning tokens:  {total_reasoning_tok:,}")
    print(f"model: {model_name}")

    return rows_for_parquet


def write_labels_parquet(rows):
    print(f"\n[labels] writing {LABELS_PATH}")
    con = duckdb.connect()
    con.register("rows_view", _rows_relation(con, rows))
    annotated_at = datetime.now(timezone.utc).isoformat()
    con.execute(f"""
        COPY (
            SELECT
                example_id,
                esci_label,
                raw_content,
                finish_reason,
                CAST('{annotated_at}' AS TIMESTAMP WITH TIME ZONE) AS annotated_at
            FROM rows_view
            ORDER BY example_id
        )
        TO '{LABELS_PATH}'
        (FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 9);
    """)
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{LABELS_PATH}')").fetchone()[0]
    print(f"  rows: {n:,}  ({LABELS_PATH.stat().st_size/1e6:.1f} MB)")


def _rows_relation(con, rows):
    """DuckDB doesn't have a direct 'register python list', so use Arrow."""
    import pyarrow as pa
    table = pa.table({
        "example_id": [r[0] for r in rows],
        "esci_label": [r[1] for r in rows],
        "raw_content": [r[2] for r in rows],
        "finish_reason": [r[3] for r in rows],
    })
    return con.from_arrow(table)


def join_labeled():
    """Produce the final dataset in the schema the training pipeline expects.

    Column renames vs the candidates-side intermediate parquet:
        candidate_id -> offer_id   (canonical name from rendering.DEFAULT_COLUMN_MAPPING)
        esci_label   -> label
    All other column names already match the canonical mapping. Note that
    `rendering.DEFAULT_COLUMN_MAPPING["offer_id"]` is `"offer_id_b64"` —
    legacy naming for the base64 offer-level UUIDs in
    queries_offers_unlabeled. Our values are 32-char hex IDs from
    offers_embedded_full.id, not base64, so we use the honest column name
    `offer_id`; consumers should set
    `data.column_mapping.offer_id: offer_id` in their training config.
    """
    print(f"\n[join] producing {LABELED_PATH}")
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute(f"""
        COPY (
            SELECT
                src.example_id,
                src.query_id,
                src.candidate_id      AS offer_id,
                lab.esci_label        AS label,
                src.query_term,
                src.normalized_qt,
                src.frequency_band,
                src.hit_band,
                src.mpn_shape,
                src.hit_count_at_search_time,
                src.platform_language,
                src.rank_hybrid_classified,
                src.rank_vector,
                src.rank_bm25,
                src.score_hybrid_classified,
                src.score_vector,
                src.score_bm25,
                src.source_legs,
                src.name,
                src.manufacturer_name,
                src.description,
                src.category_paths,
                src.ean,
                src.article_number,
                src.manufacturer_article_number,
                src.manufacturer_article_type,
                src.retrieved_at,
                lab.finish_reason     AS esci_finish_reason,
                lab.annotated_at      AS esci_annotated_at
            FROM read_parquet('{SRC_PARQUET}') src
            LEFT JOIN read_parquet('{LABELS_PATH}') lab
              ON src.example_id = lab.example_id
            ORDER BY src.example_id
        )
        TO '{LABELED_PATH}'
        (FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 9);
    """)
    n, n_lab, n_null = con.execute(f"""
        SELECT COUNT(*),
               COUNT(label),
               SUM(CASE WHEN label IS NULL THEN 1 ELSE 0 END)
        FROM read_parquet('{LABELED_PATH}')
    """).fetchone()
    sz = LABELED_PATH.stat().st_size
    print(f"  rows: {n:,}  labeled: {n_lab:,}  unlabeled: {n_null:,}")
    print(f"  size: {sz/1e6:.1f} MB")


def main():
    if not ANNOT_PATH.exists():
        sys.exit(f"annotations file not found: {ANNOT_PATH}")
    rows = verify_and_extract()
    write_labels_parquet(rows)
    join_labeled()
    print("\n[done]")


if __name__ == "__main__":
    main()

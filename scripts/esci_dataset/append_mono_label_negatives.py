#!/usr/bin/env python3
"""Append labelled mono-label-negatives rows to the canonical labeled parquet.

Workflow:
 1. Stream the annotator JSONL for the negatives batch, parse labels.
 2. Print the same sanity stats as extract_esci_labels.py (label dist,
    finish_reasons, parse failures, cost).
 3. Build a slim labels parquet (esci_labels_mono_label_negatives.parquet).
 4. Join with the materialized negatives parquet and assign a `split` value
    matching the parent query (looked up from the existing labeled parquet).
 5. UNION ALL with the existing labeled parquet and write the result back
    in place (.tmp + replace), preserving global example_id ordering.

Usage:
  uv run python scripts/esci_dataset/append_mono_label_negatives.py \
      /home/mgerer/annotatorv3/outputs/<date>/<time>/annotations-full.jsonl
"""
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import duckdb

ROOT = Path("/data/datasets/queries_offers_esci")
LABELED_DIR = ROOT / "queries_offers_merged_labeled.parquet"
LABELED_PATH = LABELED_DIR / "part-0.parquet"
LABELED_TMP = LABELED_DIR / "part-0.parquet.tmp"

NEG_MERGED = ROOT / "queries_offers_merged_mono_label_negatives.parquet" / "part-0.parquet"
LABELS_DIR = ROOT / "esci_labels_mono_label_negatives.parquet"
LABELS_DIR.mkdir(parents=True, exist_ok=True)
LABELS_PATH = LABELS_DIR / "part-0.parquet"

VALID = {"Exact", "Substitute", "Complement", "Irrelevant"}


def stream(path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def verify_and_extract(annot_path: Path):
    print(f"[verify] reading {annot_path}")
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
    rows = []  # (example_id, label, raw_content, finish_reason)
    model_name = None

    for r in stream(annot_path):
        n += 1
        ex_id = r.get("row", {}).get("example_id")
        if r.get("cached"):
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
            rows.append((ex_id, None, None, None))
            continue
        choices = resp.get("choices") or []
        if not choices:
            response_errors += 1
            rows.append((ex_id, None, None, None))
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
            label_for_storage = None
        else:
            label_for_storage = label
            label_counts[label] += 1
        rows.append((ex_id, label_for_storage, content, finish))

    print(f"\nrecords:                {n:,}")
    print(f"  cached:                   {cached_count:,}")
    print(f"  response errors:          {response_errors:,}")
    print(f"  parse failures:           {parse_failures:,}")
    total_valid = sum(label_counts.values())
    print(f"\nlabel distribution:")
    for k in ("Exact", "Substitute", "Complement", "Irrelevant"):
        c = label_counts.get(k, 0)
        pct = c / total_valid * 100 if total_valid else 0
        print(f"  {k:11} {c:>7,}  ({pct:5.1f}%)")
    print(f"\nfinish reasons: {dict(finish_reasons)}")
    print(f"\ncost: ${total_cost:.2f}  (avg ${total_cost/max(n,1):.6f}/row)")
    print(f"tokens: prompt={total_prompt_tok:,}  completion={total_completion_tok:,}  reasoning={total_reasoning_tok:,}")
    print(f"model:  {model_name}")
    return rows


def write_labels_parquet(con, rows):
    print(f"\n[labels] writing {LABELS_PATH}")
    import pyarrow as pa
    table = pa.table({
        "example_id": [r[0] for r in rows],
        "esci_label": [r[1] for r in rows],
        "raw_content": [r[2] for r in rows],
        "finish_reason": [r[3] for r in rows],
    })
    con.register("rows_view", con.from_arrow(table))
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
    n = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{LABELS_PATH}')"
    ).fetchone()[0]
    print(f"  rows: {n:,}  ({LABELS_PATH.stat().st_size/1e6:.1f} MB)")


def append_to_labeled(con):
    print(f"\n[append] producing extended labeled parquet → {LABELED_TMP}")
    con.execute(f"""
        COPY (
            WITH q_split AS (
                SELECT query_id, ANY_VALUE(split) AS split
                FROM read_parquet('{LABELED_PATH}')
                GROUP BY 1
            ),
            new_rows AS (
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
                    lab.annotated_at      AS esci_annotated_at,
                    qs.split
                FROM read_parquet('{NEG_MERGED}') src
                LEFT JOIN read_parquet('{LABELS_PATH}') lab
                  ON src.example_id = lab.example_id
                LEFT JOIN q_split qs ON src.query_id = qs.query_id
            )
            SELECT * FROM read_parquet('{LABELED_PATH}')
            UNION ALL
            SELECT * FROM new_rows
            ORDER BY example_id
        )
        TO '{LABELED_TMP}'
        (FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 9);
    """)
    LABELED_TMP.replace(LABELED_PATH)

    n, n_lab, n_null, mn, mx = con.execute(f"""
        SELECT COUNT(*), COUNT(label),
               SUM(CASE WHEN label IS NULL THEN 1 ELSE 0 END),
               MIN(example_id), MAX(example_id)
        FROM read_parquet('{LABELED_PATH}')
    """).fetchone()
    sz = LABELED_PATH.stat().st_size
    print(f"  rows: {n:,}  labeled: {n_lab:,}  unlabeled: {n_null:,}")
    print(f"  example_id range: {mn:,}..{mx:,}")
    print(f"  size: {sz/1e6:.1f} MB")

    # Mono-label query check after the addendum
    print(f"\n[check] mono-label queries after extension:")
    for label, n in con.execute(f"""
        WITH per_q AS (
            SELECT query_id,
                   COUNT(*) FILTER (WHERE label IS NOT NULL) AS n_labeled,
                   COUNT(DISTINCT label) FILTER (WHERE label IS NOT NULL) AS n_distinct,
                   ANY_VALUE(label) FILTER (WHERE label IS NOT NULL) AS only_label
            FROM read_parquet('{LABELED_PATH}')
            GROUP BY 1
        )
        SELECT only_label, COUNT(*)
        FROM per_q
        WHERE n_labeled > 0 AND n_distinct = 1
        GROUP BY 1 ORDER BY 1
    """).fetchall():
        print(f"  {label:11} {n:>5,}")


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: append_mono_label_negatives.py <annotations.jsonl>")
    annot_path = Path(sys.argv[1])
    if not annot_path.exists():
        sys.exit(f"not found: {annot_path}")

    rows = verify_and_extract(annot_path)
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='32GB'")
    write_labels_parquet(con, rows)
    append_to_labeled(con)
    print("\n[done]")


if __name__ == "__main__":
    main()

"""Pre-flatten offer bucket parquets into the shape milvus_import.py consumes.

For each source `bucket=NN.parquet` (with nested `categoryPaths` and
`vendor_listings`), write a new parquet containing:

  id                   : string
  offer_embedding      : list<halffloat>   (passed through from source)
  vendor_ids           : list<string>      (distinct)
  catalog_version_ids  : list<string>      (distinct)
  category_l1..l5      : list<string>      (distinct, joined prefixes)

Category joining matches `CategoryPath.asStringPath()`:
  - literal U+00A6 (¦) inside elements -> replaced with U+007C (|)
  - elements joined with U+00A6

The flatten runs in DuckDB (vectorized C++). The `offer_embedding` column
is passed through via pyarrow — DuckDB doesn't handle fp16 natively.

Correctness guard: after DuckDB emits its flat table and pyarrow reads
(id, offer_embedding), the id columns are compared before merge. If
DuckDB reordered rows (it shouldn't for a single-file scan without ORDER
BY), the script aborts rather than silently scrambling embeddings.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

SRC_DEFAULT = Path(
    "/mnt/HC_Volume_105463954/simplesystem/data/offers_embedded_full.parquet"
)
DST_DEFAULT = Path(
    "/mnt/HC_Volume_105463954/simplesystem/data/offers_flat.parquet"
)
BUCKET_RE = re.compile(r"^bucket=\d{2}\.parquet$")


def _level_expr(lvl: int) -> str:
    # Replace U+00A6 (chr(166)) inside each element with '|', then join first
    # `lvl` elements with U+00A6. Only paths with >= lvl elements contribute.
    return f"""
    COALESCE(list_distinct(list_transform(
      list_filter(COALESCE(categoryPaths, []), p -> p.elements IS NOT NULL AND len(p.elements) >= {lvl}),
      p -> array_to_string(
        list_transform(list_slice(p.elements, 1, {lvl}), e -> replace(COALESCE(e, ''), chr(166), '|')),
        chr(166)
      )
    )), []::VARCHAR[]) AS category_l{lvl}
    """.strip()


FLATTEN_SQL = f"""
SELECT
  id,
  COALESCE(list_distinct(list_transform(
    list_filter(COALESCE(vendor_listings, []), v -> v.vendor_id IS NOT NULL),
    v -> v.vendor_id)), []::VARCHAR[])                                      AS vendor_ids,
  COALESCE(list_distinct(list_transform(
    list_filter(COALESCE(vendor_listings, []), v -> v.catalog_version_id IS NOT NULL),
    v -> v.catalog_version_id)), []::VARCHAR[])                             AS catalog_version_ids,
  {_level_expr(1)},
  {_level_expr(2)},
  {_level_expr(3)},
  {_level_expr(4)},
  {_level_expr(5)}
FROM read_parquet(?)
"""


def process_bucket(src: Path, dst: Path, threads: int) -> None:
    print(f"--- {src.name} -> {dst.name} ---", flush=True)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")

    t0 = time.time()
    flat = con.execute(FLATTEN_SQL, [str(src)]).fetch_arrow_table()
    print(f"  duckdb flatten: {time.time()-t0:6.1f}s  rows={flat.num_rows:,}", flush=True)

    t0 = time.time()
    emb = pq.read_table(src, columns=["id", "offer_embedding"])
    print(f"  pyarrow read:   {time.time()-t0:6.1f}s", flush=True)

    if flat.num_rows != emb.num_rows:
        raise RuntimeError(
            f"row count mismatch: flat={flat.num_rows:,} emb={emb.num_rows:,}"
        )
    if not flat.column("id").combine_chunks().equals(emb.column("id").combine_chunks()):
        raise RuntimeError(
            f"id columns diverged in {src.name}; DuckDB scan reordered rows"
        )

    combined = pa.Table.from_arrays(
        [
            emb.column("id"),
            emb.column("offer_embedding"),
            flat.column("vendor_ids"),
            flat.column("catalog_version_ids"),
            flat.column("category_l1"),
            flat.column("category_l2"),
            flat.column("category_l3"),
            flat.column("category_l4"),
            flat.column("category_l5"),
        ],
        names=[
            "id",
            "offer_embedding",
            "vendor_ids",
            "catalog_version_ids",
            "category_l1",
            "category_l2",
            "category_l3",
            "category_l4",
            "category_l5",
        ],
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pq.write_table(combined, dst, compression="zstd", compression_level=1)
    size_gb = dst.stat().st_size / 1e9
    print(f"  write:          {time.time()-t0:6.1f}s  size={size_gb:.2f} GB", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-dir", type=Path, default=SRC_DEFAULT)
    p.add_argument("--dst-dir", type=Path, default=DST_DEFAULT)
    p.add_argument(
        "--bucket",
        default="",
        help="Single bucket filename (e.g., 'bucket=00.parquet'). Default: all.",
    )
    p.add_argument("--threads", type=int, default=8)
    p.add_argument(
        "--skip-existing", action="store_true",
        help="Skip a bucket if the destination file already exists.",
    )
    args = p.parse_args()

    sources = sorted(p for p in args.src_dir.iterdir() if BUCKET_RE.match(p.name))
    if args.bucket:
        sources = [s for s in sources if s.name == args.bucket]
        if not sources:
            raise SystemExit(f"No bucket matching {args.bucket!r} under {args.src_dir}")
    if not sources:
        raise SystemExit(f"No bucket=NN.parquet under {args.src_dir}")

    print(f"Flattening {len(sources)} bucket(s) with {args.threads} DuckDB threads")
    wall = time.time()
    for src in sources:
        dst = args.dst_dir / src.name
        if args.skip_existing and dst.exists():
            print(f"--- {src.name}: already exists, skipping ---")
            continue
        process_bucket(src, dst, args.threads)
    print(f"\nTotal wall: {time.time()-wall:.1f}s")


if __name__ == "__main__":
    main()

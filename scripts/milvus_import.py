"""Stream offer embeddings + filterable scalar fields into Milvus.

Uses the columnar `Collection.insert([col1, col2, ...])` API with
`FLOAT16_VECTOR` and a 2D fp16 ndarray (no fp32 cast). The collection is
dropped and recreated, rows are streamed per-bucket, then flush +
IVF_FLAT on the vector + INVERTED on each scalar array + load.

Schema (9 fields):
  - id                   VARCHAR(64)          primary
  - offer_embedding      FLOAT16_VECTOR(128)  (IVF_FLAT, nlist=4096)
  - vendor_ids           ARRAY<VARCHAR>       (INVERTED)
  - catalog_version_ids  ARRAY<VARCHAR>       (INVERTED)
  - category_l1..l5      ARRAY<VARCHAR>       (INVERTED)

Category fields mirror the app's Elasticsearch layout: each offer has
one joined prefix string per path at each depth. Separator is U+00A6
(¦); any literal ¦ inside an element is replaced with U+007C (|) before
joining, matching `CategoryPath.asStringPath()`.

Lessons baked in from prior runs (see INDEX_HOSTING.md):
  - Release the collection before streaming (prevents segment-RSS growth
    during long imports). No-op for a single bucket, but cheap insurance.
  - Poll `state == "Finished"` directly after `create_index` — the helper
    `utility.wait_for_index_building_complete` spins on stale
    `pending_index_rows` during compaction.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

DEFAULT_DATA_DIR = Path(
    "/mnt/HC_Volume_105463954/simplesystem/data/offers_embedded_full.parquet"
)
COLLECTION = "offers"
DIM = 128
BUCKET_RE = re.compile(r"^bucket=\d{2}\.parquet$")

CAT_SEP = "\u00a6"  # ¦  — matches CategoryPath.PATH_SEPARATOR
CAT_REPL = "\u007c"  # |  — matches CategoryPath.PATH_SEPARATOR_REPLACEMENT
CAT_LEVELS = (1, 2, 3, 4, 5)
CAT_FIELDS = [f"category_l{lvl}" for lvl in CAT_LEVELS]
SCALAR_ARRAY_FIELDS = ["vendor_ids", "catalog_version_ids", *CAT_FIELDS]

NESTED_READ_COLUMNS = ["id", "offer_embedding", "categoryPaths", "vendor_listings"]
FLAT_READ_COLUMNS = [
    "id",
    "offer_embedding",
    "vendor_ids",
    "catalog_version_ids",
    "category_l1",
    "category_l2",
    "category_l3",
    "category_l4",
    "category_l5",
]


def build_collection(drop_existing: bool) -> Collection:
    if utility.has_collection(COLLECTION):
        if drop_existing:
            print(f"Dropping existing collection {COLLECTION!r}.")
            utility.drop_collection(COLLECTION)
        else:
            print(f"Reusing existing collection {COLLECTION!r}.")
            return Collection(COLLECTION)

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="offer_embedding", dtype=DataType.FLOAT16_VECTOR, dim=DIM),
        FieldSchema(name="vendor_ids", dtype=DataType.ARRAY,
                    element_type=DataType.VARCHAR, max_capacity=32, max_length=64),
        FieldSchema(name="catalog_version_ids", dtype=DataType.ARRAY,
                    element_type=DataType.VARCHAR, max_capacity=2048, max_length=64),
        FieldSchema(name="category_l1", dtype=DataType.ARRAY,
                    element_type=DataType.VARCHAR, max_capacity=64, max_length=256),
        FieldSchema(name="category_l2", dtype=DataType.ARRAY,
                    element_type=DataType.VARCHAR, max_capacity=64, max_length=640),
        FieldSchema(name="category_l3", dtype=DataType.ARRAY,
                    element_type=DataType.VARCHAR, max_capacity=64, max_length=768),
        FieldSchema(name="category_l4", dtype=DataType.ARRAY,
                    element_type=DataType.VARCHAR, max_capacity=64, max_length=1024),
        FieldSchema(name="category_l5", dtype=DataType.ARRAY,
                    element_type=DataType.VARCHAR, max_capacity=64, max_length=1280),
    ]
    schema = CollectionSchema(fields, description="Offer embeddings (fp16, 128d) + filter fields")
    col = Collection(COLLECTION, schema=schema)
    print(f"Created collection {COLLECTION!r}.")
    return col


def select_buckets(data_dir: Path, filter_name: str | None) -> list[Path]:
    files = [p for p in sorted(data_dir.iterdir()) if BUCKET_RE.match(p.name)]
    if not files:
        raise SystemExit(f"No bucket=NN.parquet files under {data_dir}")
    if filter_name:
        files = [p for p in files if p.name == filter_name]
        if not files:
            raise SystemExit(f"No bucket matching {filter_name!r} under {data_dir}")
    return files


def flatten_vendors(vls: list) -> tuple[list[list[str]], list[list[str]]]:
    vendor_ids = []
    catalog_version_ids = []
    for vl in vls:
        vids: set[str] = set()
        cvids: set[str] = set()
        if vl:
            for v in vl:
                vid = v.get("vendor_id")
                cvid = v.get("catalog_version_id")
                if vid:
                    vids.add(vid)
                if cvid:
                    cvids.add(cvid)
        vendor_ids.append(sorted(vids))
        catalog_version_ids.append(sorted(cvids))
    return vendor_ids, catalog_version_ids


def flatten_categories(cps: list) -> list[list[list[str]]]:
    """Return [cat_l1, cat_l2, cat_l3, cat_l4, cat_l5], each row-aligned with cps."""
    out: list[list[list[str]]] = [[] for _ in CAT_LEVELS]
    for cp in cps:
        per_level: list[set[str]] = [set() for _ in CAT_LEVELS]
        if cp:
            for path in cp:
                els = path.get("elements") or []
                safe = [(e or "").replace(CAT_SEP, CAT_REPL) for e in els]
                for i, lvl in enumerate(CAT_LEVELS):
                    if len(safe) >= lvl:
                        per_level[i].add(CAT_SEP.join(safe[:lvl]))
        for i, s in enumerate(per_level):
            out[i].append(sorted(s))
    return out


def import_bucket(col: Collection, path: Path, batch_size: int) -> tuple[int, float]:
    pf = pq.ParquetFile(path)
    total_rows = pf.metadata.num_rows
    inserted = 0
    t0 = time.time()

    schema_names = set(pf.schema_arrow.names)
    pre_flat = "vendor_ids" in schema_names
    columns = FLAT_READ_COLUMNS if pre_flat else NESTED_READ_COLUMNS
    if pre_flat:
        print(f"  {path.name}: using pre-flattened columns (skipping flatten)")

    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        ids = batch.column("id").to_pylist()
        emb_obj = batch.column("offer_embedding").to_numpy(zero_copy_only=False)
        emb = np.stack(emb_obj)  # (batch, 128) fp16

        if pre_flat:
            vendor_ids = batch.column("vendor_ids").to_pylist()
            catalog_version_ids = batch.column("catalog_version_ids").to_pylist()
            cat_levels = [batch.column(f).to_pylist() for f in CAT_FIELDS]
        else:
            vls = batch.column("vendor_listings").to_pylist()
            cps = batch.column("categoryPaths").to_pylist()
            vendor_ids, catalog_version_ids = flatten_vendors(vls)
            cat_levels = flatten_categories(cps)

        col.insert([ids, emb, vendor_ids, catalog_version_ids, *cat_levels])
        inserted += len(ids)

        elapsed = max(time.time() - t0, 1e-6)
        pct = inserted / total_rows * 100
        rate = inserted / elapsed
        print(
            f"  {path.name}: {inserted:>12,}/{total_rows:>12,} "
            f"({pct:5.1f}%) @ {rate:>10,.0f} rows/s  [{elapsed:6.1f}s]",
            flush=True,
        )

    elapsed = time.time() - t0
    print(f"Finished {path.name} in {elapsed:.1f}s ({inserted / elapsed:,.0f} rows/s)")
    return inserted, elapsed


def wait_index_finished(col: Collection, field: str, poll_s: int = 5) -> float:
    """Poll `index_building_progress` until `state == "Finished"`.

    Per INDEX_HOSTING.md: `pending_index_rows` stays non-zero while
    background compaction produces new segments that need reindexing. The
    authoritative signal is `state`; observed in practice to flip to
    "Finished" as soon as the initial kmeans build completes, well before
    pending drains. The partially-indexed segments remain searchable.
    """
    t0 = time.time()
    while True:
        progress = utility.index_building_progress(col.name, index_name=field)
        total = progress.get("total_rows", 0)
        indexed = progress.get("indexed_rows", 0)
        pending = progress.get("pending_index_rows", 0)
        state = progress.get("state", "?")
        elapsed = time.time() - t0
        print(
            f"  [{field}] t={elapsed:6.1f}s  state={state}  "
            f"indexed={indexed:,}/{total:,}  pending={pending:,}",
            flush=True,
        )
        if state == "Finished":
            return elapsed
        time.sleep(poll_s)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument(
        "--bucket",
        default="",
        help="Single bucket filename to import (e.g., 'bucket=00.parquet'). "
             "Default: all buckets under --data-dir.",
    )
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", default="19530")
    p.add_argument("--batch-size", type=int, default=50_000)
    p.add_argument(
        "--index-type",
        default="IVF_FLAT",
        choices=["IVF_FLAT", "IVF_PQ", "FLAT", "HNSW"],
    )
    p.add_argument("--nlist", type=int, default=4096)
    p.add_argument(
        "--no-drop", action="store_true",
        help="Do not drop an existing 'offers' collection before inserting.",
    )
    args = p.parse_args()

    connections.connect(alias="default", host=args.host, port=args.port)

    if utility.has_collection(COLLECTION) and not args.no_drop:
        try:
            Collection(COLLECTION).release()
            print("Released existing collection before drop.")
        except Exception:
            pass

    col = build_collection(drop_existing=not args.no_drop)

    buckets = select_buckets(args.data_dir, args.bucket or None)
    print(f"Importing {len(buckets)} bucket(s) from {args.data_dir}")
    for b in buckets:
        print(f"  - {b.name}")

    start = time.time()
    grand_total = 0
    insert_elapsed = 0.0
    for path in buckets:
        n, e = import_bucket(col, path, args.batch_size)
        grand_total += n
        insert_elapsed += e

    print("\nFlushing...")
    t0 = time.time()
    col.flush()
    flush_s = time.time() - t0
    print(f"  flush: {flush_s:.1f}s  num_entities={col.num_entities:,}")

    print(f"\nCreating {args.index_type} index on offer_embedding (nlist={args.nlist})...")
    index_params = {
        "index_type": args.index_type,
        "metric_type": "COSINE",
    }
    if args.index_type in ("IVF_FLAT", "IVF_PQ"):
        index_params["params"] = {"nlist": args.nlist}
        if args.index_type == "IVF_PQ":
            index_params["params"].update({"m": 16, "nbits": 8})
    col.create_index(field_name="offer_embedding", index_params=index_params)
    print("Waiting for vector index state=Finished...")
    vec_index_s = wait_index_finished(col, "offer_embedding")

    scalar_index_timings: dict[str, float] = {}
    for field in SCALAR_ARRAY_FIELDS:
        print(f"\nCreating INVERTED index on {field}...")
        col.create_index(
            field_name=field,
            index_params={"index_type": "INVERTED"},
            index_name=field,
        )
        scalar_index_timings[field] = wait_index_finished(col, field)

    print("\nLoading collection...")
    t0 = time.time()
    col.load()
    load_s = time.time() - t0

    wall = time.time() - start
    print("\n=== summary ===")
    print(f"  rows inserted:     {grand_total:,}")
    print(f"  insert wall time:  {insert_elapsed:7.1f}s  "
          f"({grand_total/insert_elapsed:,.0f} rows/s)")
    print(f"  flush:             {flush_s:7.1f}s")
    print(f"  vector index:      {vec_index_s:7.1f}s  ({args.index_type})")
    for f, s in scalar_index_timings.items():
        print(f"  idx {f:<22}{s:7.1f}s")
    print(f"  load:              {load_s:7.1f}s")
    print(f"  total wall time:   {wall:7.1f}s  ({wall/60:.1f} min)")
    print(f"  num_entities:      {col.num_entities:,}")


if __name__ == "__main__":
    main()

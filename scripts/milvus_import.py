"""Import offer embeddings from parquet buckets into Milvus.

Creates the `offers` collection with a FLOAT16_VECTOR field + IVF_PQ index and
streams every `bucket=NN.parquet` file in DATA_DIR into it via the columnar
`Collection.insert` API (no fp16->fp32 conversion).
"""

from __future__ import annotations

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

URI_HOST = "localhost"
URI_PORT = "19530"
COLLECTION = "offers"
DATA_DIR = Path("/Users/max/Clients/simplesystem/data/offers_embedded.parquet")
BATCH_SIZE = 50_000
DIM = 128

BUCKET_RE = re.compile(r"^bucket=\d{2}\.parquet$")


def build_collection() -> Collection:
    if utility.has_collection(COLLECTION):
        print(f"Collection {COLLECTION!r} already exists; dropping.")
        utility.drop_collection(COLLECTION)

    fields = [
        FieldSchema(name="row_number", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="offer_embedding", dtype=DataType.FLOAT16_VECTOR, dim=DIM),
    ]
    schema = CollectionSchema(fields, description="Offer embeddings (e5, 128-dim, fp16)")
    col = Collection(COLLECTION, schema=schema)
    print(f"Created collection {COLLECTION!r}.")

    col.create_index(
        field_name="offer_embedding",
        index_params={
            "index_type": "IVF_PQ",
            "metric_type": "COSINE",
            "params": {"nlist": 4096, "m": 16, "nbits": 8},
        },
    )
    print("Created IVF_PQ index (nlist=4096, m=16, nbits=8).")
    return col


def iter_buckets() -> list[Path]:
    files = [p for p in sorted(DATA_DIR.iterdir()) if BUCKET_RE.match(p.name)]
    if not files:
        raise SystemExit(f"No bucket=NN.parquet files under {DATA_DIR}")
    return files


def import_bucket(col: Collection, path: Path) -> int:
    pf = pq.ParquetFile(path)
    total_rows = pf.metadata.num_rows
    inserted = 0
    t0 = time.time()

    for batch in pf.iter_batches(batch_size=BATCH_SIZE):
        row_numbers = batch.column("row_number").to_pylist()
        ids = batch.column("id").to_pylist()
        emb_obj = batch.column("offer_embedding").to_numpy(zero_copy_only=False)
        emb = np.stack(emb_obj)  # (batch, 128) fp16

        col.insert([row_numbers, ids, emb])
        inserted += len(row_numbers)

        pct = inserted / total_rows * 100
        rate = inserted / max(time.time() - t0, 1e-6)
        print(
            f"  {path.name}: {inserted:>12,}/{total_rows:>12,} "
            f"({pct:5.1f}%) @ {rate:>10,.0f} rows/s"
        )

    print(f"Finished {path.name} in {time.time() - t0:.1f}s")
    return inserted


def main() -> None:
    connections.connect(alias="default", host=URI_HOST, port=URI_PORT)
    col = build_collection()

    buckets = iter_buckets()
    print(f"Found {len(buckets)} buckets under {DATA_DIR}")

    grand_total = 0
    start = time.time()
    for path in buckets:
        grand_total += import_bucket(col, path)

    print("\nAll buckets inserted. Flushing...")
    col.flush()

    print("Loading collection (waits for index build)...")
    col.load()

    elapsed = time.time() - start
    print(
        f"\nDone. Inserted {grand_total:,} rows in {elapsed/60:.1f} min "
        f"(avg {grand_total/elapsed:,.0f} rows/s). num_entities={col.num_entities:,}"
    )


if __name__ == "__main__":
    main()

"""Swap the `offers` collection's index from FLAT to IVF_FLAT in place.

Sequence:
  1. release the collection (required before dropping the index)
  2. drop the existing index on offer_embedding
  3. create a new IVF_FLAT index (nlist=4096, COSINE)
  4. load the collection (blocks until the new index is fully built)
  5. report timing
"""

from __future__ import annotations

import time

from pymilvus import Collection, connections, utility

COLLECTION = "offers"
NLIST = 4096


def main() -> None:
    connections.connect(host="localhost", port="19530")
    if not utility.has_collection(COLLECTION):
        raise SystemExit(f"collection {COLLECTION!r} missing")

    col = Collection(COLLECTION)
    print(f"num_entities = {col.num_entities:,}")

    print("Releasing collection...")
    try:
        col.release()
    except Exception as e:
        print(f"  release skipped: {e}")

    existing = col.indexes
    for idx in existing:
        print(f"Dropping index: field={idx.field_name}")
        col.drop_index()

    print(f"Creating IVF_FLAT index (nlist={NLIST}, metric=COSINE)...")
    t0 = time.time()
    col.create_index(
        field_name="offer_embedding",
        index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": NLIST},
        },
    )
    create_t = time.time() - t0
    print(f"  create_index returned in {create_t:.1f}s (build runs async)")

    print("Loading collection (blocks until build completes)...")
    t1 = time.time()
    col.load()
    load_t = time.time() - t1
    print(f"  load() returned in {load_t:.1f}s")

    print("\nIndex info:")
    for idx in col.indexes:
        print(f"  field={idx.field_name} params={idx.params}")

    print(f"\nDone. Total time create->loaded: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

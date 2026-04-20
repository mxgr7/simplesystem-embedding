"""Verify the imported `offers` Milvus collection.

- Prints collection stats.
- Runs a random cosine query and shows the top-10 hits.
- Runs a self-hit sanity check: pull one existing row, search by its own
  embedding, expect the same row_number as the top-1 hit with score ~1.0.
"""

from __future__ import annotations

import numpy as np
from pymilvus import MilvusClient

URI = "http://localhost:19530"
COLLECTION = "offers"
DIM = 128


def decode_f16(v) -> np.ndarray:
    """Milvus returns FLOAT16_VECTOR values as [bytes] (single-element list)."""
    if isinstance(v, list) and len(v) == 1:
        v = v[0]
    if isinstance(v, (bytes, bytearray)):
        return np.frombuffer(v, dtype=np.float16)
    return np.asarray(v, dtype=np.float16)


def main() -> None:
    client = MilvusClient(URI)
    if not client.has_collection(COLLECTION):
        raise SystemExit(f"Collection {COLLECTION!r} does not exist")

    client.load_collection(COLLECTION)

    stats = client.get_collection_stats(COLLECTION)
    print(f"Collection stats: {stats}")

    search_params = {"metric_type": "COSINE", "params": {"nprobe": 64}}
    output_fields = ["id"]

    print("\n--- Random query ---")
    rng = np.random.default_rng(42)
    q = rng.standard_normal(DIM).astype(np.float32)
    q /= np.linalg.norm(q)
    q_f16 = q.astype(np.float16)
    results = client.search(
        collection_name=COLLECTION,
        data=[q_f16],
        limit=10,
        search_params=search_params,
        output_fields=output_fields,
    )
    for i, hit in enumerate(results[0]):
        ent = hit.get("entity", {})
        print(f"  #{i+1}: id={ent.get('id')} score={hit['distance']:.4f}")

    print("\n--- Self-hit sanity check ---")
    sample = client.query(
        collection_name=COLLECTION,
        filter='id != ""',
        output_fields=["id", "offer_embedding"],
        limit=1,
    )
    if not sample:
        raise SystemExit("query returned no rows")
    row = sample[0]
    target_id = row["id"]
    target_vec = decode_f16(row["offer_embedding"])
    print(f"  Target: id={target_id} vec[0:3]={target_vec[:3]}")

    results = client.search(
        collection_name=COLLECTION,
        data=[target_vec],
        limit=5,
        search_params=search_params,
        output_fields=output_fields,
    )
    for i, hit in enumerate(results[0]):
        ent = hit.get("entity", {})
        print(f"  #{i+1}: id={ent.get('id')} score={hit['distance']:.4f}")
    top = results[0][0]
    top_id = top.get("entity", {}).get("id")
    if top_id == target_id and top["distance"] >= 0.99:
        print(f"  OK: top-1 self hit with score {top['distance']:.4f}")
    else:
        print(f"  WARN: top-1 is id={top_id} score={top['distance']:.4f}")


if __name__ == "__main__":
    main()

"""Test Milvus search end-to-end.

Loads the `useful-cub-58` checkpoint, encodes a few example queries, runs
each through the offers Milvus collection, and joins the hex `id` hits back
to `offers_grouped.parquet` for a human-readable view.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import torch
from pymilvus import MilvusClient

from embedding_train.infer import build_tokenizer, encode_texts, resolve_device
from embedding_train.model import load_embedding_module_from_checkpoint
from embedding_train.rendering import RowTextRenderer

CHECKPOINT = Path(
    "checkpoints/useful-cub-58/best-step=4880-val_full_catalog_ndcg_at_5=0.7379.ckpt"
)
MILVUS_URI = "http://localhost:19530"
COLLECTION = "offers"
OFFERS_GROUPED = "/Users/max/Clients/simplesystem/data/offers_grouped.parquet"
TOP_K = 100
TOP_K_DISPLAY = 5

QUERIES: list[str] = [
    "schwarze damen sneaker",
    "espresso machine 2 gruppig",
    "makita akkuschrauber 18v",
    "bosch waschmaschine frontlader",
    "laptop 16gb ram ssd",
]


def encode_queries(queries: list[str], timings: dict[str, float]) -> np.ndarray:
    print(f"Loading checkpoint: {CHECKPOINT}")
    device = resolve_device("auto")
    print(f"Device: {device}")

    t0 = time.perf_counter()
    model, cfg = load_embedding_module_from_checkpoint(str(CHECKPOINT), map_location="cpu")
    model = model.to(device).eval()
    timings["checkpoint_load_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    tokenizer = build_tokenizer(cfg.model.model_name)
    renderer = RowTextRenderer(cfg.data)
    timings["tokenizer_build_s"] = time.perf_counter() - t0

    rendered = [renderer.render_query_text({"query_term": q}) for q in queries]
    for q, r in zip(queries, rendered):
        print(f"  {q!r} -> {r!r}")

    max_length = int(cfg.data.max_query_length)
    t0 = time.perf_counter()
    embs = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=rendered,
        max_length=max_length,
        encode_batch_size=len(rendered),
        device=device,
    )
    # model.encode already L2-normalizes, so `embs` are unit vectors.
    out = embs.cpu().numpy().astype(np.float16)
    timings["query_encode_s"] = time.perf_counter() - t0
    return out


def search(client: MilvusClient, embs: np.ndarray) -> list[list[dict]]:
    return client.search(
        collection_name=COLLECTION,
        data=[v for v in embs],
        limit=TOP_K,
        search_params={"metric_type": "COSINE", "params": {}},
        output_fields=["id"],
    )


def lookup_offers(hex_ids: list[str]) -> dict[str, dict]:
    if not hex_ids:
        return {}
    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT id, name, manufacturerName, ean, article_number,
               manufacturerArticleNumber, manufacturerArticleType,
               categoryPaths, description, n
        FROM read_parquet('{OFFERS_GROUPED}/*.parquet')
        WHERE id = ANY($1)
        """,
        [hex_ids],
    ).fetchdf()
    return {row["id"]: row.to_dict() for _, row in df.iterrows()}


def summarize_categories(category_paths) -> str:
    if category_paths is None:
        return ""
    try:
        first = category_paths[0]
        elements = first.get("elements") or []
        return " > ".join(str(e) for e in elements)
    except Exception:
        return ""


def main() -> None:
    if not CHECKPOINT.exists():
        sys.exit(f"checkpoint not found: {CHECKPOINT}")

    timings: dict[str, float] = {}
    t_total = time.perf_counter()

    embs = encode_queries(QUERIES, timings)
    print(f"\nEncoded embeddings: shape={embs.shape} dtype={embs.dtype}")

    t0 = time.perf_counter()
    client = MilvusClient(MILVUS_URI)
    if not client.has_collection(COLLECTION):
        sys.exit(f"Milvus collection {COLLECTION!r} missing")
    timings["milvus_connect_s"] = time.perf_counter() - t0

    print("Searching Milvus...")
    t0 = time.perf_counter()
    results = search(client, embs)
    timings["milvus_search_s"] = time.perf_counter() - t0

    all_hex = list({hit["entity"]["id"] for hits in results for hit in hits})
    print(f"Looking up {len(all_hex)} unique ids in offers_grouped.parquet...")
    t0 = time.perf_counter()
    offers = lookup_offers(all_hex)
    timings["duckdb_lookup_s"] = time.perf_counter() - t0
    print(f"  joined {len(offers)} records")

    for query, hits in zip(QUERIES, results):
        print(f"\n=== {query}  (retrieved {len(hits)}, showing top {TOP_K_DISPLAY})")
        for i, hit in enumerate(hits[:TOP_K_DISPLAY]):
            ent = hit["entity"]
            hex_id = ent["id"]
            score = hit["distance"]
            rec = offers.get(hex_id)
            print(f"  #{i+1}  score={score:.4f}  id={hex_id}")
            if rec is None:
                print(f"       (not found in offers_grouped)")
                continue
            name = (rec.get("name") or "").strip()
            brand = (rec.get("manufacturerName") or "").strip()
            cat = summarize_categories(rec.get("categoryPaths"))
            n = rec.get("n")
            print(f"       name:  {name[:120]}")
            if brand:
                print(f"       brand: {brand}")
            if cat:
                print(f"       cat:   {cat[:120]}")
            if n is not None:
                print(f"       n:     {n}")

    timings["total_s"] = time.perf_counter() - t_total
    n_q = len(QUERIES)
    search_ms = timings["milvus_search_s"] * 1000
    encode_ms = timings["query_encode_s"] * 1000
    print("\nTimings:")
    print(f"  checkpoint load : {timings['checkpoint_load_s']:7.2f} s")
    print(f"  tokenizer build : {timings['tokenizer_build_s']:7.2f} s")
    print(f"  query encode    : {timings['query_encode_s']:7.2f} s"
          f"  ({encode_ms / n_q:6.1f} ms/query, {n_q} queries)")
    print(f"  milvus connect  : {timings['milvus_connect_s']:7.2f} s")
    print(f"  milvus search   : {timings['milvus_search_s']:7.2f} s"
          f"  ({search_ms / n_q:6.1f} ms/query, batch={n_q})")
    print(f"  duckdb lookup   : {timings['duckdb_lookup_s']:7.2f} s"
          f"  ({len(all_hex)} ids)")
    print(f"  total           : {timings['total_s']:7.2f} s")


if __name__ == "__main__":
    main()

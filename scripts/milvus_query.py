"""Test Milvus search end-to-end.

Loads the `useful-cub-58` checkpoint, encodes a few example queries, runs
each through the offers Milvus collection, and joins the hex `id` hits back
to `offers_grouped.parquet` for a human-readable view.
"""

from __future__ import annotations

import sys
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
TOP_K = 5

QUERIES: list[str] = [
    "schwarze damen sneaker",
    "espresso machine 2 gruppig",
    "makita akkuschrauber 18v",
    "bosch waschmaschine frontlader",
    "laptop 16gb ram ssd",
]


def encode_queries(queries: list[str]) -> np.ndarray:
    print(f"Loading checkpoint: {CHECKPOINT}")
    device = resolve_device("auto")
    print(f"Device: {device}")

    model, cfg = load_embedding_module_from_checkpoint(str(CHECKPOINT), map_location="cpu")
    model = model.to(device).eval()

    tokenizer = build_tokenizer(cfg.model.model_name)
    renderer = RowTextRenderer(cfg.data)

    rendered = [renderer.render_query_text({"query_term": q}) for q in queries]
    for q, r in zip(queries, rendered):
        print(f"  {q!r} -> {r!r}")

    max_length = int(cfg.data.max_query_length)
    embs = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=rendered,
        max_length=max_length,
        encode_batch_size=len(rendered),
        device=device,
    )
    # model.encode already L2-normalizes, so `embs` are unit vectors.
    return embs.cpu().numpy().astype(np.float16)


def search(client: MilvusClient, embs: np.ndarray) -> list[list[dict]]:
    return client.search(
        collection_name=COLLECTION,
        data=[v for v in embs],
        limit=TOP_K,
        search_params={"metric_type": "COSINE", "params": {}},
        output_fields=["row_number", "id"],
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

    embs = encode_queries(QUERIES)
    print(f"\nEncoded embeddings: shape={embs.shape} dtype={embs.dtype}")

    client = MilvusClient(MILVUS_URI)
    if not client.has_collection(COLLECTION):
        sys.exit(f"Milvus collection {COLLECTION!r} missing")

    print("Searching Milvus...")
    results = search(client, embs)

    all_hex = list({hit["entity"]["id"] for hits in results for hit in hits})
    print(f"Looking up {len(all_hex)} unique ids in offers_grouped.parquet...")
    offers = lookup_offers(all_hex)
    print(f"  joined {len(offers)} records")

    for query, hits in zip(QUERIES, results):
        print(f"\n=== {query}")
        for i, hit in enumerate(hits):
            ent = hit["entity"]
            hex_id = ent["id"]
            rn = ent["row_number"]
            score = hit["distance"]
            rec = offers.get(hex_id)
            print(f"  #{i+1}  score={score:.4f}  rn={rn}  id={hex_id}")
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


if __name__ == "__main__":
    main()

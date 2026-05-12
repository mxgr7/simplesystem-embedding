"""Reindex local-article-index-v1 → local-article-index-v2 with fresh embeddings.

One-pass combined op: read full _source from v1, append v2 embeddings
+ embeddingModelVersion, _bulk index into v2. Old stale embedding fields
are dropped in flight (we overwrite or omit them).

Pipeline per article:
  1. Pull full _source (vendorId, offers, prices, customerArticleNumbers, ...).
  2. For each offer: resolve (vendorId, articleNumber) -> hashes via the
     in-memory parquet lookup. Union into per-article set. Fail-fast on miss.
  3. MGET fp16 vectors from Redis under tei:v2:<hash>; decode to fp32 list.
  4. Strip stale fields (embeddings, embeddingsBuiltAt, rerankTexts*) from
     the source. If the article has offers, add fresh embeddings list +
     embeddingModelVersion. Otherwise neither field is written.
  5. _bulk index to v2, same _id as v1.

Tuning targets (per BULK_IMPORT_TUNING.md):
  - bulk body ~5-15 MB
  - parallel inflight bulks tuned to disk-IO saturation, not CPU
  - keep refresh_interval=-1 on v2 throughout the run

Run:
    uv run --no-project python scripts/index_embeddings_to_es.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import duckdb
import httpx
import numpy as np
import redis.asyncio as aioredis


DEFAULT_PARQUET = (
    "/data/datasets/mongo_offers_export_20260512/article_hashes_v2/**/*.parquet"
)
DEFAULT_ES = "http://localhost:9200"
DEFAULT_REDIS = "redis://localhost:6379/0"
DEFAULT_SRC = "local-article-index-v1"
DEFAULT_DST = "local-article-index-v2"
DEFAULT_MODEL = "useful-cub-58"

EMB_DIM = 128
FP16_BYTES = EMB_DIM * 2  # 256

STALE_FIELDS = ("embeddings", "embeddingsBuiltAt", "rerankTexts", "rerankTextsBuiltAt")


def build_lookup(parquet_glob: str) -> dict[tuple[str, str], tuple[str, ...]]:
    """Load parquet and return {(vendor_id, article_number): (hash, ...)}."""
    t0 = time.time()
    print(f"loading lookup from {parquet_glob} ...", flush=True)
    con = duckdb.connect()
    con.execute(f"SET threads = {os.cpu_count() or 8}")
    con.execute("SET enable_progress_bar = false")
    arrow = con.execute(f"""
        SELECT vendor_id, article_number,
               list_distinct(list(article_hash)) AS hashes
        FROM read_parquet('{parquet_glob}', hive_partitioning = true)
        GROUP BY 1, 2
    """).fetch_arrow_table()
    print(f"  duckdb agg: {time.time() - t0:.1f}s ({arrow.num_rows:,} pairs)", flush=True)
    t = time.time()
    vendor = arrow["vendor_id"].to_pylist()
    artno = arrow["article_number"].to_pylist()
    hashes = arrow["hashes"].to_pylist()
    lookup: dict[tuple[str, str], tuple[str, ...]] = {
        (v, a): tuple(h) for v, a, h in zip(vendor, artno, hashes)
    }
    print(f"  dict build: {time.time() - t:.1f}s ({len(lookup):,} entries)", flush=True)
    return lookup


async def open_pit(client: httpx.AsyncClient, index: str, keep_alive: str = "30m") -> str:
    r = await client.post(f"/{index}/_pit?keep_alive={keep_alive}")
    r.raise_for_status()
    return r.json()["id"]


async def close_pit(client: httpx.AsyncClient, pit_id: str) -> None:
    try:
        await client.request("DELETE", "/_pit", json={"id": pit_id})
    except Exception:
        pass


async def scan_articles(client: httpx.AsyncClient, pit_id: str, page_size: int):
    """Yield pages of hits with full _source."""
    search_after: list | None = None
    while True:
        body = {
            "size": page_size,
            "track_total_hits": False,
            "pit": {"id": pit_id, "keep_alive": "30m"},
            "_source": True,
            "sort": [{"_shard_doc": "asc"}],
            "query": {"match_all": {}},
        }
        if search_after is not None:
            body["search_after"] = search_after
        r = await client.post("/_search", json=body)
        r.raise_for_status()
        data = r.json()
        hits = data["hits"]["hits"]
        if not hits:
            return
        yield hits
        search_after = hits[-1]["sort"]


def hashes_for_article(
    src: dict,
    lookup: dict[tuple[str, str], tuple[str, ...]],
    known_vendors: set[str],
) -> list[str] | None:
    """Resolve unique hashes for one article.

    Returns:
      None: article belongs to a vendor not in the parquet — caller should
        skip the article entirely (don't write to v2).
      []: known vendor but no offers / no resolvable articleNumber — caller
        should still write the article (just without embeddings).
      [hashes...]: normal case.

    Raises SystemExit on the unexpected miss where the vendor IS in the
    parquet but a specific (vendor, articleNumber) isn't — that's the
    real systemic-drift signal we want to fail on.
    """
    vendor_id = src.get("vendorId")
    offers = src.get("offers") or []
    if vendor_id is None:
        return []
    if vendor_id not in known_vendors:
        return None
    if not offers:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for off in offers:
        artno = off.get("articleNumber")
        if artno is None:
            continue
        key = (vendor_id, artno)
        hs = lookup.get(key)
        if hs is None:
            raise SystemExit(
                f"missing hash for (vendor_id={vendor_id!r}, "
                f"article_number={artno!r}) — vendor IS in parquet but this "
                f"pair isn't. Aborting per fail-fast policy."
            )
        for h in hs:
            if h not in seen:
                seen.add(h)
                out.append(h)
    return out


def decode_fp16(buf: bytes) -> list[float]:
    if len(buf) != FP16_BYTES:
        raise ValueError(f"expected {FP16_BYTES} bytes, got {len(buf)}")
    return np.frombuffer(buf, dtype=np.float16).astype(np.float32).tolist()


async def mget_vectors(r: aioredis.Redis, hashes: list[str]) -> dict[str, list[float]]:
    if not hashes:
        return {}
    keys = [f"tei:v2:{h}" for h in hashes]
    raw = await r.mget(keys)
    out: dict[str, list[float]] = {}
    for h, buf in zip(hashes, raw):
        if buf is None:
            raise SystemExit(f"missing Redis vector for hash {h!r}")
        out[h] = decode_fp16(buf)
    return out


def build_bulk_body(
    docs: list[tuple[str, dict]],
    dst_index: str,
) -> str:
    """One '{index}\\n{full doc}\\n' pair per article."""
    lines: list[str] = []
    for doc_id, doc in docs:
        meta = {"index": {"_index": dst_index, "_id": doc_id}}
        lines.append(json.dumps(meta, separators=(",", ":"), ensure_ascii=False))
        lines.append(json.dumps(doc, separators=(",", ":"), ensure_ascii=False))
    lines.append("")
    return "\n".join(lines)


async def send_bulk(
    client: httpx.AsyncClient,
    body: str,
    sem: asyncio.Semaphore,
) -> tuple[int, int]:
    async with sem:
        r = await client.post(
            "/_bulk",
            content=body,
            headers={"Content-Type": "application/x-ndjson"},
        )
        r.raise_for_status()
        data = r.json()
    n_ok = n_err = 0
    if data.get("errors"):
        for item in data["items"]:
            op = next(iter(item.values()))
            if op.get("status", 500) >= 300:
                n_err += 1
                raise SystemExit(
                    f"bulk error: status={op.get('status')} "
                    f"_id={op.get('_id')} error={op.get('error')}"
                )
            n_ok += 1
    else:
        n_ok = len(data["items"])
    return n_ok, n_err


async def drain_below(
    inflight: set[asyncio.Task], threshold: int
) -> tuple[set[asyncio.Task], int, int]:
    """Wait until len(inflight) < threshold. Return (remaining, ok_added, err_added)."""
    ok_added = err_added = 0
    while len(inflight) >= threshold:
        done, inflight = await asyncio.wait(
            inflight, return_when=asyncio.FIRST_COMPLETED
        )
        for d in done:
            ok, er = d.result()
            ok_added += ok
            err_added += er
    return inflight, ok_added, err_added


async def main_async(args: argparse.Namespace) -> None:
    lookup = build_lookup(args.parquet)
    known_vendors: set[str] = {v for v, _ in lookup.keys()}
    print(f"  known vendors: {len(known_vendors)}", flush=True)

    es_limits = httpx.Limits(
        max_connections=args.inflight * 2,
        max_keepalive_connections=args.inflight * 2,
    )
    es = httpx.AsyncClient(
        base_url=args.es,
        timeout=httpx.Timeout(300.0, connect=10.0),
        limits=es_limits,
    )
    r = aioredis.from_url(args.redis, decode_responses=False, max_connections=64)

    pit_id = await open_pit(es, args.src)
    print(f"opened PIT {pit_id[:24]}...", flush=True)

    sem = asyncio.Semaphore(args.inflight)
    inflight: set[asyncio.Task] = set()

    n_articles = 0
    n_with_offers = 0
    n_no_offers = 0
    n_skipped_orphan_vendor = 0
    skipped_orphan_vendor_counts: dict[str, int] = {}
    n_vecs_fetched = 0
    n_hashes_total = 0
    n_ok = 0
    n_err = 0
    bytes_sent = 0
    t_start = time.time()
    t_log = t_start

    try:
        async for hits in scan_articles(es, pit_id, args.page_size):
            if args.limit and n_articles >= args.limit:
                break

            # Resolve hashes per article first; collect unique hashes for one MGET.
            todo: list[tuple[str, dict, list[str]]] = []
            unique_hashes: set[str] = set()
            for hit in hits:
                src = hit.get("_source", {})
                # Drop stale legacy fields from the copy.
                for f in STALE_FIELDS:
                    src.pop(f, None)
                hs = hashes_for_article(src, lookup, known_vendors)
                if hs is None:
                    n_skipped_orphan_vendor += 1
                    v = src.get("vendorId") or "<missing>"
                    skipped_orphan_vendor_counts[v] = (
                        skipped_orphan_vendor_counts.get(v, 0) + 1
                    )
                    continue
                if hs:
                    n_with_offers += 1
                else:
                    n_no_offers += 1
                todo.append((hit["_id"], src, hs))
                unique_hashes.update(hs)

            vec_map = await mget_vectors(r, sorted(unique_hashes))
            n_vecs_fetched += len(vec_map)

            # Build docs for bulk; chunk by target body size.
            chunk: list[tuple[str, dict]] = []
            chunk_bytes = 0
            for doc_id, src, hs in todo:
                if hs:
                    src["embeddings"] = [
                        {"vector": vec_map[h], "inputHash": h} for h in hs
                    ]
                    src["embeddingModelVersion"] = args.model
                chunk.append((doc_id, src))
                n_hashes_total += len(hs)
                n_articles += 1
                # Rough size estimate: each float ~10 chars, plus baseline ~3 KB legacy.
                est = 3072 + len(hs) * EMB_DIM * 10
                chunk_bytes += est
                if chunk_bytes >= args.target_body_bytes:
                    body = build_bulk_body(chunk, args.dst)
                    bytes_sent += len(body)
                    inflight, ok, er = await drain_below(inflight, args.inflight)
                    n_ok += ok
                    n_err += er
                    inflight.add(asyncio.create_task(send_bulk(es, body, sem)))
                    chunk = []
                    chunk_bytes = 0
                if args.limit and n_articles >= args.limit:
                    break

            if chunk:
                body = build_bulk_body(chunk, args.dst)
                bytes_sent += len(body)
                inflight, ok, er = await drain_below(inflight, args.inflight)
                n_ok += ok
                n_err += er
                inflight.add(asyncio.create_task(send_bulk(es, body, sem)))

            now = time.time()
            if now - t_log >= 5.0:
                rate = n_articles / max(now - t_start, 1e-3)
                mb_s = bytes_sent / 1e6 / max(now - t_start, 1e-3)
                print(
                    f"  scanned {n_articles:,} articles "
                    f"({rate:,.0f}/s {mb_s:,.1f}MB/s)  "
                    f"with_offers={n_with_offers:,} no_offers={n_no_offers:,}  "
                    f"vecs={n_vecs_fetched:,}  "
                    f"bulk_ok={n_ok:,} inflight={len(inflight)}",
                    flush=True,
                )
                t_log = now

        # Drain remaining.
        if inflight:
            results = await asyncio.gather(*inflight)
            for ok, er in results:
                n_ok += ok
                n_err += er
    finally:
        await close_pit(es, pit_id)
        await es.aclose()
        await r.aclose()

    elapsed = time.time() - t_start
    print()
    print(f"DONE in {elapsed/60:.1f} min")
    print(f"  articles written to v2:    {n_articles:,}")
    print(f"    with offers (embedded):  {n_with_offers:,}")
    print(f"    no offers (no emb):      {n_no_offers:,}")
    print(f"  articles SKIPPED entirely: {n_skipped_orphan_vendor:,} (orphan vendor)")
    for v, c in sorted(skipped_orphan_vendor_counts.items(), key=lambda x: -x[1]):
        print(f"    {v}  {c:,}")
    print(f"  bulk ok:                   {n_ok:,}")
    print(f"  bulk errors:               {n_err:,}")
    print(f"  unique vectors fetched:    {n_vecs_fetched:,}")
    print(f"  bytes sent (bulk bodies):  {bytes_sent/1e9:.2f} GB")
    print(f"  avg hashes/article(w/off): {n_hashes_total / max(n_with_offers, 1):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--es", default=DEFAULT_ES)
    ap.add_argument("--redis", default=DEFAULT_REDIS)
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--dst", default=DEFAULT_DST)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--page-size", type=int, default=500,
                    help="ES search_after page size")
    ap.add_argument("--target-body-bytes", type=int, default=10 * 1024 * 1024,
                    help="target _bulk body size (default 10 MB)")
    ap.add_argument("--inflight", type=int, default=16,
                    help="max concurrent _bulk requests")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many articles (0 = all)")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

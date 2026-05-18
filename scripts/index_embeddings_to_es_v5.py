"""Import the staging clone -> local-article-index-v5 with embeddings +
denormalized scope fields.

Fork of scripts/index_embeddings_to_es.py adapted for the staging dataset:

  src  = stg-articles-v1-clone-20260516   (verbatim staging clone, no embeddings)
  dst  = local-article-index-v5           (target_mapping.json mapping)

Per FT_ELASTIC_IMPORT.md §2.1.3, each article doc gets, in one bulk pass:
  - nested `embeddings` = [{vector, inputHash}] + `embeddingModelVersion`
    (hashes resolved from the Mongo-export parquet lookup; fp16 vectors
     read from Redis tei:v2:<hash>, which prewarm_v2_missing.py fills via TEI)
  - denormalized `catalogVersionIds` = sorted distinct union of every
    offers[].catalogVersionIds
  - denormalized `priceKeys` = sorted distinct set of
    "{priceListId}|{currency}" over every prices[] entry
    (emitted even when empty — empty set ≠ field absent)

Added vs the v1->v2 script:
  - catalogVersionIds / priceKeys derivation (pure fn of the source doc)
  - PIT slicing: --slices N runs N concurrent sliced scans for throughput
  - --vendor restricts the source scan (bounded probe)
  - --skip-missing: drop a hash on Redis miss instead of aborting (probe-safe)

Run (probe):
    uv run --no-project python scripts/index_embeddings_to_es_v5.py \
        --parquet '/data/datasets/mongo_offers_export_20260512/article_hashes_v2/**/*.parquet' \
        --vendor <uuid> --slices 16 --limit 50000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
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
DEFAULT_SRC = "stg-articles-v1-clone-20260516"
DEFAULT_DST = "local-article-index-v5"
DEFAULT_MODEL = "useful-cub-58"

EMB_DIM = 128
FP16_BYTES = EMB_DIM * 2  # 256
STALE_FIELDS = ("embeddings", "embeddingsBuiltAt", "rerankTexts", "rerankTextsBuiltAt")


def build_lookup(parquet_glob: str) -> dict[tuple[str, str], tuple[str, ...]]:
    """{(vendor_id, article_number): (hash, ...)} from the hive parquet."""
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
    vendor = arrow["vendor_id"].to_pylist()
    artno = arrow["article_number"].to_pylist()
    hashes = arrow["hashes"].to_pylist()
    lookup = {(v, a): tuple(h) for v, a, h in zip(vendor, artno, hashes)}
    print(f"  lookup: {len(lookup):,} (vendor,artno) pairs in "
          f"{time.time()-t0:.1f}s", flush=True)
    return lookup


async def open_pit(client: httpx.AsyncClient, index: str, keep_alive="30m") -> str:
    r = await client.post(f"/{index}/_pit?keep_alive={keep_alive}")
    r.raise_for_status()
    return r.json()["id"]


async def close_pit(client: httpx.AsyncClient, pit_id: str) -> None:
    try:
        await client.request("DELETE", "/_pit", json={"id": pit_id})
    except Exception:
        pass


async def scan_slice(client, pit_id, page_size, slice_id, slices, vendor):
    """Yield pages for one PIT slice (or the whole index when slices==1)."""
    query: dict = {"match_all": {}}
    if vendor:
        query = {"term": {"vendorId": vendor}}
    search_after: list | None = None
    while True:
        body: dict = {
            "size": page_size,
            "track_total_hits": False,
            "pit": {"id": pit_id, "keep_alive": "30m"},
            "_source": True,
            "sort": [{"_shard_doc": "asc"}],
            "query": query,
        }
        if slices > 1:
            body["slice"] = {"id": slice_id, "max": slices}
        if search_after is not None:
            body["search_after"] = search_after
        r = await client.post("/_search", json=body)
        r.raise_for_status()
        hits = r.json()["hits"]["hits"]
        if not hits:
            return
        yield hits
        search_after = hits[-1]["sort"]


def hashes_for_article(src, lookup, known_vendors):
    """Resolve unique hashes for one article (see v1->v2 script for the
    None / [] / [hashes] contract). Fail-fast on the systemic-drift miss."""
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
        hs = lookup.get((vendor_id, artno))
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


def denormalize(src: dict) -> None:
    """Set catalogVersionIds + priceKeys from the doc's own offers/prices.

    Pure function of the source — no extra reads. Emitted even when empty
    so an empty set ("no offers/prices") is distinct from field-absent."""
    offers = src.get("offers") or []
    prices = src.get("prices") or []
    cvids: set[str] = set()
    for o in offers:
        for c in o.get("catalogVersionIds") or []:
            cvids.add(c)
    pkeys: set[str] = set()
    for p in prices:
        pl, cur = p.get("priceListId"), p.get("currency")
        if pl and cur:
            pkeys.add(f"{pl}|{cur}")
    src["catalogVersionIds"] = sorted(cvids)
    src["priceKeys"] = sorted(pkeys)


def decode_fp16(buf: bytes) -> list[float]:
    if len(buf) != FP16_BYTES:
        raise ValueError(f"expected {FP16_BYTES} bytes, got {len(buf)}")
    return np.frombuffer(buf, dtype=np.float16).astype(np.float32).tolist()


async def mget_vectors(r, hashes, skip_missing):
    if not hashes:
        return {}, 0
    raw = await r.mget([f"tei:v2:{h}" for h in hashes])
    out: dict[str, list[float]] = {}
    misses = 0
    for h, buf in zip(hashes, raw):
        if buf is None:
            if skip_missing:
                misses += 1
                continue
            raise SystemExit(f"missing Redis vector for hash {h!r}")
        out[h] = decode_fp16(buf)
    return out, misses


def build_bulk_body(docs: list[tuple[str, dict]], dst_index: str) -> str:
    lines: list[str] = []
    for doc_id, doc in docs:
        lines.append(json.dumps({"index": {"_index": dst_index, "_id": doc_id}},
                                 separators=(",", ":"), ensure_ascii=False))
        lines.append(json.dumps(doc, separators=(",", ":"), ensure_ascii=False))
    lines.append("")
    return "\n".join(lines)


async def send_bulk(client, body, sem):
    """Bulk-index with graceful backoff.

    429 / es_rejected_execution (HTTP-level or per-item) is transient
    backpressure, NOT a data error — retry the whole body with exponential
    backoff + jitter (idempotent: every op carries an explicit _id, so a
    replay overwrites identically). Only a non-429 item error is fatal.
    Sleep happens OUTSIDE the semaphore so a backing-off request doesn't
    hold a concurrency slot.
    """
    delay = 0.5
    for attempt in range(10):
        data = None
        async with sem:
            r = await client.post("/_bulk", content=body,
                                   headers={"Content-Type": "application/x-ndjson"})
            status = r.status_code
            if status != 429:
                r.raise_for_status()
                data = r.json()
        if status == 429:
            await asyncio.sleep(delay + random.uniform(0, delay))
            delay = min(delay * 2, 30.0)
            continue
        if not data.get("errors"):
            return len(data["items"]), 0
        n_ok = 0
        had_429 = False
        for item in data["items"]:
            op = next(iter(item.values()))
            stt = op.get("status", 500)
            if stt < 300:
                n_ok += 1
            elif stt == 429 or "rejected_execution" in str(op.get("error", "")):
                had_429 = True
            else:
                raise SystemExit(
                    f"bulk error: status={stt} _id={op.get('_id')} "
                    f"error={op.get('error')}"
                )
        if not had_429:
            return n_ok, 0
        await asyncio.sleep(delay + random.uniform(0, delay))
        delay = min(delay * 2, 30.0)
    raise SystemExit("bulk: exhausted 429/backoff retries (10)")


class Stats:
    def __init__(self) -> None:
        self.articles = 0
        self.with_emb = 0
        self.no_emb = 0
        self.skipped_orphan = 0
        self.vecs = 0
        self.redis_misses = 0
        self.ok = 0
        self.bytes = 0
        self.t0 = time.time()
        self.lock = asyncio.Lock()


async def worker(slice_id, args, es, r, lookup, known_vendors, pit_id, sem, st):
    inflight: set[asyncio.Task] = set()

    async def drain(threshold: int) -> None:
        nonlocal inflight
        while len(inflight) >= threshold:
            done, pending = await asyncio.wait(
                inflight, return_when=asyncio.FIRST_COMPLETED)
            inflight = pending
            for d in done:
                ok, _ = d.result()
                st.ok += ok

    chunk: list[tuple[str, dict]] = []
    chunk_bytes = 0
    async for hits in scan_slice(es, pit_id, args.page_size, slice_id,
                                 args.slices, args.vendor):
        todo: list[tuple[str, dict, list[str]]] = []
        uniq: set[str] = set()
        for hit in hits:
            if args.limit and st.articles >= args.limit:
                break
            src = hit.get("_source", {})
            for f in STALE_FIELDS:
                src.pop(f, None)
            hs = hashes_for_article(src, lookup, known_vendors)
            if hs is None:
                st.skipped_orphan += 1
                continue
            denormalize(src)
            todo.append((hit["_id"], src, hs))
            uniq.update(hs)
            st.articles += 1

        vec_map, misses = await mget_vectors(r, sorted(uniq), args.skip_missing)
        st.vecs += len(vec_map)
        st.redis_misses += misses

        for doc_id, src, hs in todo:
            vecs = [{"vector": vec_map[h], "inputHash": h}
                    for h in hs if h in vec_map]
            if vecs:
                src["embeddings"] = vecs
                src["embeddingModelVersion"] = args.model
                st.with_emb += 1
            else:
                st.no_emb += 1
            chunk.append((doc_id, src))
            chunk_bytes += 3072 + len(vecs) * EMB_DIM * 10
            if chunk_bytes >= args.target_body_bytes:
                body = build_bulk_body(chunk, args.dst)
                st.bytes += len(body)
                await drain(args.inflight)
                inflight.add(asyncio.create_task(send_bulk(es, body, sem)))
                chunk, chunk_bytes = [], 0

        if args.limit and st.articles >= args.limit:
            break

    if chunk:
        body = build_bulk_body(chunk, args.dst)
        st.bytes += len(body)
        await drain(args.inflight)
        inflight.add(asyncio.create_task(send_bulk(es, body, sem)))
    if inflight:
        for ok, _ in await asyncio.gather(*inflight):
            st.ok += ok


async def main_async(args) -> None:
    lookup = build_lookup(args.parquet)
    known_vendors = {v for v, _ in lookup}
    print(f"  known vendors: {len(known_vendors)}", flush=True)

    es = httpx.AsyncClient(
        base_url=args.es,
        timeout=httpx.Timeout(300.0, connect=10.0),
        limits=httpx.Limits(max_connections=args.inflight * 2,
                            max_keepalive_connections=args.inflight * 2),
    )
    r = aioredis.from_url(args.redis, decode_responses=False, max_connections=64)
    pit_id = await open_pit(es, args.src)
    print(f"opened PIT {pit_id[:24]}... slices={args.slices} "
          f"vendor={args.vendor or '<all>'}", flush=True)

    sem = asyncio.Semaphore(args.inflight)
    st = Stats()
    logger = asyncio.create_task(_log_loop(st, args))
    try:
        await asyncio.gather(*[
            worker(i, args, es, r, lookup, known_vendors, pit_id, sem, st)
            for i in range(args.slices)
        ])
    finally:
        logger.cancel()
        await close_pit(es, pit_id)
        await es.aclose()
        await r.aclose()

    el = time.time() - st.t0
    print(f"\nDONE in {el/60:.2f} min ({el:.0f}s)")
    print(f"  articles written:        {st.articles:,}")
    print(f"    with embeddings:       {st.with_emb:,}")
    print(f"    no embeddings:         {st.no_emb:,}")
    print(f"  skipped orphan vendor:   {st.skipped_orphan:,}")
    print(f"  unique vectors fetched:  {st.vecs:,}")
    print(f"  redis misses (skipped):  {st.redis_misses:,}")
    print(f"  bulk ok:                 {st.ok:,}")
    print(f"  bulk bytes:              {st.bytes/1e9:.2f} GB")
    print(f"  throughput:              {st.articles/max(el,1e-3):,.0f} articles/s")
    print(f"  model stamped:           {args.model}")


async def _log_loop(st: Stats, args) -> None:
    while True:
        await asyncio.sleep(5)
        el = time.time() - st.t0
        rate = st.articles / max(el, 1e-3)
        print(f"  {st.articles:,} arts ({rate:,.0f}/s) "
              f"emb={st.with_emb:,} no_emb={st.no_emb:,} "
              f"orphan={st.skipped_orphan:,} vecs={st.vecs:,} "
              f"miss={st.redis_misses:,} ok={st.ok:,} "
              f"{st.bytes/1e6/max(el,1e-3):,.1f}MB/s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--es", default=DEFAULT_ES)
    ap.add_argument("--redis", default=DEFAULT_REDIS)
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--dst", default=DEFAULT_DST)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="embeddingModelVersion label stamped on docs")
    ap.add_argument("--vendor", default="",
                    help="restrict source scan to this vendorId (bounded probe)")
    ap.add_argument("--slices", type=int, default=16,
                    help="concurrent PIT slices (parallel throughput)")
    ap.add_argument("--page-size", type=int, default=500)
    ap.add_argument("--target-body-bytes", type=int, default=10 * 1024 * 1024)
    ap.add_argument("--inflight", type=int, default=16,
                    help="max concurrent _bulk requests per slice")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after ~this many articles (0 = all)")
    ap.add_argument("--skip-missing", action="store_true",
                    help="drop a hash on Redis miss instead of aborting")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

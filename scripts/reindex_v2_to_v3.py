"""Reindex local-article-index-v2 -> local-article-index-v3 with fp32 vectors.

v2 carries `embeddings[].inputHash` (not doc-valued, but read from _source) and
int8-quantized vectors in the HNSW codec; v3 wants the original fp32 vectors
from Redis (`tei:v2:<hash>`). We skip v2's vectors entirely by relying on
`_source.excludes:[embeddings.vector]` — the default _source on v2 already
strips them, so the scan returns inputHash without the lossy bytes.

Per article:
  1. Pull full _source from v2 (already excludes embeddings.vector).
  2. Read embeddings[].inputHash list.
  3. MGET fp16 vectors from Redis tei:v2:<hash> ; decode to fp32 list.
  4. Rebuild embeddings list with the same inputHash plus the fp32 vector.
  5. Bulk index into v3 with the same _id.

Tuning per BULK_IMPORT_TUNING.md mirrors index_embeddings_to_es.py: target
~10 MB bulk bodies, configurable inflight bulks.

Run:
    uv run --no-project python scripts/reindex_v2_to_v3.py
"""

from __future__ import annotations

import argparse
import asyncio
import time

import httpx
import numpy as np
import orjson
import redis.asyncio as aioredis


DEFAULT_ES = "http://localhost:9200"
DEFAULT_REDIS = "redis://localhost:6379/0"
DEFAULT_SRC = "local-article-index-v2"
DEFAULT_DST = "local-article-index-v3"
DEFAULT_MODEL = "useful-cub-58"

EMB_DIM = 128
FP16_BYTES = EMB_DIM * 2  # 256


async def open_pit(client: httpx.AsyncClient, index: str, keep_alive: str = "30m") -> str:
    r = await client.post(f"/{index}/_pit?keep_alive={keep_alive}")
    r.raise_for_status()
    return r.json()["id"]


async def close_pit(client: httpx.AsyncClient, pit_id: str) -> None:
    try:
        await client.request("DELETE", "/_pit", json={"id": pit_id})
    except Exception:
        pass


async def scan_articles(
    client: httpx.AsyncClient,
    pit_id: str,
    page_size: int,
    slice_id: int = 0,
    slices: int = 1,
):
    search_after: list | None = None
    while True:
        body: dict = {
            "size": page_size,
            "track_total_hits": False,
            "pit": {"id": pit_id, "keep_alive": "30m"},
            "_source": True,
            "sort": [{"_shard_doc": "asc"}],
            "query": {"match_all": {}},
        }
        if slices > 1:
            body["slice"] = {"id": slice_id, "max": slices}
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


def build_bulk_body(docs: list[tuple[str, dict]], dst_index: str) -> bytes:
    parts: list[bytes] = []
    for doc_id, doc in docs:
        meta = {"index": {"_index": dst_index, "_id": doc_id}}
        parts.append(orjson.dumps(meta))
        parts.append(b"\n")
        parts.append(orjson.dumps(doc))
        parts.append(b"\n")
    return b"".join(parts)


async def send_bulk(
    client: httpx.AsyncClient, body: bytes, sem: asyncio.Semaphore
) -> tuple[int, int]:
    async with sem:
        r = await client.post(
            "/_bulk",
            content=body,
            headers={"Content-Type": "application/x-ndjson"},
        )
        r.raise_for_status()
        data = orjson.loads(r.content)
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


def hashes_from_source(src: dict) -> list[str]:
    """Extract the unique inputHash list from a v2 _source. Order-preserving."""
    seen: set[str] = set()
    out: list[str] = []
    for emb in src.get("embeddings") or []:
        h = emb.get("inputHash")
        if h and h not in seen:
            seen.add(h)
            out.append(h)
    return out


async def main_async(args: argparse.Namespace) -> None:
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

    if args.pit_id:
        pit_id = args.pit_id
        own_pit = False
        print(
            f"using shared PIT {pit_id[:24]}... slice {args.slice_id}/{args.slices}",
            flush=True,
        )
    else:
        pit_id = await open_pit(es, args.src)
        own_pit = True
        print(f"opened PIT {pit_id[:24]}... src={args.src} dst={args.dst}", flush=True)

    sem = asyncio.Semaphore(args.inflight)
    inflight: set[asyncio.Task] = set()

    n_articles = 0
    n_with_emb = 0
    n_no_emb = 0
    n_vecs_fetched = 0
    n_hashes_total = 0
    n_ok = 0
    n_err = 0
    bytes_sent = 0
    t_start = time.time()
    t_log = t_start

    try:
        async for hits in scan_articles(
            es, pit_id, args.page_size, slice_id=args.slice_id, slices=args.slices
        ):
            if args.limit and n_articles >= args.limit:
                break

            todo: list[tuple[str, dict, list[str]]] = []
            unique_hashes: set[str] = set()
            for hit in hits:
                src = hit.get("_source", {})
                hs = hashes_from_source(src)
                if hs:
                    n_with_emb += 1
                else:
                    n_no_emb += 1
                todo.append((hit["_id"], src, hs))
                unique_hashes.update(hs)

            vec_map = await mget_vectors(r, sorted(unique_hashes))
            n_vecs_fetched += len(vec_map)

            chunk: list[tuple[str, dict]] = []
            chunk_bytes = 0
            for doc_id, src, hs in todo:
                if hs:
                    src["embeddings"] = [
                        {"vector": vec_map[h], "inputHash": h} for h in hs
                    ]
                    src["embeddingModelVersion"] = args.model
                else:
                    src.pop("embeddings", None)
                    src.pop("embeddingModelVersion", None)
                chunk.append((doc_id, src))
                n_hashes_total += len(hs)
                n_articles += 1
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
                    f"with_emb={n_with_emb:,} no_emb={n_no_emb:,}  "
                    f"vecs={n_vecs_fetched:,}  "
                    f"bulk_ok={n_ok:,} inflight={len(inflight)}",
                    flush=True,
                )
                t_log = now

        if inflight:
            results = await asyncio.gather(*inflight)
            for ok, er in results:
                n_ok += ok
                n_err += er
    finally:
        if own_pit:
            await close_pit(es, pit_id)
        await es.aclose()
        await r.aclose()

    elapsed = time.time() - t_start
    print()
    print(f"DONE in {elapsed/60:.1f} min")
    print(f"  articles written to {args.dst}: {n_articles:,}")
    print(f"    with embeddings:              {n_with_emb:,}")
    print(f"    without embeddings:           {n_no_emb:,}")
    print(f"  bulk ok:                        {n_ok:,}")
    print(f"  bulk errors:                    {n_err:,}")
    print(f"  unique vectors fetched:         {n_vecs_fetched:,}")
    print(f"  bytes sent (bulk bodies):       {bytes_sent/1e9:.2f} GB")
    print(f"  avg hashes/article(w/emb):      {n_hashes_total / max(n_with_emb, 1):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--es", default=DEFAULT_ES)
    ap.add_argument("--redis", default=DEFAULT_REDIS)
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--dst", default=DEFAULT_DST)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--page-size", type=int, default=500)
    ap.add_argument("--target-body-bytes", type=int, default=10 * 1024 * 1024)
    ap.add_argument("--inflight", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pit-id", default=None,
                    help="reuse an existing PIT (required when sharding)")
    ap.add_argument("--slices", type=int, default=1,
                    help="total number of parallel slices (>=2 requires --pit-id)")
    ap.add_argument("--slice-id", type=int, default=0,
                    help="0-based slice index for this worker")
    args = ap.parse_args()
    if args.slices > 1 and not args.pit_id:
        ap.error("--slices > 1 requires --pit-id (open one PIT and share its id)")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

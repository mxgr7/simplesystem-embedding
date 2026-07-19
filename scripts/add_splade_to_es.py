"""In-place add of `spladeVector` to the live semantic index.

No reindex: PUT the sparse_vector field into the mapping, then scan the index
(PIT slices), read each doc's own embeddings[].inputHash, mget the prewarmed
splade:v1:<hash> vectors from Redis, max-merge them into one sparse_vector, and
write it back with a bulk `update` (partial doc). Idempotent — a crashed run just
re-runs. Coverage tracks dense: docs with no embeddings get no spladeVector.

Prereq: scripts/prewarm_splade.py has populated splade:v1:<hash> for the catalog.

Probe first:
  uv run --no-project python scripts/add_splade_to_es.py --ensure-mapping \
      --vendor <uuid> --limit 50000
Full run (all slices):
  uv run --no-project python scripts/add_splade_to_es.py --ensure-mapping --slices 16
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import httpx
import redis.asyncio as aioredis

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from indexer.sparse_codec import merge_max, unpack_sparse  # noqa: E402

DEFAULT_ES = "http://localhost:9200"
DEFAULT_REDIS = "redis://localhost:6379/0"
DEFAULT_INDEX = "prod-article-index-v1-semantic-bf16"
HASH_VERSION = "v1"


async def ensure_mapping(es, index, drop_source):
    """Add spladeVector (sparse_vector) + spladeModelVersion (keyword). Optionally
    also exclude spladeVector from stored _source (applies to rewritten docs)."""
    body = {"properties": {
        "spladeVector": {"type": "sparse_vector"},
        "spladeModelVersion": {"type": "keyword"},
    }}
    if drop_source:
        body["_source"] = {"excludes": ["embeddings.vector", "spladeVector"]}
    r = await es.put(f"/{index}/_mapping", json=body)
    r.raise_for_status()
    print(f"mapping updated: {r.json()}", flush=True)


async def open_pit(es, index, keep="30m"):
    r = await es.post(f"/{index}/_pit?keep_alive={keep}")
    r.raise_for_status()
    return r.json()["id"]


async def close_pit(es, pit_id):
    try:
        await es.request("DELETE", "/_pit", json={"id": pit_id})
    except Exception:
        pass


async def scan_slice(es, pit_id, page, slice_id, slices, vendor):
    query = {"term": {"vendorId": vendor}} if vendor else {"match_all": {}}
    after = None
    while True:
        body = {
            "size": page,
            "track_total_hits": False,
            "pit": {"id": pit_id, "keep_alive": "30m"},
            "_source": {"includes": ["embeddings.inputHash"]},
            "sort": [{"_shard_doc": "asc"}],
            "query": query,
        }
        if slices > 1:
            body["slice"] = {"id": slice_id, "max": slices}
        if after is not None:
            body["search_after"] = after
        r = await es.post("/_search", json=body)
        r.raise_for_status()
        hits = r.json()["hits"]["hits"]
        if not hits:
            return
        yield hits
        after = hits[-1]["sort"]


async def send_bulk(es, body, sem):
    delay = 0.5
    for _ in range(10):
        data = None
        async with sem:
            r = await es.post("/_bulk", content=body,
                              headers={"Content-Type": "application/x-ndjson"})
            if r.status_code != 429:
                r.raise_for_status()
                data = r.json()
        if data is None:
            await asyncio.sleep(delay + random.uniform(0, delay))
            delay = min(delay * 2, 30.0)
            continue
        if not data.get("errors"):
            return len(data["items"])
        n_ok = had_429 = 0
        for it in data["items"]:
            op = next(iter(it.values()))
            st = op.get("status", 500)
            if st < 300:
                n_ok += 1
            elif st == 429 or "rejected_execution" in str(op.get("error", "")):
                had_429 = 1
            else:
                raise SystemExit(f"bulk error {st} _id={op.get('_id')}: {op.get('error')}")
        if not had_429:
            return n_ok
        await asyncio.sleep(delay + random.uniform(0, delay))
        delay = min(delay * 2, 30.0)
    raise SystemExit("bulk: exhausted 429 retries")


def hashes_of(src):
    return [e["inputHash"] for e in (src.get("embeddings") or []) if e.get("inputHash")]


class Stats:
    def __init__(self):
        self.docs = self.updated = self.no_emb = self.no_splade = 0
        self.ok = 0
        self.t0 = time.time()


async def worker(slice_id, args, es, r, pit_id, sem, st, model):
    inflight: set = set()

    async def drain(thr):
        nonlocal inflight
        while len(inflight) >= thr:
            done, inflight = await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                st.ok += d.result()

    chunk, cbytes = [], 0
    async for hits in scan_slice(es, pit_id, args.page_size, slice_id, args.slices, args.vendor):
        todo, uniq = [], set()
        for h in hits:
            if args.limit and st.docs >= args.limit:
                break
            hs = hashes_of(h.get("_source", {}))
            st.docs += 1
            if not hs:
                st.no_emb += 1
                continue
            todo.append((h["_id"], hs))
            uniq.update(hs)
        if uniq:
            keys = [f"splade:{HASH_VERSION}:{x}" for x in uniq]
            raw = await r.mget(keys)
            vecmap = {x: unpack_sparse(b) for x, b in zip(uniq, raw) if b is not None}
        else:
            vecmap = {}
        for doc_id, hs in todo:
            dicts = [vecmap[x] for x in hs if x in vecmap]
            if not dicts:
                st.no_splade += 1
                continue
            sv = merge_max(dicts)
            chunk.append(json.dumps({"update": {"_id": doc_id}}, separators=(",", ":")))
            chunk.append(json.dumps({"doc": {"spladeVector": sv, "spladeModelVersion": model}},
                                    separators=(",", ":"), ensure_ascii=False))
            cbytes += 200 + len(sv) * 12
            st.updated += 1
            if cbytes >= args.target_body_bytes:
                await drain(args.inflight)
                inflight.add(asyncio.create_task(
                    send_bulk(es, "\n".join(chunk) + "\n", sem)))
                chunk, cbytes = [], 0
        if args.limit and st.docs >= args.limit:
            break
    if chunk:
        await drain(args.inflight)
        inflight.add(asyncio.create_task(send_bulk(es, "\n".join(chunk) + "\n", sem)))
    if inflight:
        for ok in await asyncio.gather(*inflight):
            st.ok += ok


async def log_loop(st):
    while True:
        await asyncio.sleep(5)
        el = time.time() - st.t0
        print(f"  {st.docs:,} scanned ({st.docs/max(el,1e-3):,.0f}/s) "
              f"updated={st.updated:,} no_emb={st.no_emb:,} "
              f"no_splade={st.no_splade:,} ok={st.ok:,}", flush=True)


async def main_async(args):
    es = httpx.AsyncClient(base_url=args.es, timeout=httpx.Timeout(300.0, connect=10.0),
                           limits=httpx.Limits(max_connections=args.inflight * 2))
    r = aioredis.from_url(args.redis, decode_responses=False, max_connections=64)
    if args.ensure_mapping:
        await ensure_mapping(es, args.index, args.drop_source)
    pit_id = await open_pit(es, args.index)
    print(f"PIT open, slices={args.slices}, index={args.index}", flush=True)
    sem = asyncio.Semaphore(args.inflight)
    st = Stats()
    logger = asyncio.create_task(log_loop(st))
    try:
        await asyncio.gather(*[
            worker(i, args, es, r, pit_id, sem, st, args.splade_model)
            for i in range(args.slices)])
    finally:
        logger.cancel()
        await close_pit(es, pit_id)
        await es.aclose()
        await r.aclose()
    el = time.time() - st.t0
    print(f"\nDONE {el/60:.1f}min  scanned={st.docs:,} updated={st.updated:,} "
          f"no_emb={st.no_emb:,} no_splade={st.no_splade:,} ok={st.ok:,} "
          f"({st.docs/max(el,1e-3):,.0f}/s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--es", default=DEFAULT_ES)
    ap.add_argument("--redis", default=DEFAULT_REDIS)
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--splade-model", default="capable-auk-759")
    ap.add_argument("--ensure-mapping", action="store_true")
    ap.add_argument("--drop-source", action="store_true",
                    help="also exclude spladeVector (+embeddings.vector) from _source")
    ap.add_argument("--vendor", default="")
    ap.add_argument("--slices", type=int, default=16)
    ap.add_argument("--page-size", type=int, default=500)
    ap.add_argument("--inflight", type=int, default=12)
    ap.add_argument("--target-body-bytes", type=int, default=10 * 1024 * 1024)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

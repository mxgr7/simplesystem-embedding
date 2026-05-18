"""Multiprocess v5 importer — staging clone -> local-article-index-v5.

Why this exists: the asyncio single-process variant
(index_embeddings_to_es_v5.py) is GIL-bound at ~1 core on json.dumps of
large docs while ES sits ~idle. This version runs one OS process per PIT
slice (true parallelism, the model the original clone used), so the
~30 idle cores actually do the JSON build. ES becomes the next bound.

Parent builds the (vendor,artno)->hashes lookup ONCE and forks workers,
so the multi-GB dict is inherited copy-on-write (never pickled / never
duplicated per worker).

Per article (unchanged semantics vs the asyncio variant):
  - resolve unique article_hashes from the parquet lookup
  - MGET fp16 vectors from Redis tei:v2:<hash> (cache is warm, miss≈0)
  - embeddings=[{vector,inputHash}] + embeddingModelVersion
  - denormalized catalogVersionIds / priceKeys (pure fn of the doc)
  - drop stale fields; same explicit _id
  - _bulk with 429/rejected_execution backoff (idempotent replay)

Run (probe):  ... --procs 32 --limit 200000
Run (full):   ... --procs 32
"""

from __future__ import annotations

import argparse
import binascii
import gc
import hashlib
import math
import multiprocessing as mp
import os
import queue as _queue
import random
import time

import duckdb
import httpx
import numpy as np
import orjson
import redis

# orjson serializes np.float32 vectors directly (no per-vector Python
# float list) — the #3 de-churn that flattens per-worker RSS growth.
_ORJSON = orjson.OPT_SERIALIZE_NUMPY

DEFAULT_PARQUET = (
    "/data/datasets/mongo_offers_export_20260516/article_hashes_v2/**/*.parquet"
)
DEFAULT_ES = "http://localhost:9200"
DEFAULT_REDIS = "redis://localhost:6379/0"
DEFAULT_SRC = "stg-articles-v1-clone-20260516"
DEFAULT_DST = "local-article-index-v5"
DEFAULT_MODEL = "useful-cub-58"

EMB_DIM = 128
FP16_BYTES = EMB_DIM * 2
STALE_FIELDS = ("embeddings", "embeddingsBuiltAt", "rerankTexts", "rerankTextsBuiltAt")

# The (vendor,artno)->hashes lookup, as fork-COW-SAFE numpy buffers.
#
# Why not a Python dict: a ~tens-of-M-entry dict of (str,str)->tuple[str]
# inherited via fork() is NOT actually shared. CPython mutates each
# object's refcount header on every *read*, so every touched page is
# COW-privatized per worker -> ~32x blowup -> the OOM that downed the box.
#
# These arrays are single contiguous buffers with no per-element
# PyObjects: workers only ever read them, no refcounts are touched, COW
# stays shared. Footprint is ~one copy total regardless of --procs.
#
#   _G_KEYS  structured (hi,lo) u64[N], sorted lexicographically =
#            128-bit blake2b of f"{vendor}\x00{artno}" (collision-free in
#            practice: P(coll) ~ N^2/2^129 for N~5e7 is ~1e-23)
#   _G_ORD   i64[N]  : sorted position -> ORIGINAL entry index
#   _G_HOFF  i64[N+1]: original entry -> [a,b) row range in _G_HBUF
#   _G_HBUF  u8[M,16]: every article_hash, 32-hex decoded to 16 bytes
#   _G_KNOWN u64[V]  : sorted 64-bit vendor-id digests (orphan check)
_KEY_DT = np.dtype([("h", "<u8"), ("l", "<u8")])
_G_KEYS: np.ndarray = np.empty(0, _KEY_DT)
_G_ORD: np.ndarray = np.empty(0, np.int64)
_G_HOFF: np.ndarray = np.empty(1, np.int64)
_G_HBUF: np.ndarray = np.empty((0, 16), np.uint8)
_G_KNOWN: np.ndarray = np.empty(0, np.uint64)
# Known-vendor membership as a frozenset of 64-bit digests (~2.6k ints,
# negligible, built in parent, COW-inherited). Replaces a per-article
# np.uint64 + scalar searchsorted (millions of tiny numpy allocs).
_G_VSET: frozenset[int] = frozenset()


def _k128(s: str) -> tuple[int, int]:
    d = hashlib.blake2b(s.encode("utf-8"), digest_size=16).digest()
    return (int.from_bytes(d[:8], "big"), int.from_bytes(d[8:], "big"))


def _k64(s: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def build_lookup(parquet_glob: str) -> None:
    """Populate the _G_* numpy buffers from the parquet. Runs ONCE in the
    parent before fork; sets module globals (no return)."""
    global _G_KEYS, _G_ORD, _G_HOFF, _G_HBUF, _G_KNOWN, _G_VSET
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
    n = len(vendor)

    kh = np.empty(n, np.uint64)
    kl = np.empty(n, np.uint64)
    hoff = np.empty(n + 1, np.int64)
    hoff[0] = 0
    hbuf = bytearray()
    known: set[int] = set()
    for i in range(n):
        v, a = vendor[i], artno[i]
        hi, lo = _k128(v + "\x00" + a)
        kh[i], kl[i] = hi, lo
        known.add(_k64(v))
        for h in hashes[i]:
            hbuf += bytes.fromhex(h)
        hoff[i + 1] = len(hbuf) // 16

    _G_HBUF = np.frombuffer(bytes(hbuf), np.uint8).reshape(-1, 16)
    del hbuf
    _G_HOFF = hoff
    keys = np.empty(n, _KEY_DT)
    keys["h"] = kh
    keys["l"] = kl
    del kh, kl
    _G_ORD = np.argsort(keys, kind="stable").astype(np.int64)
    _G_KEYS = keys[_G_ORD]
    _G_KNOWN = np.array(sorted(known), np.uint64)
    _G_VSET = frozenset(known)
    print(f"  lookup: {n:,} pairs, {_G_HBUF.shape[0]:,} hashes, "
          f"{_G_KNOWN.shape[0]:,} vendors, "
          f"buffers≈{(_G_KEYS.nbytes + _G_ORD.nbytes + _G_HOFF.nbytes + _G_HBUF.nbytes) / 1e9:.2f}GB"
          f" in {time.time()-t0:.1f}s", flush=True)


def open_pit(es: httpx.Client, index: str) -> str:
    r = es.post(f"/{index}/_pit?keep_alive=12h")
    r.raise_for_status()
    return r.json()["id"]


def resolve_hits(hits: list) -> list:
    """Per-PAGE batched (vendor,artno)->hashes resolution.

    Returns a list parallel to `hits`: None = orphan vendor (skip the
    article entirely); else the list[str] of unique article_hashes
    (first-seen order; empty if vendorId/offers absent). Semantics are
    identical to the old per-article path, but ALL the page's offer keys
    go through ONE np.searchsorted instead of one np.array+searchsorted
    per offer — collapsing ~tens of millions of tiny numpy temporaries
    (the residual per-worker RSS growth jemalloc couldn't fix) into one
    array build + a handful of vectorized gathers per ~500-doc page.
    """
    n = len(hits)
    res: list = [None] * n
    qh: list[int] = []
    ql: list[int] = []
    owners: list[tuple] = []  # (hit_idx, vendor_id, artno) per query key
    for idx, hit in enumerate(hits):
        src = hit.get("_source", {})
        vid = src.get("vendorId")
        if vid is None:
            res[idx] = []
            continue
        if _k64(vid) not in _G_VSET:
            res[idx] = None  # orphan vendor
            continue
        offers = src.get("offers") or []
        res[idx] = []  # non-None placeholder => not orphan
        for off in offers:
            artno = off.get("articleNumber")
            if artno is None:
                continue
            hi, lo = _k128(vid + "\x00" + artno)
            qh.append(hi)
            ql.append(lo)
            owners.append((idx, vid, artno))
    if owners:
        m = len(owners)
        qa = np.empty(m, _KEY_DT)
        qa["h"] = qh
        qa["l"] = ql
        pos = np.searchsorted(_G_KEYS, qa)
        np.clip(pos, 0, _G_KEYS.shape[0] - 1, out=pos)
        match = _G_KEYS[pos] == qa  # vectorized structured equality
        ent = _G_ORD[pos]
        for k in range(m):
            idx, vid, artno = owners[k]
            if not match[k]:
                raise SystemExit(
                    f"missing hash (vendor={vid!r}, artno={artno!r}) — "
                    f"vendor in parquet but pair absent. Aborting (fail-fast)."
                )
            e = int(ent[k])
            hx = binascii.hexlify(
                _G_HBUF[_G_HOFF[e]:_G_HOFF[e + 1]].tobytes()).decode("ascii")
            lst = res[idx]
            for j in range(0, len(hx), 32):
                h = hx[j:j + 32]
                if h not in lst:  # lst is tiny (~1-3) — cheaper than a set
                    lst.append(h)
    return res


def denormalize(src: dict) -> None:
    cvids: set[str] = set()
    for o in src.get("offers") or []:
        for c in o.get("catalogVersionIds") or []:
            cvids.add(c)
    pkeys: set[str] = set()
    for p in src.get("prices") or []:
        pl, cur = p.get("priceListId"), p.get("currency")
        if pl and cur:
            pkeys.add(f"{pl}|{cur}")
    src["catalogVersionIds"] = sorted(cvids)
    src["priceKeys"] = sorted(pkeys)


# Transient transport failures (read/connect/write timeout, conn reset,
# remote-protocol, pool) — expected under disk-saturated ES. NOT fatal:
# retry with backoff. A bare ReadTimeout was the bug that crashed a worker
# and hung the whole run (parent waits on a _done sentinel that never comes).
TRANSIENT = (httpx.TimeoutException, httpx.TransportError)


def es_search(es: httpx.Client, body: dict) -> list:
    """POST /_search with backoff on transient transport errors AND 429.
    Re-issuing the same search_after query is idempotent."""
    delay = 0.5
    for _ in range(12):
        try:
            r = es.post("/_search", json=body)
        except TRANSIENT:
            time.sleep(delay + random.uniform(0, delay))
            delay = min(delay * 2, 30.0)
            continue
        if r.status_code == 429:
            time.sleep(delay + random.uniform(0, delay))
            delay = min(delay * 2, 30.0)
            continue
        r.raise_for_status()
        return r.json()["hits"]["hits"]
    raise SystemExit("search: exhausted transient/429 retries (12)")


def post_bulk(es: httpx.Client, body: bytes) -> int:
    """Index NDJSON; back off on transient transport errors AND
    429/rejected_execution (idempotent replay — every op has an explicit
    _id). Fatal only on a genuine non-429 item error."""
    delay = 0.5
    for _ in range(12):
        try:
            r = es.post("/_bulk", content=body,
                        headers={"Content-Type": "application/x-ndjson"})
        except TRANSIENT:
            time.sleep(delay + random.uniform(0, delay))
            delay = min(delay * 2, 30.0)
            continue
        if r.status_code == 429:
            time.sleep(delay + random.uniform(0, delay))
            delay = min(delay * 2, 30.0)
            continue
        r.raise_for_status()
        data = r.json()
        if not data.get("errors"):
            return len(data["items"])
        n_ok = 0
        had_429 = False
        for item in data["items"]:
            op = next(iter(item.values()))
            st = op.get("status", 500)
            if st < 300:
                n_ok += 1
            elif st == 429 or "rejected_execution" in str(op.get("error", "")):
                had_429 = True
            else:
                raise SystemExit(
                    f"bulk error: status={st} _id={op.get('_id')} "
                    f"error={op.get('error')}"
                )
        if not had_429:
            return n_ok
        time.sleep(delay + random.uniform(0, delay))
        delay = min(delay * 2, 30.0)
    raise SystemExit("bulk: exhausted transient/429 retries (12)")


def mget_vectors(rc: redis.Redis, hashes: list[str]) -> dict:
    """hash -> np.float32[128]. Kept as an ndarray (no .tolist()): orjson
    serializes it directly, eliminating ~9M 128-element Python float
    lists per full run — the dominant obmalloc-churn source."""
    if not hashes:
        return {}
    out: dict = {}
    for i in range(0, len(hashes), 8192):
        chunk = hashes[i:i + 8192]
        raw = rc.mget([f"tei:v2:{h}" for h in chunk])
        for h, b in zip(chunk, raw):
            if b is None:
                continue  # warm cache => ~never; doc gets fewer/no vecs
            out[h] = np.frombuffer(b, dtype=np.float16).astype(np.float32)
    return out


def worker(slice_id: int, args, pit_id: str, q: mp.Queue) -> None:
    # CRITICAL: the parent does gc.freeze()+gc.disable() before fork so a
    # GC pass can't dirty the COW-shared graph. gc.disable() is INHERITED
    # here — leaving it off means per-doc reference cycles (httpx/retry/
    # JSON internals) are never collected => unbounded linear RSS growth
    # (the leak jemalloc and the de-churn both failed to fix). Re-enable
    # GC: gc.freeze() in the parent already moved the shared graph into a
    # permanent gen GC never scans, so worker collections only touch
    # objects created AFTER fork (private) — COW stays intact.
    gc.enable()
    es = httpx.Client(base_url=args.es,
                      timeout=httpx.Timeout(600.0, connect=10.0),
                      limits=httpx.Limits(max_connections=args.inflight * 2,
                                           max_keepalive_connections=args.inflight))
    rc = redis.from_url(args.redis, decode_responses=False)
    cap = math.ceil(args.limit / args.procs) if args.limit else 0

    s = dict(arts=0, emb=0, no_emb=0, orphan=0, vecs=0, miss=0, ok=0, by=0)
    produced = 0  # cumulative, never reset — drives the --limit cap
    last_push = time.time()
    chunk: list[bytes] = []
    chunk_bytes = 0
    search_after = None

    def flush() -> None:
        nonlocal chunk, chunk_bytes
        if not chunk:
            return
        body = b"\n".join(chunk) + b"\n"
        s["ok"] += post_bulk(es, body)
        s["by"] += len(body)
        chunk, chunk_bytes = [], 0

    while True:
        body_q = {
            "size": args.page_size,
            "track_total_hits": False,
            "pit": {"id": pit_id, "keep_alive": "12h"},
            "_source": True,
            "sort": [{"_shard_doc": "asc"}],
            "query": {"match_all": {}},
            "slice": {"id": slice_id, "max": args.procs},
        }
        if search_after is not None:
            body_q["search_after"] = search_after
        hits = es_search(es, body_q)
        if not hits:
            break
        search_after = hits[-1]["sort"]

        res = resolve_hits(hits)  # one batched searchsorted for the page
        todo = []
        uniq: set[str] = set()
        for i, hit in enumerate(hits):
            if cap and produced >= cap:
                break
            hs = res[i]
            if hs is None:
                s["orphan"] += 1
                continue
            src = hit.get("_source", {})
            for f in STALE_FIELDS:
                src.pop(f, None)
            denormalize(src)
            todo.append((hit["_id"], src, hs))
            uniq.update(hs)
            s["arts"] += 1
            produced += 1

        vmap = mget_vectors(rc, sorted(uniq))
        s["vecs"] += len(vmap)
        for doc_id, src, hs in todo:
            vecs = [{"vector": vmap[h], "inputHash": h} for h in hs if h in vmap]
            s["miss"] += sum(1 for h in hs if h not in vmap)
            if vecs:
                src["embeddings"] = vecs
                src["embeddingModelVersion"] = args.model
                s["emb"] += 1
            else:
                s["no_emb"] += 1
            chunk.append(orjson.dumps({"index": {"_index": args.dst,
                                                  "_id": doc_id}}))
            chunk.append(orjson.dumps(src, option=_ORJSON))
            chunk_bytes += 3072 + len(vecs) * EMB_DIM * 10
            if chunk_bytes >= args.target_body_bytes:
                flush()

        # One deterministic cyclic sweep per page (~500 docs). Negligible
        # vs the bulk POST + search round-trips; keeps RSS flat instead of
        # waiting on generational thresholds under the high alloc rate.
        gc.collect()

        now = time.time()
        if now - last_push >= 2.0:
            q.put(dict(s))
            for k in s:
                s[k] = 0
            last_push = now
        if cap and produced >= cap:
            break

    flush()
    q.put(dict(s))
    q.put({"_done": slice_id})
    es.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--es", default=DEFAULT_ES)
    ap.add_argument("--redis", default=DEFAULT_REDIS)
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--dst", default=DEFAULT_DST)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 32,
                    help="OS processes = PIT slices (true parallelism)")
    ap.add_argument("--inflight", type=int, default=2,
                    help="httpx keepalive headroom per worker (sync 1 bulk/proc)")
    ap.add_argument("--page-size", type=int, default=500)
    ap.add_argument("--target-body-bytes", type=int, default=4 * 1024 * 1024)
    ap.add_argument("--limit", type=int, default=0,
                    help="approx total article cap (probe); 0 = all")
    args = ap.parse_args()

    build_lookup(args.parquet)  # populates the _G_* numpy buffers

    es = httpx.Client(base_url=args.es, timeout=60.0)
    pit_id = open_pit(es, args.src)
    print(f"opened PIT {pit_id[:20]}...  procs={args.procs} "
          f"limit={args.limit or 'ALL'}", flush=True)

    ctx = mp.get_context("fork")
    q: mp.Queue = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(i, args, pit_id, q))
             for i in range(args.procs)]
    # Extra COW hygiene: freeze the surviving Python heap into a gen the
    # cyclic GC won't walk (a GC pass writes objects too -> dirties pages),
    # and stop GC entirely in the soon-forked workers. The numpy buffers
    # above are the real fix; this protects the small remaining graph.
    gc.collect()
    gc.freeze()
    gc.disable()
    t0 = time.time()
    for p in procs:
        p.start()

    tot = dict(arts=0, emb=0, no_emb=0, orphan=0, vecs=0, miss=0, ok=0, by=0)
    done = 0
    last_log = time.time()

    def check_crash() -> None:
        """Abort loudly if a worker died without sending its _done sentinel
        — never block on q.get() forever (the bug that hung the prior run)."""
        crashed = [(i, p.exitcode) for i, p in enumerate(procs)
                   if (not p.is_alive()) and p.exitcode not in (0, None)]
        alive = sum(p.is_alive() for p in procs)
        if crashed and done + alive < args.procs:
            for p in procs:
                if p.is_alive():
                    p.terminate()
            raise SystemExit(
                f"ABORT: worker(s) crashed (slice,exitcode)={crashed}; "
                f"v5 would be INCOMPLETE — investigate and re-run."
            )

    while done < args.procs:
        try:
            msg = q.get(timeout=15)
        except _queue.Empty:
            check_crash()
            continue
        if "_done" in msg:
            done += 1
            continue
        for k in tot:
            tot[k] += msg.get(k, 0)
        now = time.time()
        if now - last_log >= 5.0:
            check_crash()  # detect a crash even while others stream progress
            el = now - t0
            print(f"  {tot['arts']:,} arts ({tot['arts']/max(el,1e-3):,.0f}/s) "
                  f"emb={tot['emb']:,} no_emb={tot['no_emb']:,} "
                  f"orphan={tot['orphan']:,} vecs={tot['vecs']:,} "
                  f"miss={tot['miss']:,} ok={tot['ok']:,} "
                  f"{tot['by']/1e6/max(el,1e-3):,.1f}MB/s", flush=True)
            last_log = now

    for p in procs:
        p.join()
    es.request("DELETE", "/_pit", json={"id": pit_id})
    el = time.time() - t0
    print(f"\nDONE in {el/60:.2f} min ({el:.0f}s)")
    print(f"  articles written:       {tot['arts']:,}")
    print(f"    with embeddings:      {tot['emb']:,}")
    print(f"    no embeddings:        {tot['no_emb']:,}")
    print(f"  skipped orphan vendor:  {tot['orphan']:,}")
    print(f"  unique vectors:         {tot['vecs']:,}")
    print(f"  redis misses:           {tot['miss']:,}")
    print(f"  bulk ok:                {tot['ok']:,}")
    print(f"  throughput:             {tot['arts']/max(el,1e-3):,.0f} articles/s")
    print(f"  model stamped:          {args.model}")


if __name__ == "__main__":
    main()

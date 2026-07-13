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
import glob
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue as _queue
import random
import shutil
import time

import duckdb
import httpx
import numpy as np
import orjson
import redis
from dotenv import load_dotenv

# orjson serializes np.float32 vectors directly (no per-vector Python
# float list) — the #3 de-churn that flattens per-worker RSS growth.
_ORJSON = orjson.OPT_SERIALIZE_NUMPY

DEFAULT_PARQUET = (
    "/data/datasets/mongo_offers_export_20260527/article_hashes_v2/**/*.parquet"
)
DEFAULT_ES = "http://localhost:9200"
DEFAULT_REDIS = "redis://localhost:6666/0"
DEFAULT_SRC = "prod-article-index-v1"
DEFAULT_DST = ""  # required: timestamped prod-article-index-v1-semantic-<TS>
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


_LOOKUP_CACHE_FILES = ("keys.npy", "ord.npy", "hoff.npy", "hbuf.npy", "known.npy")


def _lookup_cache_key(parquet_glob: str, vendor_ids: list[str] | None) -> str:
    """Content key for the lookup cache: the parquet file set (path/size/mtime)
    plus the vendor filter. Any change invalidates the cache (rebuild)."""
    h = hashlib.md5()
    h.update(repr(sorted(vendor_ids or [])).encode())
    for f in sorted(glob.glob(parquet_glob, recursive=True)):
        st = os.stat(f)
        h.update(f"{f}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()


_BUILD_ARROW = None  # set transiently in build_lookup for the fork-Pool workers


def _build_chunk(bounds):
    """Transform arrow rows [lo:hi) -> (kh, kl, per-row hash counts, decoded-hash
    bytes, vendor-digest array). Runs in a fork-Pool worker reading the
    COW-inherited _BUILD_ARROW (never pickled in); only the compact results are
    pickled back. This is the parallelised half of the lookup build (blake2b
    keys + hex-decode of ~330M hashes) that was previously one serial core."""
    lo, hi = bounds
    a = _BUILD_ARROW
    vendor = a.column("vendor_id").slice(lo, hi - lo).to_pylist()
    artno = a.column("article_number").slice(lo, hi - lo).to_pylist()
    hashes = a.column("hashes").slice(lo, hi - lo).to_pylist()
    m = hi - lo
    kh = np.empty(m, np.uint64)
    kl = np.empty(m, np.uint64)
    counts = np.empty(m, np.int64)
    buf = bytearray()
    known: set[int] = set()
    for i in range(m):
        v = vendor[i]
        khi, klo = _k128(v + "\x00" + artno[i])
        kh[i] = khi
        kl[i] = klo
        known.add(_k64(v))
        hs = hashes[i]
        for h in hs:
            buf += bytes.fromhex(h)
        counts[i] = len(hs)
    return kh, kl, counts, bytes(buf), np.fromiter(known, np.uint64, len(known))


def build_lookup(parquet_glob: str, vendor_ids: list[str] | None = None,
                 mem_limit: str = "4GB", cache_dir: str = "") -> None:
    """Populate the _G_* numpy buffers from the parquet. Runs ONCE in the
    parent before fork; sets module globals (no return). When vendor_ids is
    given, only those vendors' pairs are loaded -- this keeps the resident
    lookup bounded (the full-fleet lookup is ~150-250M pairs and will not fit
    on a small-RAM box). The filter is pushed into DuckDB so non-matching
    parquet row groups are skipped, not materialised.

    When cache_dir is set, the built buffers are saved there (np.save) and a
    matching cache is mmap-loaded on the next run instead of rebuilt — so a
    resumed run skips the ~10-30 min build. The big arrays are mmap'd
    (mmap_mode='r'): demand-paged + fork-COW-shared, never resident in full."""
    global _G_KEYS, _G_ORD, _G_HOFF, _G_HBUF, _G_KNOWN, _G_VSET
    t0 = time.time()
    key = _lookup_cache_key(parquet_glob, vendor_ids) if cache_dir else None
    meta_path = os.path.join(cache_dir, "meta.json") if cache_dir else ""
    if cache_dir and os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path))
        except Exception:
            meta = {}
        if meta.get("key") == key and all(
                os.path.exists(os.path.join(cache_dir, f))
                for f in _LOOKUP_CACHE_FILES):
            _G_KEYS = np.load(os.path.join(cache_dir, "keys.npy"), mmap_mode="r")
            _G_ORD = np.load(os.path.join(cache_dir, "ord.npy"), mmap_mode="r")
            _G_HOFF = np.load(os.path.join(cache_dir, "hoff.npy"), mmap_mode="r")
            _G_HBUF = np.load(os.path.join(cache_dir, "hbuf.npy"), mmap_mode="r")
            _G_KNOWN = np.load(os.path.join(cache_dir, "known.npy"))  # tiny
            _G_VSET = frozenset(int(x) for x in _G_KNOWN.tolist())
            print(f"  lookup: CACHE HIT {cache_dir} — {_G_KEYS.shape[0]:,} pairs, "
                  f"{_G_HBUF.shape[0]:,} hashes (mmap'd) in {time.time()-t0:.1f}s",
                  flush=True)
            return
    print(f"loading lookup from {parquet_glob} "
          f"(vendors={len(vendor_ids) if vendor_ids else 'ALL'}) ...", flush=True)
    con = duckdb.connect()
    con.execute(f"SET threads = {os.cpu_count() or 8}")
    con.execute(f"SET memory_limit = '{mem_limit}'")
    con.execute("SET enable_progress_bar = false")
    where = ""
    if vendor_ids:
        vlist = ", ".join("'" + v.replace("'", "''") + "'" for v in vendor_ids)
        where = f"WHERE vendor_id IN ({vlist})"
    arrow = con.execute(f"""
        SELECT vendor_id, article_number,
               article_hashes AS hashes
        FROM read_parquet('{parquet_glob}', hive_partitioning = true)
        {where}
    """).fetch_arrow_table()
    # Parallelise the per-row transform (blake2b keys + hex-decode of every
    # hash) across all cores via a fork-Pool. The arrow table is COW-shared
    # (each worker .slice()s + to_pylist()s only its own range); chunks are
    # concatenated in row order so the entry ordering is identical to the old
    # serial loop. tp = build worker count.
    global _BUILD_ARROW
    n = arrow.num_rows
    tp = min(max(1, os.cpu_count() or 8), max(1, n))
    bounds = [(i * n // tp, (i + 1) * n // tp) for i in range(tp)]
    bounds = [(lo, hi) for (lo, hi) in bounds if hi > lo]
    print(f"  building lookup across {len(bounds)} cores ...", flush=True)
    _BUILD_ARROW = arrow
    try:
        if len(bounds) > 1:
            with mp.get_context("fork").Pool(len(bounds)) as pool:
                results = pool.map(_build_chunk, bounds)
        else:
            results = [_build_chunk(b) for b in bounds]
    finally:
        _BUILD_ARROW = None
    del arrow

    kh = (np.concatenate([r[0] for r in results]) if results
          else np.empty(0, np.uint64))
    kl = (np.concatenate([r[1] for r in results]) if results
          else np.empty(0, np.uint64))
    counts = (np.concatenate([r[2] for r in results]) if results
              else np.empty(0, np.int64))
    hbuf_bytes = b"".join(r[3] for r in results)
    known: set[int] = set()
    for r in results:
        known.update(r[4].tolist())
    del results

    hoff = np.empty(n + 1, np.int64)
    hoff[0] = 0
    if n:
        np.cumsum(counts, out=hoff[1:])
    _G_HBUF = np.frombuffer(hbuf_bytes, np.uint8).reshape(-1, 16)
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
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(os.path.join(cache_dir, "keys.npy"), _G_KEYS)
        np.save(os.path.join(cache_dir, "ord.npy"), _G_ORD)
        np.save(os.path.join(cache_dir, "hoff.npy"), _G_HOFF)
        np.save(os.path.join(cache_dir, "hbuf.npy"), _G_HBUF)
        np.save(os.path.join(cache_dir, "known.npy"), _G_KNOWN)
        tmp = meta_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"key": key, "pairs": int(n)}, f)
        os.replace(tmp, meta_path)
        print(f"  lookup: cached to {cache_dir} for fast resume", flush=True)


def open_pit(es: httpx.Client, index: str) -> str:
    r = es.post(f"/{index}/_pit?keep_alive=12h")
    r.raise_for_status()
    return r.json()["id"]


def pit_alive(es: httpx.Client, pit_id: str) -> bool:
    """True if the PIT still exists (resume against the same snapshot is
    possible). A size=0 search returns 200 if alive, 404 if it aged out."""
    try:
        r = es.post("/_search", json={"size": 0, "track_total_hits": False,
                                      "pit": {"id": pit_id, "keep_alive": "12h"}})
        return r.status_code == 200
    except Exception:
        return False


def resolve_hits(hits: list):
    """Per-PAGE batched (vendor,artno)->hashes resolution + classification.

    Returns three lists parallel to `hits`:
      kinds[i]     -- "no_offer"      : article has no offers[]
                      "unmappable"    : offers[] present but none carry an
                                        articleNumber (can't be looked up)
                      "orphan_vendor" : vendor absent from the lookup
                      "lookup_gap"    : vendor known but >=1 offer pair is
                                        missing from / maps to empty in the
                                        lookup (partial hashes may still exist)
                      "resolved"      : all offer pairs found, >=1 hash
      hashlists[i] -- list[str] of unique article_hashes resolved (first-seen
                      order; a partial subset when kind == "lookup_gap"; empty
                      for the no-hash kinds)
      miss_info[i] -- for "orphan_vendor"/"lookup_gap": the offer articleNumbers
                      that could not be mapped (for backfill logging). None else.

    NON-FATAL by design: a pair absent from the lookup is recorded as a
    lookup_gap, never raised. ALL the page's offer keys still go through ONE
    np.searchsorted — the COW-shared lookup hot path is unchanged.
    """
    n = len(hits)
    kinds: list = [None] * n
    hashlists: list = [None] * n
    miss_info: list = [None] * n
    qh: list[int] = []
    ql: list[int] = []
    owners: list[tuple] = []  # (hit_idx, artno) per query key
    for idx, hit in enumerate(hits):
        src = hit.get("_source", {})
        vid = src.get("vendorId")
        offers = src.get("offers") or []
        if not offers:
            kinds[idx] = "no_offer"
            hashlists[idx] = []
            continue
        artnos = [o.get("articleNumber") for o in offers
                  if o.get("articleNumber") is not None]
        if not artnos:
            kinds[idx] = "unmappable"
            hashlists[idx] = []
            continue
        if vid is None or _k64(vid) not in _G_VSET:
            kinds[idx] = "orphan_vendor"
            hashlists[idx] = []
            miss_info[idx] = artnos
            continue
        hashlists[idx] = []
        miss_info[idx] = []  # collects lookup-missing artnos for this article
        for artno in artnos:
            hi, lo = _k128(vid + "\x00" + artno)
            qh.append(hi)
            ql.append(lo)
            owners.append((idx, artno))
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
            idx, artno = owners[k]
            if not match[k]:
                miss_info[idx].append(artno)  # lookup gap (non-fatal)
                continue
            e = int(ent[k])
            hx = binascii.hexlify(
                _G_HBUF[_G_HOFF[e]:_G_HOFF[e + 1]].tobytes()).decode("ascii")
            lst = hashlists[idx]
            for j in range(0, len(hx), 32):
                h = hx[j:j + 32]
                if h not in lst:  # lst is tiny (~1-3) — cheaper than a set
                    lst.append(h)
    # Finalise vendor-known articles: a missing pair OR an all-empty mapping
    # (matched pairs that carried no hashes) is a lookup_gap; else resolved.
    for idx in range(n):
        if kinds[idx] is not None:
            continue
        if miss_info[idx] or not hashlists[idx]:
            kinds[idx] = "lookup_gap"
        else:
            kinds[idx] = "resolved"
            miss_info[idx] = None
    return kinds, hashlists, miss_info


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
    # Two ES clients: src for the PIT scan, dst for bulk writes. Same client
    # if --src-es == --dst-es.
    src_es = httpx.Client(base_url=args.src_es,
                      timeout=httpx.Timeout(600.0, connect=10.0),
                      limits=httpx.Limits(max_connections=args.inflight * 2,
                                           max_keepalive_connections=args.inflight))
    dst_es = (src_es if args.dst_es == args.src_es
              else httpx.Client(base_url=args.dst_es,
                      timeout=httpx.Timeout(600.0, connect=10.0),
                      limits=httpx.Limits(max_connections=args.inflight * 2,
                                           max_keepalive_connections=args.inflight)))
    rc = redis.from_url(args.redis, decode_responses=False)
    cap = math.ceil(args.limit / args.procs) if args.limit else 0

    s = dict(arts=0, emb=0, vecs=0, miss=0, ok=0, by=0,
             no_offer=0, unmappable=0, orphan_vendor=0,
             lookup_gap=0, cache_gap=0, full=0)
    produced = 0  # cumulative, never reset — drives the --limit cap
    last_push = time.time()
    chunk: list[bytes] = []
    chunk_bytes = 0
    search_after = None
    pending_cursor = None  # search_after of the last FULLY-buffered page

    # Resume: load this slice's saved cursor (written after the last flush, so
    # it's at-or-behind durable data). Re-covering the in-flight tail on resume
    # is safe — explicit _id makes the re-writes idempotent.
    slice_ckpt_path = os.path.join(args.checkpoint_dir, f"slice.{slice_id}.json")
    if os.path.exists(slice_ckpt_path):
        try:
            _cp = json.load(open(slice_ckpt_path))
            search_after = _cp.get("search_after")
            produced = _cp.get("count", 0)
            pending_cursor = search_after  # never checkpoint behind resume point
        except Exception:
            pass

    def _write_ckpt() -> None:
        tmp = slice_ckpt_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"pit_id": pit_id, "procs": args.procs,
                       "slice_id": slice_id, "search_after": pending_cursor,
                       "count": produced}, f)
        os.replace(tmp, slice_ckpt_path)  # atomic

    def flush() -> None:
        nonlocal chunk, chunk_bytes
        if not chunk:
            return
        body = b"\n".join(chunk) + b"\n"
        s["ok"] += post_bulk(dst_es, body)
        s["by"] += len(body)
        chunk, chunk_bytes = [], 0
        _write_ckpt()  # cursor now at-or-behind durable data

    # Per-worker miss log (opened lazily on first miss; no dedup/aggregation).
    miss_log_path = f"{args.miss_log}.{slice_id}.jsonl"
    _miss_fh = None

    def log_miss(rec: dict) -> None:
        nonlocal _miss_fh
        if _miss_fh is None:
            _miss_fh = open(miss_log_path, "ab")
        _miss_fh.write(orjson.dumps(rec) + b"\n")

    vendor_ids = [v.strip() for v in (args.vendor_ids or "").split(",") if v.strip()]
    src_query = (
        {"terms": {"vendorId": vendor_ids}} if vendor_ids else {"match_all": {}}
    )

    while True:
        body_q = {
            "size": args.page_size,
            "track_total_hits": False,
            "pit": {"id": pit_id, "keep_alive": "12h"},
            "_source": True,
            "sort": [{"_shard_doc": "asc"}],
            "query": src_query,
            "slice": {"id": slice_id, "max": args.procs},
        }
        if search_after is not None:
            body_q["search_after"] = search_after
        hits = es_search(src_es, body_q)
        if not hits:
            break
        search_after = hits[-1]["sort"]

        kinds, hashlists, miss_info = resolve_hits(hits)  # batched, non-fatal
        todo = []
        uniq: set[str] = set()
        for i, hit in enumerate(hits):
            if cap and produced >= cap:
                break
            hs = hashlists[i]
            src = hit.get("_source", {})
            for f in STALE_FIELDS:
                src.pop(f, None)
            denormalize(src)
            # Every article is written (incl. orphan_vendor) so it stays in the
            # index and can be embedding-backfilled later.
            todo.append((hit["_id"], src, hs, kinds[i], miss_info[i]))
            if hs:
                uniq.update(hs)
            s["arts"] += 1
            produced += 1

        vmap = mget_vectors(rc, sorted(uniq))
        s["vecs"] += len(vmap)
        for doc_id, src, hs, kind, m_artnos in todo:
            cache_missing = [h for h in hs if h not in vmap]
            vecs = [{"vector": vmap[h], "inputHash": h} for h in hs if h in vmap]
            s["miss"] += len(cache_missing)
            if vecs:
                src["embeddings"] = vecs
                src["embeddingModelVersion"] = args.model
                s["emb"] += 1
            vid = src.get("vendorId")
            artno = src.get("articleNumber")
            # Classify final outcome. no_offer is benign (counter only); every
            # other non-full kind is logged explicitly for backfill.
            if kind == "no_offer":
                s["no_offer"] += 1
            elif kind == "unmappable":
                s["unmappable"] += 1
                log_miss({"reason": "unmappable", "articleId": doc_id,
                          "vendorId": vid, "articleNumber": artno})
            elif kind == "orphan_vendor":
                s["orphan_vendor"] += 1
                log_miss({"reason": "orphan_vendor", "articleId": doc_id,
                          "vendorId": vid, "articleNumber": artno,
                          "offerArticleNumbers": m_artnos})
            elif kind == "lookup_gap":
                s["lookup_gap"] += 1
                log_miss({"reason": "lookup_gap", "articleId": doc_id,
                          "vendorId": vid, "articleNumber": artno,
                          "missingArticleNumbers": m_artnos,
                          "cacheMissingHashes": cache_missing})
            elif cache_missing:  # kind == "resolved" but >=1 hash uncached
                s["cache_gap"] += 1
                log_miss({"reason": "cache_gap", "articleId": doc_id,
                          "vendorId": vid, "articleNumber": artno,
                          "cacheMissingHashes": cache_missing})
            else:  # all offer pairs resolved AND all hashes cached
                s["full"] += 1
            _action = orjson.dumps({"index": {"_index": args.dst,
                                              "_id": doc_id}})
            _doc = orjson.dumps(src, option=_ORJSON)
            chunk.append(_action)
            chunk.append(_doc)
            # Size by ACTUAL serialized bytes (+2 newlines), not an estimate —
            # prod docs vary 100x (fat offers/prices arrays), so a flat-constant
            # estimate undercounts and lets a bulk body blow past ES's
            # http.max_content_length -> 413. Real bytes keep bulks at the cap.
            chunk_bytes += len(_action) + len(_doc) + 2
            if chunk_bytes >= args.target_body_bytes:
                flush()

        # The whole page is now buffered: a later flush() may checkpoint this
        # cursor. (A mid-page flush still checkpoints the PREVIOUS page's
        # cursor — conservative; the in-flight page is re-covered on resume.)
        pending_cursor = search_after

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
    if _miss_fh is not None:
        _miss_fh.close()
    src_es.close()
    if dst_es is not src_es:
        dst_es.close()


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    # Legacy single-cluster --es kept as fallback for both reads + writes.
    # Cross-cluster: pass --src-es and --dst-es (env-defaulted to
    # ELASTIC_PROD_URL / ELASTIC_URL respectively).
    ap.add_argument("--es", default=os.environ.get("ELASTIC_URL", DEFAULT_ES),
                    help="Legacy single-cluster URL; used when --src-es / "
                         "--dst-es aren't set.")
    ap.add_argument("--src-es", default=os.environ.get("ELASTIC_PROD_URL", ""),
                    help="ES URL for the PIT scan (read side). "
                         "Defaults to ELASTIC_PROD_URL.")
    ap.add_argument("--dst-es", default=os.environ.get("ELASTIC_URL", ""),
                    help="ES URL for bulk writes (write side). "
                         "Defaults to ELASTIC_URL.")
    ap.add_argument("--redis", default=DEFAULT_REDIS,
                    help="KVRocks/Redis URL for vector cache (default :6666).")
    ap.add_argument("--src", default=DEFAULT_SRC,
                    help="source index/alias to PIT-scan on --src-es.")
    ap.add_argument("--dst", default=DEFAULT_DST, required=True,
                    help="destination index to bulk-write on --dst-es.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 32,
                    help="OS processes = PIT slices (true parallelism)")
    ap.add_argument("--inflight", type=int, default=2,
                    help="httpx keepalive headroom per worker (sync 1 bulk/proc)")
    ap.add_argument("--page-size", type=int, default=500)
    ap.add_argument("--target-body-bytes", type=int, default=4 * 1024 * 1024)
    ap.add_argument("--limit", type=int, default=0,
                    help="approx total article cap (probe); 0 = all")
    ap.add_argument("--vendor-ids", default="",
                    help="Comma-separated vendorId values; restricts the PIT "
                         "scan to those vendors via a terms filter. Empty = all.")
    ap.add_argument("--miss-log", default="",
                    help="Path prefix for per-worker miss logs (JSONL); each "
                         "worker writes <prefix>.<slice>.jsonl. Logs every "
                         "genuine miss (everything except no_offer) for later "
                         "backfill. Default: import_misses.<dst>")
    ap.add_argument("--checkpoint-dir", default="",
                    help="Dir for per-slice resume checkpoints (PIT id + scan "
                         "cursor, written after every flush). Re-running with the "
                         "same dir auto-resumes each slice where it left off "
                         "(idempotent _id makes re-cover safe). --procs must "
                         "match the checkpointed run. Default: import_ckpt.<dst>")
    ap.add_argument("--lookup-cache-dir", default="lookup_cache",
                    help="Dir to cache the built lookup buffers (np.save) so a "
                         "resumed run mmap-loads them instead of the ~10-30 min "
                         "rebuild. Empty string disables.")
    args = ap.parse_args()
    if args.procs < 2:
        sys.exit("--procs must be >= 2 (ES PIT slicing requires max >= 2 slices)")

    # Resolve src/dst URLs (--src-es/--dst-es take precedence, else fall back
    # to --es). Workers use these directly so we just normalise here for the
    # PIT open + later DELETE.
    args.src_es = args.src_es or args.es
    args.dst_es = args.dst_es or args.es
    if not args.src_es or not args.dst_es:
        sys.exit("ES URLs missing: pass --src-es / --dst-es or set "
                 "ELASTIC_PROD_URL / ELASTIC_URL in .env")
    print(f"src: {args.src} on {args.src_es.split('@')[-1].split('/')[0]}",
          flush=True)
    print(f"dst: {args.dst} on {args.dst_es.split('@')[-1].split('/')[0]}",
          flush=True)

    if not args.miss_log:
        args.miss_log = f"import_misses.{args.dst}"
    _miss_dir = os.path.dirname(args.miss_log)
    if _miss_dir:
        os.makedirs(_miss_dir, exist_ok=True)
    print(f"miss log: {args.miss_log}.<slice>.jsonl", flush=True)

    if not args.checkpoint_dir:
        args.checkpoint_dir = f"import_ckpt.{args.dst}"
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    run_meta_path = os.path.join(args.checkpoint_dir, "run.json")

    vendor_ids = [v.strip() for v in (args.vendor_ids or "").split(",") if v.strip()]
    build_lookup(args.parquet, vendor_ids,
                 cache_dir=args.lookup_cache_dir)  # COW numpy buffers (cached)

    src_es = httpx.Client(base_url=args.src_es, timeout=60.0)

    # --- Resume decision: per-slice cursors live in <ckpt>/slice.<id>.json
    # (written after each flush); run.json holds the run identity + PIT id.
    prior = None
    if os.path.exists(run_meta_path):
        try:
            prior = json.load(open(run_meta_path))
        except Exception:
            prior = None
    if prior is not None:
        if (prior.get("procs") != args.procs or prior.get("src") != args.src
                or prior.get("dst") != args.dst):
            sys.exit(
                f"checkpoint in {args.checkpoint_dir} is for a different run "
                f"(mismatch {prior}). Match --procs/--src/--dst or clear the dir.")
        if prior.get("pit_id") and pit_alive(src_es, prior["pit_id"]):
            pit_id = prior["pit_id"]
            print(f"RESUME: reusing live PIT {pit_id[:20]}... (same snapshot)",
                  flush=True)
        else:
            pit_id = open_pit(src_es, args.src)
            print(f"RESUME (fresh-PIT fallback): prior PIT expired; opened new "
                  f"PIT {pit_id[:20]}... resuming from saved cursors against a "
                  f"NEWER snapshot — idempotent re-cover, minor drift OK",
                  flush=True)
        prior["pit_id"] = pit_id
        with open(run_meta_path, "w") as f:
            json.dump(prior, f)
    else:
        for stale in glob.glob(os.path.join(args.checkpoint_dir, "slice.*.json")):
            os.remove(stale)
        pit_id = open_pit(src_es, args.src)
        with open(run_meta_path, "w") as f:
            json.dump({"pit_id": pit_id, "procs": args.procs,
                       "src": args.src, "dst": args.dst}, f)
        print(f"opened PIT {pit_id[:20]}... (fresh start)", flush=True)
    print(f"  procs={args.procs} limit={args.limit or 'ALL'} "
          f"checkpoint_dir={args.checkpoint_dir}", flush=True)

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

    tot = dict(arts=0, emb=0, vecs=0, miss=0, ok=0, by=0,
               no_offer=0, unmappable=0, orphan_vendor=0,
               lookup_gap=0, cache_gap=0, full=0)
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
                  f"full={tot['full']:,} no_offer={tot['no_offer']:,} "
                  f"lookup_gap={tot['lookup_gap']:,} cache_gap={tot['cache_gap']:,} "
                  f"orphan={tot['orphan_vendor']:,} unmap={tot['unmappable']:,} "
                  f"miss={tot['miss']:,} ok={tot['ok']:,} "
                  f"{tot['by']/1e6/max(el,1e-3):,.1f}MB/s", flush=True)
            last_log = now

    for p in procs:
        p.join()
    src_es.request("DELETE", "/_pit", json={"id": pit_id})
    src_es.close()
    # Clean completion (all slices done, no crash): the resume checkpoints are
    # now useless — drop them so an accidental re-run starts fresh, not resumes
    # at end-of-scan. The lookup cache is kept (reusable). A crash never reaches
    # here (check_crash SystemExits), so checkpoints survive for resume.
    shutil.rmtree(args.checkpoint_dir, ignore_errors=True)
    el = time.time() - t0
    print(f"\nDONE in {el/60:.2f} min ({el:.0f}s)")
    print(f"  articles written:       {tot['arts']:,}")
    print(f"    with embeddings:      {tot['emb']:,}")
    print(f"  --- classification (sums to articles written) ---")
    print(f"    full (ok):            {tot['full']:,}")
    print(f"    no_offer (benign):    {tot['no_offer']:,}")
    print(f"    unmappable:           {tot['unmappable']:,}   <- logged")
    print(f"    orphan_vendor:        {tot['orphan_vendor']:,}   <- logged")
    print(f"    lookup_gap:           {tot['lookup_gap']:,}   <- logged")
    print(f"    cache_gap:            {tot['cache_gap']:,}   <- logged")
    print(f"  unique vectors:         {tot['vecs']:,}")
    print(f"  hash-level cache miss:  {tot['miss']:,}")
    print(f"  bulk ok:                {tot['ok']:,}")
    _gaps = tot['unmappable'] + tot['orphan_vendor'] + tot['lookup_gap'] + tot['cache_gap']
    print(f"  logged misses total:    {_gaps:,}  (see {args.miss_log}.*.jsonl)")
    print(f"  throughput:             {tot['arts']/max(el,1e-3):,.0f} articles/s")
    print(f"  model stamped:          {args.model}")


if __name__ == "__main__":
    main()

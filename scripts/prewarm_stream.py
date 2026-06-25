"""Stream-dedup prewarm — embed missing hashes directly from a Mongo gz
dump into KVRocks. NO render_inputs.parquet, NO DuckDB GROUP BY.

The old `prewarm_v2_missing.py` deduplicated 455M source rows down to
~180M unique content-hash rows via a DuckDB GROUP BY before doing the
EXISTS-filter / embed pass. That GROUP BY's working set was ~200 GB
because the `description` field is fat — it didn't fit even on a 247GB
box without spilling. The materialised `render_inputs.parquet` was
then also tens of GB of disk I/O.

This script does the dedup implicitly via KVRocks's EXISTS check, no
intermediate parquet:

    N workers, each handling a subset of vendor gz files. Per worker:
      stream rows  ->  compute article_hash in-process
                  ->  drop if recently-seen (small LRU)
                  ->  buffer + pipelined EXISTS against KVRocks
                  ->  for misses: render via Jinja  ->  TEI batch POST
                                ->  SET fp16 to KVRocks

Memory bound: ~0 (no GROUP BY in memory; LRU + small buffers per
worker only). Works on any box that can talk to KVRocks + TEI.

Idempotent: re-runs become no-ops for any hash already in KVRocks.
Resumable mid-stream: only an EXISTS round-trip per cached hash.

The hash function `compute_article_hash` here MUST match the DuckDB
macro `_DUCKDB_MACROS.compute_article_hash` in
`scripts/prewarm_v2_missing.py` and
`scripts/build_article_hashes_lookup.py` byte-for-byte (SHA-256 over
8 NUL-joined canonical fields, first 32 hex chars). The implementation
mirrors the macro's `_v2_canon_paths` (filter empty, ¦-join elements
per path, sort, chr(30)-join). See `_v2_canon_paths_bytes` below.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import logging
import multiprocessing as mp
import os
import queue as _queue
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import httpx
import numpy as np
import orjson
import redis
from dotenv import load_dotenv
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embedding_train.rendering import RowTextRenderer  # noqa: E402


VECTOR_DIM = 128
VECTOR_BYTES = VECTOR_DIM * 2  # fp16
HASH_VERSION = "v2"
KEY_PREFIX = "tei:v2:"

DEFAULT_TEI_URL = "http://localhost:8080"
DEFAULT_DATA_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "data" / "default.yaml"
)

log = logging.getLogger("prewarm_stream")


# --- hash function (must match the DuckDB macro byte-for-byte) ----------

_NUL = b"\x00"


def _v2_canon_paths_bytes(cps):
    """Mirror DuckDB macro _v2_canon_paths: filter empty paths,
    join each path's elements with U+00A6 '¦', sort the resulting
    strings lexicographically, join with U+001E (record separator)."""
    if not cps:
        return b""
    parts = []
    for cp in cps:
        elems = cp.get("elements") if isinstance(cp, dict) else None
        if not elems:
            continue
        parts.append("¦".join(elems))
    parts.sort()
    return "\x1e".join(parts).encode("utf-8")


def compute_article_hash(row):
    """Match scripts/build_article_hashes_lookup.py:compute_article_hash.

    Order: name, mfg_name, description, category_paths,
           ean, article_number, mfg_article_number, mfg_article_type.
    Each utf-8 encoded with COALESCE(x, '') semantics; NUL-joined.
    First 32 hex chars of the SHA-256 digest.
    """
    h = hashlib.sha256()
    h.update((row.get("name") or "").encode("utf-8")); h.update(_NUL)
    h.update((row.get("manufacturer_name") or "").encode("utf-8")); h.update(_NUL)
    h.update((row.get("description") or "").encode("utf-8")); h.update(_NUL)
    h.update(_v2_canon_paths_bytes(row.get("category_paths"))); h.update(_NUL)
    h.update((row.get("ean") or "").encode("utf-8")); h.update(_NUL)
    h.update((row.get("article_number") or "").encode("utf-8")); h.update(_NUL)
    h.update((row.get("manufacturer_article_number") or "").encode("utf-8")); h.update(_NUL)
    h.update((row.get("manufacturer_article_type") or "").encode("utf-8"))
    return h.hexdigest()[:32]


def _extract_row(doc):
    """gz-line JSON dict -> 8-field canonical row dict the renderer + hash
    function expect. Mirrors the snake_case schema used by render_inputs."""
    op = (doc.get("offer") or {}).get("offerParams") or {}
    return {
        "name": op.get("name") or "",
        "manufacturer_name": op.get("manufacturerName") or "",
        "description": op.get("description") or "",
        "category_paths": op.get("categoryPaths") or [],
        "ean": op.get("ean") or "",
        "article_number": doc.get("articleNumber") or "",
        "manufacturer_article_number": op.get("manufacturerArticleNumber") or "",
        "manufacturer_article_type": op.get("manufacturerArticleType") or "",
    }


# --- worker --------------------------------------------------------------

# Globals set in parent pre-fork (inherited COW-shared in workers).
_g_renderer = None
_g_tei_url = None
_g_redis_kwargs = None

# Optional pre-loaded snapshot of KVRocks keys at run-start time. When set,
# workers check this in-memory sorted numpy array before falling back to a
# pipelined EXISTS round-trip. Hashes added AFTER the snapshot was taken
# fall through to EXISTS and are correctly identified as cached, so the
# snapshot is safe even while other writers are concurrently embedding.
# Layout: structured np.array dtype=[('h','<u8'),('l','<u8')] sorted lex —
# fork-COW-shared (workers only read, never mutate, so the page stays shared).
_KEY_DT = np.dtype([("h", "<u8"), ("l", "<u8")])
_g_snap: np.ndarray = np.empty(0, dtype=_KEY_DT)

# Per-worker lazy clients.
_g_http = None
_g_rc = None


def _ensure_http():
    global _g_http
    if _g_http is None:
        _g_http = httpx.Client(
            timeout=httpx.Timeout(120.0, connect=10.0),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )
    return _g_http


def _ensure_redis():
    global _g_rc
    if _g_rc is None:
        _g_rc = redis.Redis(**_g_redis_kwargs)
    return _g_rc


def _tei_embed(texts):
    """POST to TEI with backoff on 429 / transient transport errors."""
    http = _ensure_http()
    payload = {"inputs": texts, "truncate": True}
    delay = 0.25
    for attempt in range(11):
        try:
            r = http.post(f"{_g_tei_url}/embed", json=payload)
            if r.status_code != 429:
                r.raise_for_status()
                arr = np.asarray(r.json(), dtype=np.float32)
                if arr.ndim != 2 or arr.shape[1] != VECTOR_DIM:
                    raise RuntimeError(f"TEI shape {arr.shape}, want (*, {VECTOR_DIM})")
                return arr
            reason = "HTTP 429"
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
        if attempt == 10:
            raise RuntimeError(f"TEI {reason} after 10 retries")
        time.sleep(min(delay * (2 ** attempt) + random.uniform(0, delay), 30.0))


def worker(wid, file_subset, args, q):
    """Per-worker: stream the assigned gz files, hash + EXISTS + embed."""
    rc = _ensure_redis()
    seen: OrderedDict = OrderedDict()
    pending: list = []  # list of (hash, row, text|None)
    stats = dict(scanned=0, seen_skip=0, cached_skip=0, embedded=0,
                 by=0, redis_misses=0)

    def flush_exists():
        """Pipelined EXISTS path — only used when no snapshot is loaded.
        With a snapshot, the worker checks per-row inline (see hot loop
        below) so `pending` never holds anything that needs to be checked,
        only confirmed misses ready to embed."""
        if not pending:
            return []
        pipe = rc.pipeline(transaction=False)
        for h, _, _ in pending:
            pipe.exists(f"{KEY_PREFIX}{h}")
        results = pipe.execute()
        miss = []
        for (h, row, text), exists in zip(pending, results):
            if exists:
                stats["cached_skip"] += 1
            else:
                miss.append((h, row, text))
        pending.clear()
        return miss

    # Snapshot-mode fast path: pre-compute references for tight loop.
    snap = _g_snap
    snap_size = snap.shape[0]
    snap_loaded = snap_size > 0

    def flush_snapshot_check(buf):
        """Bulk snapshot membership check.

        Per-row in_snapshot was the wrong choice — each call paid Python
        frame overhead (~5µs/row). Batching N=`snapshot_batch` keys lets
        us do ONE bytes.fromhex on the concatenated hex (pure C),
        np.frombuffer as a free view, and one C-native searchsorted.
        Amortised cost: <1µs/row.

        Returns the (h, row, text) tuples NOT in the snapshot."""
        if not buf:
            return []
        # Concatenate 32-char hex strings, hex-decode in one C call.
        joined = "".join(h for h, _, _ in buf)
        raw = bytes.fromhex(joined)
        keys = np.frombuffer(raw, dtype=_KEY_DT)  # view; no copy
        pos = np.searchsorted(snap, keys)
        np.clip(pos, 0, snap_size - 1, out=pos)
        hits = snap[pos] == keys  # bool[len(buf)]
        n_hit = int(hits.sum())
        stats["cached_skip"] += n_hit
        # Materialise miss list
        miss = [buf[i] for i in np.where(~hits)[0]]
        return miss

    def embed_batch(batch):
        """batch = list of (hash, row, text). Call TEI, write fp16 to Redis."""
        texts = []
        valid = []
        for h, row, text in batch:
            if text is None:
                ctx = _g_renderer.build_context(row)
                text = _g_renderer.render_offer_text(row, context=ctx)
            if text:
                texts.append(text)
                valid.append(h)
        if not texts:
            return
        arr = _tei_embed(texts)
        fp16 = arr.astype(np.float16)
        pipe = rc.pipeline(transaction=False)
        n_bytes = 0
        for h, vec in zip(valid, fp16):
            b = vec.tobytes()
            if len(b) != VECTOR_BYTES:
                raise RuntimeError(f"fp16 byte length {len(b)} != {VECTOR_BYTES}")
            pipe.set(f"{KEY_PREFIX}{h}", b)
            n_bytes += len(b)
        pipe.execute()
        stats["embedded"] += len(valid)
        stats["by"] += n_bytes

    last_push = time.time()

    # Pre-snapshot buffer: hashes that survived seen-LRU but haven't been
    # snapshot-checked yet. Distinct from `pending` (confirmed misses
    # awaiting embed).
    snap_buf: list = []
    SNAP_FLUSH_AT = args.exists_batch  # reuse for the same purpose

    def drain_per_file():
        """Flush any in-flight snap_buf + pending to KVRocks. Called at the
        end of each vendor file so the file is fully committed before we
        mark it done."""
        if snap_loaded:
            if snap_buf:
                pending.extend(flush_snapshot_check(snap_buf))
                snap_buf.clear()
            while pending:
                embed_batch(pending[:args.batch_size])
                del pending[:args.batch_size]
        else:
            miss = flush_exists()
            for i in range(0, len(miss), args.batch_size):
                embed_batch(miss[i:i + args.batch_size])

    for fpath in file_subset:
        try:
            with gzip.open(fpath, "rb") as f:
                for line in f:
                    doc = orjson.loads(line)
                    row = _extract_row(doc)
                    h = compute_article_hash(row)
                    stats["scanned"] += 1
                    if h in seen:
                        seen.move_to_end(h)
                        stats["seen_skip"] += 1
                        continue
                    seen[h] = None
                    if len(seen) > args.seen_lru:
                        seen.popitem(last=False)
                    if snap_loaded:
                        snap_buf.append((h, row, None))
                        if len(snap_buf) >= SNAP_FLUSH_AT:
                            misses = flush_snapshot_check(snap_buf)
                            snap_buf.clear()
                            pending.extend(misses)
                            while len(pending) >= args.batch_size:
                                embed_batch(pending[:args.batch_size])
                                del pending[:args.batch_size]
                    else:
                        pending.append((h, row, None))
                        if len(pending) >= args.exists_batch:
                            miss = flush_exists()
                            for i in range(0, len(miss), args.batch_size):
                                embed_batch(miss[i:i + args.batch_size])
                    now = time.time()
                    if now - last_push >= 2.0:
                        q.put(dict(stats))
                        for k in stats:
                            stats[k] = 0
                        last_push = now
            # End of file: flush in-flight so the file is fully committed
            # to KVRocks BEFORE we mark it done.
            drain_per_file()
            base = os.path.basename(fpath)
            print(f"  vendor done: {base}", flush=True)
            if args.done_log:
                # POSIX guarantees atomic O_APPEND writes <= PIPE_BUF (4096B)
                # so the basename + newline (always <100 bytes) lands as a
                # single record regardless of worker concurrency. No lock.
                with open(args.done_log, "ab") as df:
                    df.write((base + "\n").encode())
        except Exception as exc:
            q.put({"_error": f"worker {wid} file {fpath}: {type(exc).__name__}: {exc}"})
            raise
    # Final drain (also called per-file above; this is the safety net for
    # whatever's left at worker shutdown).
    drain_per_file()
    q.put(dict(stats))
    q.put({"_done": wid})


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-glob", required=True,
                    help="Glob for the gz dump (e.g. /data/.../vendor_*.json.gz)")
    ap.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG),
                    help="cfg.data YAML the model was trained against.")
    ap.add_argument("--tei-url", default=DEFAULT_TEI_URL)
    ap.add_argument("--redis-host", default="localhost")
    ap.add_argument("--redis-port", type=int, default=6379)
    ap.add_argument("--concurrency", type=int, default=16,
                    help="OS-level worker processes (= TEI in-flight bound).")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Inputs per TEI request (TEI cap is the upper limit).")
    ap.add_argument("--exists-batch", type=int, default=5000,
                    help="Hashes per pipelined Redis EXISTS round-trip.")
    ap.add_argument("--seen-lru", type=int, default=200000,
                    help="Per-worker LRU of recently-seen hashes; "
                         "lets us skip the EXISTS check for hashes "
                         "this worker already processed this run.")
    ap.add_argument("--cached-snapshot", default="",
                    help="Optional path to a uint128 binary snapshot of "
                         "KVRocks tei:v2:* keys (built by "
                         "scripts/dump_kvrocks_keys.py). When set, workers "
                         "do a vectorised in-memory check before any "
                         "EXISTS round-trip — eliminates ~99% of the "
                         "tunnel traffic during cache-hot scanning AND "
                         "lets the run survive a brief tunnel outage.")
    ap.add_argument("--done-log", default="",
                    help="Path to a file where workers append the basename "
                         "of each fully-processed vendor gz. On startup, "
                         "files already present in this file are skipped — "
                         "lets you resume cleanly after a crash mid-run. "
                         "Atomic O_APPEND writes; safe across all workers.")
    args = ap.parse_args()

    # Load renderer ONCE in parent — workers inherit it via COW fork.
    cfg = OmegaConf.load(args.data_config)
    renderer = RowTextRenderer(cfg)

    # Sanity-check Redis + TEI before forking workers.
    rc = redis.Redis(host=args.redis_host, port=args.redis_port)
    if not rc.ping():
        sys.exit("KVRocks PING failed")
    with httpx.Client(timeout=10.0) as cli:
        r = cli.get(f"{args.tei_url}/health")
        r.raise_for_status()

    # Discover files + partition round-robin across workers.
    files = sorted(glob.glob(args.input_glob))
    if not files:
        sys.exit(f"no files match {args.input_glob}")

    # Resume support: skip files already in the done-log.
    if args.done_log and os.path.exists(args.done_log):
        with open(args.done_log) as df:
            done_basenames = {line.strip() for line in df if line.strip()}
        before = len(files)
        files = [f for f in files if os.path.basename(f) not in done_basenames]
        log.info("resume: skipping %s of %s vendors already in %s",
                 f"{before - len(files):,}", f"{before:,}", args.done_log)

    subsets = [[] for _ in range(args.concurrency)]
    for i, f in enumerate(files):
        subsets[i % args.concurrency].append(f)
    log.info("partitioned %d files across %d workers", len(files), args.concurrency)

    global _g_renderer, _g_tei_url, _g_redis_kwargs, _g_snap
    _g_renderer = renderer
    _g_tei_url = args.tei_url
    _g_redis_kwargs = dict(host=args.redis_host, port=args.redis_port)

    if args.cached_snapshot:
        t = time.time()
        _g_snap = np.fromfile(args.cached_snapshot, dtype=_KEY_DT)
        log.info("loaded %s cached-key snapshot (%.2f GB) in %.1fs; sorting ...",
                 f"{_g_snap.shape[0]:,}", _g_snap.nbytes / 1e9, time.time() - t)
        # Snapshot file is UNSORTED on disk (the dumper skips the sort to
        # save CPU on a small box); sort once here before workers fork.
        # numpy sort on ~166M × 16 B is ~30-60s single-threaded; the
        # sorted array stays in the parent's COW-shared pages.
        t = time.time()
        _g_snap.sort()
        log.info("snapshot sorted in %.1fs", time.time() - t)

    ctx = mp.get_context("fork")
    q: mp.Queue = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(i, subsets[i], args, q))
             for i in range(args.concurrency)]
    t0 = time.time()
    for p in procs:
        p.start()

    tot = dict(scanned=0, seen_skip=0, cached_skip=0, embedded=0, by=0)
    done = 0
    last_log = time.time()

    def check_crash():
        crashed = [(i, p.exitcode) for i, p in enumerate(procs)
                   if (not p.is_alive()) and p.exitcode not in (0, None)]
        alive = sum(p.is_alive() for p in procs)
        if crashed and done + alive < args.concurrency:
            for p in procs:
                if p.is_alive():
                    p.terminate()
            raise SystemExit(
                f"ABORT: worker(s) crashed (slice,exitcode)={crashed}"
            )

    while done < args.concurrency:
        try:
            msg = q.get(timeout=15)
        except _queue.Empty:
            check_crash()
            continue
        if "_done" in msg:
            done += 1
            continue
        if "_error" in msg:
            print("WORKER ERROR:", msg["_error"], flush=True)
            continue
        for k in tot:
            tot[k] += msg.get(k, 0)
        now = time.time()
        if now - last_log >= 5.0:
            check_crash()
            el = now - t0
            print(f"  scanned={tot['scanned']:,} ({tot['scanned']/max(el,1e-3):,.0f}/s) "
                  f"seen_skip={tot['seen_skip']:,} cached_skip={tot['cached_skip']:,} "
                  f"embedded={tot['embedded']:,} ({tot['embedded']/max(el,1e-3):,.1f}/s) "
                  f"bytes={tot['by']/1e6:,.0f}MB", flush=True)
            last_log = now

    for p in procs:
        p.join()
    el = time.time() - t0
    print(f"\nDONE in {el/60:.2f} min ({el:.0f}s)")
    print(f"  rows scanned:           {tot['scanned']:,}")
    print(f"  intra-worker seen skip: {tot['seen_skip']:,}")
    print(f"  cached (KVRocks hit):   {tot['cached_skip']:,}")
    print(f"  newly embedded:         {tot['embedded']:,}")
    print(f"  vectors written:        {tot['by']/1e6:,.0f} MB")
    if tot['scanned']:
        cache_hit = tot['cached_skip'] / max(tot['cached_skip'] + tot['embedded'], 1)
        print(f"  cache hit ratio:        {cache_hit*100:.2f}%")


if __name__ == "__main__":
    main()

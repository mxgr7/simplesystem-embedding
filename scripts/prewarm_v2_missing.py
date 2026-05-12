"""Prewarm tei:v2:<hash> Redis entries for hashes missing from /data/redis-data.

Mirrors src/embedding_train/infer.py for text rendering: same
RowTextRenderer + same cfg.data loaded from the checkpoint. Embedding
backend is the TEI HTTP server (no local model forward), so this script
runs without a GPU.

Parallelism shape:
  - Main process: streams the parquet, runs pipelined Redis EXISTS to
    filter to missing hashes, dispatches batches to worker pool.
  - Fork-based worker pool (default --concurrency = 64): each worker has
    its own (lazy) httpx Client + Redis connection. Per batch, a worker
    renders all rows (Jinja2 template), POSTs to TEI, casts fp16,
    pipelined-SETs to Redis.
  - The renderer is loaded once in main, then inherited via fork's COW
    memory — no per-batch pickling of cfg.data or template state.

Two phases (cached):
  1. Build render_inputs.parquet — one row per unique v2 hash with the 8
     renderer-canonical columns. DuckDB over the gzipped Mongo exports,
     ~3 min. Skipped if the file already exists.
  2. Prewarm — fan out, fail-fast on TEI errors. Resumable: re-run is a
     no-op for any hash already present in Redis.
"""
from __future__ import annotations

import argparse
import gc
import logging
import multiprocessing as mp
import os
import random
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from threading import Lock

import duckdb
import httpx
import numpy as np
import pyarrow.parquet as pq
import redis
from dotenv import load_dotenv
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embedding_train.rendering import RowTextRenderer  # noqa: E402


# --- pinned constants — mirror indexer/tei_cache.py + indexer/projection.py
VECTOR_DIM = 128
VECTOR_BYTES = VECTOR_DIM * 2  # fp16
HASH_VERSION = "v2"

DEFAULT_TEI_URL = "http://217.91.60.211:17071"
# torch isn't installed in this venv, so we read cfg.data straight from the
# YAML that the checkpoint was trained against (configs/config.yaml selects
# data: default for the useful-cub-58 model). If a future checkpoint diverges
# from default.yaml, point --data-config at the matching yaml instead.
DEFAULT_DATA_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "data" / "default.yaml"
)
EXPORT_DIR = Path("/data/datasets/mongo_offers_export_20260512")
DEFAULT_INPUT_GLOB = str(EXPORT_DIR / "vendor_*.json.gz")
DEFAULT_RENDER_INPUTS = EXPORT_DIR / "render_inputs.parquet"

_DUCKDB_MACROS = r"""
CREATE OR REPLACE MACRO _v2_canon_paths(category_paths) AS
    encode(array_to_string(
        array_sort(
            list_transform(
                list_filter(
                    COALESCE(category_paths, []::STRUCT(elements VARCHAR[])[]),
                    cp -> cp.elements IS NOT NULL AND len(cp.elements) > 0
                ),
                cp -> array_to_string(cp.elements, '¦')
            )
        ),
        chr(30)
    ));

CREATE OR REPLACE MACRO compute_article_hash(
    a_name, a_mfg, a_desc, category_paths,
    a_ean, a_article_number, a_mfg_article_number, a_mfg_article_type
) AS
    substr(
        sha256(
            encode(COALESCE(a_name, '')) || '\x00'::BLOB ||
            encode(COALESCE(a_mfg, '')) || '\x00'::BLOB ||
            encode(COALESCE(a_desc, '')) || '\x00'::BLOB ||
            _v2_canon_paths(category_paths) || '\x00'::BLOB ||
            encode(COALESCE(a_ean, '')) || '\x00'::BLOB ||
            encode(COALESCE(a_article_number, '')) || '\x00'::BLOB ||
            encode(COALESCE(a_mfg_article_number, '')) || '\x00'::BLOB ||
            encode(COALESCE(a_mfg_article_type, ''))
        ),
        1, 32
    );
"""

_PINNED_COLUMNS = (
    "{"
    "'articleNumber': 'VARCHAR', "
    "'offer': 'STRUCT(offerParams STRUCT("
    "\"name\" VARCHAR, "
    "manufacturerName VARCHAR, "
    "\"description\" VARCHAR, "
    "categoryPaths STRUCT(elements VARCHAR[])[], "
    "ean VARCHAR, "
    "manufacturerArticleNumber VARCHAR, "
    "manufacturerArticleType VARCHAR"
    "))'"
    "}"
)

log = logging.getLogger("prewarm_v2")


# --- Phase 1: build render-inputs parquet --------------------------------

def build_render_inputs(input_glob: str, out_path: Path) -> None:
    if out_path.exists():
        log.info("render-inputs parquet exists at %s — skipping build", out_path)
        return
    log.info("building render-inputs parquet at %s ...", out_path)
    t0 = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET threads = {os.cpu_count() or 8}")
    con.execute("SET enable_progress_bar = false")
    con.execute("SET memory_limit = '200GB'")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET temp_directory = '{EXPORT_DIR}/duckdb_tmp'")
    con.execute(_DUCKDB_MACROS)

    sql = f"""
    COPY (
      WITH src AS (
        SELECT
          offer.offerParams.name                       AS name,
          offer.offerParams.manufacturerName           AS manufacturer_name,
          offer.offerParams.description                AS description,
          offer.offerParams.categoryPaths              AS category_paths,
          offer.offerParams.ean                        AS ean,
          articleNumber                                AS article_number,
          offer.offerParams.manufacturerArticleNumber  AS manufacturer_article_number,
          offer.offerParams.manufacturerArticleType    AS manufacturer_article_type
        FROM read_json(
          '{input_glob}',
          format='newline_delimited',
          compression='gzip',
          maximum_object_size=67108864,
          columns={_PINNED_COLUMNS}
        )
      ),
      hashed AS (
        SELECT
          compute_article_hash(name, manufacturer_name, description, category_paths,
                               ean, article_number, manufacturer_article_number,
                               manufacturer_article_type) AS article_hash,
          name, manufacturer_name, description, category_paths, ean,
          article_number, manufacturer_article_number, manufacturer_article_type
        FROM src
      )
      SELECT
        article_hash,
        any_value(name)                       AS name,
        any_value(manufacturer_name)          AS manufacturer_name,
        any_value(description)                AS description,
        any_value(category_paths)             AS category_paths,
        any_value(ean)                        AS ean,
        any_value(article_number)             AS article_number,
        any_value(manufacturer_article_number) AS manufacturer_article_number,
        any_value(manufacturer_article_type)  AS manufacturer_article_type
      FROM hashed
      GROUP BY article_hash
    ) TO '{out_path}'
    (FORMAT PARQUET, COMPRESSION 'zstd', ROW_GROUP_SIZE 50000);
    """
    con.execute(sql)
    n_rows = duckdb.sql(f"SELECT COUNT(*) FROM '{out_path}'").fetchone()[0]
    log.info("built render-inputs parquet: %s rows in %.1fs",
             f"{n_rows:,}", time.time() - t0)


# --- worker globals (set in main pre-fork) -------------------------------

_g_renderer: RowTextRenderer | None = None
_g_tei_url: str | None = None
_g_redis_kwargs: dict | None = None

# Per-worker lazy resources (one per process, created on first batch).
_g_http_client: httpx.Client | None = None
_g_redis_client: redis.Redis | None = None


def _ensure_http() -> httpx.Client:
    global _g_http_client
    if _g_http_client is None:
        # Each worker handles 1 in-flight TEI POST at a time — a tiny pool
        # is enough. Server-side concurrency comes from running N workers
        # in parallel, not from many connections per worker.
        _g_http_client = httpx.Client(
            timeout=httpx.Timeout(120.0, connect=10.0),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )
    return _g_http_client


def _ensure_redis() -> redis.Redis:
    global _g_redis_client
    if _g_redis_client is None:
        _g_redis_client = redis.Redis(**_g_redis_kwargs)
    return _g_redis_client


def worker_process_batch(rows: list[dict]) -> tuple[int, int]:
    """Render → TEI → Redis for one batch. Raises on any error.

    Returns (embedded_count, bytes_written)."""
    if not rows:
        return 0, 0

    # Render all rows. The renderer was set in module globals before fork.
    texts: list[str] = []
    valid_hashes: list[str] = []
    for row in rows:
        ctx = _g_renderer.build_context(row)  # type: ignore[union-attr]
        text = _g_renderer.render_offer_text(row, context=ctx)  # type: ignore[union-attr]
        if text:
            texts.append(text)
            valid_hashes.append(row["article_hash"])

    if not texts:
        return 0, 0

    # One TEI POST per batch. Retry with exponential backoff + jitter on
    # transient signals: HTTP 429 (server saying "slow down") and httpx
    # transport timeouts (TCP backlog full, connect refused, etc.). 5xx
    # and malformed responses still fail-fast.
    http = _ensure_http()
    payload = {"inputs": texts, "truncate": True}
    max_retries = 10
    base_delay = 0.25
    resp = None
    for attempt in range(max_retries + 1):
        try:
            resp = http.post(f"{_g_tei_url}/embed", json=payload)
            if resp.status_code != 429:
                break
            reason = f"HTTP 429"
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            # Covers Connect/Read/Write/Pool timeouts + ConnectError +
            # RemoteProtocolError + ReadError. Anything else (e.g. invalid
            # JSON response, programming errors) still surfaces immediately.
            reason = f"{type(exc).__name__}: {exc}"
        if attempt == max_retries:
            raise RuntimeError(f"TEI {reason} after {max_retries} retries")
        delay = min(base_delay * (2 ** attempt) + random.uniform(0, base_delay), 30.0)
        time.sleep(delay)
    assert resp is not None
    resp.raise_for_status()
    arr = np.asarray(resp.json(), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != VECTOR_DIM:
        raise RuntimeError(
            f"TEI returned shape {arr.shape}, expected (*, {VECTOR_DIM})"
        )
    fp16 = arr.astype(np.float16)

    # Pipelined Redis SET.
    rc = _ensure_redis()
    pipe = rc.pipeline(transaction=False)
    total_bytes = 0
    for h, vec in zip(valid_hashes, fp16):
        b = vec.tobytes()
        if len(b) != VECTOR_BYTES:
            raise RuntimeError(f"fp16 byte length {len(b)} != {VECTOR_BYTES}")
        pipe.set(f"tei:v2:{h}", b)
        total_bytes += len(b)
    pipe.execute()

    return len(valid_hashes), total_bytes


# --- main-side stats + drain -------------------------------------------

class PrewarmStats:
    def __init__(self) -> None:
        self.lock = Lock()
        self.skipped_cached = 0
        self.embedded = 0
        self.tei_calls = 0
        self.bytes_written = 0
        self.start = time.time()

    def log_progress(self, total_target: int) -> None:
        with self.lock:
            elapsed = time.time() - self.start
            rate = self.embedded / elapsed if elapsed > 0 else 0
            scanned = self.skipped_cached + self.embedded
            pct = 100 * scanned / total_target if total_target else 0
            remaining_scan = total_target - scanned
            # ETA on the embed side, assuming the missing-fraction stays
            # roughly constant.
            seen = max(scanned, 1)
            est_remaining_embed = remaining_scan * (self.embedded / seen)
            eta_s = est_remaining_embed / rate if rate > 0 else -1
        print(
            f"prewarm: scanned={scanned:,}/{total_target:,} ({pct:.1f}%) "
            f"embedded={self.embedded:,} cached_skip={self.skipped_cached:,} "
            f"elapsed={elapsed:.0f}s rate={rate:.0f} emb/s "
            f"eta={eta_s/60:.1f}min "
            f"tei_calls={self.tei_calls} bytes={self.bytes_written/1e6:.0f}MB",
            flush=True,
        )


def _drain_below(futures: set, threshold: int, stats: PrewarmStats) -> set:
    """Block until fewer than `threshold` futures remain; raises on any failure."""
    while len(futures) >= threshold:
        done, not_done = wait(futures, return_when=FIRST_COMPLETED)
        for f in done:
            embedded, bytes_w = f.result()  # fail-fast
            with stats.lock:
                stats.embedded += embedded
                stats.bytes_written += bytes_w
                stats.tei_calls += 1
        futures = not_done
    return futures


def prewarm(
    parquet_path: Path,
    renderer: RowTextRenderer,
    redis_client_main: redis.Redis,
    redis_kwargs: dict,
    tei_url: str,
    batch_size: int,
    concurrency: int,
    chunk_size: int,
    exists_batch: int,
) -> None:
    # Set module-level state BEFORE forking workers — they inherit it via COW.
    global _g_renderer, _g_tei_url, _g_redis_kwargs
    _g_renderer = renderer
    _g_tei_url = tei_url
    _g_redis_kwargs = redis_kwargs

    pq_file = pq.ParquetFile(str(parquet_path))
    total_unique = pq_file.metadata.num_rows
    log.info("scanning %s (%s unique hashes)", parquet_path, f"{total_unique:,}")

    stats = PrewarmStats()
    last_log = time.time()
    max_inflight = concurrency * 2

    pending_rows: list[dict] = []
    futures: set = set()

    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=concurrency, mp_context=ctx) as pool:
        try:
            for chunk in pq_file.iter_batches(batch_size=chunk_size):
                rows = chunk.to_pylist()
                if not rows:
                    continue

                # Pipelined EXISTS — main thread does this so workers only
                # see rows that actually need work.
                hashes_chunk = [r["article_hash"] for r in rows]
                missing_mask = [False] * len(rows)
                for i in range(0, len(hashes_chunk), exists_batch):
                    sub = hashes_chunk[i : i + exists_batch]
                    pipe = redis_client_main.pipeline(transaction=False)
                    for h in sub:
                        pipe.exists(f"tei:v2:{h}")
                    results = pipe.execute()
                    for j, r in enumerate(results):
                        missing_mask[i + j] = (r == 0)

                missing_count = sum(missing_mask)
                with stats.lock:
                    stats.skipped_cached += len(rows) - missing_count

                for row, miss in zip(rows, missing_mask):
                    if not miss:
                        continue
                    pending_rows.append(row)
                    if len(pending_rows) >= batch_size:
                        futures = _drain_below(futures, max_inflight, stats)
                        futures.add(pool.submit(worker_process_batch, pending_rows))
                        pending_rows = []

                now = time.time()
                if now - last_log >= 15.0:
                    stats.log_progress(total_unique)
                    last_log = now

            if pending_rows:
                futures = _drain_below(futures, max_inflight, stats)
                futures.add(pool.submit(worker_process_batch, pending_rows))
                pending_rows = []

            futures = _drain_below(futures, 1, stats)
        except Exception:
            # Surface the worker exception cleanly; the executor's __exit__
            # will tear down the pool.
            log.exception("prewarm failed")
            raise

    stats.log_progress(total_unique)
    elapsed = time.time() - stats.start
    print(
        f"\nDONE: embedded={stats.embedded:,} "
        f"cached_skip={stats.skipped_cached:,} "
        f"tei_calls={stats.tei_calls} "
        f"bytes_written={stats.bytes_written/1e6:.0f}MB "
        f"elapsed={elapsed:.0f}s ({elapsed/60:.1f} min) "
        f"rate={stats.embedded/elapsed:.0f} emb/s",
        flush=True,
    )


# --- entry point ----------------------------------------------------------

def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG),
                    help="Path to cfg.data YAML (the one the model was trained with).")
    ap.add_argument("--tei-url", default=DEFAULT_TEI_URL)
    ap.add_argument("--redis-host", default="localhost")
    ap.add_argument("--redis-port", type=int, default=6379)
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Texts per TEI POST.")
    ap.add_argument("--concurrency", type=int, default=64,
                    help="Worker processes (= in-flight TEI POSTs).")
    ap.add_argument("--chunk-size", type=int, default=10_000,
                    help="Parquet rows read per iter_batches call.")
    ap.add_argument("--exists-batch", type=int, default=5_000,
                    help="Hashes per pipelined Redis EXISTS round-trip.")
    ap.add_argument("--render-inputs-parquet", default=str(DEFAULT_RENDER_INPUTS))
    ap.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB,
                    help="Glob for the gzipped Mongo offer exports.")
    ap.add_argument("--build-only", action="store_true",
                    help="Build the render-inputs parquet and exit.")
    args = ap.parse_args()

    # Phase 1 — DuckDB build
    out_path = Path(args.render_inputs_parquet)
    build_render_inputs(args.input_glob, out_path)
    if args.build_only:
        return

    # Phase 2 — load cfg.data from yaml (no torch needed), then fork the pool
    log.info("loading cfg.data from %s ...", args.data_config)
    cfg_data = OmegaConf.load(args.data_config)
    gc.collect()

    renderer = RowTextRenderer(cfg_data)
    log.info("renderer ready: clean_html=%s, max_offer_length=%s",
             getattr(cfg_data, "clean_html", "?"),
             getattr(cfg_data, "max_offer_length", "?"))

    redis_kwargs = {"host": args.redis_host, "port": args.redis_port, "db": 0}
    redis_client_main = redis.Redis(**redis_kwargs)
    redis_client_main.ping()
    log.info("connected to redis at %s:%s", args.redis_host, args.redis_port)

    log.info("TEI: %s | batch_size=%d concurrency=%d (process pool)",
             args.tei_url, args.batch_size, args.concurrency)

    prewarm(
        parquet_path=out_path,
        renderer=renderer,
        redis_client_main=redis_client_main,
        redis_kwargs=redis_kwargs,
        tei_url=args.tei_url,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        chunk_size=args.chunk_size,
        exists_batch=args.exists_batch,
    )


if __name__ == "__main__":
    main()

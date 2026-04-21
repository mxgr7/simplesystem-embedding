"""Progressive field import sweep — measure marginal cost of each field.

Runs N iterations on the 'offers' collection; each iteration is a full
cleanup → create → insert → flush → index → compact → load → measure
cycle with a cumulative field subset per PROGRESSIVE_IMPORT_PLAN.md.

Deliberately standalone (does not import from sibling scripts) so it
can be launched detached via nohup.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

COLLECTION = "offers"
DIM = 128
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
DEFAULT_DATA_DIR = Path(
    "/mnt/HC_Volume_105463954/simplesystem/data/offers_flat.parquet"
)
VOLUME_MOUNT = "/mnt/HC_Volume_105463954"
MINIO_CONTAINER = "milvus-minio"
MILVUS_CONTAINER = "milvus-standalone"
MINIO_BUCKET_PREFIX = "local/a-bucket/files"
DESCRIPTION_MAX_BYTES = 65_535
INSERT_BATCH = 50_000
INDEX_NLIST = 4096
COMPACTION_POLL_S = 30
COMPACTION_MAX_WAIT_S = 30 * 60
GC_MAX_WAIT_S = 5 * 60
GC_STABLE_GB = 2.0
RESTART_MAX_WAIT_S = 3 * 60
ABORT_VOL_FREE_GB = 30.0
ABORT_RSS_GB = 170.0
ABORT_ITER_WALL_S = 50 * 60

# Scalar ARRAY fields (get INVERTED indexes + have cumulative-cost impact)
SCALAR_ARRAY_FIELDS = {
    "vendor_ids": dict(max_capacity=32, max_length=64),
    "catalog_version_ids": dict(max_capacity=2048, max_length=64),
    "category_l1": dict(max_capacity=64, max_length=256),
    "category_l2": dict(max_capacity=64, max_length=640),
    "category_l3": dict(max_capacity=64, max_length=768),
    "category_l4": dict(max_capacity=64, max_length=1024),
    "category_l5": dict(max_capacity=64, max_length=1280),
}
# VARCHAR field sizes (no index)
TEXT_FIELD_MAX_LEN = {
    "name": 256,
    "manufacturerName": 128,
    "description": 65_535,
    "ean": 32,
    "article_number": 64,
    "manufacturerArticleNumber": 128,
    "manufacturerArticleType": 512,
}


@dataclass
class IterSpec:
    idx: int
    label: str  # short name of what was added
    fields: list[str]  # cumulative list of fields beyond {id, offer_embedding}


# Ordered small → large per plan §"Field progression"
ITERATIONS: list[IterSpec] = [
    IterSpec(0, "baseline", []),
    IterSpec(1, "n", ["n"]),
    IterSpec(2, "ean", ["n", "ean"]),
    IterSpec(3, "article_number", ["n", "ean", "article_number"]),
    IterSpec(4, "manufacturerName",
             ["n", "ean", "article_number", "manufacturerName"]),
    IterSpec(5, "manufacturerArticleNumber",
             ["n", "ean", "article_number", "manufacturerName",
              "manufacturerArticleNumber"]),
    IterSpec(6, "manufacturerArticleType",
             ["n", "ean", "article_number", "manufacturerName",
              "manufacturerArticleNumber", "manufacturerArticleType"]),
    IterSpec(7, "name",
             ["n", "ean", "article_number", "manufacturerName",
              "manufacturerArticleNumber", "manufacturerArticleType", "name"]),
    IterSpec(8, "vendor_ids",
             ["n", "ean", "article_number", "manufacturerName",
              "manufacturerArticleNumber", "manufacturerArticleType", "name",
              "vendor_ids"]),
    IterSpec(9, "category_l1..l5",
             ["n", "ean", "article_number", "manufacturerName",
              "manufacturerArticleNumber", "manufacturerArticleType", "name",
              "vendor_ids",
              "category_l1", "category_l2", "category_l3", "category_l4",
              "category_l5"]),
    IterSpec(10, "catalog_version_ids",
             ["n", "ean", "article_number", "manufacturerName",
              "manufacturerArticleNumber", "manufacturerArticleType", "name",
              "vendor_ids",
              "category_l1", "category_l2", "category_l3", "category_l4",
              "category_l5",
              "catalog_version_ids"]),
    IterSpec(11, "description",
             ["n", "ean", "article_number", "manufacturerName",
              "manufacturerArticleNumber", "manufacturerArticleType", "name",
              "vendor_ids",
              "category_l1", "category_l2", "category_l3", "category_l4",
              "category_l5",
              "catalog_version_ids", "description"]),
]


# ---------- small helpers ----------

def run(cmd: list[str], *, check: bool = True, capture: bool = True,
        timeout: int | None = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, check=check, capture_output=capture, text=True, timeout=timeout
    )


def log(msg: str, *, stream) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    print(line, file=stream, flush=True)


def volume_free_gb() -> float:
    st = shutil.disk_usage(VOLUME_MOUNT)
    return st.free / 1e9


def volume_used_gb() -> float:
    st = shutil.disk_usage(VOLUME_MOUNT)
    return st.used / 1e9


def milvus_rss_gb() -> float:
    """RSS of milvus-standalone container in GB (GiB actually, reports binary).

    Uses `docker stats --no-stream` — a sync call; ~1 s.
    """
    try:
        r = run(
            ["docker", "stats", "--no-stream", "--format",
             "{{.MemUsage}}", MILVUS_CONTAINER],
            timeout=10,
        )
        token = r.stdout.strip().split("/", 1)[0].strip()
        # token like '94.87GiB' or '905.3MiB' or '12.3KiB' — normalize to GB
        m = re.match(r"([\d.]+)\s*([KMG]i?B)", token)
        if not m:
            return -1.0
        val = float(m.group(1))
        unit = m.group(2)
        factor = {"KiB": 1 / 1024**2, "MiB": 1 / 1024, "GiB": 1.0,
                  "KB": 1e-6, "MB": 1e-3, "GB": 1.0}.get(unit, 1.0)
        # convert GiB → GB (decimal) for consistency with df output
        gib = val * factor
        return gib * (1024**3) / 1e9
    except Exception as e:
        print(f"[rss err] {e}", flush=True)
        return -1.0


def mc_du_gb(prefix: str) -> float:
    """Return logical size (GB, decimal) of `local/a-bucket/files/<prefix>`.

    If prefix is empty, returns size of `files/`.
    """
    path = MINIO_BUCKET_PREFIX + (f"/{prefix}" if prefix else "")
    try:
        r = run(
            ["docker", "exec", MINIO_CONTAINER, "mc", "du", "--depth=1", path],
            timeout=30,
        )
    except Exception as e:
        print(f"[mc du err] {e}", flush=True)
        return -1.0
    # output lines like: '149GiB\t19833 objects\ta-bucket/files'
    # or subdir lines: '26GiB\t...\ta-bucket/files/insert_log'
    total_line = None
    for ln in r.stdout.splitlines():
        parts = ln.split("\t")
        if len(parts) < 3:
            continue
        tail = parts[-1].strip()
        # We want the row whose path matches `a-bucket/files[/<prefix>]`
        want = "a-bucket/files" + (f"/{prefix}" if prefix else "")
        if tail == want:
            total_line = parts
            break
    if not total_line:
        return 0.0
    size_tok = total_line[0].strip()
    m = re.match(r"([\d.]+)\s*([KMGT]i?B)", size_tok)
    if not m:
        return -1.0
    val = float(m.group(1))
    unit = m.group(2)
    factor = {"KiB": 1 / 1024**2, "MiB": 1 / 1024, "GiB": 1.0, "TiB": 1024.0,
              "KB": 1e-6, "MB": 1e-3, "GB": 1.0, "TB": 1000.0}.get(unit, 1.0)
    gib = val * factor
    return gib * (1024**3) / 1e9


def mc_du_subdirs() -> dict[str, float]:
    """Return {subpath: gb} for all direct children of files/."""
    try:
        r = run(
            ["docker", "exec", MINIO_CONTAINER, "mc", "du", "--depth=2",
             MINIO_BUCKET_PREFIX],
            timeout=30,
        )
    except Exception as e:
        print(f"[mc du subdirs err] {e}", flush=True)
        return {}
    out: dict[str, float] = {}
    prefix = "a-bucket/files/"
    for ln in r.stdout.splitlines():
        parts = ln.split("\t")
        if len(parts) < 3:
            continue
        tail = parts[-1].strip()
        if not tail.startswith(prefix):
            continue
        sub = tail[len(prefix):]
        if not sub or "/" in sub:
            continue
        size_tok = parts[0].strip()
        m = re.match(r"([\d.]+)\s*([KMGT]i?B)", size_tok)
        if not m:
            continue
        val = float(m.group(1))
        unit = m.group(2)
        factor = {"KiB": 1 / 1024**2, "MiB": 1 / 1024, "GiB": 1.0,
                  "TiB": 1024.0}.get(unit, 1.0)
        out[sub] = val * factor * (1024**3) / 1e9
    return out


# ---------- lifecycle helpers ----------

def pkill_prior_clients(stream) -> None:
    # Intentionally narrow: don't match our own argv (which contains
    # 'progressive_import'). Only target the one-off milvus_import.py
    # clients described in the plan.
    try:
        run(["pkill", "-f", "milvus_import.py"], check=False)
    except Exception:
        pass


def drop_all_collections(stream) -> None:
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    try:
        for c in list(utility.list_collections()):
            try:
                Collection(c).release()
            except Exception:
                pass
            utility.drop_collection(c)
            log(f"dropped collection {c!r}", stream=stream)
    finally:
        try:
            connections.disconnect("default")
        except Exception:
            pass


def restart_milvus(stream) -> None:
    # Ensure any prior default alias is disconnected before restart so we
    # don't hold a stale connection pool across the restart.
    for a in ("default", "probe"):
        try:
            connections.disconnect(a)
        except Exception:
            pass
    log("restarting milvus-standalone...", stream=stream)
    run(["docker", "restart", MILVUS_CONTAINER], timeout=120)
    t0 = time.time()
    while time.time() - t0 < RESTART_MAX_WAIT_S:
        try:
            r = run(["curl", "-sf", "http://localhost:9091/healthz"],
                    check=False, timeout=5)
            if r.returncode == 0:
                break
        except Exception:
            pass
        time.sleep(3)
    else:
        raise RuntimeError("milvus did not become healthy in time")
    # gRPC accept can lag /healthz. Probe with a fresh alias and explicit
    # `using=` — utility.list_collections() defaults to 'default', which
    # would not be bound.
    probe_deadline = time.time() + RESTART_MAX_WAIT_S
    last_err: Exception | None = None
    while time.time() < probe_deadline:
        try:
            connections.connect(alias="probe",
                                host=MILVUS_HOST, port=MILVUS_PORT)
            utility.list_collections(using="probe")
            connections.disconnect("probe")
            log(f"milvus healthy+grpc ready after "
                f"{time.time()-t0:.1f}s", stream=stream)
            return
        except Exception as e:
            last_err = e
            try:
                connections.disconnect("probe")
            except Exception:
                pass
            time.sleep(3)
    raise RuntimeError(f"milvus gRPC not ready: {last_err!r}")


def wait_minio_gc(stream) -> float:
    """Poll MinIO logical size until stable or < GC_STABLE_GB.

    Returns final size in GB.
    """
    t0 = time.time()
    last = None
    stable_count = 0
    while time.time() - t0 < GC_MAX_WAIT_S:
        size = mc_du_gb("")
        log(f"minio gc wait: {size:.2f} GB", stream=stream)
        if size < GC_STABLE_GB:
            return size
        if last is not None and abs(size - last) < 0.1:
            stable_count += 1
            if stable_count >= 2:
                return size
        else:
            stable_count = 0
        last = size
        time.sleep(15)
    return mc_du_gb("")


# ---------- schema / insert ----------

def build_schema(iter_fields: list[str]) -> CollectionSchema:
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR,
                    max_length=64, is_primary=True),
        FieldSchema(name="offer_embedding",
                    dtype=DataType.FLOAT16_VECTOR, dim=DIM),
    ]
    for f in iter_fields:
        if f == "n":
            fields.append(FieldSchema(name="n", dtype=DataType.INT64))
        elif f in TEXT_FIELD_MAX_LEN:
            fields.append(FieldSchema(name=f, dtype=DataType.VARCHAR,
                                      max_length=TEXT_FIELD_MAX_LEN[f]))
        elif f in SCALAR_ARRAY_FIELDS:
            spec = SCALAR_ARRAY_FIELDS[f]
            fields.append(FieldSchema(
                name=f, dtype=DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=spec["max_capacity"],
                max_length=spec["max_length"]))
        else:
            raise ValueError(f"unknown field: {f}")
    return CollectionSchema(fields,
                            description="Progressive import sweep")


def create_collection(iter_fields: list[str]) -> Collection:
    schema = build_schema(iter_fields)
    col = Collection(COLLECTION, schema=schema)
    return col


def truncate_description(col: pa.Array) -> pa.Array:
    byte_len = pc.binary_length(pc.cast(col, pa.binary()))
    over = pc.greater(byte_len, DESCRIPTION_MAX_BYTES)
    truncated = pc.utf8_slice_codeunits(col, 0, DESCRIPTION_MAX_BYTES // 4)
    return pc.if_else(over, truncated, col)


def insert_stream(col: Collection, bucket_path: Path,
                  iter_fields: list[str], stream) -> tuple[int, float]:
    columns_to_read = ["id", "offer_embedding"] + iter_fields
    # description (avg ~500B, cap 65 KB) blows the 64 MB gRPC message cap
    # at batch_size=50_000. Shrink when description is in the set.
    batch_size = 5_000 if "description" in iter_fields else INSERT_BATCH
    log(f"  using insert batch_size={batch_size}", stream=stream)
    pf = pq.ParquetFile(bucket_path)
    total_rows = pf.metadata.num_rows
    inserted = 0
    t0 = time.time()
    for batch in pf.iter_batches(batch_size=batch_size,
                                 columns=columns_to_read):
        ids = batch.column("id").to_pylist()
        emb_obj = batch.column("offer_embedding").to_numpy(
            zero_copy_only=False)
        emb = np.stack(emb_obj)

        payload = [ids, emb]
        for f in iter_fields:
            src = batch.column(f)
            if f == "description":
                src = truncate_description(src)
            if f == "n":
                payload.append(src.to_pylist())
            elif f in SCALAR_ARRAY_FIELDS:
                payload.append(src.to_pylist())
            else:
                payload.append(src.to_pylist())
        col.insert(payload)
        inserted += len(ids)
        elapsed = max(time.time() - t0, 1e-6)
        log(f"  insert {inserted:>10,}/{total_rows:>10,} "
            f"({inserted/total_rows*100:5.1f}%) "
            f"@ {inserted/elapsed:>9,.0f} rows/s "
            f"[{elapsed:6.1f}s]",
            stream=stream)
    elapsed = time.time() - t0
    return inserted, elapsed


def wait_index_finished(col: Collection, field: str, stream,
                        poll_s: int = 5) -> float:
    t0 = time.time()
    while True:
        prog = utility.index_building_progress(col.name, index_name=field)
        state = prog.get("state", "?")
        elapsed = time.time() - t0
        log(f"  [{field}] t={elapsed:6.1f}s state={state} "
            f"pending={prog.get('pending_index_rows', 0):,} "
            f"indexed={prog.get('indexed_rows', 0):,}/"
            f"{prog.get('total_rows', 0):,}",
            stream=stream)
        if state == "Finished":
            return elapsed
        time.sleep(poll_s)


def wait_compaction(col: Collection, fields_with_index: list[str],
                    stream) -> tuple[float, bool]:
    """Poll until all listed indexes show state=Finished AND pending==0
    for two consecutive checks. Cap at COMPACTION_MAX_WAIT_S.
    """
    t0 = time.time()
    consecutive = 0
    while time.time() - t0 < COMPACTION_MAX_WAIT_S:
        all_good = True
        per = []
        for f in fields_with_index:
            try:
                prog = utility.index_building_progress(
                    col.name, index_name=f)
            except Exception:
                all_good = False
                per.append(f"{f}=ERR")
                continue
            state = prog.get("state", "?")
            pending = prog.get("pending_index_rows", 0)
            per.append(f"{f}={state}/p{pending}")
            if state != "Finished" or pending != 0:
                all_good = False
        log(f"  compaction t={time.time()-t0:.0f}s consec={consecutive}  "
            + " ".join(per), stream=stream)
        if all_good:
            consecutive += 1
            if consecutive >= 2:
                return time.time() - t0, True
        else:
            consecutive = 0
        time.sleep(COMPACTION_POLL_S)
    return time.time() - t0, False


# ---------- one iteration ----------

CSV_FIELDS = [
    "iter", "field_added", "num_fields", "insert_wall_s", "rows_per_s",
    "flush_s", "vec_idx_s", "scalar_idx_total_s", "compaction_s",
    "load_wall_s", "baseline_rss_gb", "loaded_rss_gb", "rss_delta_gb",
    "minio_logical_gb", "insert_log_gb", "index_files_gb",
    "volume_used_gb", "volume_free_gb", "segment_count", "num_entities",
    "compaction_settled", "iter_wall_s",
]


def run_iteration(spec: IterSpec, bucket_path: Path, stream,
                  csv_writer: csv.DictWriter, csv_fh) -> dict:
    iter_t0 = time.time()
    log("=" * 60, stream=stream)
    log(f"iter {spec.idx} — adds {spec.label} "
        f"(fields beyond base: {len(spec.fields)})", stream=stream)
    log("=" * 60, stream=stream)

    # 1. Cleanup
    pkill_prior_clients(stream)
    drop_all_collections(stream)
    restart_milvus(stream)
    gc_size = wait_minio_gc(stream)
    log(f"minio after gc: {gc_size:.2f} GB", stream=stream)

    # abort check on disk
    free_gb = volume_free_gb()
    log(f"volume free after cleanup: {free_gb:.1f} GB", stream=stream)
    if free_gb < ABORT_VOL_FREE_GB:
        raise RuntimeError(f"disk free {free_gb:.1f} GB < "
                           f"{ABORT_VOL_FREE_GB} GB limit")

    # 2. Create collection
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    col = create_collection(spec.fields)
    log(f"created collection with {len(col.schema.fields)} fields",
        stream=stream)

    # 3. Insert
    rows, insert_s = insert_stream(col, bucket_path, spec.fields, stream)
    rps = rows / insert_s if insert_s > 0 else 0.0
    log(f"inserted {rows:,} rows in {insert_s:.1f}s ({rps:,.0f} rows/s)",
        stream=stream)

    # 4. Flush + indexes
    t0 = time.time()
    col.flush()
    flush_s = time.time() - t0
    log(f"flush: {flush_s:.1f}s num_entities={col.num_entities:,}",
        stream=stream)

    col.create_index(
        "offer_embedding",
        index_params={"index_type": "IVF_FLAT",
                      "metric_type": "COSINE",
                      "params": {"nlist": INDEX_NLIST}},
    )
    vec_s = wait_index_finished(col, "offer_embedding", stream)

    scalar_idx_total_s = 0.0
    scalar_fields_in_iter = [f for f in spec.fields if f in SCALAR_ARRAY_FIELDS]
    for f in scalar_fields_in_iter:
        col.create_index(
            f,
            index_params={"index_type": "INVERTED"},
            index_name=f,
        )
        scalar_idx_total_s += wait_index_finished(col, f, stream)

    # 5. Wait for compaction to settle
    compaction_fields = ["offer_embedding"] + scalar_fields_in_iter
    compaction_s, settled = wait_compaction(
        col, compaction_fields, stream)
    log(f"compaction: {compaction_s:.1f}s settled={settled}",
        stream=stream)

    # Baseline RSS before load
    baseline_rss = milvus_rss_gb()
    log(f"baseline RSS (pre-load): {baseline_rss:.2f} GB", stream=stream)

    # 6. Load
    t0 = time.time()
    col.load()
    load_s = time.time() - t0
    log(f"load: {load_s:.1f}s", stream=stream)

    # Allow delegator to settle
    time.sleep(30)
    loaded_rss = milvus_rss_gb()
    log(f"loaded RSS (post-load +30s): {loaded_rss:.2f} GB", stream=stream)

    # abort check on RSS
    if loaded_rss > ABORT_RSS_GB:
        raise RuntimeError(f"RSS {loaded_rss:.1f} GB > "
                           f"{ABORT_RSS_GB} GB limit")

    # 7. Measure storage
    minio_total = mc_du_gb("")
    minio_subs = mc_du_subdirs()
    insert_log_gb = minio_subs.get("insert_log", 0.0)
    index_files_gb = minio_subs.get("index_files", 0.0)
    vol_used = volume_used_gb()
    vol_free = volume_free_gb()

    # segment count: via utility.get_query_segment_info
    try:
        segs = utility.get_query_segment_info(col.name)
        segment_count = len(segs)
    except Exception as e:
        log(f"segment count err: {e}", stream=stream)
        segment_count = -1

    num_entities = col.num_entities

    row = {
        "iter": spec.idx,
        "field_added": spec.label,
        "num_fields": len(col.schema.fields),
        "insert_wall_s": round(insert_s, 1),
        "rows_per_s": round(rps, 1),
        "flush_s": round(flush_s, 1),
        "vec_idx_s": round(vec_s, 1),
        "scalar_idx_total_s": round(scalar_idx_total_s, 1),
        "compaction_s": round(compaction_s, 1),
        "load_wall_s": round(load_s, 1),
        "baseline_rss_gb": round(baseline_rss, 3),
        "loaded_rss_gb": round(loaded_rss, 3),
        "rss_delta_gb": round(loaded_rss - baseline_rss, 3),
        "minio_logical_gb": round(minio_total, 3),
        "insert_log_gb": round(insert_log_gb, 3),
        "index_files_gb": round(index_files_gb, 3),
        "volume_used_gb": round(vol_used, 1),
        "volume_free_gb": round(vol_free, 1),
        "segment_count": segment_count,
        "num_entities": num_entities,
        "compaction_settled": settled,
        "iter_wall_s": round(time.time() - iter_t0, 1),
    }
    csv_writer.writerow(row)
    csv_fh.flush()
    log(f"iter {spec.idx} done in {row['iter_wall_s']:.0f}s "
        f"rss={loaded_rss:.1f}GB minio={minio_total:.1f}GB "
        f"free={vol_free:.0f}GB", stream=stream)

    # 8. Teardown handled by next iter's cleanup step.
    try:
        connections.disconnect("default")
    except Exception:
        pass

    # abort check on iter wall time
    iter_wall = time.time() - iter_t0
    if iter_wall > ABORT_ITER_WALL_S:
        raise RuntimeError(f"iter wall {iter_wall:.0f}s > "
                           f"{ABORT_ITER_WALL_S}s limit")

    return row


# ---------- main ----------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--bucket", default="bucket=00.parquet")
    p.add_argument("--iters", default="0-11",
                   help="Range like '0-11' or '3-5'")
    p.add_argument("--csv-out", default="")
    p.add_argument("--log-out", default="")
    args = p.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = Path(args.csv_out or f"/tmp/progressive_import_{ts}.csv")
    log_path = Path(args.log_out or f"/tmp/progressive_import_{ts}.log")

    lo, hi = (int(x) for x in args.iters.split("-"))
    selected = [s for s in ITERATIONS if lo <= s.idx <= hi]
    if not selected:
        raise SystemExit(f"no iterations in range {args.iters}")

    bucket_path = args.data_dir / args.bucket
    if not bucket_path.exists():
        raise SystemExit(f"bucket not found: {bucket_path}")

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    log_fh = open(log_path, "a", buffering=1)
    csv_fh = open(csv_path, "a", newline="", buffering=1)
    csv_w = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
    if write_header:
        csv_w.writeheader()
        csv_fh.flush()

    log(f"sweep start: bucket={bucket_path.name} iters={args.iters}",
        stream=log_fh)
    log(f"csv={csv_path} log={log_path}", stream=log_fh)

    try:
        for spec in selected:
            try:
                run_iteration(spec, bucket_path, log_fh, csv_w, csv_fh)
            except Exception as e:
                log(f"!! iter {spec.idx} FAILED: {e!r}", stream=log_fh)
                # Dump abort-time state
                try:
                    ds = run(["df", "-h", VOLUME_MOUNT], check=False)
                    log(f"df: {ds.stdout}", stream=log_fh)
                    dc = run(["docker", "stats", "--no-stream",
                              MILVUS_CONTAINER], check=False)
                    log(f"docker stats: {dc.stdout}", stream=log_fh)
                    mu = run(["docker", "exec", MINIO_CONTAINER, "mc",
                              "du", "--depth=1", MINIO_BUCKET_PREFIX],
                             check=False)
                    log(f"mc du: {mu.stdout}", stream=log_fh)
                except Exception:
                    pass
                raise
    finally:
        log("sweep finished", stream=log_fh)
        csv_fh.close()
        log_fh.close()


if __name__ == "__main__":
    main()

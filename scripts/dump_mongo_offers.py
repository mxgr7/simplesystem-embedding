"""Per-vendor dump of the prod Mongo `offers` collection to gzipped NDJSON.

Output: <out-dir>/vendor_<vendorId>.json.gz, one offer per line. Format
matches what scripts/prewarm_v2_missing.py and scripts/build_article_hashes_lookup.py
consume (DuckDB read_json(format='newline_delimited', compression='gzip')).

Fields projected — exactly what the renderer + lookup builder need:
    vendorId, articleNumber,
    offer.offerParams.{name, manufacturerName, description, categoryPaths,
                       ean, manufacturerArticleNumber, manufacturerArticleType}

vendorId can be stored as either a BSON UUID or a string depending on
the writer; we query `$in: [UUID(v), str(v)]` to cover both, and emit
the string form so downstream DuckDB reads it as VARCHAR.

Resumable per vendor: writes to vendor_<id>.json.gz.tmp then renames on
clean exit; an existing final file is skipped.

Run (smoke):
    uv run python scripts/dump_mongo_offers.py \\
        --vendor-ids f508ac53-86b2-4b97-bd26-4789a3a40a1b,0928e639-fc5a-4138-8c29-9201e8eba09c \\
        --out-dir /data/datasets/mongo_offers_export_<YYYYMMDD>
"""
from __future__ import annotations

import argparse
import gzip
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import orjson
from dotenv import load_dotenv
from pymongo import MongoClient

# orjson serializes uuid.UUID natively (to its str representation) and
# releases the GIL on the C path — both reasons we use it here over stdlib
# json. The OPT_NON_STR_KEYS guard is unused; we keep options at 0.
_NL = b"\n"


PROJECTION = {
    "_id": 0,
    "vendorId": 1,
    "articleNumber": 1,
    "offer.offerParams.name": 1,
    "offer.offerParams.manufacturerName": 1,
    "offer.offerParams.description": 1,
    "offer.offerParams.categoryPaths": 1,
    "offer.offerParams.ean": 1,
    "offer.offerParams.manufacturerArticleNumber": 1,
    "offer.offerParams.manufacturerArticleType": 1,
}


def list_all_vendor_ids(coll):
    """Distinct vendorIds via $group (no 16 MB distinct() cap)."""
    t0 = time.time()
    print("listing distinct vendorIds (allowDiskUse) ...", flush=True)
    pipeline = [{"$group": {"_id": "$vendorId"}}, {"$sort": {"_id": 1}}]
    out = []
    for doc in coll.aggregate(pipeline, allowDiskUse=True):
        v = doc["_id"]
        if v is None:
            continue
        out.append(str(v))
    print(f"  found {len(out):,} distinct vendor(s) in {time.time()-t0:.1f}s",
          flush=True)
    return out


def dump_vendor(coll, vendor_id, out_path, batch_size=5000):
    if out_path.exists():
        print(f"  {vendor_id}: final file exists, skipping", flush=True)
        return 0
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)

    # Mongo stores vendorId as either a BSON UUID or a string; cover both.
    from uuid import UUID
    query = {"vendorId": {"$in": [UUID(vendor_id), vendor_id]}}
    t0 = time.time()
    n = 0
    last_log = t0
    with gzip.open(tmp, "wb") as f:
        cursor = coll.find(query, projection=PROJECTION).batch_size(batch_size)
        for doc in cursor:
            f.write(orjson.dumps(doc))
            f.write(_NL)
            n += 1
            now = time.time()
            if now - last_log >= 10.0:
                print(f"  {vendor_id}: {n:,} so far ({n / (now-t0):.0f}/s)",
                      flush=True)
                last_log = now
    tmp.rename(out_path)
    print(f"  {vendor_id}: DONE {n:,} offers in {time.time()-t0:.1f}s "
          f"-> {out_path.name}", flush=True)
    return n


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vendor-ids", default="",
                    help="Comma-separated vendorId UUIDs. Omit to dump every "
                         "distinct vendorId in the collection.")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory; created if missing.")
    ap.add_argument("--db", default="prod")
    ap.add_argument("--collection", default="offers")
    ap.add_argument("--uri-env", default="MONGODB_PROD_URI",
                    help="Env var holding the Mongo connection URI.")
    ap.add_argument("--concurrency", type=int, default=16,
                    help="Vendors dumped in parallel (thread pool).")
    ap.add_argument("--batch-size", type=int, default=5000,
                    help="Mongo cursor batch_size; bigger = fewer RTTs per "
                         "doc on the wire.")
    args = ap.parse_args()

    uri = os.environ.get(args.uri_env)
    if not uri:
        sys.exit(f"{args.uri_env} not set in env / .env")

    out_dir = Path(args.out_dir)
    # maxPoolSize >= --concurrency so threads don't queue on the conn pool.
    # readPreference=nearest spreads cursors across every replica-set member
    # within the latency window (default 15 ms) instead of pinning to the
    # primary — turns N-thread parallelism into actual ~N-node aggregate
    # throughput on a multi-replica cluster.
    # compressors=zstd: wire compression negotiated with Atlas — cuts the
    # bytes-on-wire by ~3-5x (verbose offer JSON compresses well). Necessary
    # because the dump plateaus at ~440 Mbit/s sustained on this VPS — a
    # network-side cap, not CPU.
    client = MongoClient(uri, uuidRepresentation="standard",
                         maxPoolSize=max(args.concurrency * 2, 32),
                         readPreference="nearest",
                         compressors="zstd")
    try:
        coll = client[args.db][args.collection]
        if args.vendor_ids:
            vendor_ids = [v.strip() for v in args.vendor_ids.split(",") if v.strip()]
        else:
            vendor_ids = list_all_vendor_ids(coll)
        if not vendor_ids:
            sys.exit("no vendorIds to dump")

        print(f"dumping {len(vendor_ids):,} vendor(s) from "
              f"{args.db}.{args.collection} -> {out_dir} "
              f"(concurrency={args.concurrency})", flush=True)

        total = 0
        done = 0
        lock = Lock()
        t_start = time.time()

        def _run(v):
            out = out_dir / f"vendor_{v}.json.gz"
            return dump_vendor(coll, v, out, batch_size=args.batch_size)

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(_run, v): v for v in vendor_ids}
            for f in as_completed(futures):
                n = f.result()  # propagates exceptions
                with lock:
                    total += n
                    done += 1
                    if done % 25 == 0 or done == len(vendor_ids):
                        el = time.time() - t_start
                        print(f"  [progress] {done:,}/{len(vendor_ids):,} "
                              f"vendors  {total:,} offers  {el:.0f}s elapsed "
                              f"({total/max(el,1e-3):,.0f} offers/s)",
                              flush=True)

        print(f"\nDONE: {total:,} offers across {len(vendor_ids):,} vendor(s)",
              flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()

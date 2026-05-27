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

Parallelism: ProcessPoolExecutor over vendors (not threads). Each worker
process opens its OWN MongoClient with its OWN connection pool — no
shared internal locks between vendors. (We tried ThreadPoolExecutor
first; per-cursor throughput degraded badly as thread count rose
because of shared PyMongo state. MP gives true isolation.)

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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from uuid import UUID

import orjson
from dotenv import load_dotenv
from pymongo import MongoClient


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


def _open_client(uri):
    """Open a fresh MongoClient. Called once per process — never share
    a client across fork (pymongo explicitly disallows it)."""
    return MongoClient(uri, uuidRepresentation="standard",
                       maxPoolSize=4, readPreference="nearest")


def list_all_vendor_ids(uri, db, coll_name):
    """Distinct vendorIds via $group (no 16 MB distinct() cap). Runs in
    parent, used to seed the work queue."""
    client = _open_client(uri)
    try:
        t0 = time.time()
        print("listing distinct vendorIds (allowDiskUse) ...", flush=True)
        pipeline = [{"$group": {"_id": "$vendorId"}}, {"$sort": {"_id": 1}}]
        out = []
        for doc in client[db][coll_name].aggregate(pipeline, allowDiskUse=True):
            v = doc["_id"]
            if v is None:
                continue
            out.append(str(v))
        print(f"  found {len(out):,} distinct vendor(s) in "
              f"{time.time()-t0:.1f}s", flush=True)
        return out
    finally:
        client.close()


def dump_vendor_worker(task):
    """Module-level worker — picklable for ProcessPoolExecutor.

    task = (vendor_id, out_path_str, uri, db, coll_name, batch_size)
    Each invocation opens its own MongoClient, so processes don't share
    pools or sockets. Returns (vendor_id, count).
    """
    vendor_id, out_path_str, uri, db, coll_name, batch_size = task
    out_path = Path(out_path_str)
    if out_path.exists():
        print(f"  {vendor_id}: final file exists, skipping", flush=True)
        return vendor_id, 0
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)

    client = _open_client(uri)
    try:
        coll = client[db][coll_name]
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
                    print(f"  {vendor_id}: {n:,} so far "
                          f"({n / (now-t0):.0f}/s)", flush=True)
                    last_log = now
        tmp.rename(out_path)
        print(f"  {vendor_id}: DONE {n:,} offers in {time.time()-t0:.1f}s "
              f"-> {out_path.name}", flush=True)
        return vendor_id, n
    finally:
        client.close()


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
    ap.add_argument("--concurrency", type=int, default=4,
                    help="Vendors dumped in parallel (process pool).")
    ap.add_argument("--batch-size", type=int, default=500,
                    help="Mongo cursor batch_size.")
    args = ap.parse_args()

    uri = os.environ.get(args.uri_env)
    if not uri:
        sys.exit(f"{args.uri_env} not set in env / .env")

    out_dir = Path(args.out_dir)

    if args.vendor_ids:
        vendor_ids = [v.strip() for v in args.vendor_ids.split(",") if v.strip()]
    else:
        vendor_ids = list_all_vendor_ids(uri, args.db, args.collection)
    if not vendor_ids:
        sys.exit("no vendorIds to dump")

    print(f"dumping {len(vendor_ids):,} vendor(s) from "
          f"{args.db}.{args.collection} -> {out_dir} "
          f"(procs={args.concurrency}, batch_size={args.batch_size})",
          flush=True)

    tasks = [
        (v, str(out_dir / f"vendor_{v}.json.gz"),
         uri, args.db, args.collection, args.batch_size)
        for v in vendor_ids
    ]

    total = 0
    done = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(dump_vendor_worker, t) for t in tasks]
        for f in as_completed(futures):
            _, n = f.result()  # propagates exceptions
            total += n
            done += 1
            if done % 25 == 0 or done == len(vendor_ids):
                el = time.time() - t_start
                print(f"  [progress] {done:,}/{len(vendor_ids):,} "
                      f"vendors  {total:,} offers  {el:.0f}s elapsed "
                      f"({total/max(el,1e-3):,.0f} offers/s)", flush=True)

    print(f"\nDONE: {total:,} offers across {len(vendor_ids):,} vendor(s)",
          flush=True)


if __name__ == "__main__":
    main()

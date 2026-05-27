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
import json
import os
import sys
import time
from pathlib import Path
from uuid import UUID

from dotenv import load_dotenv
from pymongo import MongoClient


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


def _coerce(obj):
    """Recursively coerce BSON-specific instances to JSON-safe values.
    UUIDs -> str; everything else passes through."""
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _coerce(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce(x) for x in obj]
    return obj


def dump_vendor(coll, vendor_id, out_path):
    if out_path.exists():
        print(f"  {vendor_id}: final file exists, skipping", flush=True)
        return 0
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)

    query = {"vendorId": {"$in": [UUID(vendor_id), vendor_id]}}
    t0 = time.time()
    n = 0
    last_log = t0
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        cursor = coll.find(query, projection=PROJECTION).batch_size(500)
        for doc in cursor:
            doc = _coerce(doc)
            f.write(json.dumps(doc, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
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
    ap.add_argument("--vendor-ids", required=True,
                    help="Comma-separated vendorId UUIDs.")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory; created if missing.")
    ap.add_argument("--db", default="prod")
    ap.add_argument("--collection", default="offers")
    ap.add_argument("--uri-env", default="MONGODB_PROD_URI",
                    help="Env var holding the Mongo connection URI.")
    args = ap.parse_args()

    uri = os.environ.get(args.uri_env)
    if not uri:
        sys.exit(f"{args.uri_env} not set in env / .env")

    vendor_ids = [v.strip() for v in args.vendor_ids.split(",") if v.strip()]
    if not vendor_ids:
        sys.exit("--vendor-ids parsed to an empty list")

    out_dir = Path(args.out_dir)
    print(f"dumping {len(vendor_ids)} vendor(s) from {args.db}.{args.collection} "
          f"-> {out_dir}", flush=True)

    client = MongoClient(uri, uuidRepresentation="standard")
    try:
        coll = client[args.db][args.collection]
        total = 0
        for v in vendor_ids:
            out = out_dir / f"vendor_{v}.json.gz"
            total += dump_vendor(coll, v, out)
        print(f"\nDONE: {total:,} offers across {len(vendor_ids)} vendor(s)",
              flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()

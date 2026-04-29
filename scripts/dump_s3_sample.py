"""Build a joined-records JSON fixture from local copies of S3 Atlas-snapshot
shards.

The Atlas snapshot in `s3://mongo-atlas-snapshot-for-lab/.../prod/` exports
each Mongo collection as gzipped JSON-lines shard files (`atlas-*.json.gz`).
For the F9 PR2b parity check we need a few-thousand offers worth of joined
records — same shape as `tests/fixtures/mongo_sample/sample_200.json` —
without scanning the full 158 GB offers collection.

Approach:
  1. Read all markers + customer_article_numbers shards into memory dicts
     keyed by (vendorId_b64, articleNumber). These are small (markers
     ~12 MB, cans ~33 MB compressed for the 5 shards we cache).
  2. Read N pricings shards into a similar dict — sparse, only DEDICATED
     entries land here (most offers have empty joined pricings).
  3. Stream offers shard, build one joined record per offer up to the
     target count, write to output JSON in the same wrapper shape as
     dump_mongo_sample.js.

Run once after pulling shards under `~/s3-cache/{offers,pricings,markers,customerArticleNumbers}/`
(see this PR's commit history / README for the AWS CLI commands). The
script does not touch S3 directly — operate on local files so each
invocation is offline-deterministic.

Coverage caveat: shard files are sharded by document `_id`, not by join
key. So a sampled subset of pricings/cans/markers shards covers a
random subset of (vendorId, articleNumber) tuples, not the offers shard
specifically. That is fine for the *projection* parity check — we just
need the inputs that DO match to exercise the join path; offers without
matching joined rows still exercise the embedded-pricings path.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

CACHE_ROOT = Path.home() / "s3-cache"


def _to_relaxed_ejson(obj):
    """Recursively unwrap canonical-EJSON number envelopes (`$numberInt`,
    `$numberLong`, `$numberDouble`) into their plain JSON counterparts.
    Keep `$binary` and `$oid` as-is — `_decode_uuid` already handles
    `$binary`, and the projection doesn't read `_id` at all (the `$oid`
    envelope shape is preserved for fixture-shape parity with the legacy
    `dump_mongo_sample.js` output).

    Production S3 exports use canonical EJSON; the legacy 200-row fixture
    was extracted via mongosh in relaxed mode (which strips small ints,
    leaves `$oid` envelopes intact). Converting numbers at dump time keeps
    the two fixture shapes identical so the parity test exercises
    projection logic only — not EJSON dialect handling. The production
    indexer will need its own EJSON normalisation pass; that lands with
    the real Stage 1 reader."""
    if isinstance(obj, dict):
        if "$numberInt" in obj and len(obj) == 1:
            return int(obj["$numberInt"])
        if "$numberLong" in obj and len(obj) == 1:
            return int(obj["$numberLong"])
        if "$numberDouble" in obj and len(obj) == 1:
            return float(obj["$numberDouble"])
        return {k: _to_relaxed_ejson(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_relaxed_ejson(x) for x in obj]
    return obj


def _stream_jsonl_gz(path: Path):
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                yield _to_relaxed_ejson(json.loads(line))


def _join_key(doc: dict) -> tuple[str, str]:
    """Index key: (vendorId_b64, articleNumber). vendorId is the base64
    payload of the EJSON binary envelope — comparing on the b64 string
    is bit-equal to comparing on raw UUID bytes and avoids per-row UUID
    decode."""
    return (doc["vendorId"]["$binary"]["base64"], doc["articleNumber"])


def _load_index(collection_dir: Path, label: str) -> dict[tuple[str, str], list[dict]]:
    """Read every shard under `collection_dir` into one dict keyed by
    `(vendorId_b64, articleNumber)`. Multiple rows per key (markers,
    cans) are collected into a list."""
    if not collection_dir.exists():
        sys.exit(f"missing cache dir: {collection_dir}")
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    files = sorted(collection_dir.glob("atlas-*.json.gz"))
    if not files:
        sys.exit(f"no atlas-*.json.gz files in {collection_dir}")
    n = 0
    for f in files:
        for doc in _stream_jsonl_gz(f):
            out[_join_key(doc)].append(doc)
            n += 1
    print(f"  {label:25} {len(out):>10} keys / {n:>10} rows / {len(files)} shards", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-count", type=int, default=10000, help="Number of joined records to emit.")
    p.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "tests/fixtures/mongo_sample/sample_10k.json")
    args = p.parse_args()

    print("Building join indexes from local cache…", flush=True)
    markers   = _load_index(args.cache_root / "coreArticleMarkers",      "markers")
    cans      = _load_index(args.cache_root / "customerArticleNumbers",  "customer_article_numbers")
    pricings  = _load_index(args.cache_root / "pricings",                "pricings")

    print("Streaming offers shard(s) and building joined records…", flush=True)
    offer_files = sorted((args.cache_root / "offers").glob("atlas-*.json.gz"))
    if not offer_files:
        sys.exit(f"no offers shards in {args.cache_root / 'offers'}")
    records: list[dict[str, Any]] = []

    n_with_joined: dict[str, int] = defaultdict(int)
    for f in offer_files:
        for offer in _stream_jsonl_gz(f):
            key = _join_key(offer)
            joined = {
                "offer": offer,
                "pricings":               pricings.get(key, []),
                "markers":                markers.get(key, []),
                "customerArticleNumbers": cans.get(key, []),
            }
            for k in ("pricings", "markers", "customerArticleNumbers"):
                if joined[k]:
                    n_with_joined[k] += 1
            records.append(joined)
            if len(records) >= args.target_count:
                break
        if len(records) >= args.target_count:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sample_size": len(records),
        "records": records,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False))
    print(f"\nWrote {len(records)} joined records → {args.output}")
    for k, n in n_with_joined.items():
        print(f"  {n:>5} / {len(records)} ({100 * n / len(records):.1f}%) have joined {k}")


if __name__ == "__main__":
    main()

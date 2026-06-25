#!/usr/bin/env python3
"""Catch up offers that changed in Kafka since the Mongo export.

We dumped `prod.offers` on 2026-05-27. Since then the portal kept publishing
article-change notifications to Kafka. This script collects every change since
a given point in time, dedupes it, and re-fetches the current offer documents
from Mongo so they can be fed back into the import pipeline.

Two phases, each resumable:

  Phase 1 — CONSUME
    Read every message on
        prod.portal.marketplace.facts.articles.changed
    with timestamp >= --since (the moment the export started), across all
    partitions, from `offsets_for_times(since)` up to each partition's current
    high watermark (a fixed cutoff captured at start, so live traffic arriving
    mid-run does not move the goalposts).

    The message VALUE is empty (`{}`); the identity lives entirely in the KEY,
    which the portal serializes as `ArticleId.stringValue()` (see next-gen
    .../backend/common/domain/ArticleId.java):

        FriendlyId(vendorId) + ":" + base64Url-nopadding(articleNumber)

    FriendlyId is base62 (alphabet 0-9A-Za-z), NOT base64 — decoding it as
    base64 yields a plausible-but-wrong UUID. We decode each key to
    (vendorId, articleNumber) and dedupe: a vendor mass-upload republishes the
    same article many times and we only need each article once.

    Output: <out-dir>/keys.tsv.gz  (vendorId<TAB>articleNumber, one per line)
            <out-dir>/consume_summary.json

  Phase 2 — FETCH
    For the unique (vendorId, articleNumber) set, fetch the current offer from
    `prod.offers` via the (vendorId, articleNumber) compound index, projecting
    exactly the fields scripts/dump_mongo_offers.py emits so the output drops
    straight into the existing DuckDB read_json pipeline. Keys that do not
    resolve to an offer (deletes, or articles not present in `offers`) are
    recorded and skipped.

    Work is split into tasks, NOT one-per-vendor: vendor key-counts are wildly
    skewed (a few vendors hold millions of keys, most hold a handful), so a
    naive one-task-per-vendor split collapses to a few busy workers at the tail
    while the rest idle. Instead each vendor is chopped into article-range
    sub-tasks of <= --max-keys-per-task, and tasks are dispatched largest-first
    (longest-processing-time scheduling) so the giants start at t=0 and idle
    workers can share one mega-vendor.

    Resumable per task: alongside each shard a `<shard>.meta` sidecar records
    that task's requested/found counts + its missing keys, written only after
    the shard lands. Re-running skips any task whose shard AND .meta both exist
    (no Mongo re-query) and replays its missing keys from the sidecar, so
    missing.tsv + the summary stay complete. Delete a shard+.meta pair to force
    that task to refetch.

    Output: <out-dir>/vendor_<vendorId>.json.gz          (single-task vendors)
            <out-dir>/vendor_<vendorId>.partNNN.json.gz  (split vendors)
            <out-dir>/vendor_*.json.gz.meta              (resume sidecars)
            <out-dir>/missing.tsv                  (vendorId<TAB>articleNumber)
            <out-dir>/fetch_summary.json
    Both shard naming forms match the `vendor_*.json.gz` glob the pipeline
    expects (the `.meta` sidecars do not).

Run:
    uv run python scripts/catchup_kafka_offers.py \\
        --since 2026-05-27T18:09:00Z \\
        --out-dir /data/datasets/catchup_20260529
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from base64 import urlsafe_b64decode
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import orjson
from confluent_kafka import Consumer, KafkaError, TopicPartition
from dotenv import load_dotenv
from pymongo import MongoClient


TOPIC_DEFAULT = "prod.portal.marketplace.facts.articles.changed"
_NL = b"\n"


def _log(msg: str) -> None:
    """Timestamped line to stdout, flushed — readable in a tailed log file."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# Same projection as scripts/dump_mongo_offers.py — keep them in lockstep so
# the catch-up shards are interchangeable with the full export.
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


# --------------------------------------------------------------------------- #
# ArticleId key decoding
# --------------------------------------------------------------------------- #

# com.devskiller.friendly_id encodes the 128-bit UUID as a big-endian unsigned
# integer in base62 with this alphabet, left-padded to 22 chars.
_FRIENDLY_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_FRIENDLY_IDX = {c: i for i, c in enumerate(_FRIENDLY_ALPHABET)}


def friendly_id_to_uuid(s: str) -> UUID:
    """Inverse of FriendlyId.toFriendlyId(UUID). Validated against the
    library's documented vector
    5wbwf6yUxVBcr48AMbz9cb -> c3587ec5-0976-497f-8374-61e0c2ea3da5."""
    n = 0
    for ch in s:
        n = n * 62 + _FRIENDLY_IDX[ch]
    return UUID(int=n)


def decode_article_id(key: str) -> tuple[UUID, str]:
    """key = FriendlyId(vendorId) ':' base64Url-nopad(articleNumber)."""
    friendly, sep, article_b64 = key.partition(":")
    if not sep:
        raise ValueError(f"key has no ':' separator: {key!r}")
    vendor_id = friendly_id_to_uuid(friendly)
    pad = "=" * (-len(article_b64) % 4)
    article_number = urlsafe_b64decode(article_b64 + pad).decode("utf-8")
    return vendor_id, article_number


# --------------------------------------------------------------------------- #
# Kafka config
# --------------------------------------------------------------------------- #

def build_kafka_conf(group_id: str) -> dict:
    return {
        "bootstrap.servers": os.environ["KAFKA_PROD_PORTAL_BOOTSTRAP_SERVERS"],
        "security.protocol": "SASL_SSL",
        "sasl.mechanism": "SCRAM-SHA-512",
        "sasl.username": os.environ["KAFKA_PROD_PORTAL_SASL_USERNAME"],
        "sasl.password": os.environ["KAFKA_PROD_PORTAL_SASL_PASSWORD"],
        "group.id": group_id,
        "enable.auto.commit": False,
        "enable.partition.eof": True,
        "fetch.max.bytes": 52_428_800,
        "queued.max.messages.kbytes": 1_048_576,
        "auto.offset.reset": "error",
    }


# --------------------------------------------------------------------------- #
# Phase 1 — consume
# --------------------------------------------------------------------------- #

def consume(topic: str, since_ms: int | None, group_id: str,
            poll_timeout: float, out_dir: Path,
            resume_from: Path | None = None) -> dict:
    consumer = Consumer(build_kafka_conf(group_id))

    md = consumer.list_topics(topic, timeout=30)
    if topic not in md.topics or md.topics[topic].error is not None:
        consumer.close()
        sys.exit(f"topic {topic} not found / errored: {md.topics.get(topic)}")
    parts = sorted(md.topics[topic].partitions.keys())

    # Start offset per partition: either resume from a previous run's captured
    # high-watermarks (offset-exact, gap-free chaining) or seek by timestamp.
    if resume_from is not None:
        prev = json.loads((resume_from / "consume_summary.json").read_text())
        prev_hi = {r["partition"]: r["hi"] for r in prev["per_partition"]}
        start_for = {p: prev_hi.get(p) for p in parts}
        _log(f"resume-from {resume_from}: starting each partition at the previous "
             f"run's hi offset (gap-free continuation)")
    else:
        starts = consumer.offsets_for_times(
            [TopicPartition(topic, p, since_ms) for p in parts], timeout=60)
        start_for = {tp.partition:
                     (None if (tp.offset is None or tp.offset == -1) else tp.offset)
                     for tp in starts}

    assignment: list[TopicPartition] = []
    ends: dict[int, int] = {}
    part_rows: list[dict] = []
    expected = 0
    retention_warn: list[int] = []

    for p in parts:
        lo, hi = consumer.get_watermark_offsets(
            TopicPartition(topic, p), timeout=20, cached=False)
        start = start_for.get(p)
        if start is None:
            # No start (no msg >= since, or partition missing from prev run).
            part_rows.append({"partition": p, "lo": lo, "hi": hi,
                              "start": None, "count": 0})
            continue
        if start < lo:
            # Our resume/seek point fell behind retention -> a gap exists.
            retention_warn.append(p)
            start = lo
        elif start <= lo and resume_from is None:
            # Seeking by ts landed at the earliest retained offset (informational).
            retention_warn.append(p)
        ends[p] = hi
        n = max(0, hi - start)
        if n > 0:
            assignment.append(TopicPartition(topic, p, start))
            expected += n
        part_rows.append({"partition": p, "lo": lo, "hi": hi,
                          "start": start, "count": n})

    _log(f"topic {topic}: {len(parts)} partitions, "
         f"{expected:,} messages with ts >= since across "
         f"{len(assignment)} active partition(s)")
    for r in part_rows:
        if r["count"]:
            _log(f"  p{r['partition']:>2}: start={r['start']} hi={r['hi']} "
                 f"-> {r['count']:,} msgs")
    if retention_warn:
        _log(f"  WARNING: partitions {retention_warn} start at their earliest "
             f"retained offset — verify nothing before `since` was trimmed by "
             f"retention.")

    seen: set[bytes] = set()
    total_read = 0
    bad_keys = 0
    bad_path = out_dir / "bad_keys.txt"
    bad_f = None
    t0 = time.time()
    last_log = t0

    if assignment:
        consumer.assign(assignment)
        done: set[int] = set()
        active = {tp.partition for tp in assignment}
        try:
            while len(done) < len(active):
                msg = consumer.poll(poll_timeout)
                if msg is None:
                    if total_read >= expected:
                        break
                    continue
                err = msg.error()
                if err is not None:
                    if err.code() == KafkaError._PARTITION_EOF:
                        done.add(msg.partition())
                        continue
                    raise RuntimeError(f"consume error: {err}")
                p = msg.partition()
                if msg.offset() >= ends[p]:
                    done.add(p)
                    continue
                total_read += 1
                key = msg.key()
                if key:
                    seen.add(key)        # dedupe on raw key bytes
                else:
                    bad_keys += 1
                    if bad_f is None:
                        bad_f = bad_path.open("w")
                    bad_f.write(f"offset={msg.offset()} partition={p} <null key>\n")
                if msg.offset() >= ends[p] - 1:
                    done.add(p)
                now = time.time()
                if now - last_log >= 5.0:
                    pct = 100.0 * total_read / expected if expected else 100.0
                    _log(f"  consumed {total_read:,}/{expected:,} ({pct:.1f}%)  "
                         f"{total_read/max(now-t0,1e-3):,.0f}/s  "
                         f"unique={len(seen):,}  parts done={len(done)}/{len(active)}")
                    last_log = now
        finally:
            consumer.close()
    else:
        consumer.close()

    # Decode unique keys -> (vendorId, articleNumber), write keys file.
    keys_path = out_dir / "keys.tsv.gz"
    unique_keys = 0
    decode_errors = 0
    with gzip.open(keys_path, "wt", encoding="utf-8") as f:
        for raw in seen:
            try:
                vid, art = decode_article_id(raw.decode("utf-8"))
            except Exception as e:  # noqa: BLE001 - want to keep going
                decode_errors += 1
                if bad_f is None:
                    bad_f = bad_path.open("w")
                bad_f.write(f"decode-error {e!r}: {raw!r}\n")
                continue
            # articleNumber may legitimately contain spaces but never a tab.
            f.write(f"{vid}\t{art}\n")
            unique_keys += 1
    if bad_f is not None:
        bad_f.close()

    summary = {
        "phase": "consume",
        "topic": topic,
        "since_ms": since_ms,
        "since_iso": (datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc)
                      .isoformat() if since_ms else None),
        "resume_from": str(resume_from) if resume_from else None,
        "partitions": len(parts),
        "messages_read": total_read,
        "messages_expected": expected,
        "unique_keys": unique_keys,
        "null_keys": bad_keys,
        "decode_errors": decode_errors,
        "retention_warn_partitions": retention_warn,
        "elapsed_s": round(time.time() - t0, 1),
        "per_partition": part_rows,
    }
    (out_dir / "consume_summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"phase 1 done in {summary['elapsed_s']}s: read {total_read:,} messages, "
         f"{unique_keys:,} unique (vendorId, articleNumber) -> {keys_path}")
    if decode_errors or bad_keys:
        _log(f"  {bad_keys} null keys, {decode_errors} decode errors -> {bad_path}")
    return summary


# --------------------------------------------------------------------------- #
# Phase 2 — fetch from Mongo
# --------------------------------------------------------------------------- #

def _open_client(uri: str) -> MongoClient:
    # One client per process; pymongo forbids sharing a client across fork.
    return MongoClient(uri, uuidRepresentation="standard",
                       maxPoolSize=4, readPreference="nearest")


def fetch_vendor_worker(task):
    """task = (vendor_id_str, [articleNumber, ...], out_path, uri, db, coll,
              chunk_size). The article list is one vendor's full set OR a
              disjoint sub-range of it; aggregation across a vendor's tasks is
              a plain sum since the ranges don't overlap.
              Returns (vendor_id, requested, found, [missing], skipped).

    Resumable: each task writes a sidecar `<out_path>.meta` (first line
    "requested found", then one missing articleNumber per line) only AFTER the
    shard is renamed into place. A task is "done" iff BOTH the shard and its
    .meta exist; on re-run such a task is skipped and its result is read back
    from the sidecar (so missing.tsv + the summary stay complete without
    re-querying Mongo). A shard without its .meta (crash mid-write) is redone."""
    vendor_id, articles, out_path_str, uri, db, coll_name, chunk_size = task
    out_path = Path(out_path_str)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    meta_path = Path(out_path_str + ".meta")

    if out_path.exists() and meta_path.exists():
        lines = meta_path.read_text().splitlines()
        requested, found = (int(x) for x in lines[0].split())
        return vendor_id, requested, found, lines[1:], True

    client = _open_client(uri)
    try:
        coll = client[db][coll_name]
        vid = UUID(vendor_id)
        remaining = set(articles)
        n = 0
        with gzip.open(tmp, "wb") as f:
            for i in range(0, len(articles), chunk_size):
                chunk = articles[i:i + chunk_size]
                cursor = coll.find(
                    {"vendorId": vid, "articleNumber": {"$in": chunk}},
                    projection=PROJECTION,
                )
                for doc in cursor:
                    f.write(orjson.dumps(doc))
                    f.write(_NL)
                    remaining.discard(doc["articleNumber"])
                    n += 1
        tmp.rename(out_path)
        # Write the resume sidecar last — its presence marks the task done.
        missing = sorted(remaining)
        meta_tmp = Path(out_path_str + ".meta.tmp")
        meta_tmp.write_text(f"{len(articles)} {n}\n" + "\n".join(missing)
                            + ("\n" if missing else ""))
        meta_tmp.rename(meta_path)
        return vendor_id, len(articles), n, missing, False
    finally:
        client.close()


def _build_tasks(by_vendor, out_dir, uri, db, coll, chunk_size, max_keys_per_task):
    """One task per vendor when small; otherwise chop the vendor's article list
    into <= max_keys_per_task sub-ranges (own sub-shard each). Dispatch
    largest-first so a late-starting mega-vendor can't dominate the tail."""
    tasks = []
    n_split = 0
    for vid, arts in by_vendor.items():
        if len(arts) <= max_keys_per_task:
            out = out_dir / f"vendor_{vid}.json.gz"
            tasks.append((vid, arts, str(out), uri, db, coll, chunk_size))
        else:
            n_split += 1
            for i, start in enumerate(range(0, len(arts), max_keys_per_task)):
                sub = arts[start:start + max_keys_per_task]
                out = out_dir / f"vendor_{vid}.part{i:03d}.json.gz"
                tasks.append((vid, sub, str(out), uri, db, coll, chunk_size))
    tasks.sort(key=lambda t: len(t[1]), reverse=True)  # longest-processing-time
    return tasks, n_split


def fetch(out_dir: Path, uri: str, db: str, coll: str,
          concurrency: int, chunk_size: int, max_keys_per_task: int) -> dict:
    keys_path = out_dir / "keys.tsv.gz"
    if not keys_path.exists():
        sys.exit(f"{keys_path} missing — run phase 1 (consume) first")

    by_vendor: dict[str, list[str]] = {}
    with gzip.open(keys_path, "rt", encoding="utf-8") as f:
        for line in f:
            vid, _, art = line.rstrip("\n").partition("\t")
            by_vendor.setdefault(vid, []).append(art)
    n_keys = sum(len(v) for v in by_vendor.values())

    tasks, n_split = _build_tasks(by_vendor, out_dir, uri, db, coll,
                                  chunk_size, max_keys_per_task)
    _log(f"phase 2: fetching {n_keys:,} keys across {len(by_vendor):,} "
         f"vendor(s) -> {len(tasks):,} tasks ({n_split:,} vendor(s) split, "
         f"max {max_keys_per_task:,} keys/task, procs={concurrency}, "
         f"$in chunk={chunk_size})")

    total_found = 0   # offer DOCS written (inflated by catalog-version multiplicity)
    total_req = 0     # KEYS requested
    total_missing = 0  # KEYS that resolved to zero offers
    skipped = 0       # tasks already complete from a prior run
    missing_path = out_dir / "missing.tsv"
    t0 = time.time()
    last_log = t0
    done = 0
    with missing_path.open("w") as mf, \
            ProcessPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(fetch_vendor_worker, t) for t in tasks]
        for fut in as_completed(futures):
            vid, requested, found, missing, was_skipped = fut.result()
            total_found += found
            total_req += requested
            total_missing += len(missing)
            skipped += 1 if was_skipped else 0
            for art in missing:
                mf.write(f"{vid}\t{art}\n")
            done += 1
            now = time.time()
            if done % 25 == 0 or done == len(tasks) or now - last_log >= 5.0:
                _log(f"  [{done:,}/{len(tasks):,} tasks, {skipped:,} resumed]  "
                     f"{total_found:,}/{total_req:,} offers found  "
                     f"{total_found/max(now-t0,1e-3):,.0f}/s  {now-t0:.0f}s")
                last_log = now

    summary = {
        "phase": "fetch",
        "unique_keys_requested": total_req,
        "keys_resolved": total_req - total_missing,
        "keys_missing": total_missing,        # no offer (deletes / not in `offers`)
        "offer_docs_written": total_found,    # > keys, due to catalog-version multiplicity
        "vendors": len(by_vendor),
        "tasks": len(tasks),
        "tasks_resumed": skipped,             # already-complete shards skipped this run
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "fetch_summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"phase 2 done in {summary['elapsed_s']}s: {total_found:,} offer docs for "
         f"{total_req - total_missing:,} resolved keys; {total_missing:,} keys had no "
         f"offer (deletes / not in `offers`) -> {missing_path}")
    return summary


# --------------------------------------------------------------------------- #

def parse_since(s: str) -> int:
    ts = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return int(ts.timestamp() * 1000)


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since",
                    help="UTC ISO timestamp the Mongo export started, e.g. "
                         "2026-05-27T18:09:00Z. Messages with kafka timestamp "
                         ">= this are collected. Mutually exclusive with --resume-from.")
    ap.add_argument("--resume-from",
                    help="Previous catch-up out-dir. Resume each partition at that "
                         "run's captured high-watermark (offset-exact, gap-free) "
                         "instead of seeking by timestamp. For chaining catch-ups.")
    ap.add_argument("--out-dir", required=True, help="Output dir; created if missing.")
    ap.add_argument("--topic", default=TOPIC_DEFAULT)
    ap.add_argument("--group-id", default="claude-catchup-consume",
                    help="Kafka group.id (we assign() explicitly; offsets are "
                         "never committed, so this only namespaces the client).")
    ap.add_argument("--poll-timeout", type=float, default=5.0)
    ap.add_argument("--db", default="prod")
    ap.add_argument("--collection", default="offers")
    ap.add_argument("--uri-env", default="MONGODB_PROD_URI")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="Vendors fetched in parallel (process pool).")
    ap.add_argument("--chunk-size", type=int, default=1000,
                    help="articleNumbers per Mongo $in batch.")
    ap.add_argument("--max-keys-per-task", type=int, default=5_000,
                    help="Split a vendor into sub-tasks above this many keys so "
                         "idle workers can share a mega-vendor (default 5k). Kept "
                         "small because cost is driven by DOC count, not keys: "
                         "high-multiplicity vendors (e.g. ~150 catalog-version "
                         "docs/article) turn even 100k keys into ~14M docs / one "
                         "long straggler. 5k keys caps a task at a parallelizable "
                         "size regardless of multiplicity.")
    ap.add_argument("--phase", choices=["consume", "fetch", "all"], default="all")
    ap.add_argument("--force-consume", action="store_true",
                    help="Re-run consume even if keys.tsv.gz already exists.")
    args = ap.parse_args()

    if bool(args.since) == bool(args.resume_from):
        sys.exit("provide exactly one of --since or --resume-from")
    resume_from = Path(args.resume_from) if args.resume_from else None
    if resume_from and not (resume_from / "consume_summary.json").exists():
        sys.exit(f"--resume-from {resume_from}: consume_summary.json not found")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    since_ms = parse_since(args.since) if args.since else None
    keys_path = out_dir / "keys.tsv.gz"

    _log(f"catchup start: "
         f"{'resume-from='+str(resume_from) if resume_from else 'since='+args.since}  "
         f"phase={args.phase}  out-dir={out_dir}")

    if args.phase in ("consume", "all"):
        if keys_path.exists() and not args.force_consume:
            _log(f"{keys_path} exists — skipping consume "
                 f"(use --force-consume to redo)")
        else:
            consume(args.topic, since_ms, args.group_id,
                    args.poll_timeout, out_dir, resume_from=resume_from)

    if args.phase in ("fetch", "all"):
        uri = os.environ.get(args.uri_env)
        if not uri:
            sys.exit(f"{args.uri_env} not set in env / .env")
        fetch(out_dir, uri, args.db, args.collection,
              args.concurrency, args.chunk_size, args.max_keys_per_task)

    return 0


if __name__ == "__main__":
    sys.exit(main())

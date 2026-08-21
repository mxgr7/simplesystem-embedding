"""Copy a PIT slice of the SPLADE index to another index WITHOUT re-encoding.

Why this exists: the latency A/B needs a baseline arm holding the *deployed*
`prod_soup` vectors over exactly the same documents the candidate arm encodes.
Two cheaper routes do not work:

  * `POST _reindex` with a manual `"slice"` is rejected by this cluster
    ("[slice] can only be used with [scroll] or [point-in-time] requests"), and
    `?slices=N` copies the whole index rather than one slice. So an ES-side copy
    cannot reproduce the tool's partition.
  * a plain `_reindex` would also silently drop the vectors: the index mapping
    has `_source.excludes = [embeddings.vector, spladeVector]`, so the *stored*
    source carries neither. They come back only under an explicit projection
    (`_source: {"includes": ["*"]}`), which is exactly what
    reindex_articles_with_splade.DEFAULT_PROJECTION asks for.

So this reuses that tool's own plumbing -- same PIT, same `slice {id, max}`,
same `_shard_doc` sort, same projection, same bulk sender -- and replaces the
encode step with a pass-through. The copied slice is therefore the *same doc
set* the tool writes for `--slices S --slice-ids I`, by construction rather than
by hope, and the candidate arm's later slices stay its exact complement.

Vectors are copied as ES returns them, i.e. after the index's own quantisation,
so the postings this produces are bit-identical to the ones being served.

    python3 copy_slice_verbatim.py --es http://localhost:9200 \
        --src prod-article-index-v1-semantic-splade-20260726 \
        --dst article-lat-s12-ps-v1 \
        --slices 12 --slice-ids 0 --state copy-ps-s0.json
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reindex_articles_with_splade as R  # noqa: E402

LOG_PATH = os.environ.get("COPY_LOG", "")


def log(message):
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a") as handle:
            handle.write(line + "\n")


def build_actions(hits, dst, stats):
    """Verbatim pass-through: keep _source exactly as the source returned it.

    The offer-share assertion is the copy's own correctness gate. An article
    with offers must carry a vector -- that is what a stripped `_source` would
    destroy, and a vector-less index would still answer queries (just wrongly
    and very fast), so a doc-count check alone would not catch it.
    """
    actions = []
    for hit in hits:
        R.validate_source_hit(hit)
        source = hit["_source"]
        vector = source.get("spladeVector")
        if source.get("offers"):
            if not vector:
                raise ValueError(
                    f"article {hit.get('_id')} has offers but no spladeVector in the "
                    f"projection -- the copy would drop the model's output"
                )
            stats["offer_docs"] += 1
            stats["nnz_sum"] += len(vector)
            stats["vec_docs"] += 1
        else:
            if vector:
                stats["nnz_sum"] += len(vector)
                stats["vec_docs"] += 1
            stats["no_offer_docs"] += 1
        operation = {"_index": dst, "_id": hit["_id"]}
        routing = R.hit_routing(hit)
        if routing is not None:
            operation["routing"] = routing
        actions.append(({"index": operation}, source))
    return actions


async def copy_slice(slice_id, args, src, dst, state, lock):
    item = state["slices"][str(slice_id)]
    if item.get("completed"):
        log(f"slice {slice_id}: already complete ({item['stats']['docs']:,} docs)")
        return
    search_after = item.get("search_after")
    while True:
        pit_id = item["pit_id"]
        hits, retries, latest = await R.search_page(
            src, pit_id, args.pit_keep_alive, args.projection, args.page_size,
            slice_id, args.slices, search_after,
        )
        if latest != pit_id:
            item["pit_id"] = latest
        if not hits:
            item["completed"] = True
            await R.close_pit(src, item.get("pit_id"))
            item["pit_closed"] = True
            async with lock:
                R.atomic_write(args.state, state)
            log(f"slice {slice_id}: done, {item['stats']['docs']:,} docs")
            return
        if args.limit:
            room = args.limit - item["stats"]["docs"]
            if room <= 0:
                item["completed"] = True
                item["limited"] = True
                await R.close_pit(src, item.get("pit_id"))
                async with lock:
                    R.atomic_write(args.state, state)
                return
            hits = hits[:room]
        page = {"docs": len(hits), "offer_docs": 0, "no_offer_docs": 0,
                "vec_docs": 0, "nnz_sum": 0, "indexed": 0, "retries": retries,
                "bytes": 0}
        actions = build_actions(hits, args.dst, page)
        for chunk in R.chunk_actions(actions, args.bulk_bytes):
            indexed, chunk_retries, sent = await R.send_bulk(dst, chunk)
            page["indexed"] += indexed
            page["retries"] += chunk_retries
            page["bytes"] += sent
        if page["indexed"] != len(hits):
            raise RuntimeError("page was not fully durable")
        search_after = hits[-1]["sort"]
        async with lock:
            for key, value in page.items():
                item["stats"][key] = item["stats"].get(key, 0) + value
            item["search_after"] = search_after
            item["updated_at"] = int(time.time())
            R.atomic_write(args.state, state)


async def progress(state, started, expected):
    """Progress with an ETA -- a multi-hour copy that prints nothing is a copy
    nobody can tell apart from a hung one."""
    while True:
        await asyncio.sleep(30)
        docs = sum(item["stats"]["docs"] for item in state["slices"].values())
        elapsed = time.time() - started
        rate = docs / elapsed if elapsed else 0
        vec = sum(item["stats"].get("vec_docs", 0) for item in state["slices"].values())
        nnz = sum(item["stats"].get("nnz_sum", 0) for item in state["slices"].values())
        eta = (expected - docs) / rate if rate and expected > docs else 0
        log(f"{docs:,}/{expected:,} docs ({100 * docs / max(1, expected):.1f}%) "
            f"{rate:,.0f}/s eta {eta / 60:.1f}min "
            f"mean_nnz {nnz / max(1, vec):.1f}")


async def main_async(args):
    src = R.make_es_client(args.src_url)
    dst = R.make_es_client(args.dst_url)
    try:
        identity, _, mapping = await R.source_snapshot(src, args.src)
        dst_identity = await R.index_identity(dst, args.dst)
        R.ensure_separate_source(identity["concrete_index"], dst_identity["concrete_index"])
        owned = R.parse_slice_ids(args.slice_ids, args.slices)
        expected = -(-identity["docs_count"] // args.slices) * len(owned)
        log(f"src {identity['concrete_index']} {identity['docs_count']:,} docs "
            f"-> dst {args.dst}; slices {owned} of {args.slices} "
            f"(~{expected:,} docs expected)")

        state = R.load_checkpoint(args.state)
        if state:
            if state["source_identity"]["uuid"] != identity["uuid"]:
                raise ValueError("pinned source identity changed")
            log(f"resuming from {args.state}")
        else:
            pits = {}
            for slice_id in owned:
                pit_id, _ = await R.open_pit(src, args.src, args.pit_keep_alive)
                pits[slice_id] = pit_id
            state = {
                "mode": "copy-verbatim",
                "source_identity": identity,
                "destination_identity": dst_identity,
                "slice_count": args.slices,
                "owned_slice_ids": owned,
                "slices": {str(slice_id): {
                    "pit_id": pits[slice_id],
                    "search_after": None,
                    "completed": False,
                    "stats": {"docs": 0, "offer_docs": 0, "no_offer_docs": 0,
                              "vec_docs": 0, "nnz_sum": 0, "indexed": 0,
                              "retries": 0, "bytes": 0},
                } for slice_id in owned},
            }
            R.atomic_write(args.state, state)

        lock = asyncio.Lock()
        started = time.time()
        watcher = asyncio.create_task(progress(state, started, expected))
        try:
            await asyncio.gather(*(
                copy_slice(slice_id, args, src, dst, state, lock)
                for slice_id in owned
            ))
        finally:
            watcher.cancel()

        total = {}
        for item in state["slices"].values():
            for key, value in item["stats"].items():
                total[key] = total.get(key, 0) + value
        elapsed = time.time() - started
        total["mean_doc_nnz"] = round(
            total["nnz_sum"] / max(1, total["vec_docs"]), 3)
        total["offer_share"] = round(total["offer_docs"] / max(1, total["docs"]), 4)
        total["elapsed_seconds"] = round(elapsed, 1)
        total["docs_per_second"] = round(total["docs"] / elapsed, 1) if elapsed else 0
        log("copy complete: " + json.dumps(total, sort_keys=True))
        print(json.dumps({"copied": args.dst, "totals": total}, indent=2, sort_keys=True))
    finally:
        await src.aclose()
        await dst.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    R.add_common(parser)
    parser.add_argument("--slices", type=int, default=12)
    parser.add_argument("--slice-ids", default="0")
    parser.add_argument("--state", default="copy-slice-state.json")
    parser.add_argument("--page-size", type=int, default=250)
    parser.add_argument("--bulk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--pit-keep-alive", default="60m")
    parser.add_argument("--limit", type=int, default=0,
                        help="cap docs per slice (smoke); 0 = whole slice")
    args = parser.parse_args()
    args.src_url = args.src_es or args.es
    args.dst_url = args.dst_es or args.es
    args.projection = json.loads(R.DEFAULT_PROJECTION)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

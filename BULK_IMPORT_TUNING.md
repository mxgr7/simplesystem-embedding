# Elasticsearch Bulk Import Tuning

Notes on the fastest way to load a large dataset into Elasticsearch, and how to decouple the raw import from the index-finalization step.

## TL;DR

- Use the `_bulk` REST API. It is the fastest client-to-cluster path; the old native transport client is gone in ES 8.x.
- The HTTP layer is almost never the bottleneck. Tuning is about index settings, batch shape, and client concurrency.
- The biggest single wins: `refresh_interval: -1` + `number_of_replicas: 0` during load, plus running multiple parallel bulk workers.

## Pre-import index settings

Set these on the target index before starting the load:

```json
PUT /my-index/_settings
{
  "index": {
    "number_of_replicas": 0,
    "refresh_interval": -1,
    "translog.durability": "async",
    "translog.flush_threshold_size": "2gb"
  }
}
```

- `number_of_replicas: 0` — replicate after the load, not during.
- `refresh_interval: -1` — disables periodic refresh; docs land on disk but are not yet searchable.
- `translog.durability: async` — fsyncs in the background instead of per request.
- `translog.flush_threshold_size` — fewer, larger flushes reduce overhead.

## Bulk request tuning

- Target **5–15 MB per `_bulk` request body**. Size matters more than doc count.
- Run **many parallel bulk workers**. Start with one per CPU on the client; increase until ES CPU saturates or you see HTTP 429 / rejected executions on the bulk thread pool.
- Prefer **ES-generated document IDs** when possible — upserts against existing IDs are slower than appends.
- Compress the request body (`Content-Encoding: gzip`) if the network is the bottleneck.

## Post-import finalize

Once the data is in, run the "indexing" phase as a separate step:

```json
PUT /my-index/_settings
{
  "index": {
    "refresh_interval": "1s",
    "number_of_replicas": 1
  }
}

POST /my-index/_refresh
POST /my-index/_forcemerge?max_num_segments=1
```

- `_refresh` makes the imported docs searchable.
- `_forcemerge` to a single segment is worthwhile for read-mostly indices; skip it for indices that will keep receiving writes.
- Restoring replicas triggers shard copies — expect a temporary network/IO spike.

## Separating import from indexing

"Import" (get bytes onto disk) and "indexing" (make them searchable / optimize structure) can be decoupled in several ways:

1. **Same cluster, two phases** — the pattern above. Load with refresh disabled, then refresh + force-merge later. Simplest approach.
2. **Staging index + reindex** — bulk into a minimal-mapping staging index, then `_reindex` into the real index with proper analyzers/mappings. Lets you redo the indexing step without re-pulling source data.
3. **Snapshot / restore** — build the index on a loader cluster, snapshot to shared storage, restore into prod. Prod never sees the write load.
4. **Offline Lucene segment build** — rare and version-fragile, but skips ES entirely until segment files are dropped in.

For most workloads, option 1 is what "separate import from indexing" actually means in practice.

## What is *not* faster than `_bulk`

All official language clients (`elasticsearch-py` `parallel_bulk`, Java API client, Go, etc.) are wrappers around HTTP `_bulk`. They help with batching and concurrency, not with the wire protocol.

The only paths faster than `_bulk` sidestep client ingestion altogether: server-side `_reindex`, snapshot/restore, or offline Lucene builds.

## Quick checklist

- [ ] `number_of_replicas: 0`
- [ ] `refresh_interval: -1`
- [ ] `translog.durability: async`
- [ ] Bulk bodies sized 5–15 MB
- [ ] N parallel bulk workers, tuned to ES CPU saturation
- [ ] ES-assigned doc IDs where possible
- [ ] After load: restore refresh + replicas, `_refresh`, optional `_forcemerge`

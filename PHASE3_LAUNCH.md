# Phase 3 launch runbook

Cross-cluster: **read prod (`prod-article-index-v1`), write nonprod (target = `prod-article-index-v1-semantic-<TS>`).** Both URLs come from `.env`.

## Pre-flight (one-time verifications)

```bash
set -a; source .env; set +a

# 1. prod ES reachable + PIT works
curl -s "$ELASTIC_PROD_URL/prod-article-index-v1/_count" | jq .
curl -s -XPOST "$ELASTIC_PROD_URL/prod-article-index-v1/_pit?keep_alive=1m" | jq .id

# 2. nonprod ES reachable + writable + no existing target
curl -s "$ELASTIC_URL/_cluster/health" | jq .cluster_name

# 3. KVRocks vector cache populated (on this box) — Phase 2 should be DONE
redis-cli -p 6666 DBSIZE

# 4. article_hashes_v2 lookup present on whichever box we run from
duckdb -c "SELECT COUNT(*) FROM read_parquet('/data/datasets/mongo_offers_export_20260527/article_hashes_v2/**/*.parquet')"
# expected: 122,254,997
```

## 3.1 Create the target index on nonprod

```bash
DST="prod-article-index-v1-semantic-$(date -u +%Y%m%d%H%M%S)"
echo "DST=$DST"

uv run python scripts/setup_es_v5.py \
    --dst "$DST"
# defaults: --settings-src=prod-article-index-v1, --settings-src-es=$ELASTIC_PROD_URL,
#          --dst-es=$ELASTIC_URL, --mapping=target_mapping.json
```

Verify the create succeeded:

```bash
curl -s "$ELASTIC_URL/$DST/_count" | jq .          # docs: 0
curl -s "$ELASTIC_URL/$DST/_settings" | jq '..|.refresh_interval? // empty'  # "-1"
```

## 3.2 Stream-enrich from prod → write to nonprod

> Where to run: **milvus is preferable** (32 cores, 247 GB RAM → 32-way PIT-slice parallelism). The article_hashes_v2 parquet is already synced to `/data/datasets/mongo_offers_export_20260527/article_hashes_v2/` on both boxes. KVRocks tunnel from milvus to `localhost:6667` should already be up (this box's `ssh -fNR 6667:localhost:6666 milvus` PID is still alive — check `pgrep -af "ssh.*6667"`).
>
> If running from THIS box: use `--redis redis://localhost:6666/0` (no tunnel). 16 workers (CPU constraint).

### Launch on milvus (recommended)

```bash
TS=$(date -u +%Y%m%d_%H%M%S)
ssh milvus "cd ~/simplesystem-embedding && \
  nohup .venv/bin/python3 -u scripts/index_embeddings_to_es_v5_mp.py \
    --dst $DST \
    --redis redis://localhost:6667/0 \
    --procs 32 \
    > logs/phase3_import_${TS}.log 2>&1 & echo PID=\$!"
# tail it:
ssh milvus "tail -f ~/simplesystem-embedding/logs/phase3_import_${TS}.log"
```

### Launch on THIS box (fallback)

```bash
TS=$(date -u +%Y%m%d_%H%M%S)
nohup uv run python -u scripts/index_embeddings_to_es_v5_mp.py \
    --dst "$DST" \
    --redis redis://localhost:6666/0 \
    --procs 16 \
    > logs/phase3_import_${TS}.log 2>&1 & echo "PID=$!"

tail -f logs/phase3_import_${TS}.log
```

What the importer does (per worker):

- Opens PIT on prod cluster
- 32-way (or 16-way) sliced scan reads articles in parallel
- For each doc: drops stale embedding fields, joins (vendor, articleNumber) → article_hashes via parquet, MGETs vectors from KVRocks, builds `embeddings[]` + denormalised `catalogVersionIds` / `priceKeys`, bulk-indexes to `$DST` on nonprod
- Backs off on 429 / transient transport errors. `_id` preserved → retries idempotent.

Watch for:
- `redis misses` — should be ~0 (Phase 2 already populated everything). Non-zero = cache miss for some hash → the embedding for that hash is silently absent. Investigate before continuing.
- `orphan` — vendor in ES doc but not in the article_hashes_v2 parquet. Skipped. A handful are expected (timing of Mongo dump vs ES PIT); large counts mean the dump is stale.

## 3.3 Post-import finalize (on nonprod)

```bash
curl -s -XPOST "$ELASTIC_URL/$DST/_refresh"
curl -s -XPOST "$ELASTIC_URL/$DST/_forcemerge?max_num_segments=1"
curl -s -XPUT  "$ELASTIC_URL/$DST/_settings" \
    -H 'Content-Type: application/json' \
    -d '{"index":{"number_of_replicas":1,"refresh_interval":"1s"}}'

# Sanity-check final count matches what importer reported:
curl -s "$ELASTIC_URL/$DST/_count" | jq .
```

## 3.4 Pre-cutover warmup (load-bearing — see FT_ELASTIC_IMPORT.md §2.1.6)

Cold HNSW first kNN query measured at ~196 s. **Do not flip the alias until p99 is in §2.1.4 territory.** Run a representative query pass:

```bash
# adapt to a known warm-pass tool / sample queries
uv run python scripts/bench_profiles_latency.py \
    --es "$ELASTIC_URL" \
    --index "$DST" \
    --queries reports/hnsw_eval/queries.parquet \
    --concurrency 4
```

Continue running query traffic until p99 ~ tens of ms.

## 3.5 Alias swap

```bash
# Determine the current serving alias on nonprod (whatever the consumer reads).
# Confirm with the consumer team before flipping.
SERVING_ALIAS="<serving-alias>"
CURRENT_BACKING=$(curl -s "$ELASTIC_URL/_alias/$SERVING_ALIAS" | jq -r 'keys[0]')

curl -s -XPOST "$ELASTIC_URL/_aliases" \
    -H 'Content-Type: application/json' \
    -d "{
      \"actions\": [
        {\"remove\": {\"index\": \"$CURRENT_BACKING\", \"alias\": \"$SERVING_ALIAS\"}},
        {\"add\":    {\"index\": \"$DST\",             \"alias\": \"$SERVING_ALIAS\"}}
      ]
    }" | jq .
```

## Reference numbers from the prior conversation

- Mongo offers exported: **455,318,596** in 753 vendor files (11 GB compressed)
- (vendor, articleNumber) pairs: **122,254,997** in `article_hashes_v2/`
- Total hash references: **181,148,463**
- KVRocks entries at Phase-2 start: 159,228,299 (carried over from prior pipeline machine; verified consistent vs current TEI by spot-check)
- Prod ES source: `prod-article-index-v1` alias → concrete `prod-article-index-v1-20260423222729`, 32 shards

# Article-search replacement — status

Snapshot of where the legacy-search → ftsearch + ACL replacement
project stands. Maps each spec packet to the code that implements it
and surfaces the deferred work.

The project specification is `issues/article-search-replacement-spec.md`;
the per-packet specs live under `issues/article-search-replacement-{ftsearch,indexer,acl}-NN-*.md`.
Each packet's own file carries a `## Status` section with the
authoritative `commit_sha` references — this doc is the bird's-eye
overview.

## Components

```
                                                    (this repo)
   next-gen ─POST /article-features/search──► ./acl/  ─POST /{col}/_search──► ./search-api/  ─pymilvus──► Milvus
                                              (FastAPI,                       (FastAPI,
                                               port 8081)                      port 8001)
                                                                                  ▲
                                                                                  │ TEI /embed
                                                                              ./playground-app/compose
                                                                              services: tei, redis,
                                                                              search-api, acl, plus
                                                                              a side-car milvus stack
                                                                              (milvus-standalone +
                                                                              milvus-minio + milvus-etcd)
```

The bulk indexer pipeline (`./indexer/`, driven by `scripts/indexer_bulk.py`)
populates the Milvus collections from a Mongo Atlas snapshot in S3.

## Done

| Packet | What it covers | Where |
| --- | --- | --- |
| F1 | `articles_v{N}` + `offers_v{N}` Milvus schemas | `scripts/create_{articles,offers}_collection.py` |
| F2 | ftsearch HTTP contract + Pydantic DTOs + OpenAPI | `search-api/{models.py,openapi.yaml}` |
| F3 | Scalar filtering + price-resolution post-pass | `search-api/{filters.py,prices.py}` |
| F4 | searchMode + sort + accurate hitCount + relevance pool bound | `search-api/{routing.py,sorting.py}` |
| F5 | Aggregations (summaries) over the full filtered hit set | `search-api/aggregations.py` |
| F6 | German identifier handling (absorbed into F9 sparse_codes) | `scripts/create_articles_collection.py` (BM25 analyzer) |
| F7 | Bounded consistency + retries + timeouts + tracing + RED metrics | `search-api/{milvus_helpers.py,embed_client.py,metrics.py,tracing.py}` |
| F8 | Per-offer price-scope envelope columns | `scripts/create_offers_collection.py` + projection in `indexer/` |
| F9 (PR1+2+3) | Article-level dedup topology — split `articles_v{N}` + `offers_v{N+1}`, dispatcher routing | `search-api/routing.py` (Path A / Path B), `indexer/projection.py` (hash + two-stream emit) |
| F9 PR2b | DuckDB-native projection + bulk indexer (ingest) | `indexer/{duckdb_projection.py,bulk.py,tei_cache.py,bulk_insert.py}`, `scripts/indexer_bulk.py` |
| I1 | Bulk projection + test-loader (Phase A); Phase B absorbed by F9 PR2b | `indexer/{projection.py,test_loader.py}` |
| I3 (partial) | Paired alias-swing CLI + post-run flush | `scripts/swing_aliases.py` |
| A1 | ACL skeleton + narrowed legacy OpenAPI | `acl/{app.py,openapi.yaml,Dockerfile,README.md}` |
| A2 | Legacy → ftsearch request mapper + httpx client | `acl/{models.py,mapping/request.py,clients/ftsearch.py}` |
| A3 | ftsearch → legacy response envelope mapper | `acl/mapping/response.py` |
| A4 | Legacy error envelope on every failure path + dropped-enum rejection | `acl/app.py` exception handlers, `acl/models.py` cross-field validators |
| A5 | ACL retries + timeouts + tracing baggage + RED metrics | `acl/{clients/ftsearch.py,tracing.py,metrics.py}` |
| A6 (partial) | End-to-end happy-path acceptance (ACL → ftsearch → Milvus) | `tests/test_acl_acceptance_e2e.py` |

## Test surface

| Suite | Count | Real infra needed |
| --- | --- | --- |
| `test_duckdb_projection_parity.py` | 6 | DuckDB only |
| `test_duckdb_aggregate_parity.py` | 12 | DuckDB only |
| `test_duckdb_raw_join.py` | 3 | Local `~/s3-cache/` shards |
| `test_tei_cache.py` | 9 | None (mocked) |
| `test_bulk_insert.py` | 19 | None (mocked) |
| `test_indexer_bulk_smoke.py` | 2 | Local `~/s3-cache/` shards (real DuckDB; mocked Milvus/TEI/Redis) |
| `test_milvus_helpers.py` | 17 | None |
| `test_embed_client.py` | 6 | None (httpx.MockTransport) |
| `test_metrics_module.py` | 10 | None |
| `test_tracing.py` | 17 | None |
| `test_filters.py` | 66 | None |
| `test_routing.py` | 33 | None (mocked Milvus) |
| `test_search_dedup_integration.py` | 27 | Live Milvus + populated `articles_v4` + `offers_v5` |
| `test_articles_collection_schema.py` | 17 | Live Milvus + `articles_v4` |
| `test_offers_collection_schema.py` | 22 | Live Milvus + `offers_v5` + `offers_v_alias` |
| `test_swing_aliases.py` | 5 | Live Milvus |
| `test_acl_skeleton.py` | 8 | None |
| `test_acl_request_mapper.py` | 13 | None |
| `test_acl_response_mapper.py` | 9 | None |
| `test_acl_error_contract.py` | 16 | None |
| `test_acl_resilience.py` | 8 | None |
| `test_acl_integration.py` | 7 | None |
| `test_acl_acceptance_e2e.py` | 7 | Live Milvus |

## Deferred

Major work blocked or scoped out:

  - **I2** — Kafka incremental indexer. **Design pass landed** in
    `issues/article-search-replacement-indexer-02-incremental.md`
    (process architecture, async coalescer with budget math,
    crash semantics, orphan-article GC, observability metrics,
    deployment shape). Implementation blocked on Kafka infra +
    topic provisioning at the target environment.
  - **I3 fully** — **Design pass landed** in
    `issues/article-search-replacement-indexer-03-zero-downtime.md`
    (state machine, second-consumer dual-write model, validation
    gates, recovery from each step, rollback paths, CLI shape,
    runbook outline). Implementation depends on I2 — see I2's design
    pass for the second-consumer mechanics.
  - **A6 expansion** — per-filter / per-sort / per-aggregation
    acceptance tests + PostHog captured-traffic smoke. The existing
    integration suite already covers each filter against ftsearch
    directly; ACL pass-through expansion is mechanical.
  - **F7 caching** — optional category / eClass hierarchy cache.
    Not on critical path; revisit when profiling shows it.

Smaller follow-ups worth picking up before a production run:

  - **Bulk-insert acceptance at full catalog scale**. The current
    smoke runs against shard 0.0 (35K offers); a full-catalog run
    needs a GPU TEI box (CPU TEI is the bottleneck — ~30 days at
    full scale vs. ~12 h on a single GPU per the F9 cost model).
  - **Legacy article-search collection drop**. The pre-F9 `offers`
    + `offers_codes` collections still exist on prod Milvus; once
    F9 PR4 cutover soaks, drop them.
  - **Documentation** — all major runbooks landed:
    `indexer/RUNBOOK.md` for the bulk pipeline,
    `scripts/SWING_ALIASES_RUNBOOK.md` for paired alias swings,
    `scripts/MILVUS_ALIAS_WORKFLOW.md` for the conceptual model,
    top-level `README.md` pointer at this status doc.

## Operating the pipeline

Local docker-compose stack:

```sh
cd playground-app && docker compose up
```

Brings up: TEI (port 8080), Redis (6379), search-api (8001), ACL (8081),
Milvus (19530) + MinIO (9000) + etcd. Pre-existing `milvus-*` containers
are external to the compose file but expected on the same host.

End-to-end indexer run (local-cache smoke):

```sh
uv run python scripts/create_articles_collection.py --version 4 --no-alias
uv run python scripts/create_offers_collection.py   --version 5 --no-alias
uv run python scripts/indexer_bulk.py \
    --local-cache ~/s3-cache \
    --offers-glob 'atlas-fkxrb3-shard-0.0.json.gz' \
    --milvus-uri http://localhost:19530 \
    --articles-collection articles_v4 --offers-collection offers_v5 \
    --tei-url http://localhost:8080 --redis-url redis://localhost:6379/0 \
    --sink-mode bulk_insert \
    --bulk-insert-checkpoint /var/run/f9_indexer_checkpoint.json
uv run python scripts/swing_aliases.py \
    --articles-target articles_v4 --offers-target offers_v5
```

Search through the ACL:

```sh
curl -X POST http://localhost:8081/article-features/search \
    -H 'Content-Type: application/json' \
    -d '{
      "searchMode": "BOTH", "searchArticlesBy": "STANDARD",
      "selectedArticleSources": {"closedCatalogVersionIds": []},
      "queryString": "schraube",
      "maxDeliveryTime": 0, "coreSortimentOnly": false,
      "closedMarketplaceOnly": false, "currency": "EUR", "explain": false
    }'
```

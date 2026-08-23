# SPLADE service

Cached serving path for the `prod_soup.ckpt` learned-sparse model.

`POST /embed` accepts one string or a list of strings. Each string contains 14
NUL-separated fields in this order:

```
name, manufacturer_name, description, category_paths, ean, article_number,
manufacturer_article_number, manufacturer_article_type, customer_artnos_text,
vendor_text, category_leaf_text, s2class_text, keywords_text, features_text
```

The description field is deliberately blanked. Source values are normalised and
German-folded before the checkpoint's kitchen-sink template is rendered. The
response is a list of top-256 `{token_id: weight}` maps.

`POST /embed-query` accepts a raw query string or list of strings in the same
`{"inputs": ...}` envelope. It normalises and applies `fold_de` inside the
service, then returns the complete positive float32 query vector. Query callers
must send the raw user text, including umlauts; they must not strip diacritics
before calling the service. Query vectors bypass the document cache.

KVRocks values use compact `(uint16 token_id, fp16 weight)` storage under the
model-scoped `splade:prod-soup-folde-top256-v1:` prefix. Cache failures fall back
to inference.

The frontend can route misses across multiple compatible backends. Add, reweight,
or drain them with `/admin/backends`; every backend must report the exact model
contract and checkpoint SHA from `/metadata`.

## Backend recovery

A backend leaves the pool after two consecutive failures and comes back only
through `verify()` -- `/metadata` plus the checkpoint and encoder-contract check.
A successful `/encode` clears the failure counter but does not promote: a backend
restarted onto a different dtype behind the same URL answers `/encode` perfectly
well, and readmitting it on that alone files its vectors under another encoder's
cache key.

Three independent ways back, so no single wedged component is fatal (MXG-166,
porting the dense wrapper's MXG-159 fix):

* The probe loop is bounded per probe and guarded, and stamps
  `splade_service_probe_loop_last_iteration_timestamp` on every iteration
  including the failing ones. A probe that outlives its bound is cancelled and
  charged as a failure -- httpx's own timeout provably does not bound a wedged
  connection pool. Without this, one silent probe stops recovery for good and
  `splade_service_backend_healthy` freezes at its last value, so the outage reads
  as steady state.
* The HTTP client is recycled after a backend has been unhealthy for
  `BACKEND_CLIENT_RECYCLE_AFTER_S`, or after
  `BACKEND_POOL_TIMEOUT_RECYCLE_AFTER` consecutive `PoolTimeout`s. A poisoned
  connection pool is indistinguishable from a dead backend from the outside, and
  it is why restarting the *backend* does not help: only a new client does.
* Half-open admission lets one trial per `BACKEND_HALF_OPEN_INTERVAL_S` through
  to an unhealthy backend, verify first and then a real chunk. With one entry in
  `BACKEND_URLS`, failing fast forever and being down are the same event.

A request that finds no healthy backend gets **503** with `Retry-After`, not an
unhandled 500, and is counted as `splade_service_requests_total{status="503"}`.

| var | default | |
| --- | --- | --- |
| `BACKEND_TIMEOUT_S` | 30 | per-request timeout to a backend; must stay well under `REQUEST_BUDGET_S` or one attempt eats the whole budget |
| `BACKEND_PROBE_INTERVAL_S` | 5 | probe cadence |
| `BACKEND_PROBE_TIMEOUT_S` | 2 | httpx timeout on one `/metadata` probe |
| `BACKEND_PROBE_ROUND_TIMEOUT_S` | 10 | hard bound outside httpx; silence is a failure |
| `BACKEND_UNHEALTHY_AFTER` | 2 | consecutive failures before a backend leaves the pool |
| `BACKEND_HALF_OPEN_INTERVAL_S` | 5 | spacing of half-open trials, per backend |
| `BACKEND_CLIENT_RECYCLE_AFTER_S` | 60 | unhealthy duration before the client is replaced |
| `BACKEND_POOL_TIMEOUT_RECYCLE_AFTER` | 3 | consecutive `PoolTimeout`s that recycle early |
| `RETRY_AFTER_S` | 1 | `Retry-After` on the 503 |

These defaults are the intended production values, so `compose.t4.yaml` does not
set them.

CUDA backends also expose `POST /encode-packed` for bulk document indexing. It
applies the special-token mask and top-256 on GPU and returns versioned batches
of the existing uint16-token/float16-weight codec. `/encode` returns the same
vectors as an unpruned JSON map, and is the transport `/embed` -- and therefore
the indexer -- uses; queries go through it too, with `document=false`, which
skips the top-256 pruning.

Both transports run the forward under the same `DOCUMENT_DTYPE`, and so do both
roles: a query vector and a document vector meet in a dot product. `/metadata`
reports the dtype in force as `document_compute_dtype` / `query_compute_dtype`,
and `document_encoding_version` -- which namespaces the cache keyspace and gates
pool membership -- names it too.

Use `compose.gpu.yaml` for the dedicated GPU backend and benchmark representative
rendered texts with `scripts/bench_splade_backend.py` before approving a full run.

On the storage-constrained dev host, the compose file mounts the CPU inference
dependencies from `/data/splade-service/runtime`. Populate that directory with:

```bash
docker run --rm -v /data/splade-service/runtime:/runtime \
  ghcr.io/astral-sh/uv:python3.12-bookworm-slim \
  uv pip install --target /runtime --no-cache \
  --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
docker run --rm -v /data/splade-service/runtime:/runtime \
  ghcr.io/astral-sh/uv:python3.12-bookworm-slim \
  uv pip install --target /runtime --no-cache \
  'transformers>=4.49,<6' 'huggingface-hub>=0.24,<2' 'tokenizers>=0.21,<1'
```

Hosts with sufficient Docker storage can instead build the self-contained
`backend-bundled` target.

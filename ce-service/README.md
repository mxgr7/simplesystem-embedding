# ce-service

Serves the cross-encoder fine ranker for the v2 search pipeline, scoring
`(query, article)` pairs **from the article token ids the indexer already
stored** rather than re-rendering and re-tokenizing per candidate per query.

MXG-144. Model: `d_mxg84_new108t_mxg66_s68` (12L×384 XLM-R student, `text_es`
profile, fp16, **`CE_MAX_LEN=128`** since 2026-08-18) — the settled output of
flag-visible teacher →
distill → MXG-66 negation overlay. **The overlay is not optional**: the unaided
student scores 0.320/0.700 on the controlled negation gates against 0.920/0.968
with it, and under the MXG-108 rendering that failure is unrecoverable.

## What it does, in one path

```
query service                    ce-service
─────────────                    ──────────
_source include ceTokenIds  ──►  base64 → little-endian int32 (splice.decode_token_ids)
(a base64 String, verbatim)      [BOS] queryIds [EOS][EOS] articleIds[:budget] [EOS]
                                 budget = maxLen − |queryIds| − 4
                                 forward → softmax → sum(p·[4,2,1,0])/4
                            ◄──  results[] in request order + skipped[] with reasons
```

The caller does **no token math**. It passes the ES `binary` field through as the
base64 string `_source` gave it, and this service owns the decode and the
splice — one decoder, one implementation of the sequence, in the language the
reference (`pipeline/bench_ce_t4.py::assemble`) and its parity gate already live
in.

## The three things that will bite

**1. `exists` on `ceTokenIds` always returns 0.** It is an ES `binary` field with
`doc_values: false`, so `{"exists": {"field": "ceTokenIds"}}` matches *nothing*
on an index where all 115,485,224 offer-bearing articles carry it. Reading it
needs an explicit `_source` include; presence checks must sample `_source` or
gate on `ceTokenCount`.

**2. Some documents carry a matching `ceTokenizerVersion` with no ids.** An
earlier in-place backfill stamped offer-less articles so it would stop revisiting
them (`SearchArticleSourceRoundTripTest.kt:99-108`); the current enricher does
not, but those documents survive until the next rebuild. **Check tokens before
version**, in that order, or a version-first predicate admits them with a null
blob. `_partition` does this and `test_ce_service_api.py` pins it.

**3. A wrong splice produces plausible scores, not an error.** Nothing
downstream can catch it. Hence `ceserve/golden/splice_fixture.json`, checked in
CI (`tests/test_ce_splice_parity.py`) **and at boot on the serving box**
(`CE_ASSERT_GOLDEN=1`). Regenerate it with
`pipeline/gen_ce_splice_fixture.py` and bump `SPLICE_VERSION` if `assemble`
ever changes.

## API

| route | notes |
|---|---|
| `POST /rerank` | the only hot path. Bearer auth when `API_KEY` is set. |
| `GET /healthz` | liveness only — says nothing about the model. |
| `GET /readyz` | 200 only after the **warmup forward** has run and while not degraded. |
| `GET /metadata` | public. Assert on `serving_contract`. |
| `GET /metrics` | Prometheus. |

Request:

```json
{ "query": "6es7 522-1bl01", "segment": "A_identifier", "max_len": 128,
  "candidates": [{ "id": "vendor42:AB-1234", "tokens_b64": "PwAAAKcD…",
                   "token_count": 137,
                   "tokenizer_version": "ce_dist_l12_v3-2026-07-18-fkr108" }] }
```

* `segment` is the ESCI hint the model was trained with. Omit it and you get
  `P_product_noun`, which is **the trained fallback**, not an arbitrary default:
  it is the `multiIf` fallback in `export_ce_data.SEGMENT_SQL` and the
  `coalesce(...)` in `build_ce_v4_dataset.py`, so a term with no classification
  at all was trained under it. An *unrecognised* value is a 400 — it would still
  tokenize, still eat query budget and still produce plausible scores. MXG-148
  will start sending real values.
* `tokenizer_version` is **per candidate**, because a partially backfilled index
  carries a mix and surviving that is the whole reason the field exists.
* `token_count` is redundant with `len(tokens_b64)`. The redundancy is the
  point: a disagreement means a truncated or rewritten `_source`.
* `max_len` is a free serve-time dial, clamped to `[8, CE_MAX_LEN]`, so an A/B
  can trade window depth against width with no redeploy. **Deployed at 128**, not
  the 192 the student was selected at: co-tenancy on this card costs a flat ~1.5×
  and there is no way to buy GPU priority back (stream priority is intra-context,
  Turing has no MIG, MPS caps rather than reserves), so the width is the lever —
  and it is a bigger one than perfect isolation would be. Measured cost on the
  DEPLOYED checkpoint against `ph.esci_test_v24` gold: nDCG@10 0.8730 → 0.8698,
  −0.32 pp with a 95% CI of [−0.79, +0.07] spanning zero.

  ⚠️ The width and the query service's `max-window-size` (200) are **one
  decision**. At width 192 a 200-deep window is 196.9 ms p95 under co-tenancy and
  breaches the 150 ms budget; at 128 it is 130.4 ms. Do not restore 192 while the
  window is 200, and do not raise the window to 256 (162 ms, outside).

Response carries `results[]` **in request order** — this service does not own
ranking policy; EAN pin-to-top, tail demotion and tie handling live in the query
service's `RerankerService` — plus `skipped[]` with a `reason` per candidate
(`tokenizer_version_mismatch` | `decode_failed` | `no_tokens`). Every input id
comes back exactly once across the two.

Each result carries **both** `ce_score` and the four class probabilities. The
score is `train_ce.py`'s `sum(softmax(logits) · [4,2,1,0]) / 4` and the field
names are byte-identical to its `scores.jsonl.gz` keys, so the agreement check is
a dict compare with no mapping table to drift. The probabilities ride along so
MXG-97 can fit and apply a calibration temperature later **with no wire change**:
a temperature-scaled score is recoverable from `ce_p_*` and irrecoverable from
`ce_score`.

Errors are `{"error": code, "detail": …}`: 400 `invalid_request` (request
*shape* only), 401, 413 `too_many_candidates`, 429 `at_capacity`, 503
`model_not_ready`, 504 `request_budget_exhausted`, 500 `inference_failed` (which
also marks the service degraded — a CUDA OOM poisons the context and a process
that keeps answering serves garbage).

A **single** corrupt blob is a `skipped` entry, never a 400: one bad `_source`
must not fail a 120-candidate window. Only a decode-failure rate above
`CE_MAX_DECODE_FAILURE_RATIO` escalates, because that is a wire-format break
rather than data noise. **All candidates skipped is a 200** with `n_scored: 0` —
a correct answer meaning "this window is not backfilled", which the caller's
fallback handles; a 5xx would fire the wrong alert.

## Deployment shape, and what was deliberately left out

One process. **No frontend/backend split, no backend pool, no `/admin/backends`,
no KVRocks cache** — every one of which `splade-service` has and earns.

* The **split** buys splade a cache tier off the GPU, a pool that can absorb an
  H100 burst backend, and two transports. Here it would add a second HTTP hop
  re-serialising a ~95 KB body and a second event loop on a 4-core box, to
  protect 70 ms of headroom rather than 140. It is also what makes "the pool
  marks its only backend unhealthy and then everything fails" possible.
* A **CE-side cache** would be keyed per `(query, article)` — cardinality
  queries × candidates, low hit rate — while the cache that matters already sits
  one layer up (`RerankCache`, Caffeine, TTL 300 s) and saves the whole round
  trip rather than just the forward. splade allows its MGET 100 ms inside a
  120 s budget; here the budget is 150 ms against a 78 ms miss.

`/metadata`'s field names and the `serving_contract` string are nonetheless kept
in the pooled shape, so that if a second GPU ever appears `backend_pool.py` lifts
across with `ENCODER_CONTRACT = ("serving_contract", "model_sha256",
"tokenizer_sha256", "splice_version")` and the caller does not notice.

**If a cache is ever added anyway**, the key must include `model_sha256` *and*
`serving_contract` — dtype, `max_len`, `splice_version`, i.e. *how* the score was
produced, which the model digest cannot express — plus a digest of the article's
`ceTokenIds` **bytes**, not the article id. That is exactly the hole MXG-111
found in the SPLADE keyspace.

## Runtime provisioning (the T4)

`semantic-search.prod.nextgen` has Docker root on `/`, at 88% with ~6 GB free.
The `service-cuda` image unpacks to 9-11 GB, so the deployed target is `service`
(no torch) with the CUDA runtime bind-mounted from `/data` — the same
arrangement `splade-backend-t4` already runs. Once, before first `up`:

```bash
docker run --rm -v /data/ce-service/runtime:/runtime \
  ghcr.io/astral-sh/uv:python3.12-bookworm-slim \
  uv pip install --target /runtime --no-cache \
  --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0
docker run --rm -v /data/ce-service/runtime:/runtime \
  ghcr.io/astral-sh/uv:python3.12-bookworm-slim \
  uv pip install --target /runtime --no-cache \
  'transformers>=5,<6' 'tokenizers>=0.22,<1' 'safetensors>=0.4,<1'
```

Then copy the **inner** checkpoint directory (`…-2026-08-14/d_mxg84_new108t_mxg66_s68/`,
excluding `scores.jsonl.gz` and `train.log`) to `/data/ce-service/model/`, and
**verify both digests from inside the container**, not from the host — the point
is to check what the process will read.

`docker compose -f compose.t4.yaml --env-file .env up -d`.

## Startup assertions

The process refuses to start on any of these, with a message naming the env var
that pins it: nested/incomplete model dir · weights sha256 · tokenizer sha256 ·
blank tokenizer version · backbone / vocab / label count / **special ids**
(`assemble` hardcodes `BOS=0, EOS=2, PAD=1`; a checkpoint with different ones
would splice a syntactically valid, semantically wrong sequence) · `max_len`
inside `[8, max_position_embeddings − 2]` · CPU without `CE_ALLOW_CPU` ·
compute capability < 7.0 · **bf16 without native bf16** · an inconsistent
`(MAX_INPUTS_PER_REQUEST, REQUEST_BUDGET_S)` pair · the golden splice fixture ·
the warmup forward.

The bf16 guard exists because on sm_75 `torch.cuda.is_bf16_supported()` answers
**True** — it defaults to `including_emulation=True` — so without it the process
starts happily on emulated bf16 at 2.48 TFLOP/s against fp16's 25.50, reporting
the same contract an H100 would.

## Tests

```bash
.venv/bin/python -m pytest tests/test_ce_splice_parity.py \
    tests/test_ce_startup_assertions.py tests/test_ce_service_api.py -q
```

`test_ce_splice_parity.py` also decodes the ids the **JVM indexer** committed in
`/next-gen/…/commons/src/test/resources/ce/token-fixtures.json`, which closes the
loop `CeTokenizer.encodePacked → ES binary → ce-service` without either side
trusting a description of the other. It skips if `/next-gen` is not checked out.

`tests/test_ce_score_agreement.py` is the deploy gate and needs the box: it
compares served scores against `train_ce.py --score-only` on the same pairs.

`ceserve/bench_ce_service.py` is the client-side k-sweep — run it idle **and**
with splade under load, since these are co-tenants.

## Why a package, not `uvicorn app:app`

`splade-service` and `embedding-service` both use the flat layout and both own
the bare `constants` / `config` / `main` module slots; `tests/conftest.py`
records the ACL service being renamed `main.py` → `app.py` for exactly that
collision. `ceserve.` costs one prefix and makes it impossible.

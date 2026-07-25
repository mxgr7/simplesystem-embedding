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

KVRocks values use compact `(uint16 token_id, fp16 weight)` storage under the
model-scoped `splade:prod-soup-folde-top256-v1:` prefix. Cache failures fall back
to inference.

The frontend can route misses across multiple compatible backends. Add, reweight,
or drain them with `/admin/backends`; every backend must report the exact model
contract and checkpoint SHA from `/metadata`.

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

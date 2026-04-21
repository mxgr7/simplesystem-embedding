# Search playground

A minimal htmx UI to try out the Milvus-backed offers search. Re-creates the
look of the productive simple system search results page, with unavailable
controls rendered but disabled so the wireframe is faithful.

```
          ┌──────────────────────┐       ┌──────────────────┐
 browser ─▶  playground-app  ───▶  EMBED_URL (TEI-shaped)   │
          │  (FastAPI+htmx)  │    └──────────────────┘
          │                   \──▶  MILVUS_URI              │
          └──────────────────────┘
```

Card metadata (`name`, `manufacturerName`, `ean`, `article_number`) is read
directly from the Milvus collection as `output_fields` on the search call —
no separate catalog lookup.

## Expected embedding service

The playground is a pure HTTP client of a TEI-compatible embedder:

```
POST {EMBED_URL}/embed
  { "inputs": ["schwarze damen sneaker"] }
→ [[0.01, -0.02, ...]]
```

HuggingFace **Text Embeddings Inference** (TEI) is the recommended choice —
it speaks this shape out of the box and is the de-facto standard for
embedding serving. BentoML, Ray Serve, or Triton also work if you wrap them
behind the same JSON contract.

Because the fine-tuned `useful-cub-58` checkpoint ships a custom
`RowTextRenderer` query template and an optional projection head that TEI
does not know about, this folder also contains a reference FastAPI server
(`reference_embed_server/main.py`) that exposes the same `POST /embed`
endpoint by wrapping the in-repo inference code. Use it to exercise the
playground end-to-end until the checkpoint is exported to a
sentence-transformers directory.

## Run

1. Configure the playground:

   ```bash
   cd playground-app
   cp .env.example .env
   # edit .env with your paths / URIs
   ```

2. Start the reference embed server (optional — skip if you already have a
   TEI instance on `EMBED_URL`):

   ```bash
   cd playground-app
   uv sync   # one-time, at repo root
   CHECKPOINT=../checkpoints/useful-cub-58/best-step=4880-val_full_catalog_ndcg_at_5=0.7379.ckpt \
     uv run uvicorn reference_embed_server.main:app --host 0.0.0.0 --port 8080
   ```

3. Start the playground via Docker Compose:

   ```bash
   cd playground-app
   docker compose up --build
   ```

   Open http://localhost:8000. The container uses `network_mode: host`, so
   `EMBED_URL=http://localhost:8080` and `MILVUS_URI=http://localhost:19530`
   from `.env` reach services running on the host as-is. `OFFERS_PARQUET_DIR`
   is bind-mounted read-only at the same path inside the container, and
   request logs land in `../logs/playground-app.log` on the host.

## What is vs. isn't wired up

Available (live against Milvus):

- Full-text search box → embedding → Milvus cosine search → DuckDB hydrate
- Load-more pagination (htmx `outerHTML` swap on the button itself)
- Per-card: brand (`manufacturerName`), product name, article number, EAN

Rendered but disabled (placeholders to match the productive UI):

- `Standard` catalog selector, `Fast order`, `OCI catalogues`
- Left sidebar: `Product categories`, `S2CLASS`
- Toolbar: video icon, `Vendor`, `Manufacturer`, `Delivery time`, `Price`,
  `More filters`, `Relevance` sort, `List/grid` view switch
- Per-card: `Customer article no.`, `S2CLASS`, delivery time, stock, price,
  quantity stepper, `Add to cart`, datasheet icon, favorite icon
- Product image (neutral SVG placeholder)

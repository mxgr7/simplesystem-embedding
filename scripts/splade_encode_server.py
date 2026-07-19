"""SPLADE encode server (box side) for the full-index deployment.

Loads capable-auk-759 once on the GPU and serves per-article SPLADE vectors over
HTTP (stdlib, no extra deps). The workspace-side updater (add_splade_via_server.py)
scans ES, ships each article's offer field-dicts here, and writes the returned
sparse_vector back to ES.

Handling the nested offers[]: for each article we render every offer's text
(description-free, the deployed config), dedup identical renderings, batch-encode
all unique texts in the request on the GPU, and max-merge each article's offer
vectors into one {token_id: weight} map (token-wise max = union of the article's
offers' activated tokens). An in-memory text->vector LRU means offer texts shared
across vendors/articles encode once.

POST /encode_articles  {"articles": [[offer_dict, ...], ...]}
    offer_dict keys (renderer-ready snake_case): name, manufacturer_name,
    manufacturer_article_number, article_number, ean, category_paths,
    manufacturer_article_type
  -> {"vectors": [{"<token_id>": weight, ...}, ...]}   (one per article; may be {})
GET /health -> {"ok": true, "nnz_cached": N}

Run on the box (ES is NOT needed here — pure encode):
  PYTORCH_ALLOC_CONF=expandable_segments:True LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1 \
    uv run --extra train python scripts/splade_encode_server.py \
      --splade-ckpt checkpoints/capable-auk-759/best-*.ckpt --port 8137
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from embedding_train.model import load_embedding_module_from_checkpoint  # noqa: E402
from embedding_train.rendering import RowTextRenderer  # noqa: E402
from embedding_train.tokenization import load_fast_tokenizer  # noqa: E402
from indexer.sparse_codec import pack_sparse  # noqa: E402


def pack_response(vectors):
    """[dict] -> bytes: uint32 N, then per vector (uint32 blen, packed bytes)."""
    parts = [struct.pack("<I", len(vectors))]
    for v in vectors:
        b = pack_sparse({int(k): w for k, w in v.items()})
        parts.append(struct.pack("<I", len(b)))
        parts.append(b)
    return b"".join(parts)


class Encoder:
    def __init__(self, ckpt, device, cache_cap):
        self.model, cfg = load_embedding_module_from_checkpoint(ckpt)
        self.model = self.model.to(device).eval()
        self.tok = load_fast_tokenizer(cfg.model.model_name)
        self.ren = RowTextRenderer(cfg.data)
        self.max_len = int(cfg.data.max_offer_length)
        self.device = device
        self.lock = threading.Lock()  # single GPU: serialize forwards
        self.cache: OrderedDict[str, dict] = OrderedDict()
        self.cache_cap = cache_cap
        self.batch = 256

    def render(self, offer: dict) -> str:
        row = {**offer, "description": ""}  # deployed config is description-free
        return self.ren.render_offer_text(row)

    @torch.inference_mode()
    def _encode(self, texts: list[str]) -> list[dict]:
        out: list[dict] = []
        for i in range(0, len(texts), self.batch):
            chunk = texts[i:i + self.batch]
            inp = self.tok(chunk, padding=True, truncation=True,
                           max_length=self.max_len, return_tensors="pt")
            rep = self.model.encode({k: v.to(self.device) for k, v in inp.items()})
            rep = rep.float().cpu().numpy()
            for vec in rep:
                idx = np.nonzero(vec > 0)[0]
                out.append({str(int(j)): float(vec[j]) for j in idx})
        return out

    def encode_articles(self, articles: list[list[dict]]) -> list[dict]:
        # 1) render + dedup unique texts (respect cache)
        per_article_texts: list[list[str]] = []
        need: "OrderedDict[str, None]" = OrderedDict()
        for offers in articles:
            texts = []
            for off in offers or []:
                t = self.render(off)
                if not t:
                    continue
                texts.append(t)
                if t not in self.cache and t not in need:
                    need[t] = None
            per_article_texts.append(texts)
        # 2) batch-encode the uncached uniques under the GPU lock
        if need:
            uniq = list(need)
            with self.lock:
                vecs = self._encode(uniq)
            for t, v in zip(uniq, vecs):
                self.cache[t] = v
                self.cache.move_to_end(t)
            while len(self.cache) > self.cache_cap:
                self.cache.popitem(last=False)
        # 3) max-merge per article
        results: list[dict] = []
        for texts in per_article_texts:
            merged: dict = {}
            seen = set()
            for t in texts:
                if t in seen:
                    continue
                seen.add(t)
                for k, w in self.cache.get(t, {}).items():
                    p = merged.get(k)
                    if p is None or w > p:
                        merged[k] = w
            results.append(merged)
        return results


ENC: Encoder | None = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "cached": len(ENC.cache)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/encode_articles":
            self._send(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        try:
            vecs = ENC.encode_articles(req.get("articles", []))
            body = pack_response(vecs)  # compact binary (tunnel-friendly)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": repr(e)})

    def log_message(self, *a):  # quiet
        pass


def main():
    global ENC
    ap = argparse.ArgumentParser()
    ap.add_argument("--splade-ckpt", required=True)
    ap.add_argument("--port", type=int, default=8137)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-cap", type=int, default=2_000_000)
    args = ap.parse_args()
    t0 = time.time()
    ENC = Encoder(args.splade_ckpt, args.device, args.cache_cap)
    print(f"loaded {args.splade_ckpt} in {time.time()-t0:.0f}s, "
          f"max_offer_length={ENC.max_len}; serving on :{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

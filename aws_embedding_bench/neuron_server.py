"""TEI-compatible HTTP wrapper for a pre-compiled Inferentia embedding model.

Exposes ``POST /embed`` with the same wire format as text-embeddings-inference
so the benchmark loadgen can hit it unchanged:

  Request:  {"inputs": ["text1", "text2", ...], "truncate": true}
  Response: [[float, ...], [float, ...], ...]

Pre-condition: the compiled model lives in NEURON_MODEL_DIR as a single
TorchScript file ``model.pt`` (produced by ``torch_neuronx.trace()`` on a
separate compile box), alongside the tokenizer files and — depending on what
the trace baked in — the SentenceTransformers post-processing modules
(``2_Dense/pytorch_model.bin``, etc.).

Compile shape constraint: the .pt is traced at a *fixed* (batch, seq_len).
This server pads / chunks every incoming request to that shape:
  - batch < FIXED_BATCH: pad with empty inputs, slice result back
  - batch > FIXED_BATCH: split into FIXED_BATCH-sized chunks
  - seq_len: tokenize with padding='max_length', max_length=MAX_SEQ_LEN

We auto-detect whether the trace was just the encoder (output [B,S,768]),
encoder+pool ([B,768]), or full pipeline ([B,128]) by running a smoke
inference at startup and applying only the post-processing the trace
hasn't already done.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer

# torch.jit.load needs the Neuron op registry. inf2 ships torch_neuronx;
# inf1 ships torch_neuron (no x). Try whichever is available.
try:
    import torch_neuronx  # noqa: F401
    _neuron_pkg = "torch_neuronx"
except ImportError:
    import torch_neuron  # type: ignore[import-not-found]  # noqa: F401
    _neuron_pkg = "torch_neuron"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03dZ | %(levelname)s | neuron-server] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logging.Formatter.converter = time.gmtime
log = logging.getLogger("neuron-server")

MODEL_DIR = os.environ.get("NEURON_MODEL_DIR", "/opt/neuron-model")
MAX_LEN = int(os.environ.get("NEURON_MAX_SEQ_LEN", "512"))
FIXED_BATCH = int(os.environ.get("NEURON_FIXED_BATCH", "32"))
MODEL_FILENAME = os.environ.get("NEURON_MODEL_FILE", "model.pt")

log.info("config: MODEL_DIR=%s MAX_LEN=%d FIXED_BATCH=%d FILE=%s neuron_pkg=%s",
         MODEL_DIR, MAX_LEN, FIXED_BATCH, MODEL_FILENAME, _neuron_pkg)

log.info("loading tokenizer from %s", MODEL_DIR)
_t0 = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
log.info("tokenizer loaded in %.2fs (model_max_length=%s vocab=%d)",
         time.perf_counter() - _t0, getattr(tokenizer, "model_max_length", "?"),
         tokenizer.vocab_size)

_pt_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
log.info("loading TorchScript model from %s", _pt_path)
_t0 = time.perf_counter()
model = torch.jit.load(_pt_path)
model.eval()
log.info("model loaded in %.2fs", time.perf_counter() - _t0)

# Try to load the SentenceTransformers Dense head (768 -> 128) in case the
# trace didn't bake it in. Tiny (~400KB), so loading speculatively is cheap.
_dense_weight: torch.Tensor | None = None
_dense_path = os.path.join(MODEL_DIR, "2_Dense", "pytorch_model.bin")
if os.path.exists(_dense_path):
    log.info("loading 2_Dense head from %s (in case trace didn't bake it in)",
             _dense_path)
    _dense_state = torch.load(_dense_path, map_location="cpu", weights_only=True)
    # SentenceTransformers Dense saves the linear.weight as 'linear.weight'.
    for k, v in _dense_state.items():
        if k.endswith("weight") and v.dim() == 2:
            _dense_weight = v
            log.info("  dense weight shape: %s", tuple(v.shape))
            break
    if _dense_weight is None:
        log.warning("  no 2D weight found in %s — dense projection disabled",
                    _dense_path)
else:
    log.info("no 2_Dense/pytorch_model.bin present — assuming trace baked it in")


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _tokenize_padded(texts: list[str]) -> dict[str, torch.Tensor]:
    """Tokenize to FIXED_BATCH × MAX_LEN. Pad batch dim with empty strings."""
    if len(texts) < FIXED_BATCH:
        texts = texts + [""] * (FIXED_BATCH - len(texts))
    elif len(texts) > FIXED_BATCH:
        raise ValueError(f"chunk size {len(texts)} > FIXED_BATCH {FIXED_BATCH}")
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    return {k: v for k, v in enc.items() if k in {"input_ids", "attention_mask", "token_type_ids"}}


def _try_call_model(enc: dict[str, torch.Tensor]) -> torch.Tensor:
    """Probe the traced model's signature. Different traces accept different
    input arities; figure out which once at startup and stash the answer in
    _CALL_FORM."""
    global _CALL_FORM
    if _CALL_FORM == "2pos":
        return model(enc["input_ids"], enc["attention_mask"])
    if _CALL_FORM == "3pos":
        tti = enc.get("token_type_ids", torch.zeros_like(enc["input_ids"]))
        return model(enc["input_ids"], enc["attention_mask"], tti)
    if _CALL_FORM == "kw":
        return model(**enc)
    # First call: try forms in order, remember whichever works.
    for form in ("2pos", "3pos", "kw"):
        try:
            if form == "2pos":
                out = model(enc["input_ids"], enc["attention_mask"])
            elif form == "3pos":
                tti = enc.get("token_type_ids", torch.zeros_like(enc["input_ids"]))
                out = model(enc["input_ids"], enc["attention_mask"], tti)
            else:
                out = model(**enc)
            _CALL_FORM = form
            log.info("model call form: %s", form)
            return out
        except Exception as e:  # noqa: BLE001 — any TorchScript error is fine to swallow during probe
            log.info("  call form %s rejected: %s", form, str(e)[:120])
    raise RuntimeError("none of (2pos, 3pos, kw) call forms worked on the traced model")


_CALL_FORM: str = ""  # set on first call


def _postprocess(raw: Any, attention_mask: torch.Tensor) -> torch.Tensor:
    """Apply whichever of pool / dense / normalize the trace didn't bake in.

    Cases (detected from output shape):
      [B, S, H=768] -> encoder only          -> pool + dense + normalize
      [B, H=768]    -> encoder + pool        -> dense + normalize
      [B, H=128]    -> full pipeline         -> nothing
    """
    if isinstance(raw, (tuple, list)):
        raw = raw[0]
    if isinstance(raw, dict) and "last_hidden_state" in raw:
        raw = raw["last_hidden_state"]

    t = raw
    if t.dim() == 3:                          # [B, S, H]
        t = _mean_pool(t, attention_mask)     # -> [B, H]
    if t.dim() != 2:
        raise RuntimeError(f"unexpected model output shape: {tuple(t.shape)}")

    if t.shape[1] == 768 and _dense_weight is not None:
        t = F.linear(t, _dense_weight)        # [B, 768] -> [B, 128]
    if t.shape[1] not in (128, 768):
        log.warning("unusual embedding dim: %d", t.shape[1])

    t = F.normalize(t, p=2, dim=1)
    return t


# Smoke inference at startup so we surface trace-shape mismatches *before*
# the first real request lands (and so we record the detected dim).
log.info("running startup smoke inference")
_t0 = time.perf_counter()
_smoke_enc = _tokenize_padded(["passage: smoke test"])
with torch.inference_mode():
    _smoke_raw = _try_call_model(_smoke_enc)
_smoke_vec = _postprocess(_smoke_raw, _smoke_enc["attention_mask"])
log.info("smoke OK in %.2fs: raw_shape=%s -> emb_shape=%s dim=%d",
         time.perf_counter() - _t0,
         tuple(_smoke_raw[0].shape if isinstance(_smoke_raw, (tuple, list)) else _smoke_raw.shape),
         tuple(_smoke_vec.shape), _smoke_vec.shape[1])


class EmbedRequest(BaseModel):
    inputs: list[str]
    truncate: bool = True


app = FastAPI()

_REQ_COUNT = 0
_LOG_EVERY_N = 100


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/embed")
def embed(req: EmbedRequest) -> list[list[float]]:
    global _REQ_COUNT
    if not req.inputs:
        raise HTTPException(status_code=400, detail="empty inputs")
    t0 = time.perf_counter()
    n = len(req.inputs)
    out_vecs: list[torch.Tensor] = []
    # Chunk into FIXED_BATCH-sized blocks; pad the last block with empty
    # strings (we slice them off below so they never reach the caller).
    for start in range(0, n, FIXED_BATCH):
        chunk = req.inputs[start:start + FIXED_BATCH]
        real_len = len(chunk)
        enc = _tokenize_padded(chunk)
        with torch.inference_mode():
            raw = _try_call_model(enc)
        vecs = _postprocess(raw, enc["attention_mask"])
        out_vecs.append(vecs[:real_len])
    result_t = torch.cat(out_vecs, dim=0)
    result = result_t.tolist()
    _REQ_COUNT += 1
    if _REQ_COUNT <= 3 or _REQ_COUNT % _LOG_EVERY_N == 0:
        log.info("req #%d: inputs=%d dim=%d latency=%.1fms",
                 _REQ_COUNT, n, len(result[0]) if result else 0,
                 (time.perf_counter() - t0) * 1000.0)
    return result

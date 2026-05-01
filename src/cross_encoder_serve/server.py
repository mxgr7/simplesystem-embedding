"""FastAPI server for the cross-encoder rerank pipeline.

Run:
  CKPT=checkpoints/apr29-soup-3way/apr29-final-soup-3way.ckpt \
  LGBM=artifacts/lgbm_soup.txt \
  TEMPERATURE=0.534 \
  ENSEMBLE_W=0.6 \
  uvicorn cross_encoder_serve.server:app --host 0.0.0.0 --port 8080

CKPT and LGBM accept either a local path or a Hugging Face Hub spec of the
form `org/repo[@revision]:filename` (e.g. `acme/ce-soup:apr29-final.ckpt`).
HF artifacts are downloaded into the standard HF cache on startup; for
private repos export `HF_TOKEN`. When LGBM is supplied as an HF spec, the
matching `<basename>.cols.json` sidecar is fetched from the same repo and
revision.

Environment variables:
  CKPT          path or HF spec for the Soup CE Lightning checkpoint (required)
  CONFIG_DIR    Hydra config dir (default: <repo>/configs)
  LGBM          path or HF spec for the saved LGBM booster (.txt). Unset = LGBM stack disabled.
  TEMPERATURE   calibration scalar (default 0.534)
  ENSEMBLE_W    CE↔LGBM mixing weight (default 0.6, only used if LGBM is loaded)
  DEVICE        force "cpu" or "cuda" (default: cuda if available)
  SERVE_DTYPE   inference autocast dtype: "bf16" | "fp16" | "fp32" | "auto"
                (default "auto" → bf16 on cuda, fp32 on cpu). bf16 ~2× max
                batch capacity at S=512 vs fp32 with no measurable accuracy
                delta on this 330M model.
  HF_TOKEN      auth for private HF repos (auto-read by huggingface_hub)
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field

from cross_encoder_serve.inference import (
    DEFAULT_ENSEMBLE_W,
    DEFAULT_TEMPERATURE,
    Reranker,
    RerankerConfig,
)


_HF_SPEC = re.compile(r"^(?P<repo>[^@:/\s]+/[^@:/\s]+)(?:@(?P<rev>[^:\s]+))?:(?P<file>\S+)$")


def _parse_hf_spec(spec: str) -> Optional[re.Match]:
    """Return a regex match if `spec` is an HF spec, else None.

    Local paths (starting with /, ./, ../, ~) are never treated as HF specs.
    """
    if spec.startswith(("/", "./", "../", "~")):
        return None
    return _HF_SPEC.match(spec)


def _resolve_artifact(spec: str) -> str:
    """Resolve `spec` to a local file path. HF specs are downloaded; paths pass through."""
    m = _parse_hf_spec(spec)
    if m is None:
        return spec
    logger.info("Resolving HF artifact: repo=%s rev=%s file=%s", m["repo"], m["rev"], m["file"])
    return hf_hub_download(repo_id=m["repo"], filename=m["file"], revision=m["rev"])


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# --- Schemas ---------------------------------------------------------------


class Offer(BaseModel):
    offer_id: str
    name: Optional[str] = None
    manufacturer_name: Optional[str] = None
    manufacturer_article_number: Optional[str] = None
    manufacturer_article_type: Optional[str] = None
    article_number: Optional[str] = None
    ean: Optional[str] = None
    category_paths: Optional[str] = None
    description: Optional[str] = None


class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1)
    offers: list[Offer]
    threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Filter results: keep offers with p_exact_calibrated >= threshold. "
            "Calibrated p_exact is well-calibrated against held-out NLL — "
            "threshold≈0.7 ≈ 70% precision on the kept set."
        ),
    )
    top_k: Optional[int] = Field(
        None, ge=1,
        description="Limit results to top-K by p_exact_calibrated (after threshold).",
    )


class OfferResult(BaseModel):
    offer_id: str
    p_exact: float
    p_substitute: float
    p_complement: float
    p_irrelevant: float
    p_exact_calibrated: float = Field(
        ..., description="Temperature-scaled p_exact (T=0.534), use for thresholding."
    )
    predicted_label: str


class RerankResponse(BaseModel):
    query: str
    n_input: int
    n_returned: int
    results: list[OfferResult]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    lgbm_loaded: bool
    temperature: float
    ensemble_w: float
    device: str
    autocast_dtype: str  # "bf16" | "fp16" | "fp32"
    attn_implementation: str  # "sdpa" | "eager" | "flash_attention_2" | ...


# --- App -------------------------------------------------------------------


app = FastAPI(
    title="Cross-Encoder Rerank",
    description="Rerank a candidate offer list against a query — Soup CE with optional LGBM stack.",
    default_response_class=ORJSONResponse,
)
_state: dict = {}


@app.on_event("startup")
def _load_model():
    ckpt_spec = os.environ.get("CKPT")
    if not ckpt_spec:
        raise RuntimeError("CKPT env var is required (path or HF spec for Soup CE checkpoint).")
    ckpt_path = _resolve_artifact(ckpt_spec)

    lgbm_spec = os.environ.get("LGBM")
    lgbm_path: Optional[str] = None
    lgbm_cols_path: Optional[str] = None
    if lgbm_spec:
        lgbm_path = _resolve_artifact(lgbm_spec)
        m = _parse_hf_spec(lgbm_spec)
        if m is not None:
            # Pull the cols sidecar from the same repo/revision; local paths
            # let RerankerConfig derive it via with_suffix(".cols.json").
            cols_filename = str(Path(m["file"]).with_suffix(".cols.json"))
            lgbm_cols_path = hf_hub_download(
                repo_id=m["repo"], filename=cols_filename, revision=m["rev"]
            )

    config_dir = os.environ.get("CONFIG_DIR") or str(Path(__file__).resolve().parents[2] / "configs")
    cfg = RerankerConfig(
        ckpt_path=ckpt_path,
        config_dir=config_dir,
        config_name=os.environ.get("SERVE_CONFIG_NAME", "cross_encoder"),
        lgbm_path=lgbm_path,
        lgbm_cols_path=lgbm_cols_path,
        temperature=float(os.environ.get("TEMPERATURE", DEFAULT_TEMPERATURE)),
        ensemble_w=float(os.environ.get("ENSEMBLE_W", DEFAULT_ENSEMBLE_W)),
        device=os.environ.get("DEVICE"),
        autocast_dtype=os.environ.get("SERVE_DTYPE"),
        max_forward_batch=int(os.environ.get("SERVE_MAX_BATCH", "128")),
        compile=os.environ.get("SERVE_COMPILE", "1") == "1",
        compile_mode=os.environ.get("SERVE_COMPILE_MODE", "max-autotune"),
        int8=os.environ.get("SERVE_INT8", "0") == "1",
        bnb8=os.environ.get("SERVE_BNB8", "0") == "1",
        runtime=os.environ.get("SERVE_RUNTIME", "torch"),
        onnx_path=os.environ.get("SERVE_ONNX_PATH"),
        onnx_providers=tuple(
            (os.environ.get("SERVE_ONNX_PROVIDERS")
             or "CUDAExecutionProvider,CPUExecutionProvider").split(",")
        ),
    )
    logger.info(
        "Loading Reranker: ckpt=%s lgbm=%s T=%.4f w=%.2f",
        cfg.ckpt_path, cfg.lgbm_path, cfg.temperature, cfg.ensemble_w,
    )
    _state["reranker"] = Reranker(cfg)
    logger.info("Reranker ready (device=%s)", _state["reranker"].device)


def _get_reranker() -> Reranker:
    rr = _state.get("reranker")
    if rr is None:
        raise HTTPException(status_code=503, detail="model not yet loaded")
    return rr


@app.get("/health", response_model=HealthResponse)
def health():
    rr = _state.get("reranker")
    if rr is None:
        return HealthResponse(
            status="loading", model_loaded=False, lgbm_loaded=False,
            temperature=0.0, ensemble_w=0.0, device="?",
            autocast_dtype="?", attn_implementation="?",
        )
    return HealthResponse(
        status="ok",
        model_loaded=True,
        lgbm_loaded=rr.booster is not None,
        temperature=rr.cfg.temperature,
        ensemble_w=rr.cfg.ensemble_w,
        device=rr.device,
        autocast_dtype=rr.autocast_label,
        attn_implementation=rr.attn_implementation,
    )


@app.post("/rerank")
async def rerank(request: Request):
    """Raw-body /rerank: parse with orjson + pass plain dicts to the
    reranker, skipping the per-offer pydantic validation that adds ~100 ms
    on 2000-offer requests. Same external JSON contract as before — the
    RerankRequest/RerankResponse Pydantic models still describe the schema
    for OpenAPI but aren't used at runtime.
    """
    import asyncio
    import orjson
    body = await request.body()
    payload = orjson.loads(body)
    query = payload.get("query")
    offers = payload.get("offers") or []
    threshold = payload.get("threshold")
    top_k = payload.get("top_k")
    if not offers:
        return ORJSONResponse({"query": query, "n_input": 0, "n_returned": 0, "results": []})

    reranker = _get_reranker()
    # Run the blocking reranker on a worker thread so the event loop stays free.
    scores = await asyncio.to_thread(reranker.rerank, query, offers)

    # Build the response as plain dicts; orjson serializes ~3× faster than
    # the default json + pydantic-model_dump path.
    results = [
        {
            "offer_id": s.offer_id,
            "p_exact": s.p_exact,
            "p_substitute": s.p_substitute,
            "p_complement": s.p_complement,
            "p_irrelevant": s.p_irrelevant,
            "p_exact_calibrated": s.p_exact_calibrated,
            "predicted_label": s.predicted_label,
        }
        for s in scores
    ]
    if threshold is not None:
        results = [r for r in results if r["p_exact_calibrated"] >= threshold]
    results.sort(key=lambda r: -r["p_exact_calibrated"])
    if top_k is not None:
        results = results[: int(top_k)]
    return ORJSONResponse({
        "query": query,
        "n_input": len(offers),
        "n_returned": len(results),
        "results": results,
    })

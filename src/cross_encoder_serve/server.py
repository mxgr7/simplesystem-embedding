"""FastAPI server for the cross-encoder rerank pipeline.

Run:
  CKPT=checkpoints/apr29-soup-3way/apr29-final-soup-3way.ckpt \
  LGBM=artifacts/lgbm_soup.txt \
  TEMPERATURE=0.534 \
  ENSEMBLE_W=0.6 \
  uvicorn cross_encoder_serve.server:app --host 0.0.0.0 --port 8080

Environment variables:
  CKPT          path to Soup CE Lightning checkpoint (required)
  CONFIG_DIR    Hydra config dir (default: <repo>/configs)
  LGBM          path to saved LGBM booster (.txt). If unset, LGBM stack is disabled.
  TEMPERATURE   calibration scalar (default 0.534)
  ENSEMBLE_W    CE↔LGBM mixing weight (default 0.6, only used if LGBM is loaded)
  DEVICE        force "cpu" or "cuda" (default: cuda if available)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from cross_encoder_serve.inference import (
    DEFAULT_ENSEMBLE_W,
    DEFAULT_TEMPERATURE,
    Reranker,
    RerankerConfig,
)


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


# --- App -------------------------------------------------------------------


app = FastAPI(
    title="Cross-Encoder Rerank",
    description="Rerank a candidate offer list against a query — Soup CE with optional LGBM stack.",
)
_state: dict = {}


@app.on_event("startup")
def _load_model():
    ckpt = os.environ.get("CKPT")
    if not ckpt:
        raise RuntimeError("CKPT env var is required (path to Soup CE checkpoint).")
    config_dir = os.environ.get("CONFIG_DIR") or str(Path(__file__).resolve().parents[2] / "configs")
    cfg = RerankerConfig(
        ckpt_path=ckpt,
        config_dir=config_dir,
        lgbm_path=os.environ.get("LGBM"),
        temperature=float(os.environ.get("TEMPERATURE", DEFAULT_TEMPERATURE)),
        ensemble_w=float(os.environ.get("ENSEMBLE_W", DEFAULT_ENSEMBLE_W)),
        device=os.environ.get("DEVICE"),
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
        )
    return HealthResponse(
        status="ok",
        model_loaded=True,
        lgbm_loaded=rr.booster is not None,
        temperature=rr.cfg.temperature,
        ensemble_w=rr.cfg.ensemble_w,
        device=rr.device,
    )


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    if not req.offers:
        return RerankResponse(query=req.query, n_input=0, n_returned=0, results=[])
    reranker = _get_reranker()
    scores = reranker.rerank(req.query, [o.dict() for o in req.offers])
    results = [
        OfferResult(
            offer_id=s.offer_id,
            p_exact=s.p_exact,
            p_substitute=s.p_substitute,
            p_complement=s.p_complement,
            p_irrelevant=s.p_irrelevant,
            p_exact_calibrated=s.p_exact_calibrated,
            predicted_label=s.predicted_label,
        )
        for s in scores
    ]
    if req.threshold is not None:
        results = [r for r in results if r.p_exact_calibrated >= req.threshold]
    results.sort(key=lambda r: -r.p_exact_calibrated)
    if req.top_k is not None:
        results = results[: req.top_k]
    return RerankResponse(
        query=req.query,
        n_input=len(req.offers),
        n_returned=len(results),
        results=results,
    )

"""FastAPI serving for the cross-encoder rerank pipeline.

Architecture:
  client → /rerank → CE soup (calibrated via T) → optional LGBM stack → ranked offers
"""
from cross_encoder_serve.inference import Reranker

__all__ = ["Reranker"]

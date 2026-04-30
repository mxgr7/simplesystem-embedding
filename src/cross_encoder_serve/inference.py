"""Reranker: load Soup CE checkpoint + optional LGBM, score (query, offers) pairs.

Production recipe:
  1. Tokenize (query, offer_text) pairs with the same tokenizer used in training.
  2. Forward through Soup CE → 4-class logits.
  3. Compute UN-scaled CE probs = softmax(logits). LGBM was trained on these,
     so we feed them into LGBM exactly as during training.
  4. Optional LGBM stack:
       - compute engineered features (EAN/ART one-hot, lex similarity)
       - compute list features within the request batch (rank/gap/zscore per query)
       - LGBM.predict() → 4-class probs
       - ensemble: w * CE_probs + (1 - w) * LGBM_probs   (w=0.6 default)
  5. Return per-offer:
       - 4-class probs (ensemble if LGBM on, else raw CE) — use these for argmax / F1.
       - p_exact_calibrated = softmax(logits / T)[:, EXACT_IDX] — temperature-scaled
         (T=0.534), well-calibrated against held-out NLL. Use this for THRESHOLDING
         in the "Exact-only" rerank: a 0.7 threshold ≈ 70% precision on Exact.

Label order: LABEL_ORDER = ("Irrelevant", "Complement", "Substitute", "Exact")
  probs[:, 0] = P(Irrelevant)
  probs[:, 3] = P(Exact)
"""
from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from cross_encoder_train.features import FeatureExtractor
from cross_encoder_train.labels import LABEL_ORDER  # ("Irrelevant","Complement","Substitute","Exact")
from cross_encoder_train.model import CrossEncoderModule
from embedding_train.rendering import RowTextRenderer

LABEL_INDEX = {label: idx for idx, label in enumerate(LABEL_ORDER)}
EXACT_IDX = LABEL_INDEX["Exact"]

# Calibration / ensemble defaults — overridable via Reranker constructor.
DEFAULT_TEMPERATURE = 0.534
DEFAULT_ENSEMBLE_W = 0.6

logger = logging.getLogger(__name__)


@dataclass
class RerankerConfig:
    ckpt_path: str
    config_dir: str
    lgbm_path: Optional[str] = None
    lgbm_cols_path: Optional[str] = None  # JSON sidecar with feature column order
    temperature: float = DEFAULT_TEMPERATURE
    ensemble_w: float = DEFAULT_ENSEMBLE_W
    device: Optional[str] = None
    config_name: str = "cross_encoder"


@dataclass
class OfferScore:
    offer_id: str
    p_irrelevant: float
    p_complement: float
    p_substitute: float
    p_exact: float
    p_exact_calibrated: float  # softmax(logits/T)[Exact] — for thresholding

    @property
    def predicted_label(self) -> str:
        idx = int(np.argmax([self.p_irrelevant, self.p_complement, self.p_substitute, self.p_exact]))
        return LABEL_ORDER[idx]


class Reranker:
    def __init__(self, cfg: RerankerConfig):
        self.cfg = cfg
        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

        with initialize_config_dir(config_dir=cfg.config_dir, version_base="1.3"):
            self.hcfg = compose(config_name=cfg.config_name)

        logger.info("Loading CE checkpoint from %s", cfg.ckpt_path)
        # Serving runs eager: torch.compile gives ~0% throughput on this model
        # but eats VRAM headroom (~25% smaller max batch) and pays a ~20s
        # recompile per new (batch_size, seq_len) shape. Override the training
        # cfg's `model.compile=true` so the encoder isn't wrapped, then strip
        # the `_orig_mod.` prefix the wrapped encoder added when the ckpt was
        # saved (no-op if the ckpt was trained without compile).
        serve_hcfg = OmegaConf.merge(
            self.hcfg, OmegaConf.create({"model": {"compile": False}})
        )
        self.model = CrossEncoderModule(cfg=serve_hcfg)
        ckpt = torch.load(cfg.ckpt_path, map_location=self.device, weights_only=False)
        state_dict = {
            k.replace("._orig_mod.", "."): v for k, v in ckpt["state_dict"].items()
        }
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.hcfg.model.model_name, use_fast=True)
        self.max_pair_length = int(self.hcfg.data.max_pair_length)
        self.renderer = RowTextRenderer(self.hcfg.data)

        # Always-on feature extractor (independent of how the CE was trained).
        self.extractor = FeatureExtractor(_serving_features_cfg())

        self.booster: Optional[lgb.Booster] = None
        self.lgbm_feature_cols: list[str] = []
        if cfg.lgbm_path:
            self.booster = lgb.Booster(model_file=cfg.lgbm_path)
            cols_path = cfg.lgbm_cols_path or str(Path(cfg.lgbm_path).with_suffix(".cols.json"))
            self.lgbm_feature_cols = json.loads(Path(cols_path).read_text())["feature_cols"]
            logger.info("Loaded LGBM booster (%d features)", len(self.lgbm_feature_cols))

    @torch.no_grad()
    def rerank(self, query: str, offers: list[dict]) -> list[OfferScore]:
        """Score every offer in `offers` against `query`. Returns one OfferScore per input.

        offer dict expected keys (any subset; missing are treated as empty):
          offer_id, name, manufacturer_name, ean, article_number,
          manufacturer_article_number, manufacturer_article_type,
          category_paths, description
        """
        if not offers:
            return []

        # 1. Build query/offer texts the same way training did.
        query_text, offer_texts, contexts = self._prepare_texts(query, offers)

        # 2. CE forward → logits.
        logits = self._forward(query_text, offer_texts).float()  # (n_offers, 4)

        # 3. Two views of the CE output:
        #    - probs_raw: softmax(logits) — what LGBM was trained on; used for
        #      ensembling and argmax/F1.
        #    - p_exact_cal: softmax(logits / T)[:, EXACT_IDX] — temperature-scaled,
        #      well-calibrated against val NLL. Use for thresholding.
        probs_raw = F.softmax(logits, dim=-1).cpu().numpy()
        p_exact_cal = F.softmax(logits / self.cfg.temperature, dim=-1)[:, EXACT_IDX].cpu().numpy()

        # 4. Optional LGBM stack — feeds LGBM the UN-scaled CE probs (matching training).
        if self.booster is not None:
            lgbm_probs = self._lgbm_predict(probs_raw, contexts, query, offers)
            probs_out = self.cfg.ensemble_w * probs_raw + (1.0 - self.cfg.ensemble_w) * lgbm_probs
        else:
            probs_out = probs_raw

        # 5. Return per-offer scores.
        return [
            OfferScore(
                offer_id=str(offer.get("offer_id", "")),
                p_irrelevant=float(p[0]),
                p_complement=float(p[1]),
                p_substitute=float(p[2]),
                p_exact=float(p[3]),
                p_exact_calibrated=float(p_cal),
            )
            for offer, p, p_cal in zip(offers, probs_out, p_exact_cal)
        ]

    def _prepare_texts(self, query: str, offers: list[dict]):
        # The renderer reads from a row-dict; we mimic the labeled-row schema.
        contexts = []
        offer_texts = []
        for offer in offers:
            row = {**offer, "query_term": query}
            ctx = self.renderer.build_context(row)
            contexts.append(ctx)
            offer_texts.append(self.renderer.render_offer_text(row, context=ctx))
        # query text is the same for every pair in this request.
        first_row = {**offers[0], "query_term": query} if offers else {"query_term": query}
        query_text = self.renderer.render_query_text(first_row, context=contexts[0] if contexts else None)
        return query_text, offer_texts, contexts

    def _forward(self, query_text: str, offer_texts: list[str]) -> torch.Tensor:
        enc = self.tokenizer(
            [query_text] * len(offer_texts),
            offer_texts,
            padding=True,
            truncation="only_second",
            max_length=self.max_pair_length,
            return_tensors="pt",
            return_token_type_ids=True,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        return self.model(enc)

    def _lgbm_predict(self, ce_probs: np.ndarray, contexts: list[dict],
                      query: str, offers: list[dict]) -> np.ndarray:
        """Build per-pair LGBM features (CE probs + engineered + list) and predict 4-class."""
        rows = []
        for i, (offer, ctx) in enumerate(zip(offers, contexts)):
            feats = _engineered_features(self.extractor, ctx, query, offer.get("name", ""))
            row = {
                "ce_p_irrelevant": float(ce_probs[i, 0]),
                "ce_p_complement": float(ce_probs[i, 1]),
                "ce_p_substitute": float(ce_probs[i, 2]),
                "ce_p_exact": float(ce_probs[i, 3]),
                **feats,
            }
            rows.append(row)
        # List features (rank/gap/zscore + group_size). Computed across THIS request batch
        # since it represents the "query group" from the caller's POV.
        rows = _add_list_features(rows)
        # Build numpy in the column order the booster expects.
        X = np.array([[row[c] for c in self.lgbm_feature_cols] for row in rows], dtype=np.float64)
        return self.booster.predict(X)


# ---------------------------------------------------------------------------
# Helpers — kept module-level for testability and to mirror the training-time
# pipeline (autoresearch/cross-encoder/{dump_predictions.py,add_list_features.py}).
# ---------------------------------------------------------------------------

_DIGIT_RUN = re.compile(r"\d+")
EAN_STATES = ("NONE", "MATCH", "MISMATCH")
ART_STATES = ("NONE", "EXACT", "SUBSTRING_ONLY", "MISMATCH", "OFFER_INVALID")


def _serving_features_cfg():
    """FeatureExtractor config for serving — emit EAN+ART states regardless of CE config."""
    return OmegaConf.create({
        "enabled": True,
        "text_mode": False,
        "slot_order": ["ean", "article"],
        "normalize": {"leading_zeros": "keep", "multivalue_separators": ",;|"},
        "ean": {
            "enabled": True, "offer_field": "ean", "validate": "gtin", "on_offer_invalid": "none",
        },
        "article": {
            "enabled": True,
            "offer_fields": ["article_number", "manufacturer_article_number"],
            "min_token_len": 4,
        },
        "spec": {"enabled": False},
    })


def _token_to_state(token: str) -> str:
    """`[EAN_MATCH]` -> `MATCH`, `[ART_SUBSTRING_ONLY]` -> `SUBSTRING_ONLY`."""
    inner = token.strip("[]")
    parts = inner.split("_", 1)
    return parts[1] if len(parts) > 1 else parts[0]


def _digit_jaccard(q: str, o: str) -> float:
    qs, os_ = set(_DIGIT_RUN.findall(q or "")), set(_DIGIT_RUN.findall(o or ""))
    if not qs or not os_:
        return 0.0
    return len(qs & os_) / len(qs | os_)


def _char3_jaccard(q: str, o: str) -> float:
    def grams(s: str):
        s = (s or "").lower()
        if len(s) < 3:
            return {s} if s else set()
        return {s[i : i + 3] for i in range(len(s) - 2)}
    qg, og = grams(q), grams(o)
    if not qg or not og:
        return 0.0
    return len(qg & og) / len(qg | og)


def _substring(q: str, o: str) -> int:
    q, o = (q or "").lower().strip(), (o or "").lower()
    return int(bool(q) and bool(o) and q in o)


def _engineered_features(extractor: FeatureExtractor, context: dict,
                         query_term: str, offer_name: str) -> dict:
    tokens = extractor.extract(context)
    state_by_slot = {slot: _token_to_state(tok) for slot, tok in zip(extractor.slot_order, tokens)}
    out = {}
    ean_state = state_by_slot.get("ean", "NONE")
    art_state = state_by_slot.get("article", "NONE")
    for s in EAN_STATES:
        out[f"ean_{s}"] = int(ean_state == s)
    for s in ART_STATES:
        out[f"art_{s}"] = int(art_state == s)
    out["lex_substring"] = _substring(query_term, offer_name)
    out["lex_digit_jaccard"] = _digit_jaccard(query_term, offer_name)
    out["lex_char3_jaccard"] = _char3_jaccard(query_term, offer_name)
    return out


def _add_list_features(rows: list[dict]) -> list[dict]:
    """Compute per-class rank/gap/zscore + group_size across this single query batch."""
    n = len(rows)
    for class_col in ("ce_p_exact", "ce_p_substitute", "ce_p_irrelevant"):
        values = np.array([r[class_col] for r in rows])
        # rank descending: 1 = best (matches pandas rank(method="dense"))
        order = (-values).argsort()
        ranks = np.empty(n, dtype=np.int32)
        # Dense rank: ties get the same rank, next distinct value gets +1.
        prev = None
        cur_rank = 0
        for idx in order:
            if prev is None or values[idx] != prev:
                cur_rank += 1
                prev = values[idx]
            ranks[idx] = cur_rank
        gmax = float(values.max())
        gmean = float(values.mean())
        gstd = float(values.std(ddof=1)) if n > 1 else 1.0
        if gstd == 0.0:
            gstd = 1.0
        for i, row in enumerate(rows):
            row[f"{class_col}_rank_desc"] = int(ranks[i])
            row[f"{class_col}_gap_from_max"] = float(gmax - values[i])
            row[f"{class_col}_zscore"] = float((values[i] - gmean) / gstd)
    for row in rows:
        row["group_size"] = n
    return rows

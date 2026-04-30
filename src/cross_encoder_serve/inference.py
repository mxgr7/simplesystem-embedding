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
    autocast_dtype: Optional[str] = None  # "bf16"|"fp16"|"fp32"|None (auto: bf16 on cuda)
    max_forward_batch: int = 256  # internal chunk size for the encoder forward;
    # 2000-offer requests at S=512 OOM on 24 GB cards if forwarded as one batch.
    # Output is bit-identical regardless (torch.cat over chunk logits).
    config_name: str = "cross_encoder"
    compile: bool = True  # torch.compile the encoder. Works best when chunk
    # size evenly divides 2000 (one shape for compile cache). ~-700 ms p95 on
    # 4090 over the eager baseline. Toggle off via SERVE_COMPILE=0 if needed.
    compile_mode: str = "max-autotune"  # "default" | "reduce-overhead" | "max-autotune"
    # max-autotune adds ~75s to first-request compile (kernel-tuning Triton matmul
    # configs) but shaves ~140 ms p95 vs "default" on this model.
    int8: bool = False  # apply torch.ao.quantization.quantize_dynamic to nn.Linear.
    # Note: quantize_dynamic uses CPU-only int8 matmul kernels; setting this forces
    # the model onto CPU. Useful for measuring quality drop at int8; for GPU int8
    # use the ONNX Runtime path (SERVE_RUNTIME=onnx) with an int8-quantized .onnx.
    runtime: str = "torch"  # "torch" | "onnx" — picks the encoder backend
    onnx_path: Optional[str] = None  # required when runtime=="onnx"
    onnx_providers: tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")
    # Set to ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
    # to attempt TRT EP first, falling back gracefully.


_AUTOCAST_ALIASES = {
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp16": torch.float16, "float16": torch.float16, "16": torch.float16,
    "fp32": None, "float32": None, "32": None, "off": None, "none": None, "": None,
}


def _resolve_autocast_dtype(name: Optional[str], device: str) -> Optional[torch.dtype]:
    """`None`/`"auto"` → bf16 on cuda, off elsewhere. Else look up in alias table."""
    if name is None or str(name).strip().lower() == "auto":
        return torch.bfloat16 if device == "cuda" else None
    key = str(name).strip().lower()
    if key not in _AUTOCAST_ALIASES:
        raise ValueError(f"Unsupported autocast dtype: {name!r}")
    return _AUTOCAST_ALIASES[key]


_DTYPE_LABEL = {torch.bfloat16: "bf16", torch.float16: "fp16"}


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
        self.runtime = cfg.runtime  # "torch" | "onnx"
        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

        with initialize_config_dir(config_dir=cfg.config_dir, version_base="1.3"):
            self.hcfg = compose(config_name=cfg.config_name)

        # Pick autocast dtype up-front (used by both runtimes for status
        # reporting; only the torch path actually wraps autocast contexts).
        self.autocast_dtype = _resolve_autocast_dtype(cfg.autocast_dtype, self.device)
        self.autocast_label = _DTYPE_LABEL.get(self.autocast_dtype, "fp32")

        self.model: Optional[CrossEncoderModule] = None
        self.ort_session = None  # set when runtime == "onnx"
        self.attn_implementation = "?"

        if self.runtime == "onnx":
            self._init_onnx()
        else:
            self._init_torch()

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

    # ------------------------------------------------------------------
    # Runtime-specific init paths
    # ------------------------------------------------------------------

    def _init_torch(self):
        """Load the Lightning ckpt → torch model with optional weight cast / compile / int8."""
        cfg = self.cfg
        logger.info("Loading CE checkpoint from %s", cfg.ckpt_path)
        # Override training cfg's model.compile=true so the encoder isn't pre-
        # wrapped, then strip the `_orig_mod.` prefix from the saved state_dict.
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

        # Cast weights to the autocast dtype. Halves weight memory traffic
        # (gelectra-large ≈ 1.3 GB fp32 → 0.67 GB bf16/fp16).
        if self.autocast_dtype is not None and self.device == "cuda":
            self.model.to(self.autocast_dtype)

        if cfg.int8:
            from torch.ao.quantization import quantize_dynamic
            logger.warning("SERVE_INT8=1: dynamic-quantizing nn.Linear; "
                           "forcing device=cpu (no CUDA kernels for qint8 mm).")
            self.model.to("cpu")
            self.model = quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            self.device = "cpu"
            self.autocast_dtype = None
            self.autocast_label = "int8"

        if cfg.compile and self.device == "cuda" and not cfg.int8:
            self.model.encoder = torch.compile(
                self.model.encoder, mode=cfg.compile_mode
            )

        self.attn_implementation = getattr(
            getattr(self.model.encoder, "config", None),
            "_attn_implementation",
            "?",
        )
        logger.info(
            "Reranker (torch) device=%s autocast=%s attn=%s",
            self.device, self.autocast_label, self.attn_implementation,
        )

    def _init_onnx(self):
        """Load ONNX session via onnxruntime-gpu. Tries the configured providers in order."""
        cfg = self.cfg
        if not cfg.onnx_path or not Path(cfg.onnx_path).exists():
            raise RuntimeError(f"SERVE_RUNTIME=onnx requires SERVE_ONNX_PATH; got {cfg.onnx_path!r}")
        import onnxruntime as ort
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Try providers in order; ORT silently falls back if a provider fails.
        providers: list = list(cfg.onnx_providers)
        # Optional TRT EP options (cache + fp16).
        prov_opts: list = []
        for p in providers:
            if p == "TensorrtExecutionProvider":
                prov_opts.append({
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(Path(cfg.onnx_path).with_suffix(".trtcache")),
                })
            elif p == "CUDAExecutionProvider":
                prov_opts.append({"arena_extend_strategy": "kSameAsRequested"})
            else:
                prov_opts.append({})
        logger.info("Loading ONNX %s with providers=%s", cfg.onnx_path, providers)
        self.ort_session = ort.InferenceSession(
            cfg.onnx_path, sess_opts, providers=providers, provider_options=prov_opts
        )
        active = self.ort_session.get_providers()
        logger.info("ONNX active providers: %s", active)
        self.attn_implementation = f"onnx:{active[0]}" if active else "onnx:?"
        # autocast/dtype is whatever the .onnx was exported as; the bench
        # /health endpoint reports "onnx" as a sentinel.
        self.autocast_label = "onnx"

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
        # Tokenize, then chunk-forward through the configured runtime. ONNX
        # path uses the ORT session; torch path autocast-wraps the LightningModule.
        # The fixture's worst-case S=512 means padding="max_length" pads to 512
        # for every chunk, which is what ONNX's exported graph expects.
        chunk = max(1, int(self.cfg.max_forward_batch))
        out_chunks: list[torch.Tensor] = []
        for start in range(0, len(offer_texts), chunk):
            sub = offer_texts[start : start + chunk]
            enc = self.tokenizer(
                [query_text] * len(sub),
                sub,
                padding="max_length" if self.runtime == "onnx" else True,
                truncation="only_second",
                max_length=self.max_pair_length,
                return_tensors="pt",
                return_token_type_ids=True,
            )
            if self.runtime == "onnx":
                # ORT eats numpy arrays directly; padding="max_length" keeps
                # shape stable at (chunk, 512) so the exported graph is reused.
                ort_inputs = {
                    "input_ids": enc["input_ids"].numpy(),
                    "attention_mask": enc["attention_mask"].numpy(),
                    "token_type_ids": enc["token_type_ids"].numpy(),
                }
                logits_np = self.ort_session.run(["logits"], ort_inputs)[0]
                out_chunks.append(torch.from_numpy(logits_np))
                continue
            enc = {k: v.to(self.device) for k, v in enc.items()}
            if self.autocast_dtype is not None and self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    out_chunks.append(self.model(enc))
            else:
                out_chunks.append(self.model(enc))
        return torch.cat(out_chunks, dim=0)

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

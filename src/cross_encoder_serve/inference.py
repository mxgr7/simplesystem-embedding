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
    max_forward_batch: int = 128  # internal chunk size for the encoder forward;
    # bench shows chunk=128 (B,S=128,512) is the 4090 sweet spot for the
    # distilled gelectra-base path — p95 3075 ms vs 4291 ms at chunk=256
    # and 4775 ms at chunk=1000. Bigger chunks thrash L2; smaller add launch
    # overhead. Output is bit-identical regardless of chunk size.
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
    bnb8: bool = False  # post-training INT8 quantization via bitsandbytes
    # (Linear8bitLt with has_fp16_weights=False). Replaces every nn.Linear inside
    # the encoder. Computes matmul in fp16 with int8 weights — saves weight-load
    # bandwidth and enables cuBLAS LtIgemm tensor cores. Quality drop typically
    # 0.005-0.02 on f1_exact for transformer models. Incompatible with
    # torch.compile (compile can't trace through bnb's custom ops); compile
    # auto-disabled when bnb8 is on.
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
        # Fast batched tokenize path used by `_pipelined_forward`. The HF
        # `tokenizer(...)` Python wrapper costs ~1000 ms of pure Python
        # overhead per 2000-pair call (arg validation, per-call padding logic,
        # tensor build) on top of the Rust BPE work. Calling
        # `tokenizer._tokenizer.encode_batch_fast(pairs)` directly skips all
        # that and returns native Encoding objects; we then assemble tensors
        # by hand with `np.empty(...).fill(e.ids)` for another ~150 ms of
        # savings vs `torch.tensor([list-comp])`. Padding/truncation are
        # pre-configured here once on the underlying Rust tokenizer; the
        # HF Python wrapper still works for the LGBM-on path which uses
        # the slow render+tokenize loop.
        self._fast_tok = self.tokenizer._tokenizer
        self._fast_tok.enable_padding(
            pad_id=self.tokenizer.pad_token_id,
            pad_token=self.tokenizer.pad_token,
            length=None,
        )
        self._fast_tok.enable_truncation(
            max_length=self.max_pair_length, strategy="only_second"
        )

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

        if cfg.bnb8 and self.device == "cuda" and not cfg.int8:
            # Walk the encoder module tree and swap nn.Linear → bnb.Linear8bitLt
            # in place. Skip the classifier head (NUM_CLASSES output is too small
            # to benefit; INT8 quantizing a 4-output linear can hurt more than
            # it saves). The first forward triggers per-row outlier extraction
            # (the threshold=6 mode of LLM.int8); subsequent forwards run the
            # cuBLAS LtIgemm path. Logs each replacement so the swap is
            # visible in the server log.
            import bitsandbytes as bnb
            replaced = 0
            def swap_linears(parent: torch.nn.Module, prefix: str = ""):
                nonlocal replaced
                for name, child in list(parent.named_children()):
                    full = f"{prefix}.{name}" if prefix else name
                    if isinstance(child, torch.nn.Linear):
                        new = bnb.nn.Linear8bitLt(
                            child.in_features,
                            child.out_features,
                            bias=child.bias is not None,
                            has_fp16_weights=False,
                            threshold=6.0,
                        )
                        # Convert weights to int8 representation by assigning
                        # an Int8Params holding the original fp16/bf16 tensor.
                        new.weight = bnb.nn.Int8Params(
                            child.weight.data,
                            has_fp16_weights=False,
                            requires_grad=False,
                        )
                        if child.bias is not None:
                            new.bias.data = child.bias.data.clone()
                        new = new.to(self.device)
                        setattr(parent, name, new)
                        replaced += 1
                    else:
                        swap_linears(child, full)
            swap_linears(self.model.encoder)
            logger.warning("SERVE_BNB8=1: replaced %d nn.Linear → bnb.Linear8bitLt", replaced)
            self.autocast_label = "int8"

        if cfg.compile and self.device == "cuda" and not cfg.int8 and not cfg.bnb8:
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
        import os, time
        debug = os.environ.get("SERVE_DEBUG_TIMING", "0") == "1"
        if not offers:
            return []

        if self.booster is None and self.runtime == "torch":
            # Fast path: pipeline CPU render+tokenize with GPU forward so the
            # ~800 ms prep cost is hidden behind the ~1.9 s forward instead
            # of being serial in front of it.
            t0 = time.perf_counter() if debug else 0
            logits = self._pipelined_forward(query, offers).float()
            t1 = time.perf_counter() if debug else 0
            p_exact_cal = F.softmax(logits / self.cfg.temperature, dim=-1)[:, EXACT_IDX].cpu().numpy()
            t2 = time.perf_counter() if debug else 0
            out = [
                OfferScore(
                    offer_id=str(offer.get("offer_id", "")),
                    p_irrelevant=0.0, p_complement=0.0, p_substitute=0.0, p_exact=0.0,
                    p_exact_calibrated=float(p_cal),
                )
                for offer, p_cal in zip(offers, p_exact_cal)
            ]
            if debug:
                t3 = time.perf_counter()
                logger.warning(
                    "[stage] pipelined_forward=%.0fms calibration=%.0fms response_objs=%.0fms"
                    " | rerank_total=%.0fms",
                    (t1 - t0) * 1000, (t2 - t1) * 1000,
                    (t3 - t2) * 1000, (t3 - t0) * 1000,
                )
            return out

        # 1. Build query/offer texts the same way training did.
        query_text, offer_texts, contexts = self._prepare_texts(query, offers)

        # 2. CE forward → logits.
        logits = self._forward(query_text, offer_texts).float()  # (n_offers, 4)

        # 3. CE output. The full 4-class softmax is only needed for the LGBM
        # stack ensemble; downstream consumes only `p_exact_calibrated`. When
        # LGBM is off (the production serving config), skip the full-softmax
        # path entirely — saves a ~85 ms cuda→cpu sync per request.
        if self.booster is not None:
            probs_raw = F.softmax(logits, dim=-1).cpu().numpy()
            p_exact_cal = F.softmax(logits / self.cfg.temperature, dim=-1)[:, EXACT_IDX].cpu().numpy()
            lgbm_probs = self._lgbm_predict(probs_raw, contexts, query, offers)
            probs_out = self.cfg.ensemble_w * probs_raw + (1.0 - self.cfg.ensemble_w) * lgbm_probs
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

        # LGBM-off fast path: only compute p_exact_calibrated; zero-fill the
        # other 4-class probs (downstream ignores them; the API schema keeps
        # them only for compatibility with the previous response shape).
        p_exact_cal = F.softmax(logits / self.cfg.temperature, dim=-1)[:, EXACT_IDX].cpu().numpy()
        return [
            OfferScore(
                offer_id=str(offer.get("offer_id", "")),
                p_irrelevant=0.0,
                p_complement=0.0,
                p_substitute=0.0,
                p_exact=0.0,
                p_exact_calibrated=float(p_cal),
            )
            for offer, p_cal in zip(offers, p_exact_cal)
        ]

    def _prepare_texts(self, query: str, offers: list[dict]):
        # Fast path replacing Jinja-based rendering. The Jinja loop costs ~1.3
        # ms/offer (≈2.5s for a 2000-offer request) and was *bigger* than the
        # GPU forward; this hand-rolled f-string builder mirrors offer_template
        # in configs/data/cross_encoder.yaml exactly. The renderer is still
        # used only to generate the per-pair `context` for the LGBM stack
        # (cheap; only when LGBM is loaded).
        if self.booster is not None:
            # Old slow path for the LGBM-on configuration — keeps the LGBM
            # feature contracts intact (engineered features rely on context).
            contexts = []
            offer_texts = []
            for offer in offers:
                row = {**offer, "query_term": query}
                ctx = self.renderer.build_context(row)
                contexts.append(ctx)
                offer_texts.append(self.renderer.render_offer_text(row, context=ctx))
            first_row = {**offers[0], "query_term": query} if offers else {"query_term": query}
            query_text = self.renderer.render_query_text(
                first_row, context=contexts[0] if contexts else None
            )
            return query_text, offer_texts, contexts

        # Lightweight str-only path (LGBM off — no need for context dicts).
        clean_html = bool(getattr(self.hcfg.data, "clean_html", True))
        offer_texts = [_render_offer_fast(o, clean_html) for o in offers]
        query_text = _normalize_text_fast(query)
        return query_text, offer_texts, []

    def _pipelined_forward(self, query: str, offers: list[dict]) -> torch.Tensor:
        """Render+tokenize chunk i+1 on a worker thread while the GPU forwards
        chunk i. Hides the ~800 ms prep cost behind the GPU's ~1.9 s forward
        on the distilled gelectra-base path. Only used when LGBM is off and
        runtime is torch (the production hot path)."""
        from concurrent.futures import ThreadPoolExecutor
        import os as _os
        import time as _time
        debug = _os.environ.get("SERVE_DEBUG_TIMING", "0") == "1"
        # Per-stage CPU-only timers — sum across all chunks. The GPU forward
        # timer is wall-clock (it overlaps with prep on a worker thread).
        sum_render = sum_tokenize = sum_h2d = sum_forward = sum_wait = 0.0
        wall_t0 = _time.perf_counter() if debug else 0
        chunk_size = max(1, int(self.cfg.max_forward_batch))
        n = len(offers)
        clean_html = bool(getattr(self.hcfg.data, "clean_html", True))
        query_text = _normalize_text_fast(query)

        def render_tok_h2d(offers_chunk):
            nonlocal sum_render, sum_tokenize, sum_h2d
            tr0 = _time.perf_counter() if debug else 0
            texts = [_render_offer_fast(o, clean_html) for o in offers_chunk]
            tr1 = _time.perf_counter() if debug else 0
            # Direct Rust call. `encode_batch_fast` releases the GIL and
            # rayon-parallelizes over pairs. Padding+truncation come from the
            # `enable_padding` / `enable_truncation` settings configured at
            # init. Output is a list of native Encoding objects.
            pairs = list(zip([query_text] * len(texts), texts))
            encs = self._fast_tok.encode_batch_fast(pairs, add_special_tokens=True)
            # Assemble into numpy arrays via pre-allocated buffers, which is
            # ~2× faster than `torch.tensor([e.ids for e in encs])` because
            # the latter has to walk the whole list-of-lists in Python before
            # the constructor sees it. `from_numpy` is zero-copy.
            n_pairs = len(encs)
            seq_len = len(encs[0].ids) if encs else 0
            ids_np = np.empty((n_pairs, seq_len), dtype=np.int64)
            mask_np = np.empty((n_pairs, seq_len), dtype=np.int64)
            ttype_np = np.empty((n_pairs, seq_len), dtype=np.int64)
            for j, e in enumerate(encs):
                ids_np[j] = e.ids
                mask_np[j] = e.attention_mask
                ttype_np[j] = e.type_ids
            input_ids = torch.from_numpy(ids_np)
            attention_mask = torch.from_numpy(mask_np)
            token_type_ids = torch.from_numpy(ttype_np)
            tr2 = _time.perf_counter() if debug else 0
            out = {
                "input_ids": input_ids.to(self.device, non_blocking=True),
                "attention_mask": attention_mask.to(self.device, non_blocking=True),
                "token_type_ids": token_type_ids.to(self.device, non_blocking=True),
            }
            tr3 = _time.perf_counter() if debug else 0
            if debug:
                sum_render += tr1 - tr0
                sum_tokenize += tr2 - tr1
                sum_h2d += tr3 - tr2
            return out

        out_chunks: list[torch.Tensor] = []
        chunks = [offers[i : i + chunk_size] for i in range(0, n, chunk_size)]
        with ThreadPoolExecutor(max_workers=1) as ex:
            next_future = ex.submit(render_tok_h2d, chunks[0])
            for i, _ in enumerate(chunks):
                tw0 = _time.perf_counter() if debug else 0
                enc = next_future.result()
                if debug:
                    sum_wait += _time.perf_counter() - tw0
                if i + 1 < len(chunks):
                    next_future = ex.submit(render_tok_h2d, chunks[i + 1])
                tf0 = _time.perf_counter() if debug else 0
                if self.autocast_dtype is not None and self.device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                        out_chunks.append(self.model(enc))
                else:
                    out_chunks.append(self.model(enc))
                if debug:
                    sum_forward += _time.perf_counter() - tf0
        result = torch.cat(out_chunks, dim=0)
        if debug:
            wall = _time.perf_counter() - wall_t0
            # `sum_wait` is the main thread's total time blocked waiting on the
            # worker thread's prep — i.e., the time the GPU spent idle because
            # prep was the critical path. `sum_forward` is the main thread's
            # GPU-forward dispatch time across 16 chunks. The render/tokenize/h2d
            # sums are worker-thread CPU time across 16 chunks; they overlap
            # with GPU forward via the ThreadPoolExecutor pipeline.
            bubble = max(0.0, sum_wait - sum_forward)  # GPU idle bubble
            logger.warning(
                "[pipe] wall=%.0fms wait=%.0fms forward(GPU dispatch)=%.0fms"
                " bubble=%.0fms | sum_render=%.0fms sum_tokenize=%.0fms"
                " sum_h2d=%.0fms | n_chunks=%d chunk_size=%d",
                wall * 1000, sum_wait * 1000, sum_forward * 1000, bubble * 1000,
                sum_render * 1000, sum_tokenize * 1000, sum_h2d * 1000,
                len(chunks), chunk_size,
            )
        return result

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

_NORMALIZE_REPLACE_CHARS = ("\x00", "\xa0", "\r\n")
_NORMALIZE_SPACE_CHARS = "\t\r\f\v"


def _normalize_text_fast(value):
    """Drop-in equivalent of embedding_train.text.normalize_text with cheap
    pre-checks. The 3 regex passes inside normalize_text cost ~0.45 ms each
    on an 8 KB string but most pre-rendered descriptions have nothing for
    them to match (single-spaces, \\n already canonical). Skipping the
    no-op subs saves ~700 ms across 2000 descriptions on the bench fixture.

    Output is byte-identical to normalize_text — verified on the fixture.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    if any(needle in value for needle in _NORMALIZE_REPLACE_CHARS):
        value = (value
                 .replace("\x00", " ")
                 .replace("\xa0", " ")
                 .replace("\r\n", "\n"))
    value = value.strip()
    if "  " in value or any(c in value for c in _NORMALIZE_SPACE_CHARS):
        value = _NORMALIZE_SPACE_RE.sub(" ", value)
    if " \n" in value or "\n " in value:
        value = _NORMALIZE_SPACE_NL_RE.sub("\n", value)
    if "\n\n\n" in value:
        value = _NORMALIZE_MULTI_NL_RE.sub("\n\n", value)
    return value.strip()


def _clean_html_text_fast(value):
    """Equivalent of embedding_train.text.clean_html_text using the fast
    normalize. Same fast-path: if no HTML markers, just normalize once."""
    value = _normalize_text_fast(value)
    if not value or ("<" not in value and "&" not in value):
        return value
    value = _BR_RE_FAST.sub("\n", value)
    value = _TAG_RE_FAST.sub(" ", value)
    import html as _html
    value = _html.unescape(value)
    return _normalize_text_fast(value)


_NORMALIZE_SPACE_RE = re.compile(r"[ \t\r\f\v]+")
_NORMALIZE_SPACE_NL_RE = re.compile(r" *\n *")
_NORMALIZE_MULTI_NL_RE = re.compile(r"\n{3,}")
# NB: the slow path's _BR_RE in embedding_train/text.py uses `\\s` (literal
# backslash-s) which never matches an actual whitespace char — so <br/>,
# <br />, <BR/> all fall through to _TAG_RE and are replaced with a space,
# not a newline. We mirror that behavior here for byte-identical output.
_BR_RE_FAST = re.compile(r"(?i)<br\\s*/?>")
_TAG_RE_FAST = re.compile(r"<[^>]+>")


# Description char cap before normalize+clean_html. The tokenizer already
# truncates the offer side to max_length=512 tokens (`truncation="only_second"`)
# which is roughly 3000 chars of German text; processing more is wasted.
# 4096 chars gives ~50 % margin for tokenizer over-counting and rare
# multi-byte char clusters while still cutting normalize+clean_html work
# in half on the bench filler-padded fixture (avg 8337 chars/desc).
# normalize_text is prefix-preserving (its operations — strip, ASCII-replace,
# whitespace collapse — operate locally on each char/whitespace cluster, so
# normalize(s[:N]) is a prefix of normalize(s)). This guarantees the first
# 512 tokens of normalize(desc[:4096]) and normalize(desc) are identical.
_DESC_CHAR_CAP = 4096


def _render_offer_fast(offer: dict, clean_html: bool) -> str:
    """Hand-rolled equivalent of configs/data/cross_encoder.yaml::offer_template
    + RowTextRenderer.build_context normalization.

    Mirrors the slow renderer's *post-normalized* output directly — first
    conditional after "Artikel: ..." gets NO leading space (slow's
    `_SPACE_NL_RE` collapses ` \\n ` → `\\n`); subsequent conditionals get a
    leading space. Each individually-rendered field is normalized once (not
    twice via final-pass normalize_text). Verified byte-equal to the slow
    renderer on the full 2000-offer bench fixture (with the description char
    cap applied — see _DESC_CHAR_CAP comment above).
    """
    normalize_text = _normalize_text_fast
    clean_html_text = _clean_html_text_fast
    name = normalize_text(offer.get("name"))
    body = f"Artikel: {name}\n"
    first = True

    def _add(label, value, sep_after_first=" "):
        nonlocal body, first
        if not value:
            return
        body += f"{label}: {value}" if first else f"{sep_after_first}{label}: {value}"
        first = False

    # All free-text fields go through normalize_text so we don't need a final
    # body-wide pass (which costs ~1ms/offer on the 8 KB filler-padded body).
    _add("EAN", normalize_text(offer.get("ean")))
    _add("Artikelnummer", normalize_text(offer.get("article_number")))
    _add("Herstellernummer", normalize_text(offer.get("manufacturer_article_number")))
    cat = offer.get("category_paths")
    if cat:
        _add("Kategorie", normalize_text(cat))
    _add("Artikeltyp", normalize_text(offer.get("manufacturer_article_type")))
    _add("Marke", normalize_text(offer.get("manufacturer_name")))
    desc = offer.get("description")
    if desc:
        # Cheap pre-truncate: cap at _DESC_CHAR_CAP before calling normalize
        # or clean_html so the regex passes don't process discarded tail.
        # Safe because tokenizer truncates the offer side to 512 tokens
        # anyway and normalize is prefix-preserving.
        if len(desc) > _DESC_CHAR_CAP:
            desc = desc[:_DESC_CHAR_CAP]
        desc = clean_html_text(desc) if clean_html else normalize_text(desc)
        if desc:
            _add("Beschreibung", desc)
    return body


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

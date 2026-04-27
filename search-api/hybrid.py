"""Hybrid search orchestration over a dense vector collection and a BM25
codes collection. Implements `hybrid_v0.md` faithfully.

Modes (`SearchParams.mode`):
  vector             — dense ANN only
  bm25               — BM25 over offers_codes only
  hybrid             — dense + bm25, RRF fused; classifier NOT consulted
  hybrid_classified  — classifier picks strict path (BM25-only, large limit)
                       or hybrid; with optional 0-result fallback to hybrid

The classifier patterns mirror §"Query classifier" exactly. Length floor 4
prevents trivial 2–3 char matches. False positives are caught by the
0-result fallback; false negatives still fall through to hybrid where BM25
will pick them up — see §"Asymmetric error handling".

Strict-path tied-score order: doc says product-level secondary sort is out
of scope. We add a deterministic ascending-`id` tiebreaker so A/B logging
is reproducible, nothing more.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Sequence

import numpy as np
from pymilvus import MilvusClient


# ──────────────────────────────────────────────────────────────────────
# Classifier (§"Query classifier")
# ──────────────────────────────────────────────────────────────────────
#
# Tightened from the original v0 patterns: short generic tokens like rj45,
# wd-40, cr2032, ffp2 were getting routed to the strict path, which gives
# the wrong cardinal answer (BM25 atomic-token retrieval returns ≤ a
# handful of hits, dense handles them well). The strict path now requires
# both length ≥7 AND a meaningful digit count, which excludes those
# industry-generic tokens while still catching real opaque MPNs and
# hyphenated codes (tze-231, gtb6-p5211, 221-413, e1987303). A static
# denylist catches anything that would still slip through.

ID_PATTERNS = [
    r"\d{8}",                                                    # EAN-8
    r"\d{12,14}",                                                # UPC-A / EAN-13 / GTIN-14
    # Hyphenated: ≥7 chars total AND ≥3 digits anywhere.
    r"(?=.{7,}$)(?=(?:[^\d]*\d){3,})[a-z0-9]+(?:-[a-z0-9]+)+",
    # Alpha-then-digit: ≥7 chars total AND ≥4 consecutive digits after the
    # letter prefix.
    r"(?=.{7,}$)[a-z]+\d{4,}[a-z0-9]*",
]
_ID_RE = re.compile("|".join(f"^{p}$" for p in ID_PATTERNS), re.IGNORECASE)

# Industry-generic tokens that pass shape checks but route incorrectly to
# strict (the right answer is a dense+BM25 hybrid). Empirically grounded —
# extend as new offenders surface in the query logs. Match is case- and
# whitespace-insensitive (handled by is_strict_identifier's normalisation).
GENERIC_TOKENS = frozenset({
    # Coin / button cell batteries
    "cr2032", "cr2025", "cr2016", "cr1632", "cr1620",
    "lr44", "lr41", "lr1130", "sr44", "sr41",
    # Network / data connectors
    "rj45", "rj11", "rj12",
    "usb-c", "usb-a", "usb-b", "hdmi", "displayport", "vga", "dvi-d",
    # Twisted-pair cable categories
    "cat5", "cat5e", "cat6", "cat6a", "cat7", "cat8",
    # Respirator filter classes
    "ffp1", "ffp2", "ffp3", "n95", "n99", "kn95",
    # Lubricants
    "wd-40", "wd40",
    # Metric thread sizes
    "m3", "m4", "m5", "m6", "m8", "m10", "m12", "m16", "m20",
    # Ingress-protection ratings
    "ip54", "ip65", "ip66", "ip67", "ip68",
})


def is_strict_identifier(q: str) -> bool:
    q = q.strip().lower()
    if q in GENERIC_TOKENS:
        return False
    if not (4 <= len(q) <= 40):
        return False
    return bool(_ID_RE.fullmatch(q))


# ──────────────────────────────────────────────────────────────────────
# Public types
# ──────────────────────────────────────────────────────────────────────

class Mode(str, Enum):
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
    HYBRID_CLASSIFIED = "hybrid_classified"


@dataclass(slots=True)
class SearchParams:
    mode: Mode = Mode.HYBRID_CLASSIFIED
    k: int = 24                       # final top_n returned to caller
    dense_limit: int = 200            # candidate pool from dense in hybrid path
    codes_limit: int = 20             # candidate pool from codes in hybrid path
    strict_codes_limit: int = 500     # codes pool in strict path
    rrf_k: int = 60
    num_candidates: int | None = None # HNSW efSearch
    enable_fallback: bool = True


@dataclass(slots=True)
class Hit:
    id: str
    score: float
    source: str                        # "dense" | "bm25" | "rrf"


# ──────────────────────────────────────────────────────────────────────
# Atoms — single-leg searches
# ──────────────────────────────────────────────────────────────────────

def _dense_search(
    client: MilvusClient,
    collection: str,
    *,
    vec: Sequence[float],
    limit: int,
    num_candidates: int | None,
    id_field: str,
    vector_field: str = "offer_embedding",
) -> list[tuple[str, float]]:
    # Collection stores fp16 vectors; matching the query precision flushes
    # subnormals to 0 instead of tripping Milvus's underflow validator.
    query = np.asarray(vec, dtype=np.float16)
    params: dict = {}
    if num_candidates is not None and num_candidates > 0:
        params["ef"] = num_candidates
    res = client.search(
        collection_name=collection,
        data=[query],
        anns_field=vector_field,
        limit=limit,
        search_params={"metric_type": "COSINE", "params": params},
        output_fields=[id_field],
    )
    raw = res[0] if res else []
    out: list[tuple[str, float]] = []
    for h in raw:
        ent = h.get("entity", {}) if isinstance(h, dict) else {}
        out.append((str(ent.get(id_field, "")), float(h["distance"])))
    return out


def _bm25_search(
    client: MilvusClient,
    collection: str,
    *,
    text: str,
    limit: int,
    sparse_field: str = "sparse_codes",
    id_field: str = "id",
    tied_id_asc: bool = False,
) -> list[tuple[str, float]]:
    res = client.search(
        collection_name=collection,
        data=[text],
        anns_field=sparse_field,
        limit=limit,
        search_params={"metric_type": "BM25"},
        output_fields=[id_field],
    )
    raw = res[0] if res else []
    out: list[tuple[str, float]] = []
    for h in raw:
        ent = h.get("entity", {}) if isinstance(h, dict) else {}
        out.append((str(ent.get(id_field, "")), float(h["distance"])))
    if tied_id_asc:
        # Stable secondary sort: by score desc, then id asc within ties.
        out.sort(key=lambda r: (-r[1], r[0]))
    return out


# ──────────────────────────────────────────────────────────────────────
# RRF fusion
# ──────────────────────────────────────────────────────────────────────

def rrf_merge(
    result_lists: Sequence[Sequence[tuple[str, float]]],
    k: int,
    top_n: int,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion — parameter-free over per-list scores. Ranks
    are 1-based; identical scores within a list are not flattened (callers
    that care can pre-sort)."""
    scores: dict[str, float] = defaultdict(float)
    for hits in result_lists:
        for rank, (hid, _score) in enumerate(hits, start=1):
            scores[hid] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])[:top_n]


# ──────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────

EmbedFn = Callable[[str], Awaitable[list[float]]]


@dataclass(slots=True)
class _LegTimings:
    dense_ms: float | None = None
    codes_ms: float | None = None
    embed_ms: float | None = None
    dense_hits: int | None = None
    codes_hits: int | None = None
    classifier_strict: bool | None = None
    fallback_fired: bool = False
    path: str = ""                     # "vector" | "bm25" | "hybrid" |
                                       # "strict" | "fallback"


async def run_search(
    q: str,
    params: SearchParams,
    *,
    dense_client: MilvusClient,
    codes_client: MilvusClient,
    embed: EmbedFn,
    dense_collection: str = "offers",
    codes_collection: str = "offers_codes",
    dense_id_field: str = "id",
) -> tuple[list[Hit], dict]:
    q = q.strip()
    timings = _LegTimings()
    if not q:
        return [], _debug_dict(timings, params)

    async def do_dense() -> list[tuple[str, float]]:
        t0 = time.perf_counter()
        vec = await embed(q)
        timings.embed_ms = (time.perf_counter() - t0) * 1000
        if not vec:
            return []
        t1 = time.perf_counter()
        out = await asyncio.to_thread(
            _dense_search, dense_client, dense_collection,
            vec=vec, limit=params.dense_limit,
            num_candidates=params.num_candidates,
            id_field=dense_id_field,
        )
        timings.dense_ms = (time.perf_counter() - t1) * 1000
        timings.dense_hits = len(out)
        return out

    async def do_bm25(limit: int, *, tied_id_asc: bool) -> list[tuple[str, float]]:
        t0 = time.perf_counter()
        out = await asyncio.to_thread(
            _bm25_search, codes_client, codes_collection,
            text=q.lower(), limit=limit, tied_id_asc=tied_id_asc,
        )
        timings.codes_ms = (time.perf_counter() - t0) * 1000
        timings.codes_hits = len(out)
        return out

    # ── mode dispatch ──
    if params.mode == Mode.VECTOR:
        timings.path = "vector"
        dense = await do_dense()
        return _to_hits(dense[: params.k], "dense"), _debug_dict(timings, params)

    if params.mode == Mode.BM25:
        timings.path = "bm25"
        bm = await do_bm25(params.codes_limit, tied_id_asc=True)
        return _to_hits(bm[: params.k], "bm25"), _debug_dict(timings, params)

    if params.mode == Mode.HYBRID:
        timings.path = "hybrid"
        dense, bm = await asyncio.gather(
            do_dense(),
            do_bm25(params.codes_limit, tied_id_asc=False),
        )
        fused = rrf_merge([dense, bm], k=params.rrf_k, top_n=params.k)
        return _to_hits(fused, "rrf"), _debug_dict(timings, params)

    # HYBRID_CLASSIFIED
    timings.classifier_strict = is_strict_identifier(q)
    if timings.classifier_strict:
        timings.path = "strict"
        bm = await do_bm25(params.strict_codes_limit, tied_id_asc=True)
        if bm:
            return _to_hits(bm[: params.k], "bm25"), _debug_dict(timings, params)
        if not params.enable_fallback:
            return [], _debug_dict(timings, params)
        timings.fallback_fired = True
        timings.path = "fallback"
        # fall through to hybrid

    # hybrid path (default route OR fallback after empty strict)
    if timings.path == "":
        timings.path = "hybrid"
    dense, bm = await asyncio.gather(
        do_dense(),
        do_bm25(params.codes_limit, tied_id_asc=False),
    )
    fused = rrf_merge([dense, bm], k=params.rrf_k, top_n=params.k)
    return _to_hits(fused, "rrf"), _debug_dict(timings, params)


def _to_hits(rows: Sequence[tuple[str, float]], source: str) -> list[Hit]:
    return [Hit(id=hid, score=score, source=source) for hid, score in rows]


def _debug_dict(t: _LegTimings, p: SearchParams) -> dict:
    return {
        "path": t.path,
        "classifier_strict": t.classifier_strict,
        "fallback_fired": t.fallback_fired,
        "embed_ms": _round(t.embed_ms),
        "dense_ms": _round(t.dense_ms),
        "codes_ms": _round(t.codes_ms),
        "dense_hits": t.dense_hits,
        "codes_hits": t.codes_hits,
        "params": {
            "mode": p.mode.value,
            "k": p.k,
            "dense_limit": p.dense_limit,
            "codes_limit": p.codes_limit,
            "strict_codes_limit": p.strict_codes_limit,
            "rrf_k": p.rrf_k,
            "num_candidates": p.num_candidates,
            "enable_fallback": p.enable_fallback,
        },
    }


def _round(v: float | None, digits: int = 1) -> float | None:
    return None if v is None else round(v, digits)

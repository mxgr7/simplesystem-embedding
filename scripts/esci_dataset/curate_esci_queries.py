#!/usr/bin/env python3
"""Curate a 20K-query subset for an ESCI-style relevance dataset.

Input:  /data/datasets/posthog_queries.parquet/<YYYY-MM-DD>.parquet (90 days)
Output: /data/datasets/queries_offers_esci/queries.parquet

Pipeline:
  1. aggregate per dedup key (lower(normalizedQueryTerm) || lower(qt))
  2. quality + bot filters
  3. stratify by (frequency_band x hit_band), sample 20K with hit-band
     targets {zero:25%, 1-9:25%, 10-99:20%, 100-999:15%, 1000+:15%}
  4. ensure >=2000 MPN-shaped queries (low-hit-band swap-in if short)
"""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

IN_ROOT = Path("/data/datasets/posthog_queries.parquet")
OUT_DIR = Path("/data/datasets/queries_offers_esci")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "queries.parquet"

TARGET = 20_000
HIT_TARGETS = {
    "zero":    int(TARGET * 0.25),  # 5000
    "1-9":     int(TARGET * 0.25),  # 5000
    "10-99":   int(TARGET * 0.20),  # 4000
    "100-999": int(TARGET * 0.15),  # 3000
    "1000+":   int(TARGET * 0.15),  # 3000
}
FREQ_QUOTA = {"head": 0.05, "torso": 0.25, "tail": 0.70}  # within each hit band
MPN_FLOOR = 2_000

ALPHA_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿẞ]")
DIGIT_RE = re.compile(r"\d")
MPN_RE = re.compile(r"^[a-z0-9][a-z0-9\-_/.]{2,}$", re.I)
SEED = 0


def is_mpn(s: str) -> bool:
    return (bool(MPN_RE.match(s))
            and any(c.isdigit() for c in s)
            and any(c.isalpha() for c in s))


def freq_band(n: int) -> str:
    if n >= 100: return "head"
    if n >= 5:   return "torso"
    return "tail"


def hit_band(h):
    if h is None: return "unknown"
    if h == 0:    return "zero"
    if h < 10:    return "1-9"
    if h < 100:   return "10-99"
    if h < 1000:  return "100-999"
    return "1000+"


def aggregate() -> dict:
    """Build per-key aggregate from all 90 daily files."""
    agg: dict[str, dict] = {}
    files = sorted(IN_ROOT.glob("*.parquet"))
    print(f"[agg] reading {len(files)} files...")
    for i, p in enumerate(files):
        t = pq.read_table(p, columns=["uuid", "timestamp", "qt",
                                      "distinct_id", "properties"])
        uuids = t["uuid"].to_pylist()
        tss = t["timestamp"].to_pylist()
        qts = t["qt"].to_pylist()
        dids = t["distinct_id"].to_pylist()
        props = t["properties"].to_pylist()
        for uuid, ts, qt, did, s in zip(uuids, tss, qts, dids, props):
            if not qt:
                continue
            try:
                d = json.loads(s)
            except Exception:
                continue
            nqt = d.get("normalizedQueryTerm")
            base = nqt if nqt else qt
            key = base.strip().lower() if base else None
            if not key:
                continue
            rec = agg.get(key)
            if rec is None:
                rec = {"events": 0, "users": set(),
                       "last_ts": ts, "last_uuid": uuid,
                       "last_qt": d.get("queryTerm") or qt,
                       "last_hits": (d.get("searchResults") or {}).get("hitCount"),
                       "lang": d.get("platformLanguage")}
                agg[key] = rec
            rec["events"] += 1
            rec["users"].add(did)
            if ts > rec["last_ts"]:
                rec["last_ts"] = ts
                rec["last_uuid"] = uuid
                rec["last_qt"] = d.get("queryTerm") or qt
                rec["last_hits"] = (d.get("searchResults") or {}).get("hitCount")
                rec["lang"] = d.get("platformLanguage")
        if (i + 1) % 15 == 0:
            print(f"  {i+1}/{len(files)} processed, {len(agg):,} keys so far")
    print(f"[agg] {len(agg):,} distinct keys")
    return agg


def filter_universe(agg: dict) -> list[dict]:
    kept = []
    drop_len, drop_noise, drop_bot = 0, 0, 0
    for k, r in agg.items():
        if not (3 <= len(k) <= 200):
            drop_len += 1; continue
        # Keep if it has a letter, OR if it has a digit AND is len>=4
        # (the latter preserves article-number / EAN / opaque-MPN queries)
        if not (ALPHA_RE.search(k) or (DIGIT_RE.search(k) and len(k) >= 4)):
            drop_noise += 1; continue
        n_users = len(r["users"])
        if r["events"] >= 50 and n_users <= 3:
            drop_bot += 1; continue
        kept.append({
            "key": k,
            "events": r["events"],
            "distinct_users": n_users,
            "last_uuid": r["last_uuid"],
            "last_ts": r["last_ts"],
            "last_qt": r["last_qt"],
            "last_hits": r["last_hits"],
            "lang": r["lang"],
            "freq_band": freq_band(r["events"]),
            "hit_band": hit_band(r["last_hits"]),
            "mpn": is_mpn(k),
        })
    print(f"[filter] kept {len(kept):,}; dropped len={drop_len:,} "
          f"noise={drop_noise:,} bot={drop_bot:,}")
    return kept


def sample(universe: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    by_hb_fb: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for q in universe:
        if q["hit_band"] == "unknown":
            continue
        by_hb_fb[q["hit_band"]][q["freq_band"]].append(q)

    print("[sample] universe per cell (hit_band x freq_band):")
    for hb in HIT_TARGETS:
        print(f"  {hb:8} "
              + " ".join(f"{fb}={len(by_hb_fb[hb][fb]):,}"
                         for fb in ("head", "torso", "tail")))

    selected: list[dict] = []
    for hb, target in HIT_TARGETS.items():
        pools = {fb: list(by_hb_fb[hb][fb]) for fb in ("head", "torso", "tail")}
        for fb in pools:
            rng.shuffle(pools[fb])

        take_head = pools["head"][: int(target * FREQ_QUOTA["head"])]
        take_torso = pools["torso"][: int(target * FREQ_QUOTA["torso"])]
        remaining_target = target - len(take_head) - len(take_torso)
        take_tail = pools["tail"][: remaining_target]
        take = take_head + take_torso + take_tail

        # If we under-filled (small head/torso), spill from any band
        if len(take) < target:
            leftovers = (
                pools["head"][len(take_head):]
                + pools["torso"][len(take_torso):]
                + pools["tail"][len(take_tail):]
            )
            rng.shuffle(leftovers)
            take += leftovers[: target - len(take)]
        selected.extend(take)
        print(f"  -> {hb:8}: head={len(take_head)} torso={len(take_torso)} "
              f"tail={len(take_tail)} (total {len(take)}/{target})")

    # MPN floor: if we are short, swap tail non-MPN for tail MPN within same hit_band
    mpn_count = sum(1 for r in selected if r["mpn"])
    print(f"[sample] MPN-shaped in selection: {mpn_count} (floor {MPN_FLOOR})")
    if mpn_count < MPN_FLOOR:
        deficit = MPN_FLOOR - mpn_count
        # Build sets for fast lookup
        chosen_keys = {r["key"] for r in selected}
        # Pull additional MPN candidates not yet chosen (any band)
        spare_mpn = [q for q in universe
                     if q["mpn"] and q["key"] not in chosen_keys
                     and q["hit_band"] != "unknown"]
        rng.shuffle(spare_mpn)
        # Find swap-out victims: non-MPN tail rows
        non_mpn_tail = [i for i, r in enumerate(selected)
                        if not r["mpn"] and r["freq_band"] == "tail"]
        rng.shuffle(non_mpn_tail)
        n_swap = min(deficit, len(spare_mpn), len(non_mpn_tail))
        for i in range(n_swap):
            selected[non_mpn_tail[i]] = spare_mpn[i]
        print(f"[sample] swapped in {n_swap} additional MPN-shaped queries")

    return selected


def write_parquet(selected: list[dict]) -> None:
    selected.sort(key=lambda r: r["last_ts"])
    n = len(selected)
    table = pa.table({
        "query_id": pa.array(list(range(1, n + 1)), type=pa.int32()),
        "normalized_qt": [r["key"] for r in selected],
        "qt_raw": [r["last_qt"] for r in selected],
        "source_event_uuid": [r["last_uuid"] for r in selected],
        "source_event_ts": pa.array([r["last_ts"] for r in selected],
                                    type=pa.timestamp("us", tz="UTC")),
        "hit_count_at_search_time": pa.array(
            [r["last_hits"] for r in selected], type=pa.int64()),
        "events_count": pa.array([r["events"] for r in selected], type=pa.int32()),
        "distinct_users": pa.array(
            [r["distinct_users"] for r in selected], type=pa.int32()),
        "frequency_band": [r["freq_band"] for r in selected],
        "hit_band": [r["hit_band"] for r in selected],
        "mpn_shape": [r["mpn"] for r in selected],
        "platform_language": [r["lang"] for r in selected],
    })
    pq.write_table(table, OUT_PATH, compression="zstd", compression_level=9)
    print(f"[write] {n} rows -> {OUT_PATH} "
          f"({OUT_PATH.stat().st_size/1e6:.1f} MB)")


def summary(selected: list[dict]) -> None:
    print("\n[summary]")
    fb = Counter(r["freq_band"] for r in selected)
    hb = Counter(r["hit_band"] for r in selected)
    print(f"  freq_band: {dict(fb)}")
    print(f"  hit_band:  {dict(hb)}")
    print(f"  mpn:       {sum(1 for r in selected if r['mpn']):,}")
    langs = Counter(r["lang"] for r in selected)
    top_langs = dict(langs.most_common(8))
    print(f"  language (top 8): {top_langs}")
    print(f"  null language:    {langs.get(None, 0):,}")


def main():
    agg = aggregate()
    universe = filter_universe(agg)
    selected = sample(universe)
    write_parquet(selected)
    summary(selected)


if __name__ == "__main__":
    main()

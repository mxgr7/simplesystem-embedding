"""Compare offer-template dedup rate with and without the per-offer SKU
fields (ean, article_number, manufacturer_article_number).

Renders each offer twice on the same sample, tokenizes both with the
production tokenizer, and reports dedup rates pre- and post-truncation.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from embedding_train.rendering import RowTextRenderer  # noqa: E402

DATA_CFG_PATH = REPO_ROOT / "configs/data/default.yaml"
OFFERS_GLOB = "/data/datasets/offers_flat.parquet/bucket=*.parquet"
MODEL_NAME = "intfloat/multilingual-e5-base"
MAX_LEN = 256  # default; overridable via --max-len

PROJECT_COLS = [
    "name",
    "manufacturerName",
    "description",
    "categoryPaths",
    "ean",
    "article_number",
    "manufacturerArticleNumber",
    "manufacturerArticleType",
]
COLUMN_RENAME = {
    "manufacturerName": "manufacturer_name",
    "categoryPaths": "category_paths",
    "manufacturerArticleNumber": "manufacturer_article_number",
    "manufacturerArticleType": "manufacturer_article_type",
}

# default.yaml's offer_template, but with the three SKU-bearing
# conditional lines removed
TEMPLATE_NO_SKUS = (
    "passage: Article Name: {{ name }}\n"
    "{% if category_text %} Category: {{ category_text }}{% endif %}\n"
    "{% if manufacturer_article_type %} Article Type: {{ manufacturer_article_type }}{% endif %}\n"
    "{% if manufacturer_name %} Brand: {{ manufacturer_name }}{% endif %}\n"
    "{% if clean_description %} Description: {{ clean_description }}{% endif %}\n"
)


def hash16(b: bytes) -> bytes:
    return hashlib.blake2b(b, digest_size=16).digest()


def make_renderer(template_override: str | None = None) -> RowTextRenderer:
    cfg = OmegaConf.load(DATA_CFG_PATH)
    cfg.column_rename = COLUMN_RENAME
    if template_override is not None:
        cfg.offer_template = template_override
    return RowTextRenderer(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-per-bucket", type=int, default=50_000)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()

    global MAX_LEN
    MAX_LEN = args.max_len

    files = sorted(glob.glob(OFFERS_GLOB))
    if not files:
        sys.exit(f"no parquet files matched {OFFERS_GLOB}")

    print(f"buckets: {len(files)}  rows/bucket: {args.rows_per_bucket}")
    print(f"target sample size: {len(files) * args.rows_per_bucket:,}")

    renderer_default = make_renderer()
    renderer_stripped = make_renderer(TEMPLATE_NO_SKUS)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    full_def: Counter[bytes] = Counter()
    trunc_def: Counter[bytes] = Counter()
    full_stripped: Counter[bytes] = Counter()
    trunc_stripped: Counter[bytes] = Counter()

    over_def = over_str = 0
    lens_def: list[int] = []
    lens_str: list[int] = []
    total = 0
    t0 = time.time()

    for fi, fp in enumerate(files):
        bt0 = time.time()
        pf = pq.ParquetFile(fp)
        rows: list[dict] = []
        for batch in pf.iter_batches(batch_size=8192, columns=PROJECT_COLS):
            for row in batch.to_pylist():
                rows.append(row)
                if len(rows) >= args.rows_per_bucket:
                    break
            if len(rows) >= args.rows_per_bucket:
                break

        texts_def: list[str] = []
        texts_str: list[str] = []
        for row in rows:
            t1 = renderer_default.render_offer_text(row)
            t2 = renderer_stripped.render_offer_text(row)
            if not t1 or not t2:
                continue
            texts_def.append(t1)
            texts_str.append(t2)

        # tokenize both in batches
        for i in range(0, len(texts_def), args.batch_size):
            chunk_d = texts_def[i : i + args.batch_size]
            chunk_s = texts_str[i : i + args.batch_size]

            enc_d = tok(chunk_d, truncation=True, max_length=MAX_LEN,
                        add_special_tokens=True,
                        return_attention_mask=False,
                        return_token_type_ids=False)
            enc_s = tok(chunk_s, truncation=True, max_length=MAX_LEN,
                        add_special_tokens=True,
                        return_attention_mask=False,
                        return_token_type_ids=False)

            for raw_d, ids_d, raw_s, ids_s in zip(
                chunk_d, enc_d["input_ids"], chunk_s, enc_s["input_ids"]
            ):
                full_def[hash16(raw_d.encode("utf-8"))] += 1
                trunc_def[
                    hash16(np.asarray(ids_d, dtype=np.int32).tobytes())
                ] += 1
                full_stripped[hash16(raw_s.encode("utf-8"))] += 1
                trunc_stripped[
                    hash16(np.asarray(ids_s, dtype=np.int32).tobytes())
                ] += 1
                lens_def.append(len(ids_d))
                lens_str.append(len(ids_s))
                if len(ids_d) >= MAX_LEN:
                    over_def += 1
                if len(ids_s) >= MAX_LEN:
                    over_str += 1
                total += 1

        print(
            f"[{fi+1:2d}/{len(files)}] {Path(fp).name}: "
            f"sampled={len(rows):,} cum={total:,} "
            f"u_def_full={len(full_def):,} u_def_tr={len(trunc_def):,} "
            f"u_str_full={len(full_stripped):,} u_str_tr={len(trunc_stripped):,} "
            f"dt={time.time()-bt0:.1f}s",
            flush=True,
        )

    elapsed = time.time() - t0
    print(f"\nelapsed: {elapsed:.1f}s ({total/elapsed:.0f} rows/s)")

    def _report(name: str, full_c: Counter, trunc_c: Counter,
                lens: list[int], over: int) -> None:
        u_full = len(full_c)
        u_tr = len(trunc_c)
        print(f"\n=== {name} (n = {total:,}) ===")
        print(
            f"unique full-text:           {u_full:>12,} "
            f"({u_full/total*100:7.4f}%)"
        )
        print(
            f"unique truncated (256 tok): {u_tr:>12,} "
            f"({u_tr/total*100:7.4f}%)"
        )
        print(
            f"full-text dup rate:         {(1-u_full/total)*100:7.4f}%   "
            f"({total - u_full:,} rows redundant)"
        )
        print(
            f"truncated dup rate:         {(1-u_tr/total)*100:7.4f}%   "
            f"({total - u_tr:,} rows redundant)"
        )
        ar = np.asarray(lens)
        print(
            f"rows >= {MAX_LEN} tok: {over:,} = {over/total*100:.2f}% "
            f"  p50={int(np.percentile(ar,50))} "
            f"p75={int(np.percentile(ar,75))} "
            f"p90={int(np.percentile(ar,90))} "
            f"p99={int(np.percentile(ar,99))}"
        )
        # cluster size top-10 + histogram (small bins only)
        sizes = Counter(trunc_c.values())
        print("  cluster-size distribution (truncated):")
        for s in sorted(sizes.keys())[:8]:
            print(f"    size {s:>4}: {sizes[s]:>10,} clusters  "
                  f"({sizes[s]*s/total*100:6.4f}% of rows)")
        if max(sizes.keys()) > 8:
            big = sum(c for s, c in sizes.items() if s > 8)
            big_rows = sum(s*c for s, c in sizes.items() if s > 8)
            print(f"    size  >8: {big:>10,} clusters  "
                  f"({big_rows/total*100:6.4f}% of rows)")
        # top-3 absolute biggest clusters
        top = trunc_c.most_common(3)
        print("  top-3 cluster sizes (truncated):")
        for _, c in top:
            print(f"    {c} rows share one truncated sequence")

    _report("DEFAULT TEMPLATE", full_def, trunc_def, lens_def, over_def)
    _report("STRIPPED TEMPLATE (no ean / article_number / mfg_article_number)",
            full_stripped, trunc_stripped, lens_str, over_str)

    # delta summary
    u_def = len(trunc_def)
    u_str = len(trunc_stripped)
    print("\n=== dedup delta ===")
    print(
        f"truncated dedup gain by stripping SKUs: "
        f"{(u_def - u_str)/total*100:.4f} percentage points "
        f"({u_def - u_str:,} more rows would be duplicates)"
    )


if __name__ == "__main__":
    main()

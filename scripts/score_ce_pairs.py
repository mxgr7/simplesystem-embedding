"""Score (query, offer) pairs with the frozen Soup cross-encoder teacher.

Produces a parquet keyed by (query_id, offer_id) with the teacher's expected
relevance gain and class probabilities, for MarginMSE distillation of the
SPLADE student (configs/model/splade_distill.yaml + data.ce_scores_path).

Usage (on the GPU box):
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1 uv run python scripts/score_ce_pairs.py \
    --checkpoint ~/checkpoints/cross-encoder/releases/v1.0-2026-04-29/soup.ckpt \
    --input data/queries_offers_labeled.parquet \
    --output data/ce_scores.parquet
"""

import argparse
import time

import pandas as pd
import torch
from omegaconf import OmegaConf

from cross_encoder_train.data import collate_pairs
from cross_encoder_train.labels import GAIN_VECTOR, LABEL_ORDER
from cross_encoder_train.model import CrossEncoderModule
from embedding_train.rendering import RowTextRenderer
from embedding_train.tokenization import load_fast_tokenizer

TEACHER_MODEL_NAME = "deepset/gelectra-large"
MAX_PAIR_LENGTH = 512

# The teacher must see inputs in its training distribution: the cross-encoder
# German templates, not the embedding-side ones.
CE_DATA_CFG = {
    "query_template": "{{ query_term }}",
    "offer_template": (
        "Artikel: {{ name }}\n"
        "{% if ean %} EAN: {{ ean }}{% endif %}\n"
        "{% if article_number %} Artikelnummer: {{ article_number }}{% endif %}\n"
        "{% if manufacturer_article_number %} Herstellernummer: {{ manufacturer_article_number }}{% endif %}\n"
        "{% if category_text %} Kategorie: {{ category_text }}{% endif %}\n"
        "{% if manufacturer_article_type %} Artikeltyp: {{ manufacturer_article_type }}{% endif %}\n"
        "{% if manufacturer_name %} Marke: {{ manufacturer_name }}{% endif %}\n"
        "{% if clean_description %} Beschreibung: {{ clean_description }}{% endif %}"
    ),
    "clean_html": True,
    "positive_label": "Exact",
    "column_mapping": {
        "query_id": "query_id",
        "offer_id": "offer_id_b64",
        "query_term": "query_term",
        "name": "name",
        "manufacturer_name": "manufacturer_name",
        "manufacturer_article_number": "manufacturer_article_number",
        "manufacturer_article_type": "manufacturer_article_type",
        "article_number": "article_number",
        "ean": "ean",
        "category_paths": "category_paths",
        "description": "description",
        "label": "label",
    },
    "column_rename": None,
}

TEACHER_CFG = {
    "model": {
        "model_name": TEACHER_MODEL_NAME,
        "head_dropout": 0.0,
        "label_smoothing": 0.0,
        "focal_gamma": 0.0,
        "use_class_weights": False,
        "gradient_checkpointing": False,
        "compile": False,
        "prune_layers": 0,
    },
    "trainer": {"precision": "bf16-mixed"},
    "data": {"log_batch_stats": False},
}


def load_teacher(checkpoint_path, device):
    teacher = CrossEncoderModule(cfg=OmegaConf.create(TEACHER_CFG))
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = {
        k.replace("._orig_mod.", "."): v for k, v in ckpt["state_dict"].items()
    }
    teacher.load_state_dict(state_dict, strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.to(torch.bfloat16).to(device)
    return teacher


def build_records(frame, renderer):
    columns = list(frame.columns)
    records = []
    for values in frame.itertuples(index=False, name=None):
        row = dict(zip(columns, values))
        context = renderer.build_context(row)
        record = renderer.build_training_record(row, context=context)
        if record is None:
            continue
        records.append(
            {
                "query_id": record["query_id"],
                "offer_id": record["offer_id"],
                "query_text": record["query_text"],
                "offer_text": record["offer_text"],
                "label_id": 0,
                "raw_label": record["raw_label"],
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit-rows", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_fast_tokenizer(TEACHER_MODEL_NAME)
    teacher = load_teacher(args.checkpoint, device)
    renderer = RowTextRenderer(OmegaConf.create(CE_DATA_CFG))

    frame = pd.read_parquet(args.input)
    if args.limit_rows:
        frame = frame.head(args.limit_rows)
    records = build_records(frame, renderer)
    print(f"scoring {len(records)} pairs on {device}", flush=True)

    gains = torch.tensor(GAIN_VECTOR, dtype=torch.float32)
    rows = []
    started = time.time()
    for start in range(0, len(records), args.batch_size):
        chunk = records[start : start + args.batch_size]
        batch = collate_pairs(chunk, tokenizer, MAX_PAIR_LENGTH)
        inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
        with torch.inference_mode():
            logits = teacher(inputs).float().cpu()
        probs = torch.softmax(logits, dim=1)
        expected_gain = probs @ gains
        for i, record in enumerate(chunk):
            row = {
                "query_id": record["query_id"],
                "offer_id_b64": record["offer_id"],
                "ce_score": float(expected_gain[i]),
            }
            for j, label in enumerate(LABEL_ORDER):
                row[f"ce_p_{label[0].lower()}"] = float(probs[i, j])
            rows.append(row)
        if (start // args.batch_size) % 50 == 0:
            done = start + len(chunk)
            rate = done / max(time.time() - started, 1e-6)
            eta = (len(records) - done) / max(rate, 1e-6)
            print(
                f"{done}/{len(records)} pairs, {rate:.0f}/s, eta {eta/60:.1f} min",
                flush=True,
            )

    out = pd.DataFrame(rows)
    out.to_parquet(args.output, index=False)
    print(f"wrote {len(out)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()

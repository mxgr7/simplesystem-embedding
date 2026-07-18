"""Post-process mined negatives for SPLADE v1 training.

The miner renders offer_text with the mining checkpoint's templates (e5
"passage:" prefixes for useful-cub-58); SPLADE trains without prefixes, so
the sidecar text must be re-rendered. Additionally, MarginMSE distillation
wants teacher scores on the mined negatives, so the mined pairs are also
emitted in the labeled-parquet schema for scripts/score_ce_pairs.py.

Usage:
  uv run python scripts/prepare_v1_sidecars.py \
    --mined data/semi_hard_negatives-uc58.parquet \
    --labeled data/queries_offers_labeled.parquet \
    --sidecar-out data/semi_hard_negatives-uc58-splade.parquet \
    --ce-pairs-out data/mined_pairs_for_ce.parquet \
    --config-overrides model=splade data=splade
"""

import argparse

import hydra
import pandas as pd

from embedding_train.config import CONFIG_DIR
from embedding_train.rendering import RowTextRenderer

OFFER_FIELD_COLUMNS = [
    "name",
    "manufacturer_name",
    "manufacturer_article_number",
    "manufacturer_article_type",
    "article_number",
    "ean",
    "category_paths",
    "description",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mined", required=True)
    parser.add_argument("--labeled", required=True)
    parser.add_argument("--sidecar-out", required=True)
    parser.add_argument("--ce-pairs-out", required=True)
    parser.add_argument("--config-overrides", nargs="*", default=[])
    args = parser.parse_args()

    with hydra.initialize_config_dir(
        version_base="1.3", config_dir=str(CONFIG_DIR)
    ):
        cfg = hydra.compose(config_name="config", overrides=args.config_overrides)
    renderer = RowTextRenderer(cfg.data)

    labeled = pd.read_parquet(args.labeled)
    offer_rows = labeled.drop_duplicates("offer_id_b64").set_index(
        "offer_id_b64"
    )
    query_terms = (
        labeled.drop_duplicates("query_id")
        .set_index("query_id")["query_term"]
        .to_dict()
    )

    mined = pd.read_parquet(args.mined)
    sidecar_rows = []
    ce_pair_rows = []
    missing_offers = 0
    missing_queries = 0

    for row in mined.itertuples(index=False):
        query_id = str(row.query_id)
        offer_id = str(row.offer_id)
        if offer_id not in offer_rows.index:
            missing_offers += 1
            continue
        query_term = query_terms.get(query_id)
        if query_term is None:
            missing_queries += 1
            continue

        offer_fields = offer_rows.loc[offer_id]
        raw_row = {
            "query_id": query_id,
            "query_term": query_term,
            "offer_id_b64": offer_id,
            "label": "Irrelevant",
        }
        for column in OFFER_FIELD_COLUMNS:
            raw_row[column] = offer_fields.get(column)

        record = renderer.build_training_record(raw_row)
        if record is None:
            continue

        sidecar_rows.append(
            {
                "query_id": query_id,
                "offer_id": offer_id,
                "offer_text": record["offer_text"],
                "provenance": getattr(row, "provenance", "semi_hard_negative"),
            }
        )
        ce_pair_rows.append(raw_row)

    pd.DataFrame(sidecar_rows).to_parquet(args.sidecar_out, index=False)
    ce_pairs = pd.DataFrame(ce_pair_rows).drop_duplicates(
        ["query_id", "offer_id_b64"]
    )
    ce_pairs.to_parquet(args.ce_pairs_out, index=False)
    print(
        f"sidecar: {len(sidecar_rows)} rows -> {args.sidecar_out}; "
        f"ce pairs: {len(ce_pairs)} rows -> {args.ce_pairs_out} "
        f"(missing offers {missing_offers}, missing queries {missing_queries})"
    )


if __name__ == "__main__":
    main()

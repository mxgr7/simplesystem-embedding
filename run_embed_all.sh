#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="checkpoints/useful-cub-58/best-step=4880-val_full_catalog_ndcg_at_5=0.7379.ckpt"
INPUT_DIR="../../data/offers_grouped.parquet"
OUTPUT_DIR="../../data/offers_embedded.parquet"
RENAME='manufacturerName=manufacturer_name,categoryPaths=category_paths,manufacturerArticleNumber=manufacturer_article_number,manufacturerArticleType=manufacturer_article_type'

mkdir -p "$OUTPUT_DIR"

# Pre-computed row offsets for each bucket (cumulative row counts)
OFFSETS=(0 9954348 19906212 29861245 39813300 49768880 59725546 69682705 79636533 89592725 99542008 109495862 119451885 129407939 139361686 149314447)

for i in $(seq 0 15); do
    BUCKET=$(printf "%02d" $i)
    INPUT_FILE="${INPUT_DIR}/bucket=${BUCKET}.parquet"
    OUTPUT_FILE="${OUTPUT_DIR}/bucket=${BUCKET}.parquet"
    OFFSET=${OFFSETS[$i]}

    if [ -f "$OUTPUT_FILE" ]; then
        echo "=== Bucket ${BUCKET} already exists, skipping ==="
        continue
    fi

    echo "=== Processing bucket=${BUCKET} (offset=${OFFSET}) ==="
    uv run embedding-infer \
        --checkpoint "$CHECKPOINT" \
        --input "$INPUT_FILE" \
        --output "$OUTPUT_FILE" \
        --mode offer \
        --embedding-precision float16 \
        --encode-batch-size 128 \
        --read-batch-size 1024 \
        --num-workers 4 \
        --column-rename "$RENAME" \
        --copy-columns id \
        --row-number-offset "$OFFSET"
    echo "=== Done bucket=${BUCKET} ==="
done

echo "=== All buckets complete ==="

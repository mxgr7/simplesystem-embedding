import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer

from embedding_train.model import load_embedding_module_from_checkpoint
from embedding_train.rendering import RowTextRenderer


DEFAULT_COPY_COLUMNS = ["query_id", "offer_id_b64", "label"]
VALID_INFERENCE_MODES = {"query", "offer", "pair_score"}


def resolve_inference_mode(mode):
    normalized = str(mode).strip().lower()

    if normalized in VALID_INFERENCE_MODES:
        return normalized

    choices = "|".join(sorted(VALID_INFERENCE_MODES))
    raise ValueError(f"Unsupported inference mode: {mode}. Expected one of {choices}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run offline embedding export or pairwise scoring over a Parquet dataset."
    )
    parser.add_argument("--checkpoint", required=True, help="Lightning checkpoint path")
    parser.add_argument("--input", required=True, help="Input Parquet path")
    parser.add_argument("--output", required=True, help="Output Parquet path")
    parser.add_argument(
        "--mode",
        default="offer",
        choices=sorted(VALID_INFERENCE_MODES),
        help="`offer` and `query` export embeddings; `pair_score` scores paired rows.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, cuda:0, or mps.",
    )
    parser.add_argument(
        "--read-batch-size",
        type=int,
        default=1024,
        help="Rows to stream from Parquet at a time.",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=128,
        help="Texts to tokenize and encode per forward pass.",
    )
    parser.add_argument(
        "--copy-columns",
        default="",
        help="Comma-separated input columns to copy to the output.",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include rendered text columns in the output.",
    )
    parser.add_argument(
        "--output-column",
        default="",
        help="Override the output column name.",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        help="Parquet compression codec.",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Optional maximum number of input rows to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser


def resolve_device(device_name):
    normalized = str(device_name).strip().lower()

    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")

    if normalized == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available")

    return torch.device(device_name)


def parse_copy_columns(raw_value, available_columns):
    if raw_value:
        selected = [value.strip() for value in raw_value.split(",") if value.strip()]
    else:
        selected = [
            column for column in DEFAULT_COPY_COLUMNS if column in available_columns
        ]

    missing_columns = [column for column in selected if column not in available_columns]
    if missing_columns:
        raise ValueError(
            "Requested copy columns are missing from the input: "
            + ", ".join(missing_columns)
        )

    return selected


def resolve_output_column(mode, output_column):
    if output_column:
        return output_column

    if mode == "query":
        return "query_embedding"

    if mode == "offer":
        return "offer_embedding"

    return "pair_score"


def build_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def encode_texts(model, tokenizer, texts, max_length, encode_batch_size, device):
    encoded_batches = []

    with torch.inference_mode():
        for start in range(0, len(texts), encode_batch_size):
            chunk = texts[start : start + encode_batch_size]
            inputs = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {name: tensor.to(device) for name, tensor in dict(inputs).items()}
            embeddings = model.encode(inputs)
            encoded_batches.append(embeddings.detach().cpu().to(dtype=torch.float32))

    return torch.cat(encoded_batches, dim=0)


def build_output_table(rows, output_column, output_values, mode, input_schema):
    arrays = {}

    for name in rows[0]:
        values = [row[name] for row in rows]

        if name in input_schema.names:
            arrays[name] = pa.array(values, type=input_schema.field(name).type)
            continue

        if name == "row_number":
            arrays[name] = pa.array(values, type=pa.int64())
            continue

        arrays[name] = pa.array(values, type=pa.string())

    if mode == "pair_score":
        arrays[output_column] = pa.array(output_values, type=pa.float32())
        return pa.table(arrays)

    arrays[output_column] = pa.array(output_values, type=pa.list_(pa.float32()))
    return pa.table(arrays)


class IncrementalParquetWriter:
    def __init__(self, output_path, compression, overwrite):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {self.output_path}. Pass --overwrite to replace it."
            )

        if self.output_path.exists():
            self.output_path.unlink()

        self.compression = compression
        self.writer = None

    def write_table(self, table):
        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.output_path,
                table.schema,
                compression=self.compression,
            )

        self.writer.write_table(table)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def process_rows(
    rows,
    renderer,
    mode,
    output_column,
    copy_columns,
    include_text,
    input_schema,
    query_max_length,
    offer_max_length,
    tokenizer,
    model,
    encode_batch_size,
    device,
    row_number,
):
    prepared_rows = []
    query_texts = []
    offer_texts = []
    skipped_rows = 0

    for row in rows:
        context = renderer.build_context(row)
        prepared_row = {"row_number": row_number}
        row_number += 1

        for column in copy_columns:
            prepared_row[column] = row.get(column)

        if mode == "query":
            query_text = renderer.render_query_text(row, context=context)
            if not query_text:
                skipped_rows += 1
                continue

            if include_text:
                prepared_row["query_text"] = query_text

            prepared_rows.append(prepared_row)
            query_texts.append(query_text)
            continue

        if mode == "offer":
            offer_text = renderer.render_offer_text(row, context=context)
            if not offer_text:
                skipped_rows += 1
                continue

            if include_text:
                prepared_row["offer_text"] = offer_text

            prepared_rows.append(prepared_row)
            offer_texts.append(offer_text)
            continue

        query_text = renderer.render_query_text(row, context=context)
        offer_text = renderer.render_offer_text(row, context=context)
        if not query_text or not offer_text:
            skipped_rows += 1
            continue

        if include_text:
            prepared_row["query_text"] = query_text
            prepared_row["offer_text"] = offer_text

        prepared_rows.append(prepared_row)
        query_texts.append(query_text)
        offer_texts.append(offer_text)

    if not prepared_rows:
        return None, row_number, skipped_rows

    if mode == "query":
        output_values = encode_texts(
            model,
            tokenizer,
            query_texts,
            query_max_length,
            encode_batch_size,
            device,
        ).tolist()
    elif mode == "offer":
        output_values = encode_texts(
            model,
            tokenizer,
            offer_texts,
            offer_max_length,
            encode_batch_size,
            device,
        ).tolist()
    else:
        query_embeddings = encode_texts(
            model,
            tokenizer,
            query_texts,
            query_max_length,
            encode_batch_size,
            device,
        )
        offer_embeddings = encode_texts(
            model,
            tokenizer,
            offer_texts,
            offer_max_length,
            encode_batch_size,
            device,
        )
        output_values = (query_embeddings * offer_embeddings).sum(dim=1).tolist()

    table = build_output_table(
        prepared_rows,
        output_column,
        output_values,
        mode,
        input_schema,
    )
    return table, row_number, skipped_rows


def run_inference(args):
    mode = resolve_inference_mode(args.mode)
    output_column = resolve_output_column(mode, args.output_column)
    device = resolve_device(args.device)

    model, cfg = load_embedding_module_from_checkpoint(
        args.checkpoint, map_location="cpu"
    )
    model = model.to(device)
    model.eval()

    tokenizer = build_tokenizer(cfg.model.model_name)
    renderer = RowTextRenderer(cfg.data)

    parquet_file = pq.ParquetFile(args.input)
    copy_columns = parse_copy_columns(args.copy_columns, parquet_file.schema.names)
    writer = IncrementalParquetWriter(args.output, args.compression, args.overwrite)

    processed_rows = 0
    written_rows = 0
    skipped_rows = 0
    row_number = 0

    try:
        for batch in parquet_file.iter_batches(batch_size=int(args.read_batch_size)):
            rows = batch.to_pylist()
            if args.limit_rows is not None:
                remaining_rows = int(args.limit_rows) - processed_rows
                if remaining_rows <= 0:
                    break
                rows = rows[:remaining_rows]

            processed_rows += len(rows)
            if not rows:
                break

            table, row_number, batch_skipped_rows = process_rows(
                rows=rows,
                renderer=renderer,
                mode=mode,
                output_column=output_column,
                copy_columns=copy_columns,
                include_text=bool(args.include_text),
                input_schema=parquet_file.schema_arrow,
                query_max_length=int(cfg.data.max_query_length),
                offer_max_length=int(cfg.data.max_offer_length),
                tokenizer=tokenizer,
                model=model,
                encode_batch_size=int(args.encode_batch_size),
                device=device,
                row_number=row_number,
            )
            skipped_rows += batch_skipped_rows

            if table is None:
                continue

            writer.write_table(table)
            written_rows += table.num_rows
    finally:
        writer.close()

    print(
        "Inference complete:",
        {
            "mode": mode,
            "device": str(device),
            "processed_rows": processed_rows,
            "written_rows": written_rows,
            "skipped_rows": skipped_rows,
            "output": str(args.output),
        },
    )


def main(argv=None):
    load_dotenv()
    torch.set_float32_matmul_precision("high")
    args = build_arg_parser().parse_args(argv)
    run_inference(args)


if __name__ == "__main__":
    main()

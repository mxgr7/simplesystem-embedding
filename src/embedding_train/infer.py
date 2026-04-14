import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer

from embedding_train.model import load_embedding_module_from_checkpoint
from embedding_train.precision import (
    quantize_embeddings,
    resolve_embedding_precision,
    score_embedding_pairs,
    serialize_embeddings,
)
from embedding_train.rendering import RowTextRenderer


DEFAULT_COPY_COLUMN_KEYS = ("query_id", "offer_id", "label")
VALID_INFERENCE_MODES = {"query", "offer", "pair_score"}


def default_copy_columns_from_renderer(renderer):
    return [renderer.column_mapping[key] for key in DEFAULT_COPY_COLUMN_KEYS]


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
        "--embedding-precision",
        default="float16",
        help="Embedding export precision: float32, float16, int8, sign, or binary.",
    )
    parser.add_argument(
        "--max-offer-length",
        type=int,
        default=0,
        help=(
            "Override cfg.data.max_offer_length for offer tokenization "
            "(0 keeps config value). Lowering below the p99 of actual token "
            "lengths trades a small fraction of truncated rows for proportional "
            "forward-pass savings."
        ),
    )
    parser.add_argument(
        "--max-query-length",
        type=int,
        default=0,
        help="Override cfg.data.max_query_length for query tokenization.",
    )
    parser.add_argument(
        "--row-number-offset",
        type=int,
        default=0,
        help=(
            "Offset added to every emitted row_number. Used for sharded runs "
            "so each worker's row_numbers are disjoint from the others."
        ),
    )
    parser.add_argument(
        "--column-rename",
        default="",
        help=(
            "Comma-separated old=new pairs to rename input columns before processing. "
            "Example: manufacturerName=manufacturer_name,categoryPaths=category_paths"
        ),
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
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Emit detailed profiling stats (padding efficiency, token throughput, etc.)",
    )
    return parser


def parse_column_rename(raw_value):
    if not raw_value:
        return {}
    rename_map = {}
    for pair in raw_value.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"Invalid column rename pair (expected old=new): {pair}")
        old, new = pair.split("=", 1)
        rename_map[old.strip()] = new.strip()
    return rename_map


def apply_column_rename(batch, rename_map):
    if not rename_map:
        return batch
    names = list(batch.schema.names)
    new_names = [rename_map.get(name, name) for name in names]
    return batch.rename_columns(new_names)


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


def parse_copy_columns(raw_value, available_columns, default_columns=None):
    if raw_value:
        selected = [value.strip() for value in raw_value.split(",") if value.strip()]
    else:
        default_columns = default_columns or []
        selected = [
            column for column in default_columns if column in available_columns
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


def _pad_and_stack(token_ids_batch, pad_token_id):
    """Pad a list of token ID lists to the batch max length, return input_ids + attention_mask tensors."""
    max_len = max(len(ids) for ids in token_ids_batch)
    batch_size = len(token_ids_batch)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, ids in enumerate(token_ids_batch):
        length = len(ids)
        input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :length] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def encode_texts(
    model,
    tokenizer,
    texts,
    max_length,
    encode_batch_size,
    device,
    phase_times=None,
):
    if not texts:
        return torch.empty(0)

    encoded_batches = []
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    profile = phase_times is not None
    sync_cuda = profile and device.type == "cuda"
    pad_token_id = tokenizer.pad_token_id or 0

    # Phase 1: Tokenize all texts at once without padding (3x faster)
    t_tok = time.perf_counter() if profile else 0.0
    all_encodings = tokenizer(
        texts,
        padding=False,
        truncation=True,
        max_length=max_length,
    )
    all_input_ids = all_encodings["input_ids"]
    if profile:
        phase_times["encode_tokenize"] = (
            phase_times.get("encode_tokenize", 0.0)
            + time.perf_counter()
            - t_tok
        )

    # Phase 2: Sort by token length for optimal batching
    sort_start = time.perf_counter() if profile else 0.0
    sort_order = sorted(range(len(all_input_ids)), key=lambda i: len(all_input_ids[i]))
    sorted_ids = [all_input_ids[i] for i in sort_order]
    if profile:
        phase_times["encode_sort"] = (
            phase_times.get("encode_sort", 0.0) + time.perf_counter() - sort_start
        )

    # Phase 3: Batch encode with local padding
    chunk_starts = list(range(0, len(sorted_ids), encode_batch_size))

    def pad_chunk(start):
        t_pad = time.perf_counter() if profile else 0.0
        chunk = sorted_ids[start : start + encode_batch_size]
        result = _pad_and_stack(chunk, pad_token_id)
        if profile:
            phase_times["encode_pad"] = (
                phase_times.get("encode_pad", 0.0)
                + time.perf_counter()
                - t_pad
            )
        return result

    with ThreadPoolExecutor(max_workers=1) as pad_executor:
        prefetched = pad_executor.submit(pad_chunk, chunk_starts[0])

        with torch.inference_mode(), amp_context:
            for i, _ in enumerate(chunk_starts):
                cpu_inputs = prefetched.result()
                if i + 1 < len(chunk_starts):
                    prefetched = pad_executor.submit(pad_chunk, chunk_starts[i + 1])

                if profile:
                    t_h2d = time.perf_counter()
                device_inputs = {
                    name: tensor.to(device, non_blocking=True)
                    for name, tensor in cpu_inputs.items()
                }
                if sync_cuda:
                    torch.cuda.synchronize()
                if profile:
                    phase_times["encode_h2d"] = (
                        phase_times.get("encode_h2d", 0.0)
                        + time.perf_counter()
                        - t_h2d
                    )
                    attn_mask = cpu_inputs["attention_mask"]
                    total_tokens = attn_mask.numel()
                    real_tokens = int(attn_mask.sum())
                    phase_times["_total_tokens"] = (
                        phase_times.get("_total_tokens", 0) + total_tokens
                    )
                    phase_times["_real_tokens"] = (
                        phase_times.get("_real_tokens", 0) + real_tokens
                    )
                    phase_times["_max_seq_len_sum"] = (
                        phase_times.get("_max_seq_len_sum", 0) + attn_mask.shape[1]
                    )
                    phase_times["_num_encode_batches"] = (
                        phase_times.get("_num_encode_batches", 0) + 1
                    )
                    phase_times["_num_sequences"] = (
                        phase_times.get("_num_sequences", 0) + attn_mask.shape[0]
                    )
                    t_fwd = time.perf_counter()

                embeddings = model.encode(device_inputs)

                if sync_cuda:
                    torch.cuda.synchronize()
                if profile:
                    phase_times["encode_forward"] = (
                        phase_times.get("encode_forward", 0.0)
                        + time.perf_counter()
                        - t_fwd
                    )
                    t_cpu = time.perf_counter()

                encoded_batches.append(embeddings.detach().cpu().to(dtype=torch.float32))

                if profile:
                    phase_times["encode_to_cpu"] = (
                        phase_times.get("encode_to_cpu", 0.0)
                        + time.perf_counter()
                        - t_cpu
                    )

    sorted_embeddings = torch.cat(encoded_batches, dim=0)
    unsort_start = time.perf_counter() if profile else 0.0
    inverse_order = torch.empty(len(sort_order), dtype=torch.long)
    inverse_order[torch.tensor(sort_order, dtype=torch.long)] = torch.arange(
        len(sort_order), dtype=torch.long
    )
    result = sorted_embeddings.index_select(0, inverse_order)
    if profile:
        phase_times["encode_sort"] = (
            phase_times.get("encode_sort", 0.0)
            + time.perf_counter()
            - unsort_start
        )
    return result


def resolve_embedding_arrow_type(embedding_precision):
    if embedding_precision == "float32":
        return pa.list_(pa.float32())

    if embedding_precision == "float16":
        return pa.list_(pa.float16())

    if embedding_precision in {"int8", "sign"}:
        return pa.list_(pa.int8())

    if embedding_precision == "binary":
        return pa.binary()

    raise ValueError(f"Unsupported embedding precision: {embedding_precision}")


def build_output_table(
    rows,
    output_column,
    output_values,
    mode,
    input_schema,
    embedding_precision,
    embedding_dim,
):
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
        table = pa.table(arrays)
        return attach_output_metadata(
            table,
            output_column,
            mode,
            embedding_precision,
            embedding_dim,
        )

    arrays[output_column] = pa.array(
        output_values,
        type=resolve_embedding_arrow_type(embedding_precision),
    )
    table = pa.table(arrays)
    return attach_output_metadata(
        table,
        output_column,
        mode,
        embedding_precision,
        embedding_dim,
    )


def attach_output_metadata(
    table,
    output_column,
    mode,
    embedding_precision,
    embedding_dim,
):
    metadata = dict(table.schema.metadata or {})
    metadata[b"output_column"] = str(output_column).encode("utf-8")

    if mode == "pair_score":
        metadata[b"scoring_embedding_precision"] = str(embedding_precision).encode(
            "utf-8"
        )
    else:
        metadata[b"embedding_precision"] = str(embedding_precision).encode("utf-8")
        metadata[b"embedding_dim"] = str(embedding_dim).encode("utf-8")

    return table.replace_schema_metadata(metadata)


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
    embedding_precision,
    phase_times=None,
):
    prepared_rows = []
    query_texts = []
    offer_texts = []
    skipped_rows = 0

    prep_start = time.perf_counter() if phase_times is not None else 0.0
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

    if phase_times is not None:
        phase_times["prepare_rows"] = (
            phase_times.get("prepare_rows", 0.0) + time.perf_counter() - prep_start
        )

    if not prepared_rows:
        return None, row_number, skipped_rows

    encode_start = time.perf_counter() if phase_times is not None else 0.0
    if mode == "query":
        embeddings = encode_texts(
            model,
            tokenizer,
            query_texts,
            query_max_length,
            encode_batch_size,
            device,
            phase_times=phase_times,
        )
        output_values = serialize_embeddings(
            quantize_embeddings(embeddings, embedding_precision),
            embedding_precision,
        )
        embedding_dim = embeddings.size(1)
    elif mode == "offer":
        embeddings = encode_texts(
            model,
            tokenizer,
            offer_texts,
            offer_max_length,
            encode_batch_size,
            device,
            phase_times=phase_times,
        )
        output_values = serialize_embeddings(
            quantize_embeddings(embeddings, embedding_precision),
            embedding_precision,
        )
        embedding_dim = embeddings.size(1)
    else:
        query_embeddings = encode_texts(
            model,
            tokenizer,
            query_texts,
            query_max_length,
            encode_batch_size,
            device,
            phase_times=phase_times,
        )
        offer_embeddings = encode_texts(
            model,
            tokenizer,
            offer_texts,
            offer_max_length,
            encode_batch_size,
            device,
            phase_times=phase_times,
        )
        output_values = score_embedding_pairs(
            query_embeddings,
            offer_embeddings,
            embedding_precision,
        ).tolist()
        embedding_dim = query_embeddings.size(1)
    if phase_times is not None:
        phase_times["encode"] = (
            phase_times.get("encode", 0.0) + time.perf_counter() - encode_start
        )

    table = build_output_table(
        prepared_rows,
        output_column,
        output_values,
        mode,
        input_schema,
        embedding_precision,
        embedding_dim,
    )
    return table, row_number, skipped_rows


def run_inference(args):
    from embedding_train.index_build import ParquetSource

    mode = resolve_inference_mode(args.mode)
    output_column = resolve_output_column(mode, args.output_column)
    device = resolve_device(args.device)
    embedding_precision = resolve_embedding_precision(args.embedding_precision)

    model, cfg = load_embedding_module_from_checkpoint(
        args.checkpoint, map_location="cpu"
    )
    model = model.to(device)
    model.eval()

    tokenizer = build_tokenizer(cfg.model.model_name)
    renderer = RowTextRenderer(cfg.data)

    column_rename = parse_column_rename(args.column_rename)

    parquet_file = ParquetSource(args.input)
    renamed_schema = parquet_file.schema_arrow
    if column_rename:
        new_names = [column_rename.get(n, n) for n in renamed_schema.names]
        renamed_schema = pa.schema(
            [renamed_schema.field(i).with_name(new_names[i]) for i in range(len(new_names))]
        )
    copy_columns = parse_copy_columns(
        args.copy_columns,
        renamed_schema.names,
        default_copy_columns_from_renderer(renderer),
    )
    writer = IncrementalParquetWriter(args.output, args.compression, args.overwrite)

    offer_max_length = int(args.max_offer_length) or int(cfg.data.max_offer_length)
    query_max_length = int(args.max_query_length) or int(cfg.data.max_query_length)
    row_number = int(getattr(args, "row_number_offset", 0) or 0)

    processed_rows = 0
    written_rows = 0
    skipped_rows = 0

    phase_times = {
        "prepare_rows": 0.0,
        "encode": 0.0,
        "write": 0.0,
    }

    try:
        for batch in parquet_file.iter_batches(batch_size=int(args.read_batch_size)):
            batch = apply_column_rename(batch, column_rename)
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
                input_schema=renamed_schema,
                query_max_length=query_max_length,
                offer_max_length=offer_max_length,
                tokenizer=tokenizer,
                model=model,
                encode_batch_size=int(args.encode_batch_size),
                device=device,
                row_number=row_number,
                embedding_precision=embedding_precision,
                phase_times=phase_times,
            )
            skipped_rows += batch_skipped_rows

            if table is None:
                continue

            t_write = time.perf_counter()
            writer.write_table(table)
            phase_times["write"] += time.perf_counter() - t_write
            written_rows += table.num_rows
    finally:
        writer.close()

    wall_time = phase_times["prepare_rows"] + phase_times["encode"] + phase_times["write"]
    rows_per_sec = written_rows / wall_time if wall_time > 0 else 0

    print(
        "Inference complete:",
        {
            "mode": mode,
            "device": str(device),
            "processed_rows": processed_rows,
            "written_rows": written_rows,
            "skipped_rows": skipped_rows,
            "embedding_precision": embedding_precision,
            "output": str(args.output),
        },
    )
    print(
        "Phase times (s):",
        {name: round(value, 3) for name, value in phase_times.items() if not name.startswith("_")},
    )
    print(f"Throughput: {rows_per_sec:.1f} rows/sec")

    if getattr(args, "profile", False) and phase_times.get("_total_tokens"):
        total_tok = phase_times["_total_tokens"]
        real_tok = phase_times["_real_tokens"]
        n_batches = phase_times["_num_encode_batches"]
        n_seqs = phase_times["_num_sequences"]
        avg_max_seq = phase_times["_max_seq_len_sum"] / n_batches if n_batches else 0
        fwd_time = phase_times.get("encode_forward", 0)
        tok_time = phase_times.get("encode_tokenize", 0)
        print("\n--- Detailed Profile ---")
        print(f"  Encode batches:        {n_batches}")
        print(f"  Total sequences:       {n_seqs}")
        print(f"  Avg batch max_seq_len: {avg_max_seq:.1f}")
        print(f"  Real tokens:           {real_tok:,}")
        print(f"  Total tokens (padded): {total_tok:,}")
        print(f"  Padding efficiency:    {real_tok/total_tok*100:.1f}%")
        print(f"  Tokens/sec (real):     {real_tok/fwd_time:,.0f}" if fwd_time > 0 else "")
        print(f"  Tokens/sec (padded):   {total_tok/fwd_time:,.0f}" if fwd_time > 0 else "")
        print(f"  Seqs/sec (forward):    {n_seqs/fwd_time:,.0f}" if fwd_time > 0 else "")
        print(f"  Seqs/sec (tokenizer):  {n_seqs/tok_time:,.0f}" if tok_time > 0 else "")
        print(f"  Time breakdown:")
        print(f"    prepare_rows:  {phase_times['prepare_rows']:7.3f}s ({phase_times['prepare_rows']/wall_time*100:5.1f}%)")
        print(f"    tokenize:      {tok_time:7.3f}s ({tok_time/wall_time*100:5.1f}%)")
        print(f"    h2d transfer:  {phase_times.get('encode_h2d',0):7.3f}s ({phase_times.get('encode_h2d',0)/wall_time*100:5.1f}%)")
        print(f"    forward:       {fwd_time:7.3f}s ({fwd_time/wall_time*100:5.1f}%)")
        print(f"    d2h transfer:  {phase_times.get('encode_to_cpu',0):7.3f}s ({phase_times.get('encode_to_cpu',0)/wall_time*100:5.1f}%)")
        print(f"    sort:          {phase_times.get('encode_sort',0):7.3f}s ({phase_times.get('encode_sort',0)/wall_time*100:5.1f}%)")
        print(f"    write:         {phase_times['write']:7.3f}s ({phase_times['write']/wall_time*100:5.1f}%)")


def main(argv=None):
    load_dotenv()
    torch.set_float32_matmul_precision("high")
    args = build_arg_parser().parse_args(argv)
    run_inference(args)


if __name__ == "__main__":
    main()

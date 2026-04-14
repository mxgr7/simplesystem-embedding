import argparse
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np
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
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes for text rendering. 0 = single-process (default).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run torch.profiler and emit CUDA kernel-level timing breakdown.",
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


# --- Multiprocess text rendering ---
# Module-level state set before fork; workers inherit it.
_g_renderer = None
_g_tokenizer = None
_g_mode = None
_g_copy_columns = None
_g_include_text = False
_g_max_length = 256


def _worker_render_and_tokenize(item):
    """Worker function: render texts and tokenize (no padding). Runs in forked child."""
    import numpy as np

    rows, row_number_start = item
    prepared_rows = []
    texts = []
    skipped = 0

    for row in rows:
        context = _g_renderer.build_context(row)
        prepared_row = {"row_number": row_number_start}
        row_number_start += 1
        for col in _g_copy_columns:
            prepared_row[col] = row.get(col)

        if _g_mode == "offer":
            text = _g_renderer.render_offer_text(row, context=context)
        elif _g_mode == "query":
            text = _g_renderer.render_query_text(row, context=context)
        else:
            raise NotImplementedError("pair_score not supported with workers")

        if not text:
            skipped += 1
            continue

        if _g_include_text:
            key = "offer_text" if _g_mode == "offer" else "query_text"
            prepared_row[key] = text

        prepared_rows.append(prepared_row)
        texts.append(text)

    token_ids = []
    if texts:
        raw_ids = _g_tokenizer(
            texts, padding=False, truncation=True, max_length=_g_max_length,
        )["input_ids"]
        # Pack as numpy for fast pickling (avoid Python list-of-list overhead)
        lengths = np.array([len(ids) for ids in raw_ids], dtype=np.int32)
        flat_ids = np.concatenate([np.array(ids, dtype=np.int32) for ids in raw_ids])
        token_ids = (flat_ids, lengths)

    return prepared_rows, token_ids, skipped, row_number_start


def _pad_and_stack(token_ids_batch, pad_token_id):
    """Pad a list of token ID lists to the batch max length, return input_ids + attention_mask tensors."""
    max_len = max(len(ids) for ids in token_ids_batch)
    batch_size = len(token_ids_batch)
    input_ids = np.full((batch_size, max_len), pad_token_id, dtype=np.int64)
    attention_mask = np.zeros((batch_size, max_len), dtype=np.int64)
    for i, ids in enumerate(token_ids_batch):
        length = len(ids)
        input_ids[i, :length] = ids
        attention_mask[i, :length] = 1
    return {
        "input_ids": torch.from_numpy(input_ids),
        "attention_mask": torch.from_numpy(attention_mask),
    }


def _encode_sorted_batches(sorted_ids, encode_batch_size, pad_token_id, model, device):
    """Run GPU forward pass on pre-sorted token ID lists. Returns list of embedding tensors."""
    chunk_starts = list(range(0, len(sorted_ids), encode_batch_size))
    encoded_batches = []
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    def pad_chunk(start):
        chunk = sorted_ids[start : start + encode_batch_size]
        return _pad_and_stack(chunk, pad_token_id)

    with ThreadPoolExecutor(max_workers=1) as pad_executor:
        prefetched = pad_executor.submit(pad_chunk, chunk_starts[0])

        with torch.inference_mode(), amp_context:
            for i, _ in enumerate(chunk_starts):
                cpu_inputs = prefetched.result()
                if i + 1 < len(chunk_starts):
                    prefetched = pad_executor.submit(pad_chunk, chunk_starts[i + 1])

                device_inputs = {
                    name: tensor.to(device, non_blocking=True)
                    for name, tensor in cpu_inputs.items()
                }
                embeddings = model.encode(device_inputs)
                encoded_batches.append(embeddings.detach().cpu().to(dtype=torch.float32))

    return encoded_batches


def encode_texts(
    model,
    tokenizer,
    texts,
    max_length,
    encode_batch_size,
    device,
):
    if not texts:
        return torch.empty(0)

    pad_token_id = tokenizer.pad_token_id or 0

    # Tokenize all texts at once without padding
    all_input_ids = tokenizer(
        texts, padding=False, truncation=True, max_length=max_length,
    )["input_ids"]

    # Sort by token length for optimal batching
    sort_order = sorted(range(len(all_input_ids)), key=lambda i: len(all_input_ids[i]))
    sorted_ids = [all_input_ids[i] for i in sort_order]

    encoded_batches = _encode_sorted_batches(
        sorted_ids, encode_batch_size, pad_token_id, model, device,
    )

    sorted_embeddings = torch.cat(encoded_batches, dim=0)
    inverse_order = torch.empty(len(sort_order), dtype=torch.long)
    inverse_order[torch.tensor(sort_order, dtype=torch.long)] = torch.arange(
        len(sort_order), dtype=torch.long
    )
    return sorted_embeddings.index_select(0, inverse_order)


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
        embeddings = encode_texts(
            model, tokenizer, query_texts, query_max_length, encode_batch_size, device,
        )
        output_values = serialize_embeddings(
            quantize_embeddings(embeddings, embedding_precision), embedding_precision,
        )
        embedding_dim = embeddings.size(1)
    elif mode == "offer":
        embeddings = encode_texts(
            model, tokenizer, offer_texts, offer_max_length, encode_batch_size, device,
        )
        output_values = serialize_embeddings(
            quantize_embeddings(embeddings, embedding_precision), embedding_precision,
        )
        embedding_dim = embeddings.size(1)
    else:
        query_embeddings = encode_texts(
            model, tokenizer, query_texts, query_max_length, encode_batch_size, device,
        )
        offer_embeddings = encode_texts(
            model, tokenizer, offer_texts, offer_max_length, encode_batch_size, device,
        )
        output_values = score_embedding_pairs(
            query_embeddings, offer_embeddings, embedding_precision,
        ).tolist()
        embedding_dim = query_embeddings.size(1)

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


def _encode_from_numpy(
    flat_ids, lengths, prepared_rows, model, encode_batch_size,
    device, pad_token_id, output_column, input_schema, embedding_precision, mode,
):
    """Sort pre-tokenized numpy-packed IDs by length, pad, forward, build table."""
    n = len(lengths)
    if n == 0:
        return None

    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths.astype(np.int64), out=offsets[1:])

    sort_order = np.argsort(lengths, kind="mergesort")
    sorted_lengths = lengths[sort_order]

    encoded_batches = []
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    chunk_starts = list(range(0, n, encode_batch_size))

    use_pinned = device.type == "cuda"

    def pad_chunk_numpy(start):
        end = min(start + encode_batch_size, n)
        batch_size = end - start
        max_len = int(sorted_lengths[end - 1])
        input_ids_np = np.full((batch_size, max_len), pad_token_id, dtype=np.int64)
        attn_mask_np = np.zeros((batch_size, max_len), dtype=np.int64)
        for i in range(batch_size):
            idx = int(sort_order[start + i])
            l = int(lengths[idx])
            s = int(offsets[idx])
            input_ids_np[i, :l] = flat_ids[s:s + l]
            attn_mask_np[i, :l] = 1
        ids_t = torch.from_numpy(input_ids_np)
        mask_t = torch.from_numpy(attn_mask_np)
        if use_pinned:
            ids_t = ids_t.pin_memory()
            mask_t = mask_t.pin_memory()
        return {"input_ids": ids_t, "attention_mask": mask_t}

    with ThreadPoolExecutor(max_workers=1) as pad_executor:
        prefetched = pad_executor.submit(pad_chunk_numpy, chunk_starts[0])

        with torch.inference_mode(), amp_context:
            for i, _ in enumerate(chunk_starts):
                cpu_inputs = prefetched.result()
                if i + 1 < len(chunk_starts):
                    prefetched = pad_executor.submit(pad_chunk_numpy, chunk_starts[i + 1])

                device_inputs = {
                    name: tensor.to(device, non_blocking=True)
                    for name, tensor in cpu_inputs.items()
                }
                embeddings = model.encode(device_inputs)
                # Keep on GPU — cat + unsort on GPU, single D2H at end
                encoded_batches.append(embeddings.detach())

    all_embeddings = torch.cat(encoded_batches, dim=0)
    inverse = torch.from_numpy(sort_order.astype(np.int64)).to(device)
    inverse_perm = torch.empty(n, dtype=torch.long, device=device)
    inverse_perm[inverse] = torch.arange(n, dtype=torch.long, device=device)
    embeddings = all_embeddings.index_select(0, inverse_perm).cpu().to(dtype=torch.float32)

    output_values = serialize_embeddings(
        quantize_embeddings(embeddings, embedding_precision), embedding_precision,
    )
    return build_output_table(
        prepared_rows, output_column, output_values, mode,
        input_schema, embedding_precision, embeddings.size(1),
    )


def _run_with_workers(
    pool, parquet_file, column_rename, limit_rows,
    read_batch_size, encode_batch_size, row_number,
    model, device, pad_token_id, output_column, renamed_schema,
    embedding_precision, mode, writer,
):
    """Stream parquet → multiprocess text rendering → GPU encode → write.

    Accumulates multiple worker results before encoding to reduce GPU idle gaps.
    """
    ACCUMULATE_TARGET = 4096

    processed_rows = 0
    written_rows = 0
    skipped_rows = 0

    accumulated_prepared = []
    accumulated_flat_ids = []
    accumulated_lengths = []

    def flush_accumulated():
        nonlocal written_rows
        if not accumulated_prepared:
            return
        all_flat = np.concatenate(accumulated_flat_ids)
        all_lengths = np.concatenate(accumulated_lengths)

        table = _encode_from_numpy(
            all_flat, all_lengths, list(accumulated_prepared),
            model, encode_batch_size, device, pad_token_id,
            output_column, renamed_schema, embedding_precision, mode,
        )
        if table is not None:
            writer.write_table(table)
            written_rows += table.num_rows

        accumulated_prepared.clear()
        accumulated_flat_ids.clear()
        accumulated_lengths.clear()

    # Collect batches of work items for imap
    def batch_items():
        nonlocal processed_rows, row_number
        for batch in parquet_file.iter_batches(batch_size=read_batch_size):
            batch = apply_column_rename(batch, column_rename)
            rows = batch.to_pylist()
            if limit_rows is not None:
                remaining = int(limit_rows) - processed_rows
                if remaining <= 0:
                    break
                rows = rows[:remaining]
            processed_rows += len(rows)
            if not rows:
                break
            rn = row_number
            row_number += len(rows)
            yield (rows, rn)

    for prepared_rows, packed_token_ids, batch_skipped, _ in pool.imap(
        _worker_render_and_tokenize, batch_items(), chunksize=1
    ):
        skipped_rows += batch_skipped

        if not prepared_rows:
            continue

        flat_ids, lengths = packed_token_ids
        accumulated_prepared.extend(prepared_rows)
        accumulated_flat_ids.append(flat_ids)
        accumulated_lengths.append(lengths)

        if len(accumulated_prepared) >= ACCUMULATE_TARGET:
            flush_accumulated()

    flush_accumulated()

    return written_rows, processed_rows, skipped_rows


def run_inference(args):
    from embedding_train.index_build import ParquetSource

    mode = resolve_inference_mode(args.mode)
    output_column = resolve_output_column(mode, args.output_column)
    device = resolve_device(args.device)
    embedding_precision = resolve_embedding_precision(args.embedding_precision)
    num_workers = int(getattr(args, "num_workers", 0) or 0)

    model, cfg = load_embedding_module_from_checkpoint(
        args.checkpoint, map_location="cpu"
    )

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

    offer_max_length = int(args.max_offer_length) or int(cfg.data.max_offer_length)
    query_max_length = int(args.max_query_length) or int(cfg.data.max_query_length)
    max_length = offer_max_length if mode == "offer" else query_max_length

    # Set up multiprocess pool BEFORE CUDA init (fork-safe)
    pool = None
    if num_workers > 0 and mode in ("offer", "query"):
        global _g_renderer, _g_tokenizer, _g_mode, _g_copy_columns, _g_include_text, _g_max_length
        _g_renderer = renderer
        _g_tokenizer = tokenizer
        _g_mode = mode
        _g_copy_columns = copy_columns
        _g_include_text = bool(args.include_text)
        _g_max_length = max_length
        ctx = multiprocessing.get_context("fork")
        pool = ctx.Pool(num_workers)

    # Now move model to GPU
    model = model.to(device)
    model.eval()

    writer = IncrementalParquetWriter(args.output, args.compression, args.overwrite)
    row_number = int(getattr(args, "row_number_offset", 0) or 0)
    pad_token_id = tokenizer.pad_token_id or 0

    processed_rows = 0
    written_rows = 0
    skipped_rows = 0

    read_batch_size = int(args.read_batch_size)
    encode_batch_size = int(args.encode_batch_size)

    use_profile = getattr(args, "profile", False)
    profiler = None
    if use_profile:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=False,
        )

    wall_start = time.perf_counter()
    prof_ctx = profiler if profiler is not None else nullcontext()

    try:
        with prof_ctx:
            if pool is not None:
                written_rows, processed_rows, skipped_rows = _run_with_workers(
                    pool, parquet_file, column_rename, args.limit_rows,
                    read_batch_size, encode_batch_size, row_number,
                    model, device, pad_token_id, output_column, renamed_schema,
                    embedding_precision, mode, writer,
                )
            else:
                for batch in parquet_file.iter_batches(batch_size=read_batch_size):
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
                        encode_batch_size=encode_batch_size,
                        device=device,
                        row_number=row_number,
                        embedding_precision=embedding_precision,
                    )
                    skipped_rows += batch_skipped_rows

                    if table is None:
                        continue

                    writer.write_table(table)
                    written_rows += table.num_rows
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        writer.close()

    wall_time = time.perf_counter() - wall_start
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
    print(f"Wall time: {wall_time:.1f}s")
    print(f"Throughput: {rows_per_sec:.1f} rows/sec")

    if profiler is not None:
        print("\n--- torch.profiler CUDA kernel summary ---")
        print(
            profiler.key_averages().table(
                sort_by="cuda_time_total", row_limit=25,
            )
        )


def main(argv=None):
    load_dotenv()
    torch.set_float32_matmul_precision("high")
    args = build_arg_parser().parse_args(argv)
    run_inference(args)


if __name__ == "__main__":
    main()

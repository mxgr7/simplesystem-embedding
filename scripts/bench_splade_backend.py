"""Benchmark the packed SPLADE document endpoint on rendered input texts."""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "splade-service"))

from codec import unpack_sparse_batch  # noqa: E402


def load_inputs(path, limit):
    inputs = []
    with open(path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("{") or line.startswith('"'):
                value = json.loads(line)
                if isinstance(value, dict):
                    value = value.get("text") or value.get("rendered")
            else:
                value = line
            if not isinstance(value, str) or not value:
                raise ValueError("each input line must contain a rendered text")
            inputs.append(value)
            if limit and len(inputs) >= limit:
                break
    if not inputs:
        raise ValueError("input file is empty")
    return inputs


def main():
    load_dotenv(REPO / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--api-key", default=os.environ.get("SPLADE_BACKEND_API_KEY", ""))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=100_000)
    parser.add_argument("--warmup-batches", type=int, default=4)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit < 1 or args.warmup_batches < 0:
        parser.error("batch size and limit must be positive; warmup must be non-negative")

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    inputs = load_inputs(args.inputs, args.limit)
    timings = []
    encoded = nonzeros = packed_bytes = 0
    with httpx.Client(
        base_url=args.url.rstrip("/"), headers=headers,
        timeout=httpx.Timeout(300, connect=10),
    ) as client:
        metadata = client.get("/metadata").raise_for_status().json()
        if not metadata.get("optimized_document_encoder"):
            raise RuntimeError("backend does not advertise an optimized document encoder")
        if "splade-u16-f16-batch-v1" not in metadata.get("document_transports", []):
            raise RuntimeError("backend does not advertise packed document transport")
        batches = [
            inputs[start:start + args.batch_size]
            for start in range(0, len(inputs), args.batch_size)
        ]
        for batch in batches[:args.warmup_batches]:
            response = client.post(
                "/encode-packed", json={"inputs": batch, "document": True}
            )
            response.raise_for_status()
            if len(unpack_sparse_batch(response.content)) != len(batch):
                raise RuntimeError("warmup response cardinality mismatch")

        started = time.perf_counter()
        for batch in batches:
            batch_started = time.perf_counter()
            response = client.post(
                "/encode-packed", json={"inputs": batch, "document": True}
            )
            response.raise_for_status()
            if response.headers.get("content-type", "").split(";")[0] != (
                "application/vnd.simplesystem.splade-u16-f16-batch-v1"
            ):
                raise RuntimeError("backend returned the wrong packed media type")
            vectors = unpack_sparse_batch(response.content)
            elapsed = time.perf_counter() - batch_started
            if len(vectors) != len(batch):
                raise RuntimeError("response cardinality mismatch")
            timings.append(elapsed)
            encoded += len(batch)
            nonzeros += sum(len(vector) for vector in vectors)
            packed_bytes += len(response.content)
        total_seconds = time.perf_counter() - started

    result = {
        "metadata": metadata,
        "documents": encoded,
        "batch_size": args.batch_size,
        "seconds": total_seconds,
        "documents_per_second": encoded / total_seconds,
        "mean_nonzeros": nonzeros / encoded,
        "packed_bytes_per_document": packed_bytes / encoded,
        "batch_latency_p50_seconds": statistics.median(timings),
        "batch_latency_p95_seconds": sorted(timings)[
            min(len(timings) - 1, int(len(timings) * 0.95))
        ],
        "projected_offer_encode_hours": 113_560_531 / (encoded / total_seconds) / 3600,
        "passes_4900_docs_per_second": encoded / total_seconds >= 4_900,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n")


if __name__ == "__main__":
    main()

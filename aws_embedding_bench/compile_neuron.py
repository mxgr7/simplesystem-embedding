"""Trace the SentenceTransformer pipeline to a single Neuron `.pt`.

The fine-tuned simplesystem-embedding model is a SentenceTransformers
module stack:

  Transformer (XLM-RoBERTa base) -> Mean-Pool(mask) -> Dense(768->128, no bias)
  -> L2-Normalize

We wrap all four steps in a single nn.Module so the Neuron tracer captures
the entire forward pass into one .pt — the server then just tokenizes and
calls model(input_ids, attention_mask), getting back normalized 128-d
embeddings directly. No post-processing on the CPU.

Args:
  --source PATH         Source SentenceTransformer dir (e.g. /opt/source-model)
  --output PATH         Where to write the traced .pt
  --batch-size N        Fixed batch dim of the trace (must match server)
  --seq-len N           Fixed sequence length (must match server tokenizer max)
  --auto-cast TYPE      "bf16" or "fp16". inf2 supports both; inf1 supports
                        fp16 only — bf16 silently downgrades to fp16 there.
  --neuron-version V    1 = Inferentia1 (torch-neuron), 2 = Inferentia2
                        (torch-neuronx). Default 2.

Output: single .pt file at --output. ~700 MB for XLM-R base at fp16/bf16.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModel

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03dZ | %(levelname)s | compile] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logging.Formatter.converter = time.gmtime
log = logging.getLogger("compile")


class STWrapper(torch.nn.Module):
    """Encoder + mean-pool + 768->128 dense [+ optional L2-normalize], baked
    into one forward(). Inputs match the HF tokenizer's output for XLM-R.

    `skip_normalize=True` for inf1 — its neuron-cc can't trace
    `F.normalize` (the internal `.expand_as()` call fails BroadcastTo at
    compile time, "Dimensions must be equal, but are 8 and 128"). The
    server's _postprocess always L2-normalizes the output anyway, so
    dropping it from the trace is functionally a no-op.
    """

    def __init__(self, source_dir: str, skip_normalize: bool = False) -> None:
        super().__init__()
        log.info("loading encoder from %s", source_dir)
        self.encoder = AutoModel.from_pretrained(source_dir)
        self.skip_normalize = skip_normalize

        dense_path = os.path.join(source_dir, "2_Dense", "pytorch_model.bin")
        log.info("loading Dense head from %s", dense_path)
        dense_state = torch.load(dense_path, map_location="cpu", weights_only=True)
        weight: torch.Tensor | None = None
        for k, v in dense_state.items():
            if k.endswith("weight") and v.dim() == 2:
                weight = v
                break
        if weight is None:
            raise RuntimeError(f"no 2D weight tensor in {dense_path}")
        log.info("dense weight shape: %s", tuple(weight.shape))
        # Register as buffer (not parameter) so it doesn't get re-initialised
        # and stays fp32 until the auto-cast pass downcasts during compile.
        self.register_buffer("dense_weight", weight)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = enc.last_hidden_state                       # [B, S, 768]
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts                                  # [B, 768]
        projected = F.linear(pooled, self.dense_weight)           # [B, 128]
        if self.skip_normalize:
            return projected                                      # [B, 128] non-normalized
        return F.normalize(projected, p=2, dim=1)                 # [B, 128]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True,
                   help="SentenceTransformer source dir")
    p.add_argument("--output", required=True,
                   help="Destination .pt path")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--auto-cast", choices=["bf16", "fp16", "none"], default="bf16")
    p.add_argument("--neuron-version", type=int, choices=[1, 2], default=2,
                   help="1=Inferentia1 (torch-neuron); 2=Inferentia2 (torch-neuronx)")
    args = p.parse_args()

    # Import the right tracer based on target chip. inf1 and inf2 are
    # separate SDK lineages with different op coverage and compiler binaries
    # (neuron-cc vs neuronx-cc). We never want both in scope at once — they
    # share namespaces in dangerous ways.
    if args.neuron_version == 1:
        import torch_neuron as neuron  # type: ignore[import-not-found]
        log.info("torch=%s torch_neuron=%s (Inferentia1)",
                 torch.__version__, neuron.__version__)
    else:
        import torch_neuronx as neuron  # type: ignore[import-not-found]
        log.info("torch=%s torch_neuronx=%s (Inferentia2)",
                 torch.__version__, neuron.__version__)

    log.info("config: source=%s output=%s bs=%d sl=%d auto-cast=%s neuron-v=%d",
             args.source, args.output, args.batch_size, args.seq_len,
             args.auto_cast, args.neuron_version)

    # Skip the on-chip F.normalize for inf1 — neuron-cc (PyTorch 1.13 era)
    # can't compile the internal expand_as call. The server's _postprocess
    # always L2-normalises the output, so this is functionally a no-op.
    skip_normalize = (args.neuron_version == 1)
    if skip_normalize:
        log.info("skipping on-chip F.normalize (server will normalize post-hoc)")
    wrapper = STWrapper(args.source, skip_normalize=skip_normalize)
    wrapper.eval()
    n_params = sum(p.numel() for p in wrapper.parameters()) \
             + sum(b.numel() for b in wrapper.buffers())
    log.info("wrapper built: %.1fM params/buffers", n_params / 1e6)

    # Smoke test on CPU before tracing — catches a broken source dir or
    # tokenizer-config mismatch in seconds, rather than after a 10-min
    # Neuron compile.
    log.info("CPU smoke test")
    smoke_ids = torch.zeros((args.batch_size, args.seq_len), dtype=torch.long)
    smoke_mask = torch.ones((args.batch_size, args.seq_len), dtype=torch.long)
    with torch.no_grad():
        smoke_out = wrapper(smoke_ids, smoke_mask)
    log.info("  cpu output shape: %s  dtype=%s  first row sample[:4]=%s",
             tuple(smoke_out.shape), smoke_out.dtype,
             smoke_out[0, :4].tolist())
    if smoke_out.shape != (args.batch_size, 128):
        raise RuntimeError(f"unexpected smoke output shape: {tuple(smoke_out.shape)}")

    # Compiler-arg names differ between Inferentia1 (neuron-cc) and
    # Inferentia2 (neuronx-cc). Most notable: inf1 lacks bf16; if asked, we
    # downgrade to fp16 with a warning.
    if args.neuron_version == 1:
        if args.auto_cast == "bf16":
            log.warning("inf1 does not support bf16; falling back to fp16")
            args.auto_cast = "fp16"
        # neuron-cc takes --fp32-cast (or --fast-math) — but the standard
        # auto-cast flag works for both. Keep it simple.
        compiler_args: list[str] = []
        if args.auto_cast == "fp16":
            compiler_args = ["--fp32-cast", "matmult"]  # matmuls go fp16, rest stays fp32
        # Note: inf1 ignores --auto-cast all syntax; we use the older
        # --fp32-cast knob.
    else:
        compiler_args = []
        if args.auto_cast == "bf16":
            compiler_args = ["--auto-cast", "all", "--auto-cast-type", "bf16"]
        elif args.auto_cast == "fp16":
            compiler_args = ["--auto-cast", "all", "--auto-cast-type", "fp16"]
    log.info("compiler_args: %s", compiler_args)

    log.info("neuron.trace start (this typically takes 5-15min)")
    t0 = time.perf_counter()
    with torch.no_grad():
        traced = neuron.trace(
            wrapper,
            (smoke_ids, smoke_mask),
            compiler_args=compiler_args,
        )
    log.info("trace done in %.1fs", time.perf_counter() - t0)

    log.info("saving to %s", args.output)
    torch.jit.save(traced, args.output)
    sz = os.path.getsize(args.output)
    log.info("saved: %.1f MB", sz / (1024 * 1024))

    log.info("post-save smoke (verify the saved file loads + runs)")
    reloaded = torch.jit.load(args.output)
    with torch.no_grad():
        check_out = reloaded(smoke_ids, smoke_mask)
    log.info("  reloaded output shape: %s  first row sample[:4]=%s",
             tuple(check_out.shape), check_out[0, :4].tolist())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

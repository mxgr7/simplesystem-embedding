"""Quality eval — CE-alone classification F1 on the training-time val split.

Usage:
  uv run python autoresearch/cross-encoder-inference/eval_val.py \\
    --ckpt /abs/path/to/soup.ckpt \\
    [--data-path /abs/path/to/queries_offers_merged_labeled.parquet] \\
    [--device cuda] [--autocast bf16] [--batch-size 64] [--num-workers 4]

Reproduces the training program's val/cls/{micro_f1, macro_f1, per-class f1}
without invoking the Lightning Trainer. Uses the *same* CrossEncoderDataModule
(query-id-based split, same val_fraction, same seed) so the metric is
bit-comparable to what `cross-encoder-train` reports.

Output (one metric per line, machine-greppable):
    val/cls/micro_f1=0.8945
    val/cls/macro_f1=0.7723
    val/cls/f1_irrelevant=0.6042
    val/cls/f1_complement=0.5377
    val/cls/f1_substitute=0.4416
    val/cls/f1_exact=0.9302
    val/cls/evaluated_pairs=76048

This script is the canonical quality measurement for the cross-encoder
inference autoresearch program. The autoresearch agent calls it, parses the
two F1 lines, and gates keep/discard against the floor in program.md.

Only Lightning .ckpt format is supported here. ONNX / TensorRT / GPTQ /
distilled-student variants must add their own loader path — extend the
`load_model` function rather than maintaining a parallel script, so the
eval pipeline (split, batching, metric) stays identical across variants.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from cross_encoder_train.data import CrossEncoderDataModule
from cross_encoder_train.metrics import compute_classification_metrics
from cross_encoder_train.model import CrossEncoderModule


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_DATA_PATH = (
    REPO_ROOT.parent / "data" / "queries_offers_esci" / "queries_offers_merged_labeled.parquet"
)

AUTOCAST_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": None,
    "off": None,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", required=True, help="Path to Lightning .ckpt.")
    p.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    p.add_argument("--config-name", default="cross_encoder")
    p.add_argument("--data-path", default=str(DEFAULT_DATA_PATH),
                   help="Absolute path to the labeled parquet (overrides cfg.data.path).")
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto).")
    p.add_argument("--autocast", default="bf16", choices=list(AUTOCAST_DTYPES.keys()))
    p.add_argument("--weights-dtype", default="fp32",
                   choices=["fp32", "bf16", "fp16"],
                   help="Cast model weights to this dtype after load. fp32 = baseline "
                        "(weights as saved), bf16/fp16 mirrors serve-side weight cast.")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Eval batch size (default: 64; larger than train default since no grads).")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--subset-rows", type=int, default=10000,
                   help="Eval on a deterministic uniform-random subset of N val rows "
                        "(seed=0). 0 = full val set (~76k rows, ~25 min on a 4090). "
                        "Default 10k is the iteration-time canonical metric "
                        "(~3-4 min, macro_f1 noise ~0.013, micro_f1 noise ~0.005).")
    p.add_argument("--subset-seed", type=int, default=0,
                   help="Seed for the subset draw (default 0; keep fixed across runs).")
    p.add_argument("--limit-batches", type=int, default=None,
                   help="Smoke test: only evaluate N batches. Skips final F1 print if set.")
    return p.parse_args()


_WEIGHTS_DTYPE = {
    "fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16,
}


def load_model(ckpt_path: str, cfg, device: str,
               weights_dtype: str = "fp32") -> CrossEncoderModule:
    """Load a Lightning .ckpt into a CrossEncoderModule on `device`.

    Mirrors `Reranker.__init__` in src/cross_encoder_serve/inference.py:
    forces model.compile=false at load time and strips the `_orig_mod.`
    prefix the wrapped encoder added when the ckpt was saved.

    `weights_dtype` mirrors the serve-side weight cast (fp32 keeps baseline,
    bf16/fp16 quantizes weights to that dtype). Activation autocast is
    independent and configured by the caller.
    """
    serve_cfg = OmegaConf.merge(cfg, OmegaConf.create({"model": {"compile": False}}))
    model = CrossEncoderModule(cfg=serve_cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = {k.replace("._orig_mod.", "."): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    cast_dtype = _WEIGHTS_DTYPE[weights_dtype]
    if cast_dtype is not None and device == "cuda":
        model.to(cast_dtype)
    return model


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = AUTOCAST_DTYPES[args.autocast]

    with initialize_config_dir(config_dir=args.config_dir, version_base="1.3"):
        cfg = compose(config_name=args.config_name)
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create({
            "data": {
                "path": args.data_path,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
            },
            "model": {"compile": False},
        }),
    )

    print(f"[eval] device={device} autocast={args.autocast} batch_size={args.batch_size} "
          f"data={args.data_path}", file=sys.stderr)
    print(f"[eval] ckpt={args.ckpt}", file=sys.stderr)

    dm = CrossEncoderDataModule(cfg)
    dm.setup("fit")

    # Optional: subsample val to speed up iteration. Uniform random with a fixed
    # seed → deterministic across runs, so subset F1 is bit-comparable across
    # checkpoints. Class distribution is preserved in expectation.
    full_val_rows = len(dm.val_dataset)
    if args.subset_rows and args.subset_rows > 0 and args.subset_rows < full_val_rows:
        import random
        rng = random.Random(args.subset_seed)
        all_records = list(dm.val_dataset.records)
        rng.shuffle(all_records)
        dm.val_dataset.records = all_records[: args.subset_rows]
        print(f"[eval] val subset: {args.subset_rows}/{full_val_rows} rows "
              f"(seed={args.subset_seed}; uniform random)", file=sys.stderr)
    else:
        print(f"[eval] val full: {full_val_rows} rows "
              f"(no subset)", file=sys.stderr)

    val_loader = dm.val_dataloader()
    print(f"[eval] val_rows={len(dm.val_dataset)} "
          f"val_queries_full={dm.dataset_stats.get('val_queries', '?')}", file=sys.stderr)

    is_onnx = args.ckpt.endswith(".onnx")
    if is_onnx:
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        sess = ort.InferenceSession(args.ckpt, providers=providers)
        model = None
        print(f"[eval] runtime=onnx providers={sess.get_providers()}", file=sys.stderr)
    else:
        model = load_model(args.ckpt, cfg, device, weights_dtype=args.weights_dtype)
        sess = None
        print(f"[eval] weights_dtype={args.weights_dtype}", file=sys.stderr)

    preds: list[int] = []
    targets: list[int] = []
    with torch.no_grad():
        for n_done, batch in enumerate(val_loader, start=1):
            if is_onnx:
                ort_inputs = {
                    "input_ids": batch["inputs"]["input_ids"].numpy(),
                    "attention_mask": batch["inputs"]["attention_mask"].numpy(),
                    "token_type_ids": batch["inputs"]["token_type_ids"].numpy(),
                }
                logits_np = sess.run(["logits"], ort_inputs)[0]
                logits = torch.from_numpy(logits_np)
            else:
                inputs = {k: v.to(device, non_blocking=True) for k, v in batch["inputs"].items()}
                if autocast_dtype is not None and device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                        logits = model(inputs)
                else:
                    logits = model(inputs)
            preds.extend(logits.float().argmax(dim=-1).cpu().tolist())
            targets.extend(batch["labels"].tolist())
            if n_done % 50 == 0:
                print(f"[eval] batch {n_done}: seen {len(preds)} pairs", file=sys.stderr)
            if args.limit_batches is not None and n_done >= args.limit_batches:
                print(f"[eval] --limit-batches={args.limit_batches} hit; stopping early "
                      "(metrics below are partial)", file=sys.stderr)
                break

    metrics = compute_classification_metrics(preds, targets)
    print(f"val/cls/micro_f1={metrics['micro_f1']:.4f}")
    print(f"val/cls/macro_f1={metrics['macro_f1']:.4f}")
    print(f"val/cls/f1_irrelevant={metrics['f1_irrelevant']:.4f}")
    print(f"val/cls/f1_complement={metrics['f1_complement']:.4f}")
    print(f"val/cls/f1_substitute={metrics['f1_substitute']:.4f}")
    print(f"val/cls/f1_exact={metrics['f1_exact']:.4f}")
    print(f"val/cls/evaluated_pairs={int(metrics['evaluated_pairs'])}")


if __name__ == "__main__":
    main()

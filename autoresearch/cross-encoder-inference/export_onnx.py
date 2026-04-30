"""Export a Lightning CE checkpoint to ONNX.

Loads the student CrossEncoderModule via the same code path as eval/serve,
traces with `torch.onnx.export` using:
  - 3 inputs: input_ids, attention_mask, token_type_ids — all (B, S=512)
  - dynamic batch dim only (S frozen at 512 to match the bench worst-case)
  - 1 output: logits (B, 4)
After export, re-loads with onnxruntime, runs on a 2-offer fixture, and
compares logits to the torch path. Fails loudly if max-abs diff > rtol/atol.

Run:
  LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \\
    uv run --extra train python autoresearch/cross-encoder-inference/export_onnx.py \\
    --ckpt /abs/path/to/student.ckpt \\
    --out  /abs/path/to/student.onnx \\
    [--config-name distill_cross_encoder] [--weights-dtype bf16] [--opset 17]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_val import load_model, _WEIGHTS_DTYPE  # reuse the canonical loader


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"
SEQ_LEN = 512


class _EncoderHead(torch.nn.Module):
    """Wrap encoder + classifier into a single forward(input_ids, attention_mask, token_type_ids) -> logits."""

    def __init__(self, lightning_module):
        super().__init__()
        self.encoder = lightning_module.encoder
        self.dropout = lightning_module.dropout  # eval: identity
        self.classifier = lightning_module.classifier

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        return self.classifier(cls)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    p.add_argument("--config-name", default="distill_cross_encoder")
    p.add_argument("--weights-dtype", default="fp32",
                   choices=list(_WEIGHTS_DTYPE.keys()))
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--validate-batch", type=int, default=2,
                   help="batch size to use for parity validation (default 2).")
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--dynamic-seq", action="store_true",
                   help="Also make sequence dim dynamic (slightly slower at runtime "
                        "but lets eval feed variable-S batches). Default off — bench "
                        "fixture is always S=512 so a fixed-S graph is faster.")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[export] device={device} ckpt={args.ckpt}", file=sys.stderr)

    with initialize_config_dir(config_dir=args.config_dir, version_base="1.3"):
        cfg = compose(config_name=args.config_name)
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"model": {"compile": False}}))

    module = load_model(args.ckpt, cfg, device, weights_dtype=args.weights_dtype)
    module.eval()
    head = _EncoderHead(module).to(device).eval()

    # Dummy inputs at (B=validate_batch, S=SEQ_LEN). Use real-ish token IDs (>= 100)
    # so the embedding lookup hits valid rows.
    B = int(args.validate_batch)
    rng = np.random.default_rng(0)
    input_ids = torch.tensor(rng.integers(100, 30000, size=(B, SEQ_LEN), dtype=np.int64), device=device)
    attention_mask = torch.ones((B, SEQ_LEN), dtype=torch.long, device=device)
    token_type_ids = torch.zeros((B, SEQ_LEN), dtype=torch.long, device=device)

    if args.dynamic_seq:
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "token_type_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        }
    else:
        dynamic_axes = {
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "token_type_ids": {0: "batch"},
            "logits": {0: "batch"},
        }
    print(f"[export] tracing → {out_path} (opset={args.opset}, dynamic_seq={args.dynamic_seq})", file=sys.stderr)
    with torch.no_grad():
        torch.onnx.export(
            head,
            (input_ids, attention_mask, token_type_ids),
            str(out_path),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
        )
    print(f"[export] wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)", file=sys.stderr)

    # Validate round-trip
    print("[export] validating ONNX vs torch…", file=sys.stderr)
    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(out_path), providers=providers)

    with torch.no_grad():
        torch_logits = head(input_ids, attention_mask, token_type_ids).float().cpu().numpy()
    onnx_logits = sess.run(
        ["logits"],
        {
            "input_ids": input_ids.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy(),
            "token_type_ids": token_type_ids.cpu().numpy(),
        },
    )[0]

    diff = np.max(np.abs(torch_logits - onnx_logits))
    print(f"[export] max|torch - onnx| = {diff:.6f}", file=sys.stderr)
    if diff > args.atol:
        print(f"[export] FAIL: diff exceeds atol={args.atol}", file=sys.stderr)
        sys.exit(1)
    print("[export] OK", file=sys.stderr)
    print(f"onnx_path={out_path}")
    print(f"max_abs_diff={diff:.6f}")


if __name__ == "__main__":
    main()

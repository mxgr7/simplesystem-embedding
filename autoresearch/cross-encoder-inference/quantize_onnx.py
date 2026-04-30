"""INT8 dynamic quantization of an ONNX model via onnxruntime.quantization.

Takes a float (fp32 or fp16) .onnx produced by export_onnx.py and writes an
int8 dynamic-quantized .onnx. Dynamic quant: weights → int8, activations
quantized at runtime. CUDAExecutionProvider has int8 matmul kernels (via
cuBLAS), TensorrtExecutionProvider can also run int8 with calibration.

Run:
  uv run python autoresearch/cross-encoder-inference/quantize_onnx.py \\
    --in /abs/path/to/student.onnx \\
    --out /abs/path/to/student.int8.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--weight-type", default="QInt8", choices=["QInt8", "QUInt8"])
    args = p.parse_args()

    from onnxruntime.quantization import quantize_dynamic, QuantType

    in_path = Path(args.in_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    qtype = QuantType.QInt8 if args.weight_type == "QInt8" else QuantType.QUInt8
    print(f"[quant] {in_path} → {out_path} (weight_type={args.weight_type})", file=sys.stderr)
    quantize_dynamic(str(in_path), str(out_path), weight_type=qtype)
    print(f"[quant] wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)", file=sys.stderr)
    print(f"int8_path={out_path}")


if __name__ == "__main__":
    main()

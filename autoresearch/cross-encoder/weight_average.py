"""Weight-average a list of Lightning checkpoints into one.

Loads N CrossEncoderModule checkpoints, averages all state_dict tensors
element-wise, saves a new checkpoint with the averaged weights. Useful
as a single-model alternative to prob-ensembling at inference (Model Soup
/ SWA style).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", nargs="+", required=True, help="Input checkpoint paths")
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    args = parser.parse_args()

    print(f"Averaging {len(args.ckpts)} checkpoints:")
    for c in args.ckpts:
        print(f"  {c}")

    state_dicts = []
    base_ckpt = None
    for path in args.ckpts:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if base_ckpt is None:
            base_ckpt = ckpt
        sd = ckpt["state_dict"]
        state_dicts.append(sd)

    # Sanity: same keys
    keys = list(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], 1):
        if set(sd.keys()) != set(keys):
            missing = set(keys) - set(sd.keys())
            extra = set(sd.keys()) - set(keys)
            raise ValueError(f"checkpoint {i} keys differ. missing={missing} extra={extra}")

    # Average each tensor
    averaged = {}
    for k in keys:
        tensors = [sd[k] for sd in state_dicts]
        # cast all to float32 for averaging stability, then back to original dtype
        orig_dtype = tensors[0].dtype
        stacked = torch.stack([t.to(torch.float32) for t in tensors], dim=0)
        avg = stacked.mean(dim=0).to(orig_dtype)
        averaged[k] = avg

    base_ckpt["state_dict"] = averaged
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base_ckpt, str(out))
    print(f"Wrote averaged checkpoint to {out}")


if __name__ == "__main__":
    main()

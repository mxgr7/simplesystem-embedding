"""Export a fine-tuned Lightning checkpoint to a sentence-transformers
directory that TEI can serve.

Layout written::

    <output>/
      config.json, model.safetensors, tokenizer.*    # base encoder
      sentence_bert_config.json                      # max_seq_length
      modules.json                                   # Transformer→Pooling→Dense→Normalize
      1_Pooling/config.json                          # mean pooling
      2_Dense/config.json + pytorch_model.bin        # 768→128 projection (no bias)
      3_Normalize/                                   # no params

The `RowTextRenderer` query template (typically ``query: {{ query_term }}``)
is applied by the caller before POSTing, so TEI does not need to know it.

Usage::

    uv run python scripts/export_for_tei.py \\
      --checkpoint /mnt/.../useful-cub-58/best-step=4880-...ckpt \\
      --output     /mnt/.../models/useful-cub-58-st
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


_ENCODER_PREFIX = "encoder."
_COMPILED_PREFIX = "encoder._orig_mod."


def strip_encoder_prefix(state_dict: dict) -> dict:
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith(_COMPILED_PREFIX):
            out[k[len(_COMPILED_PREFIX):]] = v
        elif k.startswith(_ENCODER_PREFIX):
            out[k[len(_ENCODER_PREFIX):]] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.output)
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists (use --overwrite): {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    print(f"loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    model_name = hp["model"]["model_name"]
    output_dim = int(hp["model"]["output_dim"])
    pooling = hp["model"]["pooling"]
    max_seq_length = int(hp["data"]["max_query_length"])

    if pooling != "mean":
        raise SystemExit(f"unsupported pooling: {pooling!r} (only 'mean' is exported)")

    state_dict = ckpt["state_dict"]
    encoder_sd = strip_encoder_prefix(state_dict)
    projection_weight = state_dict["projection.weight"]
    hidden_size = int(projection_weight.shape[1])
    if int(projection_weight.shape[0]) != output_dim:
        raise SystemExit(
            f"projection shape {tuple(projection_weight.shape)} "
            f"does not match output_dim={output_dim}"
        )

    print(f"exporting base encoder: {model_name} (hidden={hidden_size}, out={output_dim})")
    encoder = AutoModel.from_pretrained(model_name)
    missing, unexpected = encoder.load_state_dict(encoder_sd, strict=False)
    if unexpected:
        print(f"  warn: {len(unexpected)} unexpected keys (first few: {unexpected[:3]})")
    if missing:
        print(f"  warn: {len(missing)} missing keys (first few: {missing[:3]})")
    encoder.save_pretrained(out)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(out)

    (out / "sentence_bert_config.json").write_text(json.dumps(
        {"max_seq_length": max_seq_length, "do_lower_case": False}, indent=2
    ))

    (out / "modules.json").write_text(json.dumps([
        {"idx": 0, "name": "0", "path": "",
         "type": "sentence_transformers.models.Transformer"},
        {"idx": 1, "name": "1", "path": "1_Pooling",
         "type": "sentence_transformers.models.Pooling"},
        {"idx": 2, "name": "2", "path": "2_Dense",
         "type": "sentence_transformers.models.Dense"},
        {"idx": 3, "name": "3", "path": "3_Normalize",
         "type": "sentence_transformers.models.Normalize"},
    ], indent=2))

    pooling_dir = out / "1_Pooling"
    pooling_dir.mkdir()
    (pooling_dir / "config.json").write_text(json.dumps({
        "word_embedding_dimension": hidden_size,
        "pooling_mode_cls_token": False,
        "pooling_mode_mean_tokens": True,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
        "pooling_mode_weightedmean_tokens": False,
        "pooling_mode_lasttoken": False,
        "include_prompt": True,
    }, indent=2))

    dense_dir = out / "2_Dense"
    dense_dir.mkdir()
    (dense_dir / "config.json").write_text(json.dumps({
        "in_features": hidden_size,
        "out_features": output_dim,
        "bias": False,
        "activation_function": "torch.nn.modules.linear.Identity",
    }, indent=2))
    torch.save({"linear.weight": projection_weight.contiguous()},
               dense_dir / "pytorch_model.bin")

    (out / "3_Normalize").mkdir()

    print(f"done: {out}")


if __name__ == "__main__":
    main()

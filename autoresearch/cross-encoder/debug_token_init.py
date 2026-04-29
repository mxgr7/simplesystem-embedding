"""Inspect token IDs and embedding rows after resize_token_embeddings.

Reproduces the init path of CrossEncoderModule and CrossEncoderDataModule for
features.enabled=true, then prints:
  - tokenizer ids of the 8 feature tokens
  - shape of encoder.embeddings.word_embeddings.weight before/after resize
  - L2 norm distribution of existing rows vs new rows (mean+cov init)
  - effect of feeding a feature token through the encoder forward (no compile)
"""
from pathlib import Path
import torch
from hydra import compose, initialize_config_dir
from transformers import AutoModel, AutoTokenizer

from cross_encoder_train.features import feature_token_names

REPO = Path(__file__).resolve().parents[2]


def main():
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="cross_encoder",
            overrides=["data.features.enabled=true"],
        )

    model_name = cfg.model.model_name
    print(f"model: {model_name}")

    # ---- tokenizer step (mirrors data.py) ----
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    pre_len = len(tokenizer)
    feat_tokens = feature_token_names(cfg.data.features)
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": list(feat_tokens)}
    )
    print(f"tokenizer pre_len={pre_len}  added={added}  post_len={len(tokenizer)}")
    print(f"feature tokens: {feat_tokens}")
    feat_ids = tokenizer.convert_tokens_to_ids(feat_tokens)
    print(f"feature ids   : {feat_ids}")

    # Round-trip through tokenizer to confirm ids
    sample_text = " ".join(feat_tokens) + " test query"
    enc = tokenizer(sample_text, add_special_tokens=False)
    print(f"round-trip ids: {enc['input_ids'][:12]}")

    # ---- encoder step (mirrors model.py) ----
    encoder = AutoModel.from_pretrained(model_name, dtype=torch.float32)
    pre_vocab = encoder.config.vocab_size
    pre_emb = encoder.embeddings.word_embeddings
    pre_shape = tuple(pre_emb.weight.shape)
    pre_norms = pre_emb.weight.detach().float().norm(dim=1)
    print(
        f"encoder pre_vocab={pre_vocab}  pre_emb.shape={pre_shape}  "
        f"pre row-norm mean={pre_norms.mean():.4f}  std={pre_norms.std():.4f}"
    )

    extra = len(feat_tokens)
    new_size = pre_vocab + extra
    encoder.resize_token_embeddings(new_size, mean_resizing=False)
    post_emb = encoder.embeddings.word_embeddings
    post_shape = tuple(post_emb.weight.shape)
    print(
        f"encoder new_size={new_size}  post_emb.shape={post_shape}  "
        f"config.vocab_size={encoder.config.vocab_size}"
    )

    new_rows = post_emb.weight[pre_vocab:].detach().float()
    new_norms = new_rows.norm(dim=1)
    print(
        f"new rows norm: mean={new_norms.mean():.4f}  std={new_norms.std():.4f}  "
        f"min={new_norms.min():.4f}  max={new_norms.max():.4f}"
    )
    # cosine sim: new vs existing centroid
    centroid = pre_emb.weight.detach().float().mean(dim=0)
    cos = torch.nn.functional.cosine_similarity(new_rows, centroid.unsqueeze(0))
    print(f"new row cos-sim to existing centroid: {cos.tolist()}")
    # pairwise cos sim within new rows (should not be near 1 for all)
    nrn = new_rows / new_rows.norm(dim=1, keepdim=True)
    pair = nrn @ nrn.T
    off_diag = pair - torch.eye(extra)
    print(f"new-row pairwise cos: max_off_diag={off_diag.max():.4f}  mean_off_diag={(off_diag.sum()/(extra*(extra-1))):.4f}")

    # ---- forward pass: feed a single feature token, check [CLS] response ----
    encoder.eval()
    with torch.no_grad():
        cls = tokenizer.cls_token_id
        sep = tokenizer.sep_token_id
        feat = feat_ids[1]  # [EAN_MATCH]
        # input: [CLS] [EAN_MATCH] foo [SEP]
        plain_ids = torch.tensor([[cls, tokenizer.convert_tokens_to_ids("test"), sep]])
        feat_ids_t = torch.tensor([[cls, feat, tokenizer.convert_tokens_to_ids("test"), sep]])
        out_plain = encoder(plain_ids).last_hidden_state[:, 0, :]
        out_feat = encoder(feat_ids_t).last_hidden_state[:, 0, :]
        delta = (out_feat - out_plain).norm().item() / out_plain.norm().item()
    print(f"[CLS] delta norm with vs without feature token (relative): {delta:.4f}")


if __name__ == "__main__":
    main()

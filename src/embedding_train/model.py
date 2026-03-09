import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoModel

from embedding_train.losses import (
    cosine_bce_loss,
    in_batch_contrastive_loss,
    in_batch_triplet_loss,
)
from embedding_train.metrics import compute_ranking_metrics


def resolve_model_dtype(precision):
    normalized = str(precision).strip().lower()

    if normalized in {
        "16",
        "16-mixed",
        "bf16",
        "bf16-mixed",
        "32",
        "32-true",
        "transformer-engine",
        "transformer-engine-float16",
        "transformer-engine-bfloat16",
    }:
        return torch.float32

    if normalized == "16-true":
        return torch.float16

    if normalized == "bf16-true":
        return torch.bfloat16

    if normalized in {"64", "64-true"}:
        return torch.float64

    raise ValueError(f"Unsupported trainer precision: {precision}")


def resolve_loss_type(loss_type):
    normalized = str(loss_type).strip().lower()

    if normalized in {"bce", "contrastive", "triplet"}:
        return normalized

    raise ValueError(f"Unsupported loss type: {loss_type}")


class EmbeddingModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss_type = resolve_loss_type(cfg.model.loss_type)
        self.model_dtype = resolve_model_dtype(cfg.trainer.precision)
        self.encoder = AutoModel.from_pretrained(
            cfg.model.model_name,
            dtype=self.model_dtype,
        )
        self.validation_rows = []

        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

        if cfg.model.gradient_checkpointing and hasattr(
            self.encoder, "gradient_checkpointing_enable"
        ):
            self.encoder.gradient_checkpointing_enable()

        self.encoder = self.encoder.to(dtype=self.model_dtype)
        self.encoder.train()

    def forward(self, query_inputs, offer_inputs):
        query_embeddings = self.encode(query_inputs)
        offer_embeddings = self.encode(offer_inputs)
        scores = (query_embeddings * offer_embeddings).sum(dim=1)
        return query_embeddings, offer_embeddings, scores

    def training_step(self, batch, batch_idx):
        query_embeddings, offer_embeddings, scores = self(
            batch["query_inputs"], batch["offer_inputs"]
        )
        self.assert_finite(batch["labels"], "labels", batch_idx)
        self.assert_finite(scores, "scores", batch_idx)
        loss = self.compute_loss(
            batch,
            query_embeddings,
            offer_embeddings,
            scores,
        )
        self.assert_finite(loss, "train_loss", batch_idx)
        self.log(
            f"train/{self.loss_type}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch["labels"].size(0),
        )
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["labels"].size(0),
        )
        return loss

    def on_validation_epoch_start(self):
        self.validation_rows = []

    def validation_step(self, batch, batch_idx):
        query_embeddings, offer_embeddings, scores = self(
            batch["query_inputs"], batch["offer_inputs"]
        )
        self.assert_finite(batch["labels"], "labels", batch_idx)
        self.assert_finite(scores, "scores", batch_idx)
        loss = self.compute_loss(
            batch,
            query_embeddings,
            offer_embeddings,
            scores,
        )
        self.assert_finite(loss, "val_loss", batch_idx)

        self.log(
            f"val/{self.loss_type}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch["labels"].size(0),
        )
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["labels"].size(0),
        )

        cpu_scores = scores.detach().cpu().tolist()
        raw_labels = batch["raw_labels"]

        for query_id, score, raw_label in zip(
            batch["query_ids"], cpu_scores, raw_labels
        ):
            self.validation_rows.append(
                {
                    "query_id": query_id,
                    "score": float(score),
                    "raw_label": raw_label,
                }
            )

        return loss

    def on_validation_epoch_end(self):
        metrics = compute_ranking_metrics(self.validation_rows)

        self.log("val/ndcg_at_1", metrics["ndcg@1"], prog_bar=True)
        self.log("val/ndcg_at_5", metrics["ndcg@5"], prog_bar=True)
        self.log("val/ndcg_at_10", metrics["ndcg@10"], prog_bar=False)
        self.log("val/eligible_queries", metrics["eligible_queries"], prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.cfg.optimizer.lr),
            weight_decay=float(self.cfg.optimizer.weight_decay),
        )

    def compute_loss(self, batch, query_embeddings, offer_embeddings, scores):
        scale = float(self.cfg.model.similarity_scale)

        if self.loss_type == "bce":
            return cosine_bce_loss(
                scores,
                batch["labels"],
                scale=scale,
            )

        if self.loss_type == "contrastive":
            return in_batch_contrastive_loss(
                query_embeddings,
                offer_embeddings,
                batch["query_ids"],
                batch["labels"],
                scale=scale,
            )

        if self.loss_type == "triplet":
            return in_batch_triplet_loss(
                query_embeddings,
                offer_embeddings,
                batch["query_ids"],
                batch["labels"],
                margin=float(self.cfg.model.triplet_margin),
            )

        raise RuntimeError(f"Unsupported loss type: {self.loss_type}")

    def encode(self, inputs):
        outputs = self.encoder(**inputs)
        self.assert_finite(outputs.last_hidden_state, "last_hidden_state")
        pooled = self.pool_last_hidden_state(
            outputs.last_hidden_state, inputs["attention_mask"]
        )
        self.assert_finite(pooled, "pooled_embeddings")
        normalized = F.normalize(pooled, p=2, dim=1)
        self.assert_finite(normalized, "normalized_embeddings")
        return normalized

    def pool_last_hidden_state(self, hidden_state, attention_mask):
        if self.cfg.model.pooling != "mean":
            raise ValueError(f"Unsupported pooling: {self.cfg.model.pooling}")

        mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
        mask = mask.expand(hidden_state.size())
        masked_hidden_state = hidden_state * mask
        summed = masked_hidden_state.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def assert_finite(self, tensor, name, batch_idx=None):
        if torch.isfinite(tensor).all():
            return

        message = f"Non-finite tensor detected: {name}"
        if batch_idx is not None:
            message = f"{message} at batch {batch_idx}"
        raise RuntimeError(message)

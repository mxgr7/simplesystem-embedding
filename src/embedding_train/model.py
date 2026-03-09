import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoModel

from embedding_train.losses import cosine_bce_loss
from embedding_train.metrics import compute_ranking_metrics


class EmbeddingModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = AutoModel.from_pretrained(cfg.model.model_name)
        self.validation_rows = []

        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

        if cfg.model.gradient_checkpointing and hasattr(
            self.encoder, "gradient_checkpointing_enable"
        ):
            self.encoder.gradient_checkpointing_enable()

        self.encoder.train()

    def forward(self, query_inputs, offer_inputs):
        query_embeddings = self.encode(query_inputs)
        offer_embeddings = self.encode(offer_inputs)
        scores = (query_embeddings * offer_embeddings).sum(dim=1)
        return query_embeddings, offer_embeddings, scores

    def training_step(self, batch, batch_idx):
        _, _, scores = self(batch["query_inputs"], batch["offer_inputs"])
        self.assert_finite(batch["labels"], "labels", batch_idx)
        self.assert_finite(scores, "scores", batch_idx)
        loss = cosine_bce_loss(
            scores,
            batch["labels"],
            scale=float(self.cfg.model.similarity_scale),
        )
        self.assert_finite(loss, "train_loss", batch_idx)
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
        _, _, scores = self(batch["query_inputs"], batch["offer_inputs"])
        self.assert_finite(batch["labels"], "labels", batch_idx)
        self.assert_finite(scores, "scores", batch_idx)
        loss = cosine_bce_loss(
            scores,
            batch["labels"],
            scale=float(self.cfg.model.similarity_scale),
        )
        self.assert_finite(loss, "val_loss", batch_idx)

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["labels"].size(0),
        )

        cpu_scores = scores.detach().cpu().tolist()
        cpu_labels = batch["labels"].detach().cpu().tolist()

        for query_id, score, label in zip(batch["query_ids"], cpu_scores, cpu_labels):
            self.validation_rows.append(
                {
                    "query_id": query_id,
                    "score": float(score),
                    "label": int(label),
                }
            )

        return loss

    def on_validation_epoch_end(self):
        metrics = compute_ranking_metrics(self.validation_rows)

        self.log("val/recall_at_1", metrics["recall@1"], prog_bar=True)
        self.log("val/recall_at_5", metrics["recall@5"], prog_bar=True)
        self.log("val/recall_at_10", metrics["recall@10"], prog_bar=False)
        self.log("val/mrr_at_10", metrics["mrr@10"], prog_bar=True)
        self.log("val/eligible_queries", metrics["eligible_queries"], prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.cfg.optimizer.lr),
            weight_decay=float(self.cfg.optimizer.weight_decay),
        )

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

        mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
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

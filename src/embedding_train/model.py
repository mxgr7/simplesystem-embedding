import lightning as L
import torch
import torch.distributed
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoModel

from embedding_train.config import build_config_from_hyperparameters
from embedding_train.losses import (
    cosine_bce_loss,
    in_batch_contrastive_loss,
    in_batch_triplet_loss,
)
from embedding_train.metrics import (
    compute_exact_retrieval_metrics,
    compute_ranking_metrics,
)


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


def resolve_output_dim(output_dim):
    if output_dim is None:
        return None

    resolved_output_dim = int(output_dim)
    if resolved_output_dim < 1:
        raise ValueError(f"output_dim must be at least 1, got: {output_dim}")

    return resolved_output_dim


def load_embedding_module_from_checkpoint(checkpoint_path, map_location="cpu"):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )

    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise ValueError(f"Checkpoint is missing state_dict: {checkpoint_path}")

    cfg = build_config_from_hyperparameters(checkpoint.get("hyper_parameters"))
    model = EmbeddingModule(cfg)
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg


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
        self.encoder_hidden_size = int(self.encoder.config.hidden_size)
        self.output_dim = resolve_output_dim(cfg.model.output_dim)
        self.projection = None
        self.records_seen = 0
        self.validation_rows = []
        self.validation_loss_total = 0.0
        self.validation_loss_examples = 0

        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

        if cfg.model.gradient_checkpointing and hasattr(
            self.encoder, "gradient_checkpointing_enable"
        ):
            self.encoder.gradient_checkpointing_enable()

        if self.output_dim is not None:
            self.projection = torch.nn.Linear(
                self.encoder_hidden_size,
                self.output_dim,
                dtype=self.model_dtype,
            )

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
        batch_size = batch["labels"].size(0)
        self.assert_finite(batch["labels"], "labels", batch_idx)
        self.assert_finite(scores, "scores", batch_idx)
        loss = self.compute_loss(
            batch,
            query_embeddings,
            offer_embeddings,
            scores,
        )
        self.assert_finite(loss, "train_loss", batch_idx)
        self.records_seen += self.resolve_batch_record_count(batch)
        self.log(
            f"train/{self.loss_type}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log_metrics_by_records(
            {
                f"train/{self.loss_type}_loss": loss,
                "train/loss": loss,
            }
        )
        self.log_training_batch_stats(batch)
        return loss

    def on_validation_epoch_start(self):
        self.validation_rows = []
        self.validation_loss_total = 0.0
        self.validation_loss_examples = 0

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
        batch_size = batch["labels"].size(0)
        self.validation_loss_total += float(loss.detach().item()) * batch_size
        self.validation_loss_examples += batch_size

        self.log(
            f"val/{self.loss_type}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
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
        exact_metrics = compute_exact_retrieval_metrics(self.validation_rows)
        record_metrics = {
            "val/ndcg_at_1": metrics["ndcg@1"],
            "val/ndcg_at_5": metrics["ndcg@5"],
            "val/ndcg_at_10": metrics["ndcg@10"],
            "val/exact_success_at_1": exact_metrics["exact_success@1"],
            "val/exact_mrr": exact_metrics["exact_mrr"],
            "val/exact_recall_at_5": exact_metrics["exact_recall@5"],
            "val/exact_recall_at_10": exact_metrics["exact_recall@10"],
            "val/eligible_queries": metrics["eligible_queries"],
            "val/evaluated_queries": exact_metrics["evaluated_queries"],
        }

        if self.validation_loss_examples:
            average_validation_loss = (
                self.validation_loss_total / self.validation_loss_examples
            )
            record_metrics[f"val/{self.loss_type}_loss"] = average_validation_loss
            record_metrics["val/loss"] = average_validation_loss

        self.log("val/ndcg_at_1", metrics["ndcg@1"], prog_bar=True)
        self.log("val/ndcg_at_5", metrics["ndcg@5"], prog_bar=True)
        self.log("val/ndcg_at_10", metrics["ndcg@10"], prog_bar=False)
        self.log(
            "val/exact_success_at_1",
            exact_metrics["exact_success@1"],
            prog_bar=True,
        )
        self.log("val/exact_mrr", exact_metrics["exact_mrr"], prog_bar=True)
        self.log(
            "val/exact_recall_at_5",
            exact_metrics["exact_recall@5"],
            prog_bar=False,
        )
        self.log(
            "val/exact_recall_at_10",
            exact_metrics["exact_recall@10"],
            prog_bar=False,
        )
        self.log("val/eligible_queries", metrics["eligible_queries"], prog_bar=False)
        self.log(
            "val/evaluated_queries",
            exact_metrics["evaluated_queries"],
            prog_bar=False,
        )
        self.log_metrics_by_records(record_metrics)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.cfg.optimizer.lr),
            weight_decay=float(self.cfg.optimizer.weight_decay),
        )

    def log_training_batch_stats(self, batch):
        if not bool(self.cfg.data.log_batch_stats):
            return

        batch_stats = batch.get("batch_stats")
        if not batch_stats:
            return

        batch_size = batch["labels"].size(0)
        stats_to_log = {
            "train/batch_positive_count": batch_stats["positive_count"],
            "train/batch_same_query_negative_count": batch_stats[
                "same_query_negative_count"
            ],
            "train/batch_cross_query_negative_count": batch_stats[
                "cross_query_negative_count"
            ],
        }

        for name, value in stats_to_log.items():
            self.log(
                name,
                float(value),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )

        self.log_metrics_by_records(stats_to_log)

    def resolve_batch_record_count(self, batch):
        local_batch_size = int(batch["labels"].size(0))

        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return local_batch_size

        batch_size = batch["labels"].new_tensor([local_batch_size], dtype=torch.long)
        torch.distributed.all_reduce(batch_size, op=torch.distributed.ReduceOp.SUM)
        return int(batch_size.item())

    def record_aligned_metric_name(self, name):
        return f"{name}_by_records"

    def log_metrics_by_records(self, metrics):
        logger = self.logger
        log_metrics = getattr(logger, "log_metrics", None)
        if logger is None or not callable(log_metrics):
            return

        record_metrics = {}
        for name, value in metrics.items():
            record_metrics[self.record_aligned_metric_name(name)] = (
                self.metric_value_to_float(value)
            )

        log_metrics(record_metrics, step=int(self.records_seen))

    def metric_value_to_float(self, value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().item())

        return float(value)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["records_seen"] = int(self.records_seen)

    def on_load_checkpoint(self, checkpoint):
        self.records_seen = int(checkpoint.get("records_seen", 0))

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
        projected = self.project_embeddings(pooled)
        self.assert_finite(projected, "projected_embeddings")
        normalized = F.normalize(projected, p=2, dim=1)
        self.assert_finite(normalized, "normalized_embeddings")
        return normalized

    def project_embeddings(self, embeddings):
        if self.projection is None:
            return embeddings

        return self.projection(embeddings)

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

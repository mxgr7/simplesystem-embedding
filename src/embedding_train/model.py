import lightning as L
import torch
import torch.distributed
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoModel
from transformers import get_scheduler

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


def resolve_scheduler_name(scheduler_name):
    normalized = str(scheduler_name).strip().lower()

    if normalized in {"none", "off", "null"}:
        return "none"

    if normalized in {"linear", "cosine"}:
        return normalized

    raise ValueError(f"Unsupported optimizer scheduler: {scheduler_name}")


def resolve_warmup_steps(optimizer_cfg, total_training_steps):
    total_training_steps = int(total_training_steps)
    warmup_steps = optimizer_cfg.get("warmup_steps")

    if warmup_steps is not None:
        resolved_warmup_steps = int(warmup_steps)
        if resolved_warmup_steps < 0:
            raise ValueError("optimizer.warmup_steps must be at least 0")
        return min(resolved_warmup_steps, total_training_steps)

    warmup_ratio = float(optimizer_cfg.get("warmup_ratio", 0.0))
    if warmup_ratio < 0.0 or warmup_ratio > 1.0:
        raise ValueError("optimizer.warmup_ratio must be between 0.0 and 1.0")

    resolved_warmup_steps = int(total_training_steps * warmup_ratio)
    if warmup_ratio > 0.0 and resolved_warmup_steps == 0 and total_training_steps > 0:
        resolved_warmup_steps = 1

    return min(resolved_warmup_steps, total_training_steps)


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
        self.pending_train_record_metrics = {}
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
        loss_metric_name = self.batch_aligned_metric_name(
            "train", f"{self.loss_type}_loss"
        )
        train_metric_name = self.batch_aligned_metric_name("train", "loss")
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
            loss_metric_name,
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            train_metric_name,
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        batch_stats_to_log = self.log_training_batch_stats(batch)
        self.pending_train_record_metrics = {
            loss_metric_name: loss.detach(),
            train_metric_name: loss.detach(),
            **batch_stats_to_log,
        }
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        del outputs, batch_idx

        if not self.pending_train_record_metrics:
            return

        self.records_seen += self.resolve_batch_record_count(batch)
        self.log_metrics_by_records(self.pending_train_record_metrics)
        self.pending_train_record_metrics = {}

    def on_train_epoch_start(self):
        self.pending_train_record_metrics = {}

    def on_validation_start(self):
        if self.is_sanity_checking():
            self.validation_rows = []
            self.validation_loss_total = 0.0
            self.validation_loss_examples = 0

    def on_validation_end(self):
        if self.is_sanity_checking():
            self.validation_rows = []
            self.validation_loss_total = 0.0
            self.validation_loss_examples = 0

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
        loss_metric_name = self.batch_aligned_metric_name(
            "val", f"{self.loss_type}_loss"
        )
        validation_metric_name = self.batch_aligned_metric_name("val", "loss")

        self.log(
            loss_metric_name,
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            validation_metric_name,
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
            self.batch_aligned_metric_name("val", "ndcg_at_1"): metrics["ndcg@1"],
            self.batch_aligned_metric_name("val", "ndcg_at_5"): metrics["ndcg@5"],
            self.batch_aligned_metric_name("val", "ndcg_at_10"): metrics["ndcg@10"],
            self.batch_aligned_metric_name("val", "exact_success_at_1"): exact_metrics[
                "exact_success@1"
            ],
            self.batch_aligned_metric_name("val", "exact_mrr"): exact_metrics[
                "exact_mrr"
            ],
            self.batch_aligned_metric_name("val", "exact_recall_at_5"): exact_metrics[
                "exact_recall@5"
            ],
            self.batch_aligned_metric_name("val", "exact_recall_at_10"): exact_metrics[
                "exact_recall@10"
            ],
            self.batch_aligned_metric_name("val", "eligible_queries"): metrics[
                "eligible_queries"
            ],
            self.batch_aligned_metric_name("val", "evaluated_queries"): exact_metrics[
                "evaluated_queries"
            ],
        }

        if self.validation_loss_examples:
            average_validation_loss = (
                self.validation_loss_total / self.validation_loss_examples
            )
            record_metrics[
                self.batch_aligned_metric_name("val", f"{self.loss_type}_loss")
            ] = average_validation_loss
            record_metrics[self.batch_aligned_metric_name("val", "loss")] = (
                average_validation_loss
            )

        self.log(
            self.batch_aligned_metric_name("val", "ndcg_at_1"),
            metrics["ndcg@1"],
            prog_bar=True,
        )
        self.log(
            self.batch_aligned_metric_name("val", "ndcg_at_5"),
            metrics["ndcg@5"],
            prog_bar=True,
        )
        self.log(
            self.batch_aligned_metric_name("val", "ndcg_at_10"),
            metrics["ndcg@10"],
            prog_bar=False,
        )
        self.log(
            self.batch_aligned_metric_name("val", "exact_success_at_1"),
            exact_metrics["exact_success@1"],
            prog_bar=True,
        )
        self.log(
            self.batch_aligned_metric_name("val", "exact_mrr"),
            exact_metrics["exact_mrr"],
            prog_bar=True,
        )
        self.log(
            self.batch_aligned_metric_name("val", "exact_recall_at_5"),
            exact_metrics["exact_recall@5"],
            prog_bar=False,
        )
        self.log(
            self.batch_aligned_metric_name("val", "exact_recall_at_10"),
            exact_metrics["exact_recall@10"],
            prog_bar=False,
        )
        self.log(
            self.batch_aligned_metric_name("val", "eligible_queries"),
            metrics["eligible_queries"],
            prog_bar=False,
        )
        self.log(
            self.batch_aligned_metric_name("val", "evaluated_queries"),
            exact_metrics["evaluated_queries"],
            prog_bar=False,
        )

        if self.is_sanity_checking() or self.records_seen < 1:
            return

        self.log_metrics_by_records(record_metrics)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.cfg.optimizer.lr),
            weight_decay=float(self.cfg.optimizer.weight_decay),
        )

        scheduler_name = resolve_scheduler_name(
            self.cfg.optimizer.get("scheduler", "none")
        )
        if scheduler_name == "none":
            return optimizer

        total_training_steps = int(self.trainer.estimated_stepping_batches)
        if total_training_steps < 1:
            return optimizer

        warmup_steps = resolve_warmup_steps(self.cfg.optimizer, total_training_steps)
        scheduler = get_scheduler(
            scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def log_training_batch_stats(self, batch):
        if not bool(self.cfg.data.log_batch_stats):
            return {}

        batch_stats = batch.get("batch_stats")
        if not batch_stats:
            return {}

        batch_size = batch["labels"].size(0)
        stats_to_log = {
            self.batch_aligned_metric_name(
                "train", "batch_positive_count"
            ): batch_stats["positive_count"],
            self.batch_aligned_metric_name(
                "train", "batch_same_query_negative_count"
            ): batch_stats["same_query_negative_count"],
            self.batch_aligned_metric_name(
                "train", "batch_cross_query_negative_count"
            ): batch_stats["cross_query_negative_count"],
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

        return stats_to_log

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
        if "/by_batch/" in name:
            return name.replace("/by_batch/", "/by_records/", 1)

        first_separator = name.find("/")
        if first_separator == -1:
            return f"by_records/{name}"

        return f"{name[:first_separator]}/by_records/{name[first_separator + 1 :]}"

    def batch_aligned_metric_name(self, split, name):
        return f"{split}/by_batch/{name}"

    def log_metrics_by_records(self, metrics):
        logger = self.logger
        log_metrics = getattr(logger, "log_metrics", None)
        if logger is None or not callable(log_metrics) or self.records_seen < 1:
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

    def is_sanity_checking(self):
        trainer = getattr(self, "trainer", None)
        return bool(getattr(trainer, "sanity_checking", False))

    def on_save_checkpoint(self, checkpoint):
        checkpoint["records_seen"] = int(self.records_seen)

    def on_load_checkpoint(self, checkpoint):
        self.records_seen = int(checkpoint.get("records_seen", 0))
        self.pending_train_record_metrics = {}

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

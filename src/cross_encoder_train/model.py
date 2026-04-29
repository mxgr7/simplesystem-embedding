import logging

import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoModel, get_scheduler

from embedding_train.metrics import (
    compute_binary_retrieval_metrics,
    compute_ranking_metrics,
)

from cross_encoder_train.features import feature_token_names
from cross_encoder_train.labels import GAIN_VECTOR, NUM_CLASSES
from cross_encoder_train.metrics import compute_classification_metrics

logger = logging.getLogger(__name__)


def resolve_model_dtype(precision):
    normalized = str(precision).strip().lower()
    if normalized in {
        "16",
        "16-mixed",
        "bf16",
        "bf16-mixed",
        "32",
        "32-true",
    }:
        return torch.float32
    if normalized == "16-true":
        return torch.float16
    if normalized == "bf16-true":
        return torch.bfloat16
    if normalized in {"64", "64-true"}:
        return torch.float64
    raise ValueError(f"Unsupported trainer precision: {precision}")


def resolve_scheduler_name(scheduler_name):
    normalized = str(scheduler_name).strip().lower()
    if normalized in {"none", "off", "null"}:
        return "none"
    if normalized in {"linear", "cosine", "constant_with_warmup"}:
        return normalized
    raise ValueError(f"Unsupported optimizer scheduler: {scheduler_name}")


def resolve_warmup_steps(optimizer_cfg, steps_per_epoch, total_training_steps):
    steps_per_epoch = int(steps_per_epoch)
    total_training_steps = int(total_training_steps)
    warmup_steps = optimizer_cfg.get("warmup_steps")

    if warmup_steps is not None:
        resolved = int(warmup_steps)
        if resolved < 0:
            raise ValueError("optimizer.warmup_steps must be at least 0")
        return min(resolved, total_training_steps)

    warmup_ratio = float(optimizer_cfg.get("warmup_ratio", 0.0))
    if warmup_ratio < 0.0 or warmup_ratio > 1.0:
        raise ValueError("optimizer.warmup_ratio must be between 0.0 and 1.0")

    resolved = int(steps_per_epoch * warmup_ratio)
    if warmup_ratio > 0.0 and resolved == 0 and steps_per_epoch > 0:
        resolved = 1
    return min(resolved, total_training_steps)


class CrossEncoderModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_dtype = resolve_model_dtype(cfg.trainer.precision)
        self.encoder = AutoModel.from_pretrained(
            cfg.model.model_name, dtype=self.model_dtype
        )
        features_cfg = cfg.data.get("features", None) if hasattr(
            cfg.data, "get"
        ) else getattr(cfg.data, "features", None)
        extra_token_count = len(feature_token_names(features_cfg))
        if extra_token_count > 0:
            new_size = self.encoder.config.vocab_size + extra_token_count
            self.encoder.resize_token_embeddings(new_size)
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = torch.nn.Dropout(float(cfg.model.head_dropout))
        self.classifier = torch.nn.Linear(
            hidden_size, NUM_CLASSES, dtype=self.model_dtype
        )
        self.label_smoothing = float(cfg.model.label_smoothing)
        self.focal_gamma = float(cfg.model.get("focal_gamma", 0.0))
        self.use_class_weights = bool(cfg.model.use_class_weights)
        self.register_buffer(
            "class_weights",
            torch.ones(NUM_CLASSES, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "gain_vector",
            torch.tensor(GAIN_VECTOR, dtype=torch.float32),
            persistent=False,
        )
        self.validation_rows = []

        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

        if cfg.model.gradient_checkpointing and hasattr(
            self.encoder, "gradient_checkpointing_enable"
        ):
            self.encoder.gradient_checkpointing_enable()

        if getattr(cfg.model, "compile", False):
            self.encoder = torch.compile(self.encoder)

    def on_fit_start(self):
        self._sync_class_weights()

    def on_validation_start(self):
        self._sync_class_weights()

    def _sync_class_weights(self):
        if not self.use_class_weights:
            return
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            return
        weights = getattr(datamodule, "class_weights", None)
        if not weights:
            return
        tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
        self.class_weights = tensor

    def forward(self, inputs):
        outputs = self.encoder(**inputs)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        cls_hidden = self.dropout(cls_hidden)
        return self.classifier(cls_hidden)

    def compute_loss(self, logits, labels):
        weight = self.class_weights if self.use_class_weights else None
        if self.focal_gamma > 0.0:
            log_probs = F.log_softmax(logits.float(), dim=-1)
            log_pt = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            pt = log_pt.exp()
            focal_weight = (1.0 - pt).pow(self.focal_gamma)
            loss = -focal_weight * log_pt
            if weight is not None:
                loss = loss * weight[labels]
            return loss.mean()
        return F.cross_entropy(
            logits.float(),
            labels,
            weight=weight,
            label_smoothing=self.label_smoothing,
        )

    def training_step(self, batch, batch_idx):
        logits = self(batch["inputs"])
        loss = self.compute_loss(logits, batch["labels"])
        batch_size = batch["labels"].size(0)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def on_validation_epoch_start(self):
        self.validation_rows = []

    def validation_step(self, batch, batch_idx):
        logits = self(batch["inputs"])
        loss = self.compute_loss(logits, batch["labels"])
        batch_size = batch["labels"].size(0)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        probabilities = F.softmax(logits.float(), dim=-1)
        predictions = probabilities.argmax(dim=-1)
        ranking_scores = (probabilities * self.gain_vector).sum(dim=-1)

        cpu_predictions = predictions.detach().cpu().tolist()
        cpu_targets = batch["labels"].detach().cpu().tolist()
        cpu_scores = ranking_scores.detach().cpu().tolist()

        for i, query_id in enumerate(batch["query_ids"]):
            self.validation_rows.append(
                {
                    "query_id": query_id,
                    "offer_id": batch["offer_ids"][i],
                    "raw_label": batch["raw_labels"][i],
                    "score": float(cpu_scores[i]),
                    "predicted_id": int(cpu_predictions[i]),
                    "target_id": int(cpu_targets[i]),
                }
            )

        return loss

    def on_validation_epoch_end(self):
        if not self.validation_rows:
            return

        cls_metrics = compute_classification_metrics(
            [row["predicted_id"] for row in self.validation_rows],
            [row["target_id"] for row in self.validation_rows],
        )
        for name, value in cls_metrics.items():
            self.log(f"val/cls/{name}", float(value), prog_bar=(name == "macro_f1"))

        rank_metrics = compute_ranking_metrics(self.validation_rows)
        for k in (1, 5, 10):
            self.log(
                f"val/rank/ndcg_at_{k}",
                float(rank_metrics[f"ndcg@{k}"]),
                prog_bar=(k == 5),
            )
        self.log(
            "val/rank/eligible_queries",
            float(rank_metrics["eligible_queries"]),
        )

        exact_metrics = compute_binary_retrieval_metrics(
            self.validation_rows,
            relevant_labels=("Exact",),
            metric_prefix="exact",
        )
        self.log("val/rank/exact_mrr", float(exact_metrics["exact_mrr"]))
        for k in (5, 10):
            self.log(
                f"val/rank/exact_recall_at_{k}",
                float(exact_metrics[f"exact_recall@{k}"]),
            )

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

        max_epochs = max(int(self.cfg.trainer.get("max_epochs", 1) or 1), 1)
        steps_per_epoch = max(total_training_steps // max_epochs, 1)
        warmup_steps = resolve_warmup_steps(
            self.cfg.optimizer, steps_per_epoch, total_training_steps
        )
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

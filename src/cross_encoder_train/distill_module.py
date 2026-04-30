"""Distillation Lightning module: train a smaller student CE against a frozen teacher.

The student is a fresh `CrossEncoderModule` configured for the smaller backbone
(e.g. `deepset/gelectra-base`); the teacher is loaded from a Lightning .ckpt of
the production CE (Soup CE / soup.ckpt) and frozen.

Loss: alpha * T^2 * KL(softmax(student/T) || softmax(teacher/T)) + (1 - alpha)
* CrossEntropy(student, hard_labels). KL uses softened logits with temperature
T (>1 spreads probability mass; the T^2 term restores the gradient scale per
Hinton et al.). The hard-label CE keeps the student honest when the teacher is
wrong.

Validation reuses the training program's classification metrics on the
student's argmax predictions, so floors are bit-comparable across teacher /
student / quantized variants.
"""
from __future__ import annotations

import logging

import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from cross_encoder_train.labels import GAIN_VECTOR, NUM_CLASSES
from cross_encoder_train.metrics import compute_classification_metrics
from cross_encoder_train.model import CrossEncoderModule, resolve_model_dtype, resolve_scheduler_name, resolve_warmup_steps
from embedding_train.metrics import (
    compute_binary_retrieval_metrics,
    compute_ranking_metrics,
)
from transformers import AutoModel, get_scheduler

logger = logging.getLogger(__name__)


class CrossEncoderDistillModule(L.LightningModule):
    """Student CE + frozen teacher CE, trained with KL + hard-label CE."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_dtype = resolve_model_dtype(cfg.trainer.precision)

        # ---- Student: fresh encoder + classifier head ----
        student_name = cfg.model.student_name
        self.encoder = AutoModel.from_pretrained(student_name, dtype=self.model_dtype)
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = torch.nn.Dropout(float(cfg.model.head_dropout))
        self.classifier = torch.nn.Linear(hidden_size, NUM_CLASSES, dtype=self.model_dtype)

        # ---- Teacher: load full Lightning ckpt of the production CE ----
        teacher_cfg_overrides = OmegaConf.create({"model": {"compile": False}})
        teacher_cfg = OmegaConf.merge(cfg, teacher_cfg_overrides)
        # The teacher's model_name lives under cfg.model.teacher_name; rewrite
        # cfg.model.model_name temporarily so CrossEncoderModule loads the
        # teacher backbone.
        teacher_cfg = OmegaConf.merge(
            teacher_cfg,
            OmegaConf.create({"model": {"model_name": cfg.model.teacher_name}}),
        )
        teacher = CrossEncoderModule(cfg=teacher_cfg)
        ckpt = torch.load(cfg.model.teacher_ckpt, map_location="cpu", weights_only=False)
        state_dict = {
            k.replace("._orig_mod.", "."): v for k, v in ckpt["state_dict"].items()
        }
        teacher.load_state_dict(state_dict, strict=True)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        # Cast teacher to bf16 to halve memory + match autocast — saves ~700 MB
        # VRAM at training time; teacher is no_grad so gradient precision N/A.
        teacher.to(torch.bfloat16)
        # Register as a submodule so .to(device) propagates, but flag it so
        # configure_optimizers doesn't try to update it.
        self.teacher = teacher

        # ---- Hyperparameters ----
        self.distill_temperature = float(cfg.model.distill_temperature)
        self.distill_alpha = float(cfg.model.distill_alpha)  # weight on KL
        self.label_smoothing = float(cfg.model.label_smoothing)

        self.register_buffer(
            "gain_vector",
            torch.tensor(GAIN_VECTOR, dtype=torch.float32),
            persistent=False,
        )
        self.validation_rows = []
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

        if getattr(cfg.model, "compile", False):
            self.encoder = torch.compile(self.encoder)

    def forward(self, inputs):
        outputs = self.encoder(**inputs)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        cls_hidden = self.dropout(cls_hidden)
        return self.classifier(cls_hidden)

    def _teacher_logits(self, inputs):
        with torch.no_grad():
            return self.teacher(inputs).float()

    def training_step(self, batch, batch_idx):
        student_logits = self(batch["inputs"]).float()
        teacher_logits = self._teacher_logits(batch["inputs"])

        T = self.distill_temperature
        # softened distributions
        student_log_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        # KL(student || teacher) — match teacher's distribution
        kl = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean") * (T * T)

        ce = F.cross_entropy(
            student_logits, batch["labels"], label_smoothing=self.label_smoothing
        )

        loss = self.distill_alpha * kl + (1.0 - self.distill_alpha) * ce

        bs = batch["labels"].size(0)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("train/kl", kl, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train/ce", ce, on_step=False, on_epoch=True, batch_size=bs)
        return loss

    def on_validation_epoch_start(self):
        self.validation_rows = []

    def validation_step(self, batch, batch_idx):
        logits = self(batch["inputs"]).float()
        ce = F.cross_entropy(logits, batch["labels"], label_smoothing=self.label_smoothing)
        bs = batch["labels"].size(0)
        self.log("val/loss", ce, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)

        probabilities = F.softmax(logits, dim=-1)
        predictions = probabilities.argmax(dim=-1)
        ranking_scores = (probabilities * self.gain_vector).sum(dim=-1)

        cpu_predictions = predictions.detach().cpu().tolist()
        cpu_targets = batch["labels"].detach().cpu().tolist()
        cpu_scores = ranking_scores.detach().cpu().tolist()

        for i, query_id in enumerate(batch["query_ids"]):
            self.validation_rows.append({
                "query_id": query_id,
                "offer_id": batch["offer_ids"][i],
                "raw_label": batch["raw_labels"][i],
                "score": float(cpu_scores[i]),
                "predicted_id": int(cpu_predictions[i]),
                "target_id": int(cpu_targets[i]),
            })

        return ce

    def on_validation_epoch_end(self):
        if not self.validation_rows:
            return
        cls_metrics = compute_classification_metrics(
            [r["predicted_id"] for r in self.validation_rows],
            [r["target_id"] for r in self.validation_rows],
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
        self.log("val/rank/eligible_queries", float(rank_metrics["eligible_queries"]))

        exact_metrics = compute_binary_retrieval_metrics(
            self.validation_rows, relevant_labels=("Exact",), metric_prefix="exact",
        )
        self.log("val/rank/exact_mrr", float(exact_metrics["exact_mrr"]))
        for k in (5, 10):
            self.log(
                f"val/rank/exact_recall_at_{k}",
                float(exact_metrics[f"exact_recall@{k}"]),
            )

    def on_save_checkpoint(self, checkpoint):
        # Don't ship the teacher in the checkpoint — it's frozen and 3× larger
        # than the student. Strips `teacher.*` keys so the saved ckpt loads
        # cleanly into a plain CrossEncoderModule (configured with the student
        # backbone name) for eval and serving.
        sd = checkpoint.get("state_dict", {})
        for k in list(sd.keys()):
            if k.startswith("teacher."):
                del sd[k]

    def configure_optimizers(self):
        # Only train the student — explicitly drop teacher params from the
        # optimizer set (they're frozen but Lightning wouldn't know that
        # otherwise without the requires_grad filter).
        student_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            student_params,
            lr=float(self.cfg.optimizer.lr),
            weight_decay=float(self.cfg.optimizer.weight_decay),
        )
        scheduler_name = resolve_scheduler_name(self.cfg.optimizer.get("scheduler", "none"))
        if scheduler_name == "none":
            return optimizer
        total = int(self.trainer.estimated_stepping_batches)
        if total < 1:
            return optimizer
        max_epochs = max(int(self.cfg.trainer.get("max_epochs", 1) or 1), 1)
        steps_per_epoch = max(total // max_epochs, 1)
        warmup = resolve_warmup_steps(self.cfg.optimizer, steps_per_epoch, total)
        scheduler = get_scheduler(scheduler_name, optimizer=optimizer,
                                  num_warmup_steps=warmup, num_training_steps=total)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

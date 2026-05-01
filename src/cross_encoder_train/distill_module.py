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
import math

import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoModel, get_scheduler

from cross_encoder_train.labels import GAIN_VECTOR, LABEL_TO_ID, NUM_CLASSES
from cross_encoder_train.metrics import compute_classification_metrics
from cross_encoder_train.model import CrossEncoderModule, resolve_model_dtype, resolve_scheduler_name, resolve_warmup_steps
from embedding_train.metrics import (
    compute_binary_retrieval_metrics,
    compute_ranking_metrics,
)

EXACT_IDX = LABEL_TO_ID["Exact"]
SUBSTITUTE_IDX = LABEL_TO_ID["Substitute"]
LN3 = math.log(3.0)

logger = logging.getLogger(__name__)


class CrossEncoderDistillModule(L.LightningModule):
    """Student CE + frozen teacher CE, trained with KL + hard-label CE."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_dtype = resolve_model_dtype(cfg.trainer.precision)

        # ---- Student: fresh encoder + classifier head ----
        # Two paths:
        # - student_name + (optional) prune_layers: load student backbone, keep
        #   only the first N layers (warm-start from teacher when student_name
        #   == teacher_name; otherwise fresh + truncated).
        # - default: load student_name as-is (e.g. gelectra-base, MiniLM-L12).
        student_name = cfg.model.student_name
        self.encoder = AutoModel.from_pretrained(student_name, dtype=self.model_dtype)
        prune_layers = int(cfg.model.get("prune_layers", 0) or 0)
        if prune_layers > 0:
            full_n = len(self.encoder.encoder.layer)
            if prune_layers > full_n:
                raise ValueError(
                    f"prune_layers={prune_layers} > encoder layers={full_n}"
                )
            # Keep the first N layers — empirically the lower layers have
            # broader coverage; for a more uniform downsample, prune evenly.
            kept = torch.nn.ModuleList(list(self.encoder.encoder.layer)[:prune_layers])
            self.encoder.encoder.layer = kept
            self.encoder.config.num_hidden_layers = prune_layers
            logger.info("Pruned student encoder %d → %d layers", full_n, prune_layers)
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = torch.nn.Dropout(float(cfg.model.head_dropout))
        # Binary head: train Linear(hidden, 1) → BCE+KL on Exact-vs-rest, but
        # save a 4-channel-shaped state_dict at checkpoint time so the existing
        # eval / serve loaders (which expect Linear(hidden, NUM_CLASSES)) work
        # unchanged. See on_save_checkpoint below for the conversion.
        self.binary_head = bool(cfg.model.get("binary_head", False))
        out_dim = 1 if self.binary_head else NUM_CLASSES
        self.classifier = torch.nn.Linear(hidden_size, out_dim, dtype=self.model_dtype)

        # ---- Teacher: load full Lightning ckpt of the production CE ----
        # Override compile=False, point model_name at teacher backbone, and
        # CRITICALLY null out prune_layers so the teacher keeps its full depth
        # (the student-side prune_layers must not leak into the teacher cfg).
        teacher_cfg = OmegaConf.merge(
            cfg,
            OmegaConf.create({"model": {
                "compile": False,
                "model_name": cfg.model.teacher_name,
                "prune_layers": 0,
            }}),
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
        self.class_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

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
        if self.binary_head:
            # Squash teacher 4-class softmax to a Bernoulli on Exact-vs-rest;
            # student head emits a single logit per row. KL(Bernoulli||Bernoulli)
            # ∝ BCE(student_p, teacher_p) (the teacher-entropy term is a
            # constant w.r.t. the student parameters and drops from the
            # gradient). T^2 restores the gradient scale a-la Hinton, mirroring
            # the 4-class path.
            student_logit = student_logits.squeeze(-1)
            teacher_p_exact = F.softmax(teacher_logits / T, dim=-1)[:, EXACT_IDX]
            student_p_T = torch.sigmoid(student_logit / T)
            kl = F.binary_cross_entropy(student_p_T, teacher_p_exact) * (T * T)
            binary_target = (batch["labels"] == EXACT_IDX).float()
            ce = F.binary_cross_entropy_with_logits(student_logit, binary_target)
        else:
            # softened distributions
            student_log_soft = F.log_softmax(student_logits / T, dim=-1)
            teacher_soft = F.softmax(teacher_logits / T, dim=-1)
            # KL(student || teacher) — match teacher's distribution
            kl = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean") * (T * T)

            weight = self.class_weights if self.use_class_weights else None
            if self.focal_gamma > 0.0:
                log_probs = F.log_softmax(student_logits, dim=-1)
                log_pt = log_probs.gather(1, batch["labels"].unsqueeze(1)).squeeze(1)
                pt = log_pt.exp()
                focal = (1.0 - pt).pow(self.focal_gamma)
                ce_per = -focal * log_pt
                if weight is not None:
                    ce_per = ce_per * weight[batch["labels"]]
                ce = ce_per.mean()
            else:
                ce = F.cross_entropy(
                    student_logits, batch["labels"], weight=weight,
                    label_smoothing=self.label_smoothing,
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
        bs = batch["labels"].size(0)

        if self.binary_head:
            # Binary head: predict Exact iff sigmoid(logit) > 0.5; map non-Exact
            # to SUBSTITUTE so compute_classification_metrics still gets a 4-way
            # prediction. Only `f1_exact` is meaningful in this mode (and is the
            # only gating metric under the new floor); the other class F1s
            # collapse by construction.
            student_logit = logits.squeeze(-1)
            binary_target = (batch["labels"] == EXACT_IDX).float()
            ce = F.binary_cross_entropy_with_logits(student_logit, binary_target)
            self.log("val/loss", ce, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
            sigm = torch.sigmoid(student_logit)
            predictions = torch.where(
                sigm > 0.5,
                torch.full_like(student_logit, EXACT_IDX, dtype=torch.long),
                torch.full_like(student_logit, SUBSTITUTE_IDX, dtype=torch.long),
            )
            # Ranking score = p_exact (calibration-free monotone score)
            ranking_scores = sigm
        else:
            ce = F.cross_entropy(logits, batch["labels"], label_smoothing=self.label_smoothing)
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
        if self.binary_head:
            # Inflate Linear(hidden, 1) → Linear(hidden, NUM_CLASSES) so the
            # checkpoint loads cleanly into CrossEncoderModule (which expects
            # the 4-class shape). Mapping: rows 0..2 = zero, row EXACT_IDX =
            # binary row + ln(3) bias offset. With this offset, at T=1
            # softmax([0,0,0, b+ln3])[EXACT] = sigmoid(b) — preserves the
            # binary scoring exactly. At T != 1 the mapping is monotone in b,
            # so a re-fitted serving temperature handles the rest of the
            # calibration.
            w_key = "classifier.weight"
            b_key = "classifier.bias"
            if w_key in sd and sd[w_key].shape[0] == 1:
                hidden = sd[w_key].shape[1]
                inflated_w = torch.zeros(NUM_CLASSES, hidden, dtype=sd[w_key].dtype)
                inflated_w[EXACT_IDX] = sd[w_key][0]
                sd[w_key] = inflated_w
                inflated_b = torch.zeros(NUM_CLASSES, dtype=sd[b_key].dtype)
                inflated_b[EXACT_IDX] = sd[b_key][0] + LN3
                sd[b_key] = inflated_b

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

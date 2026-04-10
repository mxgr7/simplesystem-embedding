import logging
import random
from collections import defaultdict

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
    RELEVANCE_GAINS,
    compute_exact_retrieval_metrics,
    compute_ranking_metrics,
)

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


VALID_VALIDATION_MODES = {"full_catalog", "pairwise_proxy"}
VALID_VALIDATION_METRICS = {"ndcg_at_5", "ndcg_at_10", "mrr", "recall_at_10", "recall_at_100"}
VALID_VALIDATION_SIMILARITIES = {"dot", "cosine"}


def resolve_validation_mode(validation_mode):
    normalized = str(validation_mode).strip().lower()
    if normalized in VALID_VALIDATION_MODES:
        return normalized
    choices = "|".join(sorted(VALID_VALIDATION_MODES))
    raise ValueError(
        f"Unsupported validation_mode: {validation_mode}. Expected one of {choices}"
    )


def resolve_validation_metric(validation_metric):
    normalized = str(validation_metric).strip().lower()
    if normalized in VALID_VALIDATION_METRICS:
        return normalized
    choices = "|".join(sorted(VALID_VALIDATION_METRICS))
    raise ValueError(
        f"Unsupported validation_metric: {validation_metric}. Expected one of {choices}"
    )


def resolve_validation_similarity(validation_similarity):
    normalized = str(validation_similarity).strip().lower()
    if normalized in VALID_VALIDATION_SIMILARITIES:
        return normalized
    choices = "|".join(sorted(VALID_VALIDATION_SIMILARITIES))
    raise ValueError(
        f"Unsupported validation_similarity: {validation_similarity}. "
        f"Expected one of {choices}"
    )


def build_full_catalog_monitor_metric(validation_metric):
    return f"val/full_catalog/{validation_metric}"


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


def subsample_catalog(catalog_rows, sample_size, judgments_by_query):
    judged_offer_ids = set()
    for offers in judgments_by_query.values():
        judged_offer_ids.update(offers)

    judged_rows = []
    unjudged_rows = []
    for row in catalog_rows:
        if row["offer_id"] in judged_offer_ids:
            judged_rows.append(row)
        else:
            unjudged_rows.append(row)

    if len(judged_rows) >= sample_size:
        return judged_rows[:sample_size]

    remaining = sample_size - len(judged_rows)
    rng = random.Random(42)
    rng.shuffle(unjudged_rows)
    return judged_rows + unjudged_rows[:remaining]


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
        self.validation_mode = resolve_validation_mode(
            getattr(cfg.trainer, "validation_mode", "full_catalog")
        )
        self.validation_metric = resolve_validation_metric(
            getattr(cfg.trainer, "validation_metric", "ndcg_at_5")
        )
        self.validation_similarity = resolve_validation_similarity(
            getattr(cfg.trainer, "validation_similarity", "dot")
        )
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
        del outputs

        if not self.pending_train_record_metrics:
            return

        self.records_seen += self.resolve_batch_record_count(batch)
        if self.should_log_records_on_batch(batch_idx):
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

        for i, (query_id, score, raw_label) in enumerate(
            zip(batch["query_ids"], cpu_scores, raw_labels)
        ):
            row = {
                "query_id": query_id,
                "score": float(score),
                "raw_label": raw_label,
            }
            if self.validation_mode == "full_catalog":
                row["offer_id"] = batch["offer_ids"][i]
                row["query_text"] = batch["query_texts"][i]
                row["offer_text"] = batch["offer_texts"][i]
            self.validation_rows.append(row)

        return loss

    def on_validation_epoch_end(self):
        record_metrics = self._compute_pairwise_validation_metrics()

        if self.validation_mode == "full_catalog":
            catalog_metrics = self._compute_full_catalog_validation_metrics()
            record_metrics.update(catalog_metrics)

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

        if self.is_sanity_checking() or self.records_seen < 1:
            return

        self.log_metrics_by_records(record_metrics)

    def _compute_pairwise_validation_metrics(self):
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

        return record_metrics

    def _compute_full_catalog_validation_metrics(self):
        from embedding_train.catalog_benchmark import (
            parse_relevant_labels,
            resolve_similarity,
            score_queries_against_catalog,
        )

        query_rows, catalog_rows, judgments_by_query = (
            self._build_full_catalog_validation_data()
        )

        if not query_rows or not catalog_rows:
            logger.warning(
                "Full-catalog validation skipped: %d queries, %d catalog items",
                len(query_rows),
                len(catalog_rows),
            )
            return {}

        tokenizer = self.trainer.datamodule.tokenizer
        device = self.device
        encode_batch_size = int(
            getattr(self.cfg.trainer, "encode_batch_size", 128)
        )
        score_batch_size = int(
            getattr(self.cfg.trainer, "score_batch_size", 128)
        )
        max_query_length = int(self.cfg.data.max_query_length)
        max_offer_length = int(self.cfg.data.max_offer_length)

        query_embeddings = self._encode_texts_batched(
            tokenizer,
            [row["query_text"] for row in query_rows],
            max_query_length,
            encode_batch_size,
            device,
        )
        catalog_embeddings = self._encode_texts_batched(
            tokenizer,
            [row["offer_text"] for row in catalog_rows],
            max_offer_length,
            encode_batch_size,
            device,
        )

        query_embeddings, catalog_embeddings = resolve_similarity(
            self.validation_similarity,
            query_embeddings,
            catalog_embeddings,
        )

        ks = (5, 10, 100)
        relevant_labels = parse_relevant_labels(
            getattr(self.cfg.trainer, "validation_relevant_labels", "Exact")
        )

        catalog_metrics = score_queries_against_catalog(
            query_rows=query_rows,
            query_embeddings=query_embeddings,
            catalog_rows=catalog_rows,
            catalog_embeddings=catalog_embeddings,
            judgments_by_query=judgments_by_query,
            ks=ks,
            relevant_labels=set(relevant_labels),
            score_batch_size=score_batch_size,
        )

        metric_mapping = {
            "val/full_catalog/mrr": catalog_metrics["mrr"],
            "val/full_catalog/evaluated_queries": catalog_metrics[
                "evaluated_queries"
            ],
            "val/full_catalog/ndcg_eligible_queries": catalog_metrics[
                "ndcg_eligible_queries"
            ],
            "val/full_catalog/retrieval_eligible_queries": catalog_metrics[
                "retrieval_eligible_queries"
            ],
            "val/full_catalog/catalog_size": float(len(catalog_rows)),
        }
        for k in ks:
            metric_mapping[f"val/full_catalog/ndcg_at_{k}"] = catalog_metrics[
                f"ndcg@{k}"
            ]
            metric_mapping[f"val/full_catalog/recall_at_{k}"] = catalog_metrics[
                f"recall@{k}"
            ]

        monitor_metric = build_full_catalog_monitor_metric(self.validation_metric)
        self.log(
            monitor_metric,
            float(metric_mapping[monitor_metric]),
            prog_bar=True,
            logger=False,
        )

        catalog_sample = getattr(self.cfg.trainer, "validation_catalog_sample", None)
        mode_label = (
            "sampled" if catalog_sample is not None else "exhaustive"
        )
        logger.info(
            "Full-catalog validation (%s): %d queries, %d catalog items, %s=%.4f",
            mode_label,
            len(query_rows),
            len(catalog_rows),
            monitor_metric,
            float(metric_mapping.get(monitor_metric, 0.0)),
        )

        return metric_mapping

    def _build_full_catalog_validation_data(self):
        query_rows_by_id = {}
        catalog_rows_by_offer_id = {}
        judgments_by_query = defaultdict(dict)

        for row in self.validation_rows:
            query_id = row["query_id"]
            offer_id = row["offer_id"]
            raw_label = row["raw_label"]

            if query_id not in query_rows_by_id:
                query_rows_by_id[query_id] = {
                    "query_id": query_id,
                    "query_text": row["query_text"],
                }

            if offer_id not in catalog_rows_by_offer_id:
                catalog_rows_by_offer_id[offer_id] = {
                    "offer_id": offer_id,
                    "offer_text": row["offer_text"],
                }

            existing_label = judgments_by_query[query_id].get(offer_id)
            if existing_label is None:
                judgments_by_query[query_id][offer_id] = raw_label
            elif RELEVANCE_GAINS.get(raw_label, 0.0) > RELEVANCE_GAINS.get(
                existing_label, 0.0
            ):
                judgments_by_query[query_id][offer_id] = raw_label

        catalog_rows = list(catalog_rows_by_offer_id.values())

        catalog_sample = getattr(self.cfg.trainer, "validation_catalog_sample", None)
        if catalog_sample is not None and int(catalog_sample) < len(catalog_rows):
            catalog_rows = subsample_catalog(
                catalog_rows,
                int(catalog_sample),
                judgments_by_query,
            )

        return (
            list(query_rows_by_id.values()),
            catalog_rows,
            dict(judgments_by_query),
        )

    def _encode_texts_batched(
        self, tokenizer, texts, max_length, encode_batch_size, device
    ):
        encoded_batches = []
        was_training = self.training
        self.eval()

        with torch.inference_mode():
            for start in range(0, len(texts), encode_batch_size):
                chunk = texts[start : start + encode_batch_size]
                inputs = tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs = {
                    name: tensor.to(device)
                    for name, tensor in dict(inputs).items()
                }
                embeddings = self.encode(inputs)
                encoded_batches.append(
                    embeddings.detach().cpu().to(dtype=torch.float32)
                )

        if was_training:
            self.train()

        return torch.cat(encoded_batches, dim=0)

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
            self.batch_aligned_metric_name(
                "train", "batch_hard_negative_count"
            ): batch_stats.get("hard_negative_count", 0),
        }

        total_negatives = (
            batch_stats["same_query_negative_count"]
            + batch_stats["cross_query_negative_count"]
            + batch_stats.get("hard_negative_count", 0)
        )
        if total_negatives > 0:
            stats_to_log[
                self.batch_aligned_metric_name("train", "batch_hard_negative_share")
            ] = batch_stats.get("hard_negative_count", 0) / total_negatives

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
            return name.replace("/by_batch/", "/", 1)

        first_separator = name.find("/")
        if first_separator == -1:
            return name

        return name

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

    def should_log_records_on_batch(self, batch_idx):
        log_every_n_steps = self.resolve_log_every_n_steps()
        return (batch_idx + 1) % log_every_n_steps == 0

    def resolve_log_every_n_steps(self):
        trainer = getattr(self, "trainer", None)
        log_every_n_steps = int(getattr(trainer, "log_every_n_steps", 1) or 1)
        return max(log_every_n_steps, 1)

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

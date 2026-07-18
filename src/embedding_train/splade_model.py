import torch
from transformers import AutoModelForMaskedLM

from embedding_train.tokenization import load_fast_tokenizer

from embedding_train.losses import (
    flops_regularizer,
    kl_distillation_loss,
    margin_mse_loss,
    quadratic_warmup,
)
from embedding_train.model import EmbeddingModule, resolve_output_dim


class SpladeModule(EmbeddingModule):
    """SPLADE-max learned sparse retriever.

    Reuses the full EmbeddingModule pipeline (batching, contrastive loss,
    full-catalog validation, checkpointing) by overriding encode() to return
    vocabulary-sized activations instead of dense embeddings. Similarity is
    the raw dot product between sparse activation vectors — no L2 norm.
    """

    def __init__(self, cfg):
        if resolve_output_dim(cfg.model.output_dim) is not None:
            raise ValueError(
                "SPLADE has no projection head; set model.output_dim: null"
            )

        super().__init__(cfg)

        self.flops_lambda_q = float(getattr(cfg.model, "flops_lambda_q", 0.0))
        self.flops_lambda_d = float(getattr(cfg.model, "flops_lambda_d", 0.0))
        self.flops_warmup_steps = int(getattr(cfg.model, "flops_warmup_steps", 0))

        loss_weights = getattr(cfg.model, "loss_weights", None) or {}
        self.margin_mse_weight = float(loss_weights.get("margin_mse", 0.0))
        self.kl_weight = float(loss_weights.get("kl", 0.0))
        # Teacher expected gains live in [0, 1] while raw SPLADE dots reach
        # logit scale; without scaling, MarginMSE squeezes all margins to <1
        # and fights the contrastive term (and explodes early in training).
        self.margin_mse_teacher_scale = float(
            getattr(cfg.model, "margin_mse_teacher_scale", 1.0)
        )
        self.margin_mse_warmup_steps = int(
            getattr(cfg.model, "margin_mse_warmup_steps", 0)
        )
        if self.kl_weight > 0.0 and self._teacher_ref is None:
            raise ValueError(
                "loss_weights.kl > 0 requires cfg.model.teacher_checkpoint"
            )

        tokenizer = load_fast_tokenizer(cfg.model.model_name)
        vocab_size = int(self.encoder.config.vocab_size)
        # Special tokens ([CLS]/[SEP]/[PAD]/[MASK]/...) appear in every
        # sequence, so their activations would dominate every vector.
        vocab_mask = torch.ones(vocab_size, dtype=torch.float32)
        special_ids = [
            token_id
            for token_id in tokenizer.all_special_ids
            if 0 <= token_id < vocab_size
        ]
        vocab_mask[special_ids] = 0.0
        self.register_buffer(
            "special_token_vocab_mask", vocab_mask, persistent=False
        )

        self._collect_encode_nnz = None

    def build_encoder(self, cfg):
        try:
            return AutoModelForMaskedLM.from_pretrained(
                cfg.model.model_name,
                dtype=self.model_dtype,
            )
        except ValueError:
            # Older hub repos (e.g. deepset/gbert-base) predate the
            # model_type key that the Auto classes require; fall back to
            # the class named in the config's `architectures`.
            import transformers

            config_dict, _ = transformers.PretrainedConfig.get_config_dict(
                cfg.model.model_name
            )
            for architecture_name in config_dict.get("architectures") or []:
                model_class = getattr(transformers, architecture_name, None)
                if model_class is not None:
                    return model_class.from_pretrained(
                        cfg.model.model_name,
                        dtype=self.model_dtype,
                    )
            raise

    def encode(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        # The [batch, seq, vocab] logits and each elementwise intermediate
        # are ~8 GB at batch 512; without checkpointing the autograd graph
        # holds several of them at once and OOMs an 80 GB H100.
        if self.training and torch.is_grad_enabled():
            representations = torch.utils.checkpoint.checkpoint(
                self._encode_representations,
                input_ids,
                attention_mask,
                use_reentrant=False,
            )
        else:
            representations = self._encode_representations(
                input_ids, attention_mask
            )
        self.assert_finite(representations, "splade_representations")
        return representations

    def _encode_representations(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        logits = outputs.logits
        activations = torch.log1p(torch.relu(logits))
        mask = attention_mask.unsqueeze(-1).to(activations.dtype)
        activations = activations * mask
        representations = activations.max(dim=1).values
        return representations * self.special_token_vocab_mask.to(
            representations.dtype
        )

    def compute_loss(self, batch, query_embeddings, offer_embeddings, scores):
        loss = super().compute_loss(
            batch, query_embeddings, offer_embeddings, scores
        )

        if self.margin_mse_weight > 0.0:
            teacher_scores = batch.get("ce_scores")
            if teacher_scores is None:
                raise RuntimeError(
                    "loss_weights.margin_mse > 0 requires ce_scores in the "
                    "batch; set data.ce_scores_path"
                )
            margin_weight = self.margin_mse_weight * quadratic_warmup(
                self.global_step, self.margin_mse_warmup_steps, 1.0
            )
            margin_term = margin_mse_loss(
                scores,
                batch["query_ids"],
                batch["labels"],
                teacher_scores * self.margin_mse_teacher_scale,
            )
            loss = loss + margin_weight * margin_term
            self.log(
                self.batch_aligned_metric_name(
                    "train" if self.training else "val", "margin_mse"
                ),
                margin_term.detach(),
                on_step=self.training,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch["labels"].size(0),
            )

        if self.kl_weight > 0.0:
            teacher = self._teacher_ref[0]
            with torch.no_grad():
                teacher_query_embeddings = teacher.encode(batch["query_inputs"])
                teacher_offer_embeddings = teacher.encode(batch["offer_inputs"])
            student_sims = torch.matmul(
                query_embeddings, offer_embeddings.transpose(0, 1)
            )
            teacher_sims = torch.matmul(
                teacher_query_embeddings,
                teacher_offer_embeddings.transpose(0, 1),
            )
            loss = loss + self.kl_weight * kl_distillation_loss(
                student_sims,
                teacher_sims,
                temperature=float(self.cfg.model.distill_temperature),
            )

        # Deduplicate query rows (each query appears n_pos+n_neg times per
        # batch) so the query-side FLOPS term is not over-weighted.
        unique_query_embeddings = self._unique_query_rows(
            query_embeddings, batch["query_ids"]
        )
        flops_q = flops_regularizer(unique_query_embeddings)
        flops_d = flops_regularizer(offer_embeddings)
        lambda_q = quadratic_warmup(
            self.global_step, self.flops_warmup_steps, self.flops_lambda_q
        )
        lambda_d = quadratic_warmup(
            self.global_step, self.flops_warmup_steps, self.flops_lambda_d
        )
        total = loss + lambda_q * flops_q + lambda_d * flops_d

        self._log_splade_stats(
            batch,
            unique_query_embeddings,
            offer_embeddings,
            flops_q,
            flops_d,
            lambda_q,
            lambda_d,
        )
        return total

    def _unique_query_rows(self, query_embeddings, query_ids):
        seen = set()
        first_row_indexes = []
        for index, query_id in enumerate(query_ids):
            if query_id not in seen:
                seen.add(query_id)
                first_row_indexes.append(index)
        return query_embeddings[
            torch.tensor(first_row_indexes, device=query_embeddings.device)
        ]

    def _log_splade_stats(
        self,
        batch,
        unique_query_embeddings,
        offer_embeddings,
        flops_q,
        flops_d,
        lambda_q,
        lambda_d,
    ):
        split = "train" if self.training else "val"
        batch_size = batch["labels"].size(0)
        query_avg_nnz = (
            (unique_query_embeddings > 0).sum(dim=1).float().mean()
        )
        offer_avg_nnz = (offer_embeddings > 0).sum(dim=1).float().mean()

        stats = {
            "flops_q": flops_q.detach(),
            "flops_d": flops_d.detach(),
            "flops_lambda_q": lambda_q,
            "flops_lambda_d": lambda_d,
            "query_avg_nnz": query_avg_nnz.detach(),
            "offer_avg_nnz": offer_avg_nnz.detach(),
        }
        for name, value in stats.items():
            self.log(
                self.batch_aligned_metric_name(split, name),
                value,
                on_step=split == "train",
                on_epoch=True,
                prog_bar=name == "offer_avg_nnz",
                batch_size=batch_size,
            )

    def _encode_texts_batched(self, *args, **kwargs):
        embeddings = super()._encode_texts_batched(*args, **kwargs)
        if self._collect_encode_nnz is not None:
            self._collect_encode_nnz.append(
                float((embeddings > 0).sum(dim=1).float().mean().item())
            )
        return embeddings

    def _compute_full_catalog_validation_metrics(self):
        # Parent encodes queries first, then the catalog, so the collected
        # nnz averages arrive in that order.
        self._collect_encode_nnz = []
        metrics = super()._compute_full_catalog_validation_metrics()
        nnz_by_call = self._collect_encode_nnz
        self._collect_encode_nnz = None

        if metrics and len(nnz_by_call) >= 2:
            metrics["val/full_catalog/query_avg_nnz"] = nnz_by_call[0]
            metrics["val/full_catalog/doc_avg_nnz"] = nnz_by_call[1]

        return metrics

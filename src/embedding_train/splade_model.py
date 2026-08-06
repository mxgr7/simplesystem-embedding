import re
import unicodedata

import torch
from transformers import AutoModelForMaskedLM

from embedding_train.tokenization import load_fast_tokenizer


def is_cased_token(tok):
    """True if the token string carries uppercase or a diacritic — i.e. text
    folding (lowercase + strip diacritics) would change it. Used to build the
    folded-vocab output mask for models trained on folded input."""
    s = tok.replace("##", "")
    if re.search(r"[A-ZÄÖÜ]", s):
        return True
    return any(unicodedata.combining(c) for c in unicodedata.normalize("NFD", s))

from embedding_train.losses import (
    df_activation_paper,
    df_flops_regularizer,
    flops_regularizer,
    kl_distillation_loss,
    l1_regularizer,
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
        # Validate everything readable from cfg BEFORE super().__init__ builds a
        # 109M-param backbone (and a second one for untied). A typo would
        # otherwise select a different regularizer than the arm claims, or fail
        # minutes into a launch after the model load.
        _activation = str(getattr(cfg.model, "df_activation", "sigmoid"))
        if _activation not in ("sigmoid", "paper"):
            raise ValueError(
                f"Unsupported df_activation: {_activation!r}. "
                "Expected 'sigmoid' or 'paper'."
            )
        for _name in ("reg_type_q", "reg_type_d"):
            _reg = str(getattr(cfg.model, _name, "flops"))
            if _reg not in ("flops", "l1", "df_flops"):
                raise ValueError(
                    f"Unsupported {_name}: {_reg!r}. "
                    "Expected 'flops', 'l1' or 'df_flops'."
                )
        _half = float(getattr(cfg.model, "df_half", 0.10))
        if not 0.0 < _half < 1.0:
            # Otherwise this only raises inside df_activation_paper at step 1 —
            # on EVERY arm, since the diagnostics log the paper form regardless.
            raise ValueError(f"df_half must be in (0, 1), got {_half}")
        _momentum = float(getattr(cfg.model, "df_ema_momentum", 0.99))
        if not 0.0 <= _momentum < 1.0:
            raise ValueError(f"df_ema_momentum must be in [0, 1), got {_momentum}")
        if bool(getattr(cfg.model, "untied_encoders", False)) and bool(
            getattr(cfg.model, "compile", False)
        ):
            raise ValueError(
                "untied_encoders + compile is unsupported: torch.compile renames "
                "encoder keys to encoder._orig_mod.*, and train.py's warm-start "
                "mirroring would emit query_encoder._orig_mod.* against a module "
                "expecting query_encoder.* — the strict load then fails."
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
        # Optional train-time stoplist: zero always-on stopword/punctuation dims so
        # the model never spends capacity (or FLOPS budget) on non-discriminative
        # terms and reallocates expansion onto content terms.
        stopword_ids = [
            int(i)
            for i in (getattr(cfg.model, "stopword_mask_ids", None) or [])
            if 0 <= int(i) < vocab_size
        ]
        if stopword_ids:
            vocab_mask[stopword_ids] = 0.0
        # Optional folded-vocab mask: for models trained on folded (lowercase,
        # diacritic-stripped) input, the cased/diacritic output dims are largely
        # redundant case-twins of a lowercase token (System vs system). Zeroing
        # them at train time frees capacity + FLOPS budget for content terms and
        # consolidates each lemma onto its lowercase dimension.
        if bool(getattr(cfg.model, "fold_vocab_mask", False)):
            # index by the REAL token id (not enumerate position) so gaps / added
            # tokens beyond vocab_size can't misalign the mask.
            cased_ids = [i for tok, i in tokenizer.get_vocab().items()
                         if i < vocab_size and is_cased_token(tok)]
            vocab_mask[cased_ids] = 0.0
        self.register_buffer(
            "special_token_vocab_mask", vocab_mask, persistent=False
        )

        # Regularizer selection. Defaults reproduce plain FLOPS bit-for-bit, so
        # every existing launch script and checkpoint keeps working unchanged.
        self.reg_type_q = str(getattr(cfg.model, "reg_type_q", "flops"))
        self.reg_type_d = str(getattr(cfg.model, "reg_type_d", "flops"))
        self.df_alpha = float(getattr(cfg.model, "df_alpha", 40.0))
        self.df_beta = float(getattr(cfg.model, "df_beta", 0.08))
        # 'paper' = the DF-FLOPS generalized logistic (arXiv:2505.15070), a
        # near-hard high-pass on df. 'sigmoid' = our broader variant.
        self.df_activation = str(getattr(cfg.model, "df_activation", "sigmoid"))
        self.df_half = float(getattr(cfg.model, "df_half", 0.10))
        self.df_sharp = float(getattr(cfg.model, "df_sharp", 10.0))
        # Untied query/doc encoders. Two independent papers find query-side
        # regularization is near-inert with a shared encoder ("nothing to
        # differentiate between them"), and untying cut latency 59% at zero
        # quality cost. Costs 2x encoder params.
        self.untied_encoders = bool(getattr(cfg.model, "untied_encoders", False))
        # Freezing the doc encoder is only meaningful when untied — with a shared
        # encoder it would freeze the query side too and nothing would train.
        self.freeze_doc_encoder = bool(getattr(cfg.model, "freeze_doc_encoder", False))
        if self.freeze_doc_encoder and not self.untied_encoders:
            raise ValueError(
                "freeze_doc_encoder requires untied_encoders=true: with a shared "
                "encoder it would freeze the query side as well."
            )
        if self.untied_encoders:
            self.query_encoder = self.build_encoder(cfg)
            # Mirror the setup super().__init__() already applied to self.encoder.
            # gradient checkpointing especially: without it the query encoder
            # recomputes all layers at once instead of layer-by-layer.
            if bool(getattr(cfg.model, "gradient_checkpointing", False)) and hasattr(
                self.query_encoder, "gradient_checkpointing_enable"
            ):
                self.query_encoder.gradient_checkpointing_enable()
            # from_pretrained returns eval mode and HF gates gradient
            # checkpointing on module.training; self.encoder gets this in
            # super().__init__, so mirror it here.
            self.query_encoder.train()
            if self.freeze_doc_encoder:
                # postings(q) = sum_{j in q} df_j * N, and df_j is a property of
                # the DOCUMENT index. Holding the doc encoder fixed keeps the
                # existing 113M-doc index byte-valid, so a sparser query encoder
                # still cuts retrieval cost with NO reindex — deployment becomes
                # "swap the query encoder" instead of a ~6.5h re-encode.
                for parameter in self.encoder.parameters():
                    parameter.requires_grad_(False)
                self.encoder.eval()
        self.df_ema_momentum = float(getattr(cfg.model, "df_ema_momentum", 0.99))
        self._df_seen = False
        # persistent=False is REQUIRED: train.py warm-starts via a strict
        # load_state_dict from v1a_best.ckpt, and a new persistent buffer would
        # raise on the missing key. Training from scratch degenerates, so
        # breaking the warm start silently costs a whole run.
        self.register_buffer(
            "df_ema", torch.zeros(vocab_size, dtype=torch.float32), persistent=False
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

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "freeze_doc_encoder", False):
            # Lightning calls model.train() each epoch, which would re-enable
            # dropout in the frozen encoder and make the "fixed" doc vectors
            # drift between epochs.
            self.encoder.eval()
        return self

    def forward(self, query_inputs, offer_inputs):
        # Overridden so the query side can route to the untied query encoder.
        query_embeddings = self.encode(query_inputs, is_query=True)
        offer_embeddings = self.encode(offer_inputs, is_query=False)
        scores = (query_embeddings * offer_embeddings).sum(dim=1)
        return query_embeddings, offer_embeddings, scores

    def encode(self, inputs, is_query=None):
        """is_query defaults None so every existing call site keeps the doc
        encoder; only paths that know they hold queries opt in. With
        untied_encoders=False (the default) both paths are the same weights and
        behaviour is unchanged."""
        if self.untied_encoders and is_query is None:
            raise ValueError(
                "This model has untied encoders, so encode() needs an explicit "
                "is_query=True/False. A caller that omits it would silently "
                "encode queries with the DOCUMENT encoder and produce "
                "wrong-but-plausible vectors."
            )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        encoder = (
            self.query_encoder
            if (is_query and self.untied_encoders)
            else self.encoder
        )
        # A frozen doc encoder produces CONSTANTS, so build no graph at all.
        # Without this the doc side still enters torch.utils.checkpoint, which
        # with no grad-requiring parameter degenerates into a passthrough that
        # RETAINS the [batch, seq, vocab] activation chain rather than
        # recomputing it — freezing then costs more memory than not freezing,
        # and OOMs an 80GB H100 at batch 512. The contrastive loss still
        # differentiates correctly: gradients flow through the query side.
        if self.freeze_doc_encoder and not is_query:
            with torch.no_grad():
                representations = self._encode_representations(
                    input_ids, attention_mask, encoder
                )
            self.assert_finite(representations, "splade_representations")
            return representations
        # The [batch, seq, vocab] logits and each elementwise intermediate
        # are ~8 GB at batch 512; without checkpointing the autograd graph
        # holds several of them at once and OOMs an 80 GB H100.
        if self.training and torch.is_grad_enabled():
            representations = torch.utils.checkpoint.checkpoint(
                self._encode_representations,
                input_ids,
                attention_mask,
                encoder,
                use_reentrant=False,
            )
        else:
            representations = self._encode_representations(
                input_ids, attention_mask, encoder
            )
        self.assert_finite(representations, "splade_representations")
        return representations

    def _encode_representations(self, input_ids, attention_mask, encoder=None):
        encoder = encoder if encoder is not None else self.encoder
        outputs = encoder(
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
                teacher_query_embeddings = teacher.encode(batch["query_inputs"], is_query=True)
                teacher_offer_embeddings = teacher.encode(batch["offer_inputs"], is_query=False)
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
        self._update_document_frequency(offer_embeddings)
        flops_q = self._regularize(unique_query_embeddings, self.reg_type_q)
        flops_d = self._regularize(offer_embeddings, self.reg_type_d)
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

    def _update_document_frequency(self, offer_embeddings):
        """EMA of per-dimension document frequency over the doc side of the batch.

        Document frequency is a doc-side property, but the weight it produces is
        applied to BOTH regularizer terms: the query's high-df tokens are
        precisely what inflates the candidate set, so they must be penalized on
        the query side too.
        """
        # Maintain the EMA even when no df regularizer is active, so the
        # reg_df_flops_* diagnostics are real measurements. Previously this
        # returned early on non-df arms, leaving df_ema all zeros; the lambda
        # calibration probe (reg_type=flops) then logged
        # sigmoid(alpha*(0-beta)) * flops = 0.0392 * flops and that artifact was
        # mistaken for the df_flops magnitude, setting every df arm's lambda ~25x
        # too high.
        #
        # Train batches only: compute_loss also runs in validation_step, and with
        # num_sanity_val_steps=2 the first seed would come from a sanity-check
        # batch, after which a full val pass would drag the EMA toward the
        # validation corpus's df — making the training penalty depend on val set
        # size and ordering.
        if not self.training:
            return
        with torch.no_grad():
            batch_df = (offer_embeddings > 0).to(self.df_ema.dtype).mean(dim=0)
            if self._df_seen:
                self.df_ema.mul_(self.df_ema_momentum).add_(
                    batch_df, alpha=1.0 - self.df_ema_momentum
                )
            else:
                # Seed from the first batch; starting at zeros would leave every
                # dimension unpenalized for the first few hundred steps, which is
                # most of a short screening run.
                self.df_ema.copy_(batch_df)
                self._df_seen = True

    def _df_flops(self, representations):
        """Single implementation used by BOTH the loss and the diagnostics.

        The two used to be separate, so `df_activation=paper` was optimised while
        the logged magnitude was still the sigmoid form — the two differ by ~270x
        at df=2%, so the paper arm's real pressure was never observed.
        """
        document_frequency = self.df_ema.to(representations.dtype)
        if self.df_activation == "paper":
            weights = df_activation_paper(
                document_frequency, self.df_half, self.df_sharp
            )
            return (weights * representations.abs().mean(dim=0) ** 2).sum()
        return df_flops_regularizer(
            representations, document_frequency, self.df_alpha, self.df_beta
        )

    def _regularize(self, representations, reg_type):
        if reg_type == "flops":
            return flops_regularizer(representations)
        if reg_type == "l1":
            return l1_regularizer(representations)
        if reg_type == "df_flops":
            return self._df_flops(representations)
        raise ValueError(
            f"Unsupported reg_type: {reg_type}. Expected 'flops', 'l1' or 'df_flops'."
        )

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
        # Log every regularizer's raw magnitude regardless of which is active.
        # L1 is unsquared so its natural lambda sits 1-2 orders below FLOPS;
        # without these a lambda-calibration probe would have to be re-run once
        # per reg_type, and guessing lambda yields either a no-op or a
        # degenerate model. Cheap: three reductions over [batch, vocab].
        with torch.no_grad():
            for side, representations in (
                ("q", unique_query_embeddings),
                ("d", offer_embeddings),
            ):
                abs_mean = representations.abs().mean(dim=0)
                mean_sq_all = abs_mean ** 2
                stats[f"reg_l1_{side}"] = abs_mean.sum()
                stats[f"reg_flops_{side}"] = mean_sq_all.sum()
                # Only meaningful once df_ema holds a real statistic; logging it
                # against an all-zero df yields sigmoid(-alpha*beta)*flops, which
                # is a constant rescale of FLOPS masquerading as a measurement.
                if self._df_seen:
                    df = self.df_ema.to(representations.dtype)
                    mean_sq = mean_sq_all
                    # Log BOTH activations so a single lambda-calibration probe
                    # covers the sigmoid and paper arms; they differ by ~270x at
                    # df=2%, so calibrating one against the other is meaningless.
                    stats[f"reg_df_flops_{side}"] = (
                        torch.sigmoid(self.df_alpha * (df - self.df_beta)) * mean_sq
                    ).sum()
                    stats[f"reg_df_paper_{side}"] = (
                        df_activation_paper(df, self.df_half, self.df_sharp) * mean_sq
                    ).sum()
                    # Share of FLOPS mass the df weighting actually keeps.
                    # ~1.0 means the regularizer is indistinguishable from plain
                    # FLOPS; this single number would have caught the 0.039x
                    # artifact immediately.
                    total = mean_sq_all.sum().clamp(min=1e-12)
                    stats[f"df_mass_share_{side}"] = (
                        stats[f"reg_df_flops_{side}"] / total
                    )
                    stats[f"df_paper_mass_share_{side}"] = (
                        stats[f"reg_df_paper_{side}"] / total
                    )
                    # df percentiles make df_beta / df_half auditable against the
                    # distribution they are actually applied to (batch-level
                    # untruncated doc df), which is NOT the top-256 corpus df they
                    # were calibrated on.
                    if side == "d":
                        q = torch.tensor([0.5, 0.9, 0.99], device=df.device,
                                         dtype=torch.float32)
                        p10, p90, p99 = torch.quantile(df.float(), q).tolist()
                        stats["df_ema_p50"] = p10
                        stats["df_ema_p90"] = p90
                        stats["df_ema_p99"] = p99
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

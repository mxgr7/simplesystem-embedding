import random

import torch
from torch.utils.data import IterableDataset


SYNTHETIC_NEGATIVE_LABEL = "SyntheticNegative"
HARD_NEGATIVE_LABEL = "HardNegative"
SEMI_HARD_NEGATIVE_LABEL = "SemiHardNegative"


def build_synthetic_negative_record(
    anchor_query_id, anchor_query_text, synthetic_offer
):
    if synthetic_offer["offer_source_query_id"] == anchor_query_id:
        raise ValueError("Synthetic negative offer must come from a foreign query")

    return {
        "query_id": anchor_query_id,
        "query_text": anchor_query_text,
        "offer_id": synthetic_offer["offer_id"],
        "offer_text": synthetic_offer["offer_text"],
        "label": 0.0,
        "raw_label": SYNTHETIC_NEGATIVE_LABEL,
    }


def build_hard_negative_record(anchor_query_id, anchor_query_text, hard_negative):
    return {
        "query_id": anchor_query_id,
        "query_text": anchor_query_text,
        "offer_id": hard_negative["offer_id"],
        "offer_text": hard_negative["offer_text"],
        "label": 0.0,
        "raw_label": HARD_NEGATIVE_LABEL,
    }


def build_semi_hard_negative_record(
    anchor_query_id, anchor_query_text, semi_hard_negative
):
    return {
        "query_id": anchor_query_id,
        "query_text": anchor_query_text,
        "offer_id": semi_hard_negative["offer_id"],
        "offer_text": semi_hard_negative["offer_text"],
        "label": 0.0,
        "raw_label": SEMI_HARD_NEGATIVE_LABEL,
    }


def build_batch_stats(records, anchor_query_id=None):
    positive_count = 0
    same_query_negative_count = 0
    cross_query_negative_count = 0
    hard_negative_count = 0
    semi_hard_negative_count = 0

    for record in records:
        if record["label"] > 0.5:
            positive_count += 1
            continue

        if record["raw_label"] == HARD_NEGATIVE_LABEL:
            hard_negative_count += 1
            continue

        if record["raw_label"] == SEMI_HARD_NEGATIVE_LABEL:
            semi_hard_negative_count += 1
            continue

        if record["raw_label"] == SYNTHETIC_NEGATIVE_LABEL:
            cross_query_negative_count += 1
            continue

        same_query_negative_count += 1

    batch_stats = {
        "positive_count": positive_count,
        "same_query_negative_count": same_query_negative_count,
        "cross_query_negative_count": cross_query_negative_count,
        "hard_negative_count": hard_negative_count,
        "semi_hard_negative_count": semi_hard_negative_count,
    }

    if anchor_query_id is not None:
        batch_stats["anchor_query_id"] = anchor_query_id

    return batch_stats


class QuerySamplingBuilder:
    def __init__(
        self,
        positive_records_by_query,
        negative_records_by_query,
        eligible_query_ids,
        synthetic_negative_offer_pool,
        n_pos_samples_per_query,
        n_neg_samples_per_query,
        seed,
        hard_negative_records_by_query=None,
        semi_hard_negative_records_by_query=None,
    ):
        self.positive_records_by_query = {
            query_id: list(records)
            for query_id, records in positive_records_by_query.items()
        }
        self.negative_records_by_query = {
            query_id: list(records)
            for query_id, records in negative_records_by_query.items()
        }
        self.eligible_query_ids = list(eligible_query_ids)
        self.synthetic_offer_pool = list(synthetic_negative_offer_pool)
        self.n_pos_samples_per_query = int(n_pos_samples_per_query)
        self.n_neg_samples_per_query = int(n_neg_samples_per_query)
        self.randomizer = random.Random(seed)

        self._validate_eligible_queries()

        self._query_text_by_query_id = self._build_query_text_index()
        self.hard_negative_records_by_query = self._prebuild_mined_records(
            hard_negative_records_by_query, build_hard_negative_record
        )
        self.semi_hard_negative_records_by_query = self._prebuild_mined_records(
            semi_hard_negative_records_by_query, build_semi_hard_negative_record
        )

    def _build_query_text_index(self):
        index = {}
        for query_id, records in self.positive_records_by_query.items():
            if records:
                index[query_id] = records[0]["query_text"]
        for query_id, records in self.negative_records_by_query.items():
            if query_id not in index and records:
                index[query_id] = records[0]["query_text"]
        return index

    def _prebuild_mined_records(self, raw_by_query, builder_fn):
        prebuilt = {}
        if not raw_by_query:
            return prebuilt
        for query_id, raw_records in raw_by_query.items():
            query_text = self._query_text_by_query_id.get(query_id)
            if query_text is None or not raw_records:
                continue
            prebuilt[query_id] = [
                builder_fn(query_id, query_text, raw) for raw in raw_records
            ]
        return prebuilt

    def _validate_eligible_queries(self):
        if not self.eligible_query_ids:
            raise ValueError(
                "QuerySamplingBuilder requires at least one eligible query"
            )

        for query_id in self.eligible_query_ids:
            positive_records = self.positive_records_by_query.get(query_id, [])
            if len(positive_records) < self.n_pos_samples_per_query:
                raise ValueError(
                    "Eligible query does not satisfy n_pos_samples_per_query: "
                    f"{query_id}"
                )

    def _validate_anchor_query_id(self, anchor_query_id):
        if anchor_query_id not in self.eligible_query_ids:
            raise ValueError(f"Anchor query is not eligible: {anchor_query_id}")

    def _build_query_minimum_records(self, anchor_query_id):
        self._validate_anchor_query_id(anchor_query_id)
        anchor_positive_records = self._shuffled_copy(
            self.positive_records_by_query.get(anchor_query_id, [])
        )
        anchor_negative_records = self._shuffled_copy(
            self.negative_records_by_query.get(anchor_query_id, [])
        )
        anchor_hard_negative_records = self._shuffled_copy(
            self.hard_negative_records_by_query.get(anchor_query_id, [])
        )
        anchor_semi_hard_negative_records = self._shuffled_copy(
            self.semi_hard_negative_records_by_query.get(anchor_query_id, [])
        )
        anchor_query_text = self._resolve_anchor_query_text(anchor_query_id)

        if len(anchor_positive_records) < self.n_pos_samples_per_query:
            raise ValueError(
                f"Anchor query does not have enough positive rows: {anchor_query_id}"
            )

        records = []
        used_offer_ids = set()

        for _ in range(self.n_pos_samples_per_query):
            self._append_real_record(
                records, anchor_positive_records.pop(), used_offer_ids
            )

        for _ in range(self.n_neg_samples_per_query):
            if anchor_negative_records:
                self._append_real_record(
                    records,
                    anchor_negative_records.pop(),
                    used_offer_ids,
                )
                continue

            if anchor_hard_negative_records and self._append_prebuilt_negative(
                records, anchor_hard_negative_records, used_offer_ids
            ):
                continue

            if anchor_semi_hard_negative_records and self._append_prebuilt_negative(
                records, anchor_semi_hard_negative_records, used_offer_ids
            ):
                continue

            self._append_synthetic_negative(
                records,
                anchor_query_id,
                anchor_query_text,
                used_offer_ids,
            )

        return {
            "records": records,
            "remaining_positive_records": anchor_positive_records,
            "remaining_negative_records": anchor_negative_records,
            "remaining_hard_negative_records": anchor_hard_negative_records,
            "remaining_semi_hard_negative_records": anchor_semi_hard_negative_records,
            "anchor_query_text": anchor_query_text,
            "used_offer_ids": used_offer_ids,
        }

    def _append_real_record(self, batch, record, used_offer_ids):
        batch.append(record)
        used_offer_ids.add(record["offer_id"])

    def _append_prebuilt_negative(self, batch, candidates, used_offer_ids):
        while candidates:
            candidate = candidates.pop()
            if candidate["offer_id"] in used_offer_ids:
                continue
            batch.append(candidate)
            used_offer_ids.add(candidate["offer_id"])
            return True

        return False

    def _append_synthetic_negative(
        self,
        batch,
        anchor_query_id,
        anchor_query_text,
        used_offer_ids,
    ):
        synthetic_offer = self._sample_synthetic_offer(
            anchor_query_id, used_offer_ids
        )

        batch.append(
            build_synthetic_negative_record(
                anchor_query_id, anchor_query_text, synthetic_offer
            )
        )
        used_offer_ids.add(synthetic_offer["offer_id"])

    def _sample_synthetic_offer(self, anchor_query_id, used_offer_ids):
        pool = self.synthetic_offer_pool
        pool_size = len(pool)
        if pool_size == 0:
            raise ValueError(
                "No synthetic offers available to sample a negative from"
            )
        # Rejection-sample from the full pool by random index, so we don't
        # copy+shuffle the entire pool on every call.
        max_attempts = max(pool_size * 2, 1024)
        for _ in range(max_attempts):
            offer = pool[self.randomizer.randrange(pool_size)]
            if offer["offer_source_query_id"] == anchor_query_id:
                continue
            if offer["offer_id"] in used_offer_ids:
                continue
            return offer

        raise ValueError(
            "No unique non-anchor query offers remain for synthetic negatives"
        )

    def _resolve_anchor_query_text(self, anchor_query_id):
        query_text = self._query_text_by_query_id.get(anchor_query_id)
        if query_text is None:
            raise ValueError(f"Anchor query has no records: {anchor_query_id}")
        return query_text

    def _shuffled_copy(self, items):
        shuffled = list(items)
        self.randomizer.shuffle(shuffled)
        return shuffled


class AnchorQueryBatchBuilder(QuerySamplingBuilder):
    def __init__(
        self,
        positive_records_by_query,
        negative_records_by_query,
        eligible_query_ids,
        synthetic_negative_offer_pool,
        batch_size,
        n_pos_samples_per_query,
        n_neg_samples_per_query,
        seed,
        hard_negative_records_by_query=None,
        semi_hard_negative_records_by_query=None,
    ):
        super().__init__(
            positive_records_by_query=positive_records_by_query,
            negative_records_by_query=negative_records_by_query,
            eligible_query_ids=eligible_query_ids,
            synthetic_negative_offer_pool=synthetic_negative_offer_pool,
            n_pos_samples_per_query=n_pos_samples_per_query,
            n_neg_samples_per_query=n_neg_samples_per_query,
            seed=seed,
            hard_negative_records_by_query=hard_negative_records_by_query,
            semi_hard_negative_records_by_query=semi_hard_negative_records_by_query,
        )
        self.batch_size = int(batch_size)
        self._validate_config()

    def build_batch(self, anchor_query_id=None):
        if anchor_query_id is None:
            anchor_query_id = self.randomizer.choice(self.eligible_query_ids)

        build_context = self._build_query_minimum_records(anchor_query_id)
        batch = build_context["records"]
        anchor_positive_records = build_context["remaining_positive_records"]
        anchor_negative_records = build_context["remaining_negative_records"]
        anchor_hard_negative_records = build_context["remaining_hard_negative_records"]
        anchor_semi_hard_negative_records = build_context[
            "remaining_semi_hard_negative_records"
        ]
        anchor_query_text = build_context["anchor_query_text"]
        used_offer_ids = build_context["used_offer_ids"]

        while len(batch) < self.batch_size and (
            anchor_positive_records or anchor_negative_records
        ):
            if anchor_positive_records and len(batch) < self.batch_size:
                self._append_real_record(
                    batch,
                    anchor_positive_records.pop(),
                    used_offer_ids,
                )

            if anchor_negative_records and len(batch) < self.batch_size:
                self._append_real_record(
                    batch,
                    anchor_negative_records.pop(),
                    used_offer_ids,
                )

        while len(batch) < self.batch_size and anchor_hard_negative_records:
            appended = self._append_prebuilt_negative(
                batch,
                anchor_hard_negative_records,
                used_offer_ids,
            )
            if not appended:
                break

        while len(batch) < self.batch_size and anchor_semi_hard_negative_records:
            appended = self._append_prebuilt_negative(
                batch,
                anchor_semi_hard_negative_records,
                used_offer_ids,
            )
            if not appended:
                break

        while len(batch) < self.batch_size:
            self._append_synthetic_negative(
                batch,
                anchor_query_id,
                anchor_query_text,
                used_offer_ids,
            )

        return {
            "records": batch,
            "batch_stats": build_batch_stats(batch, anchor_query_id=anchor_query_id),
        }

    def _validate_config(self):
        minimum_batch_size = self.n_pos_samples_per_query + self.n_neg_samples_per_query
        if self.batch_size < minimum_batch_size:
            raise ValueError(
                "batch_size must be at least n_pos_samples_per_query + "
                "n_neg_samples_per_query"
            )


class RandomQueryPoolBuilder(QuerySamplingBuilder):
    def build_pool(self):
        records = []

        for query_id in self.eligible_query_ids:
            query_text = self._query_text_by_query_id[query_id]
            real_positives = self.positive_records_by_query.get(query_id, [])
            real_negatives = self.negative_records_by_query.get(query_id, [])
            hard_negatives = self.hard_negative_records_by_query.get(query_id, [])
            semi_hard_negatives = self.semi_hard_negative_records_by_query.get(
                query_id, []
            )

            used_offer_ids = set()

            for record in real_positives:
                offer_id = record["offer_id"]
                if offer_id in used_offer_ids:
                    continue
                used_offer_ids.add(offer_id)
                records.append(record)

            real_negatives_added = 0
            for record in real_negatives:
                offer_id = record["offer_id"]
                if offer_id in used_offer_ids:
                    continue
                used_offer_ids.add(offer_id)
                records.append(record)
                real_negatives_added += 1

            hard_negatives_added = 0
            for record in hard_negatives:
                offer_id = record["offer_id"]
                if offer_id in used_offer_ids:
                    continue
                used_offer_ids.add(offer_id)
                records.append(record)
                hard_negatives_added += 1

            semi_hard_negatives_added = 0
            for record in semi_hard_negatives:
                offer_id = record["offer_id"]
                if offer_id in used_offer_ids:
                    continue
                used_offer_ids.add(offer_id)
                records.append(record)
                semi_hard_negatives_added += 1

            available_negatives = (
                real_negatives_added
                + hard_negatives_added
                + semi_hard_negatives_added
            )
            synthetic_needed = self.n_neg_samples_per_query - available_negatives
            for _ in range(synthetic_needed):
                synthetic_offer = self._sample_synthetic_offer(
                    query_id, used_offer_ids
                )
                records.append(
                    build_synthetic_negative_record(
                        query_id, query_text, synthetic_offer
                    )
                )
                used_offer_ids.add(synthetic_offer["offer_id"])

        self.randomizer.shuffle(records)
        return records


class AnchorQueryBatchDataset(IterableDataset):
    def __init__(self, batch_builder, batches_per_epoch):
        self.batch_builder = batch_builder
        self.batches_per_epoch = int(batches_per_epoch)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        if info is None:
            n_batches = self.batches_per_epoch
        else:
            base, rem = divmod(self.batches_per_epoch, info.num_workers)
            n_batches = base + (1 if info.id < rem else 0)
            # torch.initial_seed() is derived from (base_seed, worker_id) and
            # rotated per epoch by the DataLoader, so this gives each worker a
            # disjoint, epoch-varying RNG stream. Without this, every worker
            # replays the identical sequence from the pickled builder state.
            self.batch_builder.randomizer.seed(torch.initial_seed())
        for _ in range(n_batches):
            yield self.batch_builder.build_batch()

    def __len__(self):
        return self.batches_per_epoch

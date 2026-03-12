import random

from torch.utils.data import IterableDataset


SYNTHETIC_NEGATIVE_LABEL = "SyntheticNegative"


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


def build_batch_stats(records, anchor_query_id=None):
    positive_count = 0
    same_query_negative_count = 0
    cross_query_negative_count = 0

    for record in records:
        if record["label"] > 0.5:
            positive_count += 1
            continue

        if record["raw_label"] == SYNTHETIC_NEGATIVE_LABEL:
            cross_query_negative_count += 1
            continue

        same_query_negative_count += 1

    batch_stats = {
        "positive_count": positive_count,
        "same_query_negative_count": same_query_negative_count,
        "cross_query_negative_count": cross_query_negative_count,
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
        self.synthetic_offer_pool = [
            dict(record) for record in synthetic_negative_offer_pool
        ]
        self.n_pos_samples_per_query = int(n_pos_samples_per_query)
        self.n_neg_samples_per_query = int(n_neg_samples_per_query)
        self.randomizer = random.Random(seed)

        self._validate_eligible_queries()

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
        anchor_query_text = self._resolve_anchor_query_text(anchor_query_id)
        synthetic_offer_candidates = self._shuffled_copy(self.synthetic_offer_pool)

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

            self._append_synthetic_negative(
                records,
                anchor_query_id,
                anchor_query_text,
                synthetic_offer_candidates,
                used_offer_ids,
            )

        return {
            "records": records,
            "remaining_positive_records": anchor_positive_records,
            "remaining_negative_records": anchor_negative_records,
            "anchor_query_text": anchor_query_text,
            "synthetic_offer_candidates": synthetic_offer_candidates,
            "used_offer_ids": used_offer_ids,
        }

    def _append_real_record(self, batch, record, used_offer_ids):
        batch.append(dict(record))
        used_offer_ids.add(record["offer_id"])

    def _append_synthetic_negative(
        self,
        batch,
        anchor_query_id,
        anchor_query_text,
        synthetic_offer_candidates,
        used_offer_ids,
    ):
        synthetic_offer = self._sample_synthetic_offer(
            anchor_query_id,
            synthetic_offer_candidates,
            used_offer_ids,
        )

        batch.append(
            build_synthetic_negative_record(
                anchor_query_id, anchor_query_text, synthetic_offer
            )
        )
        used_offer_ids.add(synthetic_offer["offer_id"])

    def _sample_synthetic_offer(
        self, anchor_query_id, synthetic_offer_candidates, used_offer_ids
    ):
        while synthetic_offer_candidates:
            offer = synthetic_offer_candidates.pop()
            if offer["offer_source_query_id"] == anchor_query_id:
                continue
            if offer["offer_id"] in used_offer_ids:
                continue
            return offer

        raise ValueError(
            "No unique non-anchor query offers remain for synthetic negatives"
        )

    def _resolve_anchor_query_text(self, anchor_query_id):
        positive_records = self.positive_records_by_query.get(anchor_query_id, [])
        if positive_records:
            return positive_records[0]["query_text"]

        negative_records = self.negative_records_by_query.get(anchor_query_id, [])
        if negative_records:
            return negative_records[0]["query_text"]

        raise ValueError(f"Anchor query has no records: {anchor_query_id}")

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
    ):
        super().__init__(
            positive_records_by_query=positive_records_by_query,
            negative_records_by_query=negative_records_by_query,
            eligible_query_ids=eligible_query_ids,
            synthetic_negative_offer_pool=synthetic_negative_offer_pool,
            n_pos_samples_per_query=n_pos_samples_per_query,
            n_neg_samples_per_query=n_neg_samples_per_query,
            seed=seed,
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
        anchor_query_text = build_context["anchor_query_text"]
        synthetic_offer_candidates = build_context["synthetic_offer_candidates"]
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

        while len(batch) < self.batch_size:
            self._append_synthetic_negative(
                batch,
                anchor_query_id,
                anchor_query_text,
                synthetic_offer_candidates,
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
        query_ids = self._shuffled_copy(self.eligible_query_ids)

        for query_id in query_ids:
            query_records = self._build_query_records(query_id)
            records.extend(query_records)

        self.randomizer.shuffle(records)
        return records

    def _build_query_records(self, query_id):
        build_context = self._build_query_minimum_records(query_id)
        records = build_context["records"]
        remaining_positive_records = build_context["remaining_positive_records"]
        remaining_negative_records = build_context["remaining_negative_records"]

        extra_records = []
        for record in remaining_positive_records:
            extra_records.append(dict(record))

        for record in remaining_negative_records:
            extra_records.append(dict(record))

        self.randomizer.shuffle(extra_records)
        records.extend(extra_records)
        return records


class AnchorQueryBatchDataset(IterableDataset):
    def __init__(self, batch_builder, batches_per_epoch):
        self.batch_builder = batch_builder
        self.batches_per_epoch = int(batches_per_epoch)

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            yield self.batch_builder.build_batch()

    def __len__(self):
        return self.batches_per_epoch

import math


RELEVANCE_GAINS = {
    "Exact": 1.0,
    "Substitute": 0.1,
    "Complement": 0.01,
    "Irrelevant": 0.0,
}

EXACT_LABEL = "Exact"
DEFAULT_BINARY_RELEVANT_LABELS = (EXACT_LABEL,)


def compute_ranking_metrics(rows, ks=(1, 5, 10)):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["query_id"], []).append(row)

    ndcg_totals = {k: 0.0 for k in ks}
    eligible_queries = 0

    for items in grouped.values():
        scored_items = []

        for item in items:
            raw_label = item["raw_label"]
            if raw_label not in RELEVANCE_GAINS:
                raise ValueError(f"Unknown relevance label: {raw_label}")

            scored_items.append(
                {
                    "score": item["score"],
                    "gain": RELEVANCE_GAINS[raw_label],
                }
            )

        ranked = sorted(scored_items, key=lambda item: item["score"], reverse=True)
        ideal = sorted(scored_items, key=lambda item: item["gain"], reverse=True)

        query_ndcgs = {}
        has_positive_gain = False

        for k in ks:
            dcg = compute_dcg(ranked, k)
            idcg = compute_dcg(ideal, k)
            if idcg > 0:
                has_positive_gain = True
                query_ndcgs[k] = dcg / idcg
            else:
                query_ndcgs[k] = 0.0

        if not has_positive_gain:
            continue

        eligible_queries += 1
        for k in ks:
            ndcg_totals[k] += query_ndcgs[k]

    metrics = {
        "eligible_queries": float(eligible_queries),
        "evaluated_queries": float(len(grouped)),
    }

    if eligible_queries == 0:
        for k in ks:
            metrics[f"ndcg@{k}"] = 0.0
        return metrics

    for k in ks:
        metrics[f"ndcg@{k}"] = ndcg_totals[k] / eligible_queries
    return metrics


def compute_exact_retrieval_metrics(
    rows,
    ks=(1, 5, 10),
    evaluated_query_ids=None,
    eligible_query_ids=None,
):
    return compute_binary_retrieval_metrics(
        rows,
        ks=ks,
        evaluated_query_ids=evaluated_query_ids,
        eligible_query_ids=eligible_query_ids,
        relevant_labels=DEFAULT_BINARY_RELEVANT_LABELS,
        metric_prefix="exact",
    )


def compute_binary_retrieval_metrics(
    rows,
    ks=(1, 5, 10),
    evaluated_query_ids=None,
    eligible_query_ids=None,
    relevant_labels=None,
    metric_prefix="",
):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["query_id"], []).append(row)

    relevant_label_set = resolve_relevant_label_set(relevant_labels)

    if evaluated_query_ids is None:
        evaluated_query_ids = list(grouped)
    else:
        evaluated_query_ids = list(dict.fromkeys(evaluated_query_ids))

    if eligible_query_ids is None:
        eligible_query_ids = [
            query_id
            for query_id, items in grouped.items()
            if any(item.get("raw_label") in relevant_label_set for item in items)
        ]
    else:
        eligible_query_ids = list(dict.fromkeys(eligible_query_ids))

    eligible_query_id_set = set(eligible_query_ids)
    evaluated_eligible_query_ids = [
        query_id
        for query_id in evaluated_query_ids
        if query_id in eligible_query_id_set
    ]

    success_totals = {k: 0.0 for k in ks}
    reciprocal_rank_total = 0.0

    for query_id in evaluated_eligible_query_ids:
        ranked_items = sort_ranked_items(grouped.get(query_id, []))
        first_exact_rank = None

        for index, item in enumerate(ranked_items, start=1):
            if item.get("raw_label") in relevant_label_set:
                first_exact_rank = index
                break

        if first_exact_rank is None:
            continue

        reciprocal_rank_total += 1.0 / float(first_exact_rank)
        for k in ks:
            if first_exact_rank <= int(k):
                success_totals[k] += 1.0

    eligible_queries = len(evaluated_eligible_query_ids)
    metrics = {
        "eligible_queries": float(eligible_queries),
        "evaluated_queries": float(len(evaluated_query_ids)),
        resolve_mrr_metric_name(metric_prefix): 0.0,
    }

    for k in ks:
        metric_name = resolve_recall_metric_name(k, metric_prefix)
        metrics[metric_name] = 0.0

    if eligible_queries == 0:
        return metrics

    metrics[resolve_mrr_metric_name(metric_prefix)] = reciprocal_rank_total / float(
        eligible_queries
    )
    for k in ks:
        metric_name = resolve_recall_metric_name(k, metric_prefix)
        metrics[metric_name] = success_totals[k] / float(eligible_queries)

    return metrics


def compute_precision_metrics(
    rows,
    ks=(1, 5, 10),
    evaluated_query_ids=None,
    eligible_query_ids=None,
    relevant_labels=None,
    metric_prefix="",
):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["query_id"], []).append(row)

    relevant_label_set = resolve_relevant_label_set(relevant_labels)

    if evaluated_query_ids is None:
        evaluated_query_ids = list(grouped)
    else:
        evaluated_query_ids = list(dict.fromkeys(evaluated_query_ids))

    if eligible_query_ids is None:
        eligible_query_ids = [
            query_id
            for query_id, items in grouped.items()
            if any(item.get("raw_label") in relevant_label_set for item in items)
        ]
    else:
        eligible_query_ids = list(dict.fromkeys(eligible_query_ids))

    eligible_query_id_set = set(eligible_query_ids)
    evaluated_eligible_query_ids = [
        query_id
        for query_id in evaluated_query_ids
        if query_id in eligible_query_id_set
    ]

    precision_totals = {k: 0.0 for k in ks}
    eligible_queries = len(evaluated_eligible_query_ids)
    metrics = {
        "eligible_queries": float(eligible_queries),
        "evaluated_queries": float(len(evaluated_query_ids)),
    }

    for k in ks:
        metrics[resolve_precision_metric_name(k, metric_prefix)] = 0.0

    if eligible_queries == 0:
        return metrics

    for query_id in evaluated_eligible_query_ids:
        ranked_items = sort_ranked_items(grouped.get(query_id, []))
        for k in ks:
            relevant_hits = 0.0
            for item in ranked_items[: int(k)]:
                if item.get("raw_label") in relevant_label_set:
                    relevant_hits += 1.0
            precision_totals[k] += relevant_hits / float(int(k))

    for k in ks:
        metric_name = resolve_precision_metric_name(k, metric_prefix)
        metrics[metric_name] = precision_totals[k] / float(eligible_queries)

    return metrics


def compute_dcg(items, k):
    dcg = 0.0
    for index, item in enumerate(items[:k], start=1):
        dcg += item["gain"] / math.log2(index + 1)
    return dcg


def resolve_exact_metric_name(k):
    if int(k) == 1:
        return "exact_success@1"

    return f"exact_recall@{int(k)}"


def resolve_recall_metric_name(k, metric_prefix=""):
    if metric_prefix == "exact":
        return resolve_exact_metric_name(k)

    metric_name = f"recall@{int(k)}"
    if not metric_prefix:
        return metric_name
    return f"{metric_prefix}_{metric_name}"


def resolve_mrr_metric_name(metric_prefix=""):
    if metric_prefix == "exact":
        return "exact_mrr"

    if not metric_prefix:
        return "mrr"
    return f"{metric_prefix}_mrr"


def resolve_precision_metric_name(k, metric_prefix=""):
    metric_name = f"precision@{int(k)}"
    if not metric_prefix:
        return metric_name
    return f"{metric_prefix}_{metric_name}"


def resolve_relevant_label_set(relevant_labels):
    if relevant_labels is None:
        relevant_labels = DEFAULT_BINARY_RELEVANT_LABELS

    return {str(label).strip() for label in relevant_labels if str(label).strip()}


def sort_ranked_items(items):
    return sorted(items, key=build_rank_sort_key)


def build_rank_sort_key(item):
    rank = item.get("rank")
    if rank is not None:
        return (0, int(rank), 0.0)

    score = float(item.get("score", 0.0))
    return (1, 0, -score)

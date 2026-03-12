import math


RELEVANCE_GAINS = {
    "Exact": 1.0,
    "Substitute": 0.1,
    "Complement": 0.01,
    "Irrelevant": 0.0,
}

EXACT_LABEL = "Exact"


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
    grouped = {}
    for row in rows:
        grouped.setdefault(row["query_id"], []).append(row)

    if evaluated_query_ids is None:
        evaluated_query_ids = list(grouped)
    else:
        evaluated_query_ids = list(dict.fromkeys(evaluated_query_ids))

    if eligible_query_ids is None:
        eligible_query_ids = [
            query_id
            for query_id, items in grouped.items()
            if any(item.get("raw_label") == EXACT_LABEL for item in items)
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
            if item.get("raw_label") == EXACT_LABEL:
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
        "exact_mrr": 0.0,
    }

    for k in ks:
        metric_name = resolve_exact_metric_name(k)
        metrics[metric_name] = 0.0

    if eligible_queries == 0:
        return metrics

    metrics["exact_mrr"] = reciprocal_rank_total / float(eligible_queries)
    for k in ks:
        metric_name = resolve_exact_metric_name(k)
        metrics[metric_name] = success_totals[k] / float(eligible_queries)

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


def sort_ranked_items(items):
    return sorted(items, key=build_rank_sort_key)


def build_rank_sort_key(item):
    rank = item.get("rank")
    if rank is not None:
        return (0, int(rank), 0.0)

    score = float(item.get("score", 0.0))
    return (1, 0, -score)

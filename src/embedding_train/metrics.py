import math


RELEVANCE_GAINS = {
    "Exact": 1.0,
    "Substitute": 0.1,
    "Complement": 0.01,
    "Irrelevant": 0.0,
}


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


def compute_dcg(items, k):
    dcg = 0.0
    for index, item in enumerate(items[:k], start=1):
        dcg += item["gain"] / math.log2(index + 1)
    return dcg

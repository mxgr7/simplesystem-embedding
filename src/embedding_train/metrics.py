def compute_ranking_metrics(rows, ks=(1, 5, 10), mrr_k=10):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["query_id"], []).append(row)

    recall_hits = {k: 0 for k in ks}
    eligible_queries = 0
    mrr_total = 0.0

    for items in grouped.values():
        ranked = sorted(items, key=lambda item: item["score"], reverse=True)
        first_positive_rank = None

        for index, item in enumerate(ranked, start=1):
            if item["label"] > 0:
                first_positive_rank = index
                break

        if first_positive_rank is None:
            continue

        eligible_queries += 1

        for k in ks:
            if first_positive_rank <= k:
                recall_hits[k] += 1

        if first_positive_rank <= mrr_k:
            mrr_total += 1.0 / first_positive_rank

    metrics = {
        "eligible_queries": float(eligible_queries),
        "evaluated_queries": float(len(grouped)),
    }

    if eligible_queries == 0:
        for k in ks:
            metrics[f"recall@{k}"] = 0.0
        metrics[f"mrr@{mrr_k}"] = 0.0
        return metrics

    for k in ks:
        metrics[f"recall@{k}"] = recall_hits[k] / eligible_queries
    metrics[f"mrr@{mrr_k}"] = mrr_total / eligible_queries
    return metrics
